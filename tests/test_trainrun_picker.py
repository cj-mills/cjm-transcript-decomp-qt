"""Offscreen tests for the training-run picker (work item a13ebc66): the
cursor parks on the EFFECTIVE model (pick > pin > newest), j/k/enter drive
the native list, a pick hands the manifest to on_pick and closes, esc and
the header's close anchor cancel, the empty index is a no-op placeholder."""

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent, Qt, QUrl
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication

from cjm_transcript_decomp_qt.trainrun_picker import TrainingRunPicker


def _runs():
    return [{"run_id": "trainrun_20260827_001206_b1072a9f", "_path": "/ws/b",
             "classes": ["speech", "inhale"]},
            {"run_id": "trainrun_20260810_113513_f4717dc5", "_path": "/ws/f",
             "classes": ["speech", "inhale"]},
            {"run_id": "trainrun_20260731_002436_6f803b12", "_path": "/ws/6",
             "classes": ["speech", "inhale"]}]


def _key(key, text=""):
    return QKeyEvent(QEvent.KeyPress, key, Qt.NoModifier, text)


def test_picker_parks_cursor_on_effective_model_and_picks():
    QApplication.instance() or QApplication(sys.argv[:1])
    picked = []
    dlg = TrainingRunPicker(None, on_pick=picked.append)
    # pin = the middle run, no pick: the cursor parks on the pin row
    dlg.open_for(_runs(), pin="f4717dc5", pick=None)
    assert dlg.list.count() == 3 and dlg.list.currentRow() == 1
    assert "← pin" in dlg.list.item(1).text()
    assert "newest" in dlg.list.item(0).text()
    assert dlg.list.item(1).text().startswith("● ")
    dlg.keyPressEvent(_key(Qt.Key_J, "j"))
    assert dlg.list.currentRow() == 2
    for _ in range(3):
        dlg.keyPressEvent(_key(Qt.Key_K, "k"))
    assert dlg.list.currentRow() == 0                 # clamps at the head
    dlg.keyPressEvent(_key(Qt.Key_Return))
    assert [r["run_id"][-8:] for r in picked] == ["b1072a9f"]
    assert not dlg.isVisible()
    # reopened with a pick: it outranks the pin for the ● / cursor, the pin
    # row keeps its chip so the standing default stays visible
    dlg.open_for(_runs(), pin="f4717dc5", pick="/ws/6")
    assert dlg.list.currentRow() == 2
    assert dlg.list.item(2).text().startswith("● ")
    assert dlg.list.item(1).text().startswith("  ")
    assert "← pin" in dlg.list.item(1).text()
    # esc and the header's close anchor cancel without a pick
    dlg.keyPressEvent(_key(Qt.Key_Escape))
    assert len(picked) == 1 and not dlg.isVisible()
    dlg.open_for(_runs())
    assert dlg.list.currentRow() == 0                 # no pin, no pick: newest
    dlg._on_anchor(QUrl("close:"))
    assert len(picked) == 1 and not dlg.isVisible()
    # empty index: the placeholder row, enter is a no-op
    dlg.open_for([])
    assert dlg.list.count() == 1 and dlg.current_run() is None
    dlg.keyPressEvent(_key(Qt.Key_Return))
    assert len(picked) == 1
    dlg.close()


def test_picker_lifecycle_verbs_archive_toggle_and_delete_two_step():
    """Work item b20cb911 (DEC 14678fc9): archived runs stay hidden until h,
    paint dim with a ⌂ chip and are never pickable; a hands the cursored run
    + verb to on_lifecycle and repaints from the lists it returns (cursor
    kept); x arms on an archived row only and deletes on the second press,
    any other key cancels; the hook's note rides the foot."""
    QApplication.instance() or QApplication(sys.argv[:1])
    picked, calls = [], []
    state = {"runs": _runs()[:2],
             "archived": [dict(_runs()[2], _lifecycle="archived")]}

    def hook(run, verb):
        calls.append((run["run_id"][-8:], verb))
        keep = lambda rows: [r for r in rows if r["_path"] != run["_path"]]
        if verb == "archive":
            state["runs"] = keep(state["runs"])
            state["archived"] = state["archived"] + [dict(run, _lifecycle="archived")]
            note = f"…{run['run_id'][-8:]} archived"
        elif verb == "unarchive":
            state["archived"] = keep(state["archived"])
            state["runs"] = state["runs"] + [dict(run, _lifecycle="active")]
            note = f"…{run['run_id'][-8:]} active"
        else:
            state["archived"] = keep(state["archived"])
            note = f"deleted …{run['run_id'][-8:]}"
        return state["runs"], state["archived"], note

    dlg = TrainingRunPicker(None, on_pick=picked.append, on_lifecycle=hook)
    dlg.open_for(state["runs"], pin="f4717dc5", archived=state["archived"])
    assert dlg.list.count() == 2
    assert "1 archived (h shows)" in dlg.head.toPlainText()
    dlg.keyPressEvent(_key(Qt.Key_H, "h"))
    assert dlg.list.count() == 3 and "⌂ archived" in dlg.list.item(2).text()
    assert "1 archived (shown)" in dlg.head.toPlainText()
    # archived rows are not pickable
    dlg.list.setCurrentRow(2)
    dlg.keyPressEvent(_key(Qt.Key_Return))
    assert picked == [] and dlg.isVisible() and "not pickable" in dlg.foot.text()
    # a on an archived row unarchives; on an active row archives
    dlg.keyPressEvent(_key(Qt.Key_A, "a"))
    assert calls[-1] == ("6f803b12", "unarchive") and dlg.list.count() == 3
    assert "…6f803b12 active" in dlg.foot.text()
    assert "⌂ archived" not in dlg.list.item(2).text()
    dlg.list.setCurrentRow(0)
    dlg.keyPressEvent(_key(Qt.Key_A, "a"))
    assert calls[-1] == ("b1072a9f", "archive")
    assert dlg.list.count() == 3          # shown: 2 active + the fresh archive
    assert "⌂ archived" in dlg.list.item(2).text()
    # the cursor stays put; the pin row is now the effective head
    assert dlg.list.currentRow() == 0 and dlg.list.item(0).text().startswith("● ")
    assert "← pin" in dlg.list.item(0).text() and "newest" in dlg.list.item(0).text()
    # x: refused on an active row, two-step on an archived one, other keys cancel
    dlg.keyPressEvent(_key(Qt.Key_X, "x"))
    assert "ARCHIVED runs only" in dlg.foot.text() and dlg._delete_armed is None
    dlg.list.setCurrentRow(2)
    dlg.keyPressEvent(_key(Qt.Key_X, "x"))
    assert dlg._delete_armed == "/ws/b" and "again: DELETE" in dlg.foot.text()
    dlg.keyPressEvent(_key(Qt.Key_K, "k"))
    assert dlg._delete_armed is None and dlg.list.currentRow() == 1
    dlg.list.setCurrentRow(2)
    dlg.keyPressEvent(_key(Qt.Key_X, "x"))
    dlg.keyPressEvent(_key(Qt.Key_X, "x"))
    assert calls[-1] == ("b1072a9f", "delete") and dlg.list.count() == 2
    assert "deleted …b1072a9f" in dlg.foot.text()
    # h hides the (now empty) archived side again
    dlg.keyPressEvent(_key(Qt.Key_H, "h"))
    assert dlg.list.count() == 2 and not dlg.show_archived
    assert "archived" not in dlg.head.toPlainText()
    dlg.close()
    # no hook wired: a is a silent no-op
    bare = TrainingRunPicker(None, on_pick=picked.append)
    bare.open_for(_runs())
    bare.keyPressEvent(_key(Qt.Key_A, "a"))
    assert bare.list.count() == 3 and bare.foot.text()
    bare.close()
