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
