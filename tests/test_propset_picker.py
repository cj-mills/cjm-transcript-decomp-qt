"""Offscreen tests for the propset lifecycle picker (b20cb911's unarchive
rung on the 8d29f0f0 kit shell): the cursor parks on the EFFECTIVE set,
enter picks an active set and closes, archived rows refuse the pick, h
shows/hides the archived half, a routes archive/unarchive through the
window hook, x arms two-step delete on archived rows only."""

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent, Qt, QUrl
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication

from cjm_transcript_decomp_qt.propset_picker import PropsetPicker


def _sets():
    return [{"proposal_set_id": "propset_20260828_aaaa1111", "_path": "/ws/a",
             "counts": {"inhale": 9}, "created_at": 1756400000.0},
            {"proposal_set_id": "propset_20260810_bbbb2222", "_path": "/ws/b",
             "counts": {"inhale": 4}}]


def _archived():
    return [{"proposal_set_id": "propset_20260701_cccc3333", "_path": "/ws/c",
             "counts": {"inhale": 2}, "_lifecycle": "archived"}]


def _key(key, text=""):
    return QKeyEvent(QEvent.KeyPress, key, Qt.NoModifier, text)


def test_parks_on_effective_picks_active_and_refuses_archived():
    QApplication.instance() or QApplication(sys.argv[:1])
    picked = []
    dlg = PropsetPicker(None, on_pick=picked.append)
    dlg.open_for("ep1.mp3", _sets(), archived=_archived(), effective="/ws/b")
    assert dlg.body.count() == 2            # archived hidden by default
    assert dlg.body.cursor == 1             # parked on the effective set
    assert "● " in dlg.body.plain_text()
    assert "1 archived (h shows)" in dlg.head.toPlainText()
    dlg.keyPressEvent(_key(Qt.Key_H, "h"))  # show the archived half
    assert dlg.body.count() == 3
    assert "⌂ archived" in dlg.body.plain_text()
    dlg.keyPressEvent(_key(Qt.Key_J, "j"))  # onto the archived row
    dlg.keyPressEvent(_key(Qt.Key_Return))  # refused, dialog stays
    assert picked == [] and dlg.isVisible()
    assert "not pickable" in dlg.foot.text()
    dlg.keyPressEvent(_key(Qt.Key_K, "k"))
    dlg.keyPressEvent(_key(Qt.Key_K, "k"))
    dlg.keyPressEvent(_key(Qt.Key_Return))  # the head row picks + closes
    assert [p["_path"] for p in picked] == ["/ws/a"]
    assert not dlg.isVisible()
    dlg.close()


def test_lifecycle_hook_routes_verbs_and_delete_arms_on_archived_only():
    QApplication.instance() or QApplication(sys.argv[:1])
    verbs = []

    def hook(m, verb):
        verbs.append((m["_path"], verb))
        return _sets(), _archived(), f"{verb} ok"

    dlg = PropsetPicker(None, on_pick=lambda m: None, on_lifecycle=hook)
    dlg.open_for("ep1.mp3", _sets(), archived=_archived(), effective="/ws/a")
    dlg.keyPressEvent(_key(Qt.Key_A, "a"))            # active row: archive
    assert verbs == [("/ws/a", "archive")]
    assert "archive ok" in dlg.foot.text()
    dlg.keyPressEvent(_key(Qt.Key_X, "x"))            # active row: refused
    assert "ARCHIVED sets only" in dlg.foot.text()
    dlg.keyPressEvent(_key(Qt.Key_H, "h"))
    dlg.body.set_cursor(2)                            # the archived row
    dlg.keyPressEvent(_key(Qt.Key_A, "a"))            # archived: unarchive
    assert verbs[-1] == ("/ws/c", "unarchive")
    dlg.keyPressEvent(_key(Qt.Key_H, "h"))
    dlg.keyPressEvent(_key(Qt.Key_H, "h"))
    dlg.body.set_cursor(2)
    dlg.keyPressEvent(_key(Qt.Key_X, "x"))            # arms
    assert "again: DELETE" in dlg.foot.text()
    assert not any(v == "delete" for _, v in verbs)
    dlg.keyPressEvent(_key(Qt.Key_J, "j"))            # any other key disarms
    assert "again: DELETE" not in dlg.foot.text()
    dlg.body.set_cursor(2)
    dlg.keyPressEvent(_key(Qt.Key_X, "x"))
    dlg.keyPressEvent(_key(Qt.Key_X, "x"))            # second press deletes
    assert verbs[-1] == ("/ws/c", "delete")
    dlg.close()


def test_empty_placeholder_and_close_anchor():
    QApplication.instance() or QApplication(sys.argv[:1])
    dlg = PropsetPicker(None, on_pick=lambda m: None)
    dlg.open_for("ep1.mp3", [], archived=[], effective=None)
    assert dlg.body.count() == 0
    assert "no proposal sets" in dlg.body.plain_text()
    dlg.keyPressEvent(_key(Qt.Key_Return))            # no-op on placeholder
    assert dlg.isVisible()
    dlg._on_anchor(QUrl("close:"))                    # the kit mouse close
    assert not dlg.isVisible()
    dlg.close()
