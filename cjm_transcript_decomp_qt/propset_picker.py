"""Modal proposal-set lifecycle picker (the b20cb911 propset-unarchive
rung, riding the 8d29f0f0 kit): X on the runs stage lists the focused
run's cursor-source proposal sets on the kit FormShell — fixed header
(mouse close) + native PickerList rows + fixed footer — where the Shift+E
ring alone could never SHOW an archived set to unarchive it.

Rows: the effective set marked ● (what the hand-off would carve), archived
sets dimmed + chipped ⌂ behind h (hidden by default), each row dated.
Keys: j/k or click move · enter/space or double-click pick the set into
the Shift+E slot (archived rows refuse — a unarchives first) · a archive/
unarchive · x twice deletes an ARCHIVED set · esc or ✕ cancel.

The dialog owns no policy: on_pick hands the chosen manifest to the window
(which writes its propset_pick slot), and on_lifecycle(set, verb) drives
the substrate sidecar seam and hands back (sets, archived, note) — the
TrainingRunPicker contract, re-expressed on the kit shell."""

import time as _time
from typing import Any, Callable, Dict, List, Optional, Tuple

from cjm_substrate_qt_kit.formdialog import FormShell
from cjm_substrate_qt_kit.keyhints import keycaps
from cjm_substrate_qt_kit.theme import current_theme
from cjm_transcript_decomp_core.runs import PropsetIndex
from PySide6.QtCore import Qt

LifecycleHook = Callable[[Dict[str, Any], str],
                         Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str]]


class PropsetPicker(FormShell):
    """One source's proposal-set ring as a native lifecycle picker."""

    def __init__(self, parent, *,
                 on_pick: Callable[[Dict[str, Any]], None],
                 on_lifecycle: Optional[LifecycleHook] = None):
        super().__init__(parent, on_activate=lambda _key: self._pick_current())
        self._on_pick = on_pick
        self._on_lifecycle = on_lifecycle
        self.sets: List[Dict[str, Any]] = []       # active, newest first
        self.archived: List[Dict[str, Any]] = []   # listed only when shown
        self.show_archived = False
        self._title = ""
        self._effective: Optional[str] = None      # ring-resolved set _path
        self._note: Optional[str] = None
        self._delete_armed: Optional[str] = None   # the x two-step (_path)

    def open_for(self, title: str,
                 sets: List[Dict[str, Any]],
                 archived: Optional[List[Dict[str, Any]]] = None,
                 effective: Optional[str] = None) -> None:
        """Paint the source's sets (newest first, as the index lists them)
        and open sized to the rows, cursor parked on the EFFECTIVE set."""
        self._title = title
        self.sets = list(sets)
        self.archived = list(archived or [])
        self._effective = effective
        self._note, self._delete_armed = None, None
        self._render()
        self.open_sized(min_width=640)

    # ---- paint -----------------------------------------------------------

    def _rows(self) -> List[Dict[str, Any]]:
        """The manifests the list paints, cursor index = position here."""
        return self.sets + (self.archived if self.show_archived else [])

    def _render(self, keep_row: Optional[int] = None) -> None:
        theme = current_theme()
        arch = ""
        if self.archived:
            arch = " · %d archived (%s)" % (
                len(self.archived), "shown" if self.show_archived else "h shows")
        self.set_header(
            "PROPOSAL SETS — %s<span style='color:%s'> · %d set(s)%s · pick "
            "lands in the Shift+E slot for this sitting</span>"
            % (self._title, theme["content-dim"], len(self.sets), arch))
        rows: List[Dict[str, Any]] = []
        cursor = 0
        for i, m in enumerate(self._rows()):
            is_archived = m.get("_lifecycle") == "archived"
            is_effective = (not is_archived and self._effective
                            and m.get("_path") == self._effective)
            spans = [("● " if is_effective else "  ",
                      "green" if is_effective else ""),
                     (PropsetIndex.summary(m),
                      "dim" if is_archived else
                      "green" if is_effective else "")]
            created = float(m.get("created_at") or 0.0)
            if created:
                spans.append(("  ·  " + _time.strftime(
                    "%Y-%m-%d %H:%M", _time.localtime(created)), "dim"))
            if is_archived:
                spans.append(("  ⌂ archived", "dim"))
            rows.append({"kind": "item", "spans": spans, "key": i})
            if is_effective:
                cursor = i
        if not rows:
            rows.append({"kind": "note", "spans": [
                ("  (no proposal sets for this source — Shift+P proposes one)",
                 "dim")]})
        self.body.set_rows(rows, cursor=(cursor if keep_row is None
                                         else keep_row))
        self._paint_foot()

    def _paint_foot(self) -> None:
        theme = current_theme()
        if self._note:
            text = "%s · %s" % (self._note, keycaps("esc", theme))
        elif self._delete_armed:
            text = ("%s again: DELETE this archived set from disk · any other "
                    "key cancels" % keycaps("x", theme))
        else:
            text = ("%s move · %s pick · %s archive/unarchive · %s archived · "
                    "%s delete (archived) · %s or ✕ cancel"
                    % (keycaps("j/k", theme), keycaps("enter", theme),
                       keycaps("a", theme), keycaps("h", theme),
                       keycaps("x", theme), keycaps("esc", theme)))
        self.set_footer(text)

    # ---- selection -------------------------------------------------------

    def current_set(self) -> Optional[Dict[str, Any]]:
        """The manifest under the cursor (None on the empty placeholder)."""
        rows = self._rows()
        i = self.body.current_key()
        return rows[i] if isinstance(i, int) and 0 <= i < len(rows) else None

    def _pick_current(self) -> None:
        m = self.current_set()
        if m is None:
            return
        if m.get("_lifecycle") == "archived":
            self._note = "archived sets are not pickable — a unarchives it"
            self._paint_foot()
            return
        self.accept()
        self._on_pick(m)

    # ---- lifecycle verbs (b20cb911) --------------------------------------

    def _lifecycle(self, verb: str) -> None:
        """Run a verb on the cursored set through the window's hook and
        repaint from the lists it hands back (cursor kept in place)."""
        m = self.current_set()
        if m is None or self._on_lifecycle is None:
            return
        row = self.body.cursor
        self.sets, self.archived, self._note = self._on_lifecycle(m, verb)
        self._delete_armed = None
        self._render(keep_row=row)

    def _toggle_archived(self) -> None:
        self.show_archived = not self.show_archived
        self._note, self._delete_armed = None, None
        self._render(keep_row=self.body.cursor)

    def _delete(self) -> None:
        """x: two-step — arms on an ARCHIVED row (an active set refuses:
        archive it first), the second press deletes through the hook (which
        refuses loud while anything still references the set)."""
        m = self.current_set()
        if m is None:
            return
        if m.get("_lifecycle") != "archived":
            self._note = "delete serves ARCHIVED sets only — a archives it first"
            self._delete_armed = None
            self._paint_foot()
            return
        if self._delete_armed == m.get("_path"):
            self._lifecycle("delete")
            return
        self._delete_armed = m.get("_path")
        self._note = None
        self._paint_foot()

    # ---- keys ------------------------------------------------------------

    def keyPressEvent(self, event) -> None:
        key = event.text()
        if key != "x" and self._delete_armed:
            self._delete_armed = None
            self._paint_foot()
        if key == "j" or event.key() == Qt.Key_Down:
            self.body.move(1)
        elif key == "k" or event.key() == Qt.Key_Up:
            self.body.move(-1)
        elif key == " " or event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self._pick_current()
        elif key == "a":
            m = self.current_set()
            if m is not None:
                self._lifecycle("unarchive"
                                if m.get("_lifecycle") == "archived"
                                else "archive")
        elif key == "h":
            self._toggle_archived()
        elif key == "x":
            self._delete()
        else:
            super().keyPressEvent(event)
