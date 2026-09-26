"""Modal training-run picker for the propose pre-stage (work item a13ebc66):
the decomp shell's model choice made in-app instead of by hand-editing the
workspace marker's flywheel.event_training_run pin. A frameless kit modal
(modal_header mouse close, 140a7b3c) over a NATIVE QListWidget — rows from
panes.trainrun_rows (manifest facts, the effective model marked ●, the pin
row chipped), click-to-cursor, j/k/arrows move, enter/space or double-click
pick, esc cancel. The dialog holds no model policy: the window resolves
PICK > PIN > NEWEST through TrainingRunIndex.resolve_pick and the pick
lives for the sitting only — the yaml pin keeps its provenance comments.

Lifecycle verbs (work item b20cb911, DEC 14678fc9): `a` archives or
unarchives the cursored run, `h` shows/hides the archived rows (dimmed,
chipped ⌂ archived — hidden by default so a test run stops diluting the
list), `x` twice deletes an ARCHIVED run outright. The dialog owns none of
it: on_lifecycle(run, verb) is the window's — it drives the substrate
lifecycle seam, reloads its index and hands back (runs, archived, note)."""

from typing import Any, Callable, Dict, List, Optional, Tuple

from cjm_substrate_qt_kit.keyhints import is_close_anchor, keycaps, modal_header
from cjm_substrate_qt_kit.style import apply_row_style
from cjm_substrate_qt_kit.theme import current_theme, make_font
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QLabel, QListWidget, QListWidgetItem, QTextBrowser,
                               QVBoxLayout)

from .panes import trainrun_rows

LifecycleHook = Callable[[Dict[str, Any], str],
                         Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str]]


class TrainingRunPicker(QDialog):
    """The propose-model picker: a header row (kit modal_header, mouse
    close), the native run list, a hint line. open_for paints the rows and
    parks the cursor on the EFFECTIVE model (pick > pin > newest); a pick
    closes the dialog and hands the chosen manifest to on_pick — the
    CALLER owns what a pick means (the window's session-scoped
    training_run_pick) and what a lifecycle verb does (on_lifecycle)."""

    def __init__(self, parent, *,
                 on_pick: Callable[[Dict[str, Any]], None],
                 on_lifecycle: Optional[LifecycleHook] = None):
        super().__init__(parent, Qt.WindowType.FramelessWindowHint)
        self.setModal(True)
        self._on_pick = on_pick
        self._on_lifecycle = on_lifecycle
        self.runs: List[Dict[str, Any]] = []       # active, newest first
        self.archived: List[Dict[str, Any]] = []   # archived, listed only when shown
        self.show_archived = False
        self._pin: Optional[str] = None
        self._pick: Optional[str] = None
        self._note: Optional[str] = None            # the last lifecycle verb's echo
        self._delete_armed: Optional[str] = None    # the x two-step (a run's _path)
        self.head = QTextBrowser(self)
        self.head.setFocusPolicy(Qt.NoFocus)
        self.head.setOpenLinks(False)
        self.head.anchorClicked.connect(self._on_anchor)
        self.head.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # The DIALOG owns the keys: a focused QListWidget keyboard-searches
        # on letter presses, which would eat j/k.
        self.list = QListWidget(self)
        self.list.setFocusPolicy(Qt.NoFocus)
        self.list.setFont(make_font(kind="mono"))
        self.list.itemDoubleClicked.connect(lambda _item: self._pick_current())
        self.foot = QLabel(self)
        self.foot.setTextFormat(Qt.RichText)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(1, 1, 1, 1)
        lay.setSpacing(0)
        lay.addWidget(self.head)
        lay.addWidget(self.list, 1)
        lay.addWidget(self.foot)

    def open_for(self, runs: List[Dict[str, Any]],
                 pin: Optional[str] = None,
                 pick: Optional[str] = None,
                 archived: Optional[List[Dict[str, Any]]] = None) -> None:
        """Paint the rows (newest first, as the index lists them) and open
        centered over the owner, sized to the rows (the keyhints sizing
        recipe: size to the rendered content, capped by the owner)."""
        self.runs = list(runs)
        self.archived = list(archived or [])
        self._pin, self._pick = pin, pick
        self._note, self._delete_armed = None, None
        self._render()
        self._size_to_rows()
        self.open()
        self.setFocus()

    def _size_to_rows(self) -> None:
        owner = self.parentWidget()
        avail_w = (owner.width() - 64) if owner is not None else 960
        avail_h = (owner.height() - 64) if owner is not None else 640
        width = min(avail_w, max(self.list.sizeHintForColumn(0) + 48, 640))
        doc = self.head.document()
        doc.setTextWidth(width - 24)
        head_h = int(doc.size().height()) + 12
        self.head.setFixedHeight(head_h)
        row_h = max(self.list.sizeHintForRow(0), 18)
        height = min(avail_h, head_h + row_h * (self.list.count() + 1)
                     + self.foot.sizeHint().height() + 24)
        self.resize(width, height)
        if owner is not None:
            center = owner.mapToGlobal(owner.rect().center())
            self.move(center.x() - self.width() // 2,
                      center.y() - self.height() // 2)

    # ---- paint -----------------------------------------------------------

    def _rows(self) -> List[Dict[str, Any]]:
        """The manifests the list paints, cursor index = position here."""
        return self.runs + (self.archived if self.show_archived else [])

    def _render(self, keep_row: Optional[int] = None) -> None:
        theme = current_theme()
        # Chrome comes from the design system's stylesheet (token schema v1);
        # the footer paints through the state channel.
        self.foot.setProperty("role", "dim")
        arch = ""
        if self.archived:
            arch = " · %d archived (%s)" % (
                len(self.archived), "shown" if self.show_archived else "h shows")
        self.head.setHtml("<div style='color:%s'>%s</div>" % (
            theme["text"],
            modal_header("TRAINING RUN — propose model"
                         "<span style='color:%s'> · %d run(s)%s · pick "
                         "outranks the workspace pin for this sitting</span>"
                         % (theme["dim"], len(self.runs), arch), theme)))
        self.list.clear()
        cur = 0
        for r in trainrun_rows(self._rows(), self._pin, self._pick):
            item = QListWidgetItem(str(r.get("text", "")))
            apply_row_style(item, r.get("style"))
            self.list.addItem(item)
            if r.get("style") == "green":
                cur = self.list.count() - 1
        if keep_row is not None and self.list.count():
            cur = max(0, min(self.list.count() - 1, keep_row))
        self.list.setCurrentRow(cur)
        self._paint_foot()

    def _paint_foot(self) -> None:
        theme = current_theme()
        if self._note:
            text = "%s · %s" % (self._note, keycaps("esc", theme))
        elif self._delete_armed:
            text = ("%s again: DELETE this archived run from disk · any other "
                    "key cancels" % keycaps("x", theme))
        else:
            text = ("%s move · %s pick · %s archive/unarchive · %s archived · "
                    "%s delete (archived) · %s or ✕ cancel"
                    % (keycaps("j/k", theme), keycaps("enter", theme),
                       keycaps("a", theme), keycaps("h", theme),
                       keycaps("x", theme), keycaps("esc", theme)))
        self.foot.setText(text)

    # ---- selection -------------------------------------------------------

    def current_run(self) -> Optional[Dict[str, Any]]:
        """The manifest under the cursor (None on the empty placeholder)."""
        rows = self._rows()
        i = self.list.currentRow()
        return rows[i] if 0 <= i < len(rows) else None

    def _move(self, delta: int) -> None:
        n = self.list.count()
        if n:
            self.list.setCurrentRow(max(0, min(n - 1,
                                               self.list.currentRow() + delta)))

    def _pick_current(self) -> None:
        run = self.current_run()
        if run is None:
            return
        if run.get("_lifecycle") == "archived":
            self._note = "archived runs are not pickable — a unarchives it"
            self._paint_foot()
            return
        self.accept()
        self._on_pick(run)

    def _on_anchor(self, url) -> None:
        """The header's mouse close (140a7b3c) — the one link painted."""
        if is_close_anchor(url):
            self.reject()

    # ---- lifecycle verbs (b20cb911) ---------------------------------------

    def _lifecycle(self, verb: str) -> None:
        """Run a verb on the cursored run through the window's hook and
        repaint from the lists it hands back (cursor kept in place)."""
        run = self.current_run()
        if run is None or self._on_lifecycle is None:
            return
        row = self.list.currentRow()
        self.runs, self.archived, self._note = self._on_lifecycle(run, verb)
        self._delete_armed = None
        self._render(keep_row=row)

    def _toggle_archived(self) -> None:
        self.show_archived = not self.show_archived
        self._note, self._delete_armed = None, None
        self._render(keep_row=self.list.currentRow())
        self._size_to_rows()

    def _delete(self) -> None:
        """x: two-step — the first press arms on an ARCHIVED row (an active
        run refuses: archive it first), the second deletes through the hook
        (which refuses loud while anything still references the run)."""
        run = self.current_run()
        if run is None:
            return
        if run.get("_lifecycle") != "archived":
            self._note = "delete serves ARCHIVED runs only — a archives it first"
            self._delete_armed = None
            self._paint_foot()
            return
        if self._delete_armed == run.get("_path"):
            self._lifecycle("delete")
            return
        self._delete_armed = run.get("_path")
        self._note = None
        self._paint_foot()

    # ---- keys ------------------------------------------------------------

    def keyPressEvent(self, event) -> None:
        key = event.text()
        if key != "x" and self._delete_armed:
            self._delete_armed = None
            self._paint_foot()
        if key == "j" or event.key() == Qt.Key_Down:
            self._move(1)
        elif key == "k" or event.key() == Qt.Key_Up:
            self._move(-1)
        elif key == " " or event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self._pick_current()
        elif key == "a":
            run = self.current_run()
            if run is not None:
                self._lifecycle("unarchive" if run.get("_lifecycle") == "archived"
                                else "archive")
        elif key == "h":
            self._toggle_archived()
        elif key == "x":
            self._delete()
        else:
            super().keyPressEvent(event)
