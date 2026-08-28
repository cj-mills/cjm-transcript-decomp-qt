"""Modal training-run picker for the propose pre-stage (work item a13ebc66):
the decomp shell's model choice made in-app instead of by hand-editing the
workspace marker's flywheel.event_training_run pin. A frameless kit modal
(modal_header mouse close, 140a7b3c) over a NATIVE QListWidget — rows from
panes.trainrun_rows (manifest facts, the effective model marked ●, the pin
row chipped), click-to-cursor, j/k/arrows move, enter/space or double-click
pick, esc cancel. The dialog holds no model policy: the window resolves
PICK > PIN > NEWEST through TrainingRunIndex.resolve_pick and the pick
lives for the sitting only — the yaml pin keeps its provenance comments."""

from typing import Any, Callable, Dict, List, Optional

from cjm_substrate_qt_kit.keyhints import is_close_anchor, keycaps, modal_header
from cjm_substrate_qt_kit.style import apply_row_style
from cjm_substrate_qt_kit.theme import current_theme, make_font
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QLabel, QListWidget, QListWidgetItem, QTextBrowser,
                               QVBoxLayout)

from .panes import trainrun_rows


class TrainingRunPicker(QDialog):
    """The propose-model picker: a header row (kit modal_header, mouse
    close), the native run list, a hint line. open_for paints the rows and
    parks the cursor on the EFFECTIVE model (pick > pin > newest); a pick
    closes the dialog and hands the chosen manifest to on_pick — the
    CALLER owns what a pick means (the window's session-scoped
    training_run_pick)."""

    def __init__(self, parent, *,
                 on_pick: Callable[[Dict[str, Any]], None]):
        super().__init__(parent, Qt.WindowType.FramelessWindowHint)
        self.setModal(True)
        self._on_pick = on_pick
        self.runs: List[Dict[str, Any]] = []
        self._pin: Optional[str] = None
        self._pick: Optional[str] = None
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
                 pick: Optional[str] = None) -> None:
        """Paint the rows (newest first, as the index lists them) and open
        centered over the owner, sized to the rows (the keyhints sizing
        recipe: size to the rendered content, capped by the owner)."""
        self.runs = list(runs)
        self._pin, self._pick = pin, pick
        self._render()
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
        self.open()
        self.setFocus()

    # ---- paint -----------------------------------------------------------

    def _render(self) -> None:
        theme = current_theme()
        chrome = ("background: %s; border: 1px solid %s; padding: 6px;"
                  % (theme["surface"], theme["border"]))
        self.head.setStyleSheet("QTextBrowser { %s }" % chrome)
        self.list.setStyleSheet("QListWidget { %s }" % chrome)
        self.foot.setStyleSheet("QLabel { %s color: %s; }"
                                % (chrome, theme["content-dim"]))
        self.head.setHtml("<div style='color:%s'>%s</div>" % (
            theme["content"],
            modal_header("TRAINING RUN — propose model"
                         "<span style='color:%s'> · %d run(s) · pick "
                         "outranks the workspace pin for this sitting</span>"
                         % (theme["content-dim"], len(self.runs)), theme)))
        self.list.clear()
        cur = 0
        for r in trainrun_rows(self.runs, self._pin, self._pick):
            item = QListWidgetItem(str(r.get("text", "")))
            apply_row_style(item, r.get("style"))
            self.list.addItem(item)
            if r.get("style") == "green":
                cur = self.list.count() - 1
        self.list.setCurrentRow(cur)
        self.foot.setText("%s move · %s pick · %s or ✕ cancel"
                          % (keycaps("j/k", theme), keycaps("enter", theme),
                             keycaps("esc", theme)))

    # ---- selection -------------------------------------------------------

    def current_run(self) -> Optional[Dict[str, Any]]:
        """The manifest under the cursor (None on the empty placeholder)."""
        i = self.list.currentRow()
        return self.runs[i] if 0 <= i < len(self.runs) else None

    def _move(self, delta: int) -> None:
        n = self.list.count()
        if n:
            self.list.setCurrentRow(max(0, min(n - 1,
                                               self.list.currentRow() + delta)))

    def _pick_current(self) -> None:
        run = self.current_run()
        if run is None:
            return
        self.accept()
        self._on_pick(run)

    def _on_anchor(self, url) -> None:
        """The header's mouse close (140a7b3c) — the one link painted."""
        if is_close_anchor(url):
            self.reject()

    # ---- keys ------------------------------------------------------------

    def keyPressEvent(self, event) -> None:
        key = event.text()
        if key == "j" or event.key() == Qt.Key_Down:
            self._move(1)
        elif key == "k" or event.key() == Qt.Key_Up:
            self._move(-1)
        elif key == " " or event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self._pick_current()
        else:
            super().keyPressEvent(event)
