"""The Qt decomp-batch shell: the same batch queueing over transcription-run
manifests as the Textual TUI (RUNS -> confirm, plus the RESULTS drill into
SEGMENTS with focusable gap rows and the VADFORM/PROBEVIEW probe pair),
repainted with real typography and handed the same exit contract — confirm
leaves a grouped batch plan for the CLI's headless hand-off, so Qt-queued
batches stay byte-identical to hand-launched ones.

Shell-only by design (DEC 6c574c89): every stateful decision lives in the
imported spine (SourceRunIndex / DecompIndex / PropsetIndex / TrainingRunIndex
/ group_batches / the segments.py fold+probe helpers / kit ConfigForm) or in
DecompSession, whose loop thread owns the read-only graph seat and the
propose seat. This module materializes row dicts from panes.py into Qt
widgets and forwards the VERBATIM Textual key vocabulary (j/k, enter/space,
t, v, r, n, b, c, p, s, Shift+R respine, e, Shift+E propset, x propset
lifecycle picker, Shift+X archive, m model, Shift+P propose,
[ ], escape, q — stage-dispatched exactly like the Textual actions). Text
entry rides modal QInputDialogs; detail text paints setPlainText.

Async discipline (the c4b0d6e5 seam): the segments open, the probe chain, and
the propose batch all resolve as loop-thread Futures landing here through
QUEUED Signals — the paint thread never blocks on graph reads or model
probes. No blocked-reason poll: decomp loads no heavy comparison stack."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from cjm_substrate.utils.lifecycle import (artifact_id, ArtifactLifecycle, find_holders,
                                           LifecycleRefusal)
from cjm_substrate_qt_kit.findbar import FindBar
from cjm_substrate_qt_kit.keyhints import KeyHintsOverlay
from cjm_substrate_qt_kit.keymap import KeymapRegistry
from cjm_substrate_qt_kit.player import SpanPlayer
from cjm_substrate_qt_kit.statusstrip import StatusStrip
from cjm_substrate_qt_kit.style import apply_row_style
from cjm_substrate_qt_kit.theme import style_text_pane
from cjm_substrate_tui_kit.form import ConfigForm
from cjm_substrate_tui_kit.viewport import tail
from cjm_transcript_decomp_core.runs import (DecompIndex, group_batches, PropsetIndex,
                                             SourceRunIndex, TrainingRunIndex)
from cjm_transcript_decomp_core.segments import (aseg_index_for, build_display,
                                                 capability_config_schema, fmt_ts, locate_span,
                                                 predicted_rows, probe_compare, realign_rows,
                                                 split_predicted, vad_summary)
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QInputDialog, QLabel, QListWidget, QListWidgetItem, QMainWindow,
                               QPlainTextEdit, QSplitter, QStackedWidget, QVBoxLayout, QWidget)

from .panes import (batch_html, config_diffs, decomp_header, decomp_rows, drill_header_html,
                    drill_source_rows, form_rows, probe_detail, probe_rows, probe_summary_html,
                    probe_target_html, probeview_header_html, run_detail_html, run_rows,
                    runs_header, segment_detail, segment_rows, segments_header_html,
                    vadform_header_html)
from .propset_picker import PropsetPicker
from .session import DecompSession
from .trainrun_picker import TrainingRunPicker

HINTS = {
    "runs": "enter/space pick · t text-from · s split · Shift+R respine · "
            "e event · Shift+E propset · x propsets · Shift+X archive · "
            "m model · Shift+P propose · v decomp runs · "
            "r reload · n confirm · q quit",
    "results": "enter open run · j/k walk · b back · q quit",
    "results_drill": "enter segments · j/k source · b decomp list · q quit",
    "segments": "j/k walk · r play · [ ] speed · c vad config · b back · "
                "q quit",
    "vadform": "j/k field · enter cycle/edit · p probe · s split · "
               "v skeleton · b back · q quit",
    "probeview": "j/k walk · r play · s split · b form · q quit",
}


class DecompWindow(QMainWindow):
    """Batch-setup window; exit with `plan` set = the CLI hands off headless."""

    seg_opened = Signal(object)     # loop-thread Future -> Qt thread (queued)
    probe_done = Signal(object)
    propose_done = Signal(object)
    progress_note = Signal(str)     # loop-thread narration -> status bar

    def __init__(self, manifests_dir: str,          # Capability manifests directory
                 *, runs_dir: str = "runs",         # Both cores' manifest dir
                 sysmon_capability: Optional[str] = None,
                 graph_capability: str = "cjm-capability-graph-sqlite",
                 graph_db_path: Optional[str] = None,
                 gap_threshold: float = 2.0,
                 sentence_split: bool = True,
                 respine: bool = False,
                 proposals_dir: str = "proposals",
                 event_split: bool = False,
                 propset_pin: Optional[str] = None,
                 training_runs_dir: str = "training-runs",
                 training_run_pin: Optional[str] = None,
                 event_capability: str = "cjm-capability-pyannote"):
        super().__init__()
        self.setWindowTitle("cjm transcript decomp setup (qt)")
        self.resize(1080, 780)
        self.plan: Optional[Dict[str, Any]] = None  # set by confirm; CLI reads it
        self.manifests_dir = manifests_dir
        self.sysmon_capability = sysmon_capability
        self.graph_capability = graph_capability
        self.graph_db_path = graph_db_path
        self.gap_threshold = gap_threshold
        self.sentence_split = sentence_split
        self.respine = respine
        self.event_split = event_split
        self.propset_pin = propset_pin
        self.propset_pick: Dict[str, str] = {}      # "<run path>#<source idx>" -> set path (Shift+E)
        self.propset_cursor: Dict[str, int] = {}    # run path -> position in its (source, set) ring
        self.training_run_pin = training_run_pin
        self.event_capability = event_capability
        self.src_index = SourceRunIndex(runs_dir)
        self.dec_index = DecompIndex(runs_dir)
        self.propset_index = PropsetIndex(proposals_dir)
        self.trainrun_index = TrainingRunIndex(training_runs_dir)
        self.propose_armed: Optional[Dict[str, Any]] = None
        self.propose_busy = False
        # The propose-model pick for this sitting (a13ebc66): a run's _path,
        # outranking the workspace pin in resolve_pick; never written back
        # to cjm-workspace.yaml (its model-register comments survive).
        self.training_run_pick: Optional[str] = None
        self.trainrun_picker = TrainingRunPicker(
            self, on_pick=self._on_trainrun_picked,
            on_lifecycle=self._on_trainrun_lifecycle)
        # The propset lifecycle picker (b20cb911's unarchive rung): x lists
        # the cursor source's sets INCLUDING archived — the Shift+E ring
        # alone can never show one to unarchive.
        self.propset_picker = PropsetPicker(
            self, on_pick=self._on_propset_picked,
            on_lifecycle=self._on_propset_lifecycle)
        self._propset_picker_ctx: Optional[Tuple[str, int]] = None
        # The Shift+X two-step (b20cb911): the propset path the first press
        # named; the second press archives it. Any reload drops it.
        self.archive_armed: Optional[str] = None
        self.stage = "runs"
        self.picked: List[str] = []          # ordered batch, keyed by _path
        self.text_from: Dict[str, str] = {}
        self._decomp_counts: Dict[str, int] = {}
        self.results_run: Optional[int] = None
        self.busy: Optional[str] = None
        self.error: Optional[str] = None
        self.notice: Optional[str] = None
        # Segments drill state (filled by _on_seg_opened from the manifest +
        # the session's spine read; mirrors the Textual seg_* block).
        self.seg_rows: List[Dict[str, Any]] = []
        self.seg_display: List[Tuple[str, int, float]] = []
        self.seg_title = ""
        self.seg_vad: Optional[str] = None
        self.seg_missing = 0
        self.seg_source_id = ""
        self.seg_vad_id = "cjm-capability-silero-vad"
        self.seg_vad_cfg: Dict[str, Any] = {}
        self.seg_fa_id = "cjm-capability-qwen3-forced-aligner"
        self.seg_fa_cfg: Dict[str, Any] = {}
        self.seg_seg_id = "cjm-capability-pysbd"
        self.seg_seg_cfg: Dict[str, Any] = {}
        self.seg_text_from = ""
        self.seg_asegs: List[Dict[str, Any]] = []
        self.player: Optional[SpanPlayer] = None
        self.speed = 1.0
        self.vad_form: Optional[ConfigForm] = None
        self.probe_aseg: Optional[int] = None
        self.probe_result: Optional[Dict[str, Any]] = None
        self.probe_busy = False
        self.probe_split = False
        self._probe_ctx: Optional[Tuple[Dict[str, Any],
                                        List[Tuple[float, float]],
                                        Optional[str],
                                        Optional[List[Tuple[str, float, float]]],
                                        Optional[List[Tuple[int, int]]]]] = None
        self.probe_view_rows: List[Dict[str, Any]] = []
        self.probe_view_display: List[Tuple[str, int, float]] = []
        self._seg_pending: Optional[Tuple[Dict[str, Any], Dict[str, Any],
                                          List[str]]] = None
        self._armed_run: Optional[Dict[str, Any]] = None
        self._armed_targets: List[Dict[str, Any]] = []
        self.sess = DecompSession(manifests_dir, graph_capability)
        self.sess.start()
        self._build_widgets()
        self._bind_keys()
        self.seg_opened.connect(self._on_seg_opened)
        self.probe_done.connect(self._on_probe_done)
        self.propose_done.connect(self._on_propose_done)
        self.progress_note.connect(self._on_progress_note)
        self._reload_indexes()
        self._show_stage("runs")

    # ---- widget tree ----------------------------------------------------

    def _build_widgets(self) -> None:
        def page(*widgets) -> QWidget:
            w = QWidget()
            lay = QVBoxLayout(w)
            lay.setContentsMargins(8, 8, 8, 8)
            for child in widgets:
                lay.addWidget(child)
            return w

        def label() -> QLabel:
            lab = QLabel()
            lab.setTextFormat(Qt.RichText)
            lab.setWordWrap(True)
            return lab

        def rowlist() -> QListWidget:
            lst = QListWidget()
            lst.setWordWrap(True)
            lst.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            return lst

        def text_pane() -> QPlainTextEdit:
            pane = QPlainTextEdit()
            pane.setReadOnly(True)
            style_text_pane(pane, live=True)
            return pane

        self.runs_head = label()
        self.runs_list = rowlist()
        self.runs_list.itemActivated.connect(lambda _i: self.on_select())
        self.runs_list.currentRowChanged.connect(
            lambda _r: self._paint_run_detail())
        self.runs_detail = label()
        self.runs_batch = label()
        self.runs_page = page(self.runs_head, self.runs_list,
                              self.runs_detail, self.runs_batch)

        self.res_head = label()
        self.res_list = rowlist()
        self.res_list.itemActivated.connect(lambda _i: self.on_select())
        self.results_page = page(self.res_head, self.res_list)

        self.seg_head = label()
        self.seg_list = rowlist()
        self.seg_list.currentRowChanged.connect(
            lambda _r: self._paint_seg_detail())
        self.seg_text = text_pane()
        seg_split = QSplitter(Qt.Vertical)
        seg_split.addWidget(self.seg_list)
        seg_split.addWidget(self.seg_text)
        seg_split.setStretchFactor(0, 2)
        seg_split.setStretchFactor(1, 1)
        self.segments_page = page(self.seg_head, seg_split)

        self.form_head = label()
        self.form_list = rowlist()
        self.form_list.itemActivated.connect(lambda _i: self.on_select())
        self.form_target = label()
        self.form_summary = label()
        self.vadform_page = page(self.form_head, self.form_list,
                                 self.form_target, self.form_summary)

        self.pv_head = label()
        self.pv_list = rowlist()
        self.pv_list.currentRowChanged.connect(
            lambda _r: self._paint_pv_detail())
        self.pv_text = text_pane()
        pv_split = QSplitter(Qt.Vertical)
        pv_split.addWidget(self.pv_list)
        pv_split.addWidget(self.pv_text)
        pv_split.setStretchFactor(0, 2)
        pv_split.setStretchFactor(1, 1)
        self.probeview_page = page(self.pv_head, pv_split)

        self.stack = QStackedWidget()
        for p in (self.runs_page, self.results_page, self.segments_page,
                  self.vadform_page, self.probeview_page):
            self.stack.addWidget(p)
        self.findbar = FindBar(self.seg_text)  # focus-follow covers pv_text too
        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self.stack, 1)
        outer.addWidget(self.findbar)
        self.strip = StatusStrip()
        outer.addWidget(self.strip)
        self.hints_overlay = KeyHintsOverlay(self)
        self.setCentralWidget(central)
        # Decomp ALWAYS journals — extension IS graph writing — so the chip
        # shows the target, never a NOT-JOURNALED state. It rides the strip's
        # chip row now (DEC 2a42c028 — QStatusBar retired).

    def _bind_keys(self) -> None:
        # Kit KeymapRegistry (adoption rung caa33c98): the declarative table is
        # the discovery surface; guard_text_entry gates character verbs off
        # while the FindBar field has focus.
        self.keymap = KeymapRegistry(self)
        add = self.keymap.add
        add("next", "Next row", "J", lambda: self.move_cursor(1), group="Rows")
        add("prev", "Previous row", "K", lambda: self.move_cursor(-1),
            group="Rows")
        add("select", "Select / descend", "Return", self.on_select,
            group="Rows")
        add("select-space", "Select (Space)", "Space", self.on_select,
            group="Rows")
        add("back", "Back", "B", self.on_back, group="Rows")
        add("cycle-text-from", "Cycle text-from", "T", self.on_cycle_text_from,
            group="Stage")
        add("results", "Results view", "V", self.on_results, group="Stage")
        add("reload", "Reload", "R", self.on_reload, group="Stage")
        add("confirm", "Confirm", "N", self.on_confirm, group="Stage")
        add("vad-config", "VAD config", "C", self.on_vad_config, group="Stage")
        add("probe", "Probe", "P", self.on_probe, group="Stage")
        add("toggle-split", "Toggle split", "S", self.on_toggle_split,
            group="Proposals")
        add("toggle-respine", "Toggle respine", "Shift+R",
            self.on_toggle_respine, group="Proposals")
        add("toggle-event-split", "Toggle event split", "E",
            self.on_toggle_event_split, group="Proposals")
        add("cycle-propset", "Cycle propset", "Shift+E", self.on_cycle_propset,
            group="Proposals")
        add("pick-model", "Pick training run (propose model)", "M",
            self.on_pick_model, group="Proposals")
        add("propose-missing", "Propose (picked; regenerates)", "Shift+P",
            self.on_propose_missing, group="Proposals")
        add("archive-propset", "Archive propset (two-step)", "Shift+X",
            self.on_archive_propset, group="Proposals")
        add("propset-picker", "Proposal sets (lifecycle picker)", "X",
            self.on_propset_picker, group="Proposals")
        add("slower", "Playback slower", "[", lambda: self.nudge_speed(-0.25),
            group="Audio")
        add("faster", "Playback faster", "]", lambda: self.nudge_speed(0.25),
            group="Audio")
        add("stop-audio", "Stop audio", "Escape", self.on_stop_audio,
            group="Audio")
        add("quit", "Quit", "Q", self.on_quit, group="App")
        add("find", "Find in text pane", "Ctrl+F", self.open_find)
        add("find-next", "Find next", "F3", self.findbar.next)
        add("find-previous", "Find previous", "Shift+F3", self.findbar.previous)
        add("keys", "Keyboard hints", "?", self.hints_overlay.toggle)
        self.keymap.guard_text_entry()

    def open_find(self) -> None:
        """Ctrl+F: find over the visible text pane (kit FindBar; focus-follow
        re-attaches when the user clicks into another searchable pane)."""
        pane = self.pv_text if self.stage == "probeview" else self.seg_text
        self.findbar.attach(pane)
        self.findbar.open()

    # ---- painting -------------------------------------------------------

    def _populate(self, lst: QListWidget, rows: List[Dict[str, Any]],
                  keep_row: Optional[int] = None) -> None:
        """Materialize row dicts; UserRole carries each row's spine index."""
        lst.blockSignals(True)   # repaint is not a cursor gesture
        lst.clear()
        for r in rows:
            item = QListWidgetItem(str(r.get("text", "")))
            apply_row_style(item, r.get("style"))
            item.setData(Qt.UserRole, r.get("index"))
            lst.addItem(item)
        if lst.count():
            row = keep_row if keep_row is not None else 0
            lst.setCurrentRow(max(0, min(row, lst.count() - 1)))
        lst.blockSignals(False)

    def _active_list(self) -> QListWidget:
        return {"runs": self.runs_list, "results": self.res_list,
                "segments": self.seg_list, "vadform": self.form_list,
                "probeview": self.pv_list}[self.stage]

    def _show_stage(self, stage: str) -> None:
        self.stage = stage
        pages = {"runs": self.runs_page, "results": self.results_page,
                 "segments": self.segments_page, "vadform": self.vadform_page,
                 "probeview": self.probeview_page}
        self._paint_stage()
        self.stack.setCurrentWidget(pages[stage])
        self._active_list().setFocus()
        self._refresh_status()

    def _paint_stage(self) -> None:
        if self.stage == "runs":
            self._paint_runs()
        elif self.stage == "results":
            self._paint_results()
        elif self.stage == "segments":
            self._paint_segments()
        elif self.stage == "vadform":
            self._paint_vadform()
        elif self.stage == "probeview":
            self._paint_probeview()

    def _paint_runs(self) -> None:
        self.runs_head.setText(runs_header(len(self.src_index.runs),
                                           self.src_index.runs_dir))
        keep = self.runs_list.currentRow()
        self._populate(self.runs_list,
                       run_rows(self.src_index.runs, self.picked,
                                self.text_from, self._decomp_counts,
                                self.event_split, self._propset_chip),
                       keep_row=keep if keep >= 0 else None)
        self._paint_run_detail()
        self.runs_batch.setText(batch_html(self._batches(), self._run_id_for,
                                           self.sentence_split, self.respine,
                                           self.event_split))

    def _paint_run_detail(self) -> None:
        if self.stage != "runs":
            return
        m = self._focused_run()
        self.runs_detail.setText(
            run_detail_html(m, self._resolved_graph_db(m)) if m else "")

    def _paint_results(self) -> None:
        if self.results_run is None:
            self.res_head.setText(decomp_header(len(self.dec_index.runs),
                                                self.dec_index.runs_dir))
            self._populate(self.res_list, decomp_rows(self.dec_index.runs))
        else:
            m = self.dec_index.runs[self.results_run]
            self.res_head.setText(drill_header_html(m))
            self._populate(self.res_list, drill_source_rows(m))

    def _paint_segments(self) -> None:
        gaps = sum(1 for kind, _, _ in self.seg_display if kind == "gap")
        self.seg_head.setText(segments_header_html(
            self.seg_title, len(self.seg_rows), gaps, self.seg_missing,
            self.gap_threshold, self.seg_vad))
        keep = self.seg_list.currentRow()
        self._populate(self.seg_list, segment_rows(self.seg_display,
                                                   self.seg_rows),
                       keep_row=keep if keep >= 0 else None)
        self._paint_seg_detail()

    def _paint_seg_detail(self) -> None:
        i = self.seg_list.currentRow()
        if 0 <= i < len(self.seg_display):
            entry = self.seg_display[i]
            self.seg_text.setPlainText(segment_detail(entry,
                                                      self.seg_rows[entry[1]]))
        else:
            self.seg_text.setPlainText("")

    def _paint_vadform(self) -> None:
        self.form_head.setText(vadform_header_html(self.seg_vad_id,
                                                   self.seg_title,
                                                   self.probe_split))
        keep = self.form_list.currentRow()
        self._populate(self.form_list, form_rows(self.vad_form,
                                                 self.seg_vad_cfg),
                       keep_row=keep if keep >= 0 else None)
        self.form_target.setText(probe_target_html(self.probe_aseg,
                                                   self.seg_asegs))
        self.form_summary.setText(probe_summary_html(self.probe_result,
                                                     self.probe_busy))

    def _paint_probeview(self) -> None:
        realigned = bool(self.probe_view_rows
                         and self.probe_view_rows[0].get("realigned"))
        self.pv_head.setText(probeview_header_html(
            config_diffs(self.vad_form, self.seg_vad_cfg), self.probe_aseg,
            self.seg_asegs, self.probe_result, realigned))
        keep = self.pv_list.currentRow()
        self._populate(self.pv_list, probe_rows(self.probe_view_display,
                                                self.probe_view_rows),
                       keep_row=keep if keep >= 0 else None)
        self._paint_pv_detail()

    def _paint_pv_detail(self) -> None:
        i = self.pv_list.currentRow()
        if 0 <= i < len(self.probe_view_display):
            entry = self.probe_view_display[i]
            self.pv_text.setPlainText(
                probe_detail(entry, self.probe_view_rows[entry[1]]))
        else:
            self.pv_text.setPlainText("")

    def _refresh_status(self) -> None:
        """Footer decomposition (DEC 2a42c028): stage + journal chips, the
        error/busy/notice ladder on the readout, the stage legend on the
        hint line, and the ?-overlay tracking the registry."""
        chip = (f"<span style='color:#3f9d55'> graph→{self.graph_capability} "
                "</span>")
        if self.graph_db_path:
            chip += (f"<span style='color:#8a9299'>@"
                     f"{tail(self.graph_db_path, 28)} </span>")
        chips = [("stage", self.stage.upper()), ("journal", chip)]
        model = self._model_chip()
        if model:
            chips.append(("model", model))
        self.strip.set_chips(chips)
        hints_key = ("results_drill" if self.stage == "results"
                     and self.results_run is not None else self.stage)
        if self.error:
            self.strip.set_readout(f"⚠ {self.error}", role="warn")
        elif self.busy:
            self.strip.set_readout(self.busy)
        elif self.notice:
            self.strip.set_readout(self.notice)
        else:
            self.strip.clear_readout()
        self.strip.set_hints(HINTS[hints_key] + " · ? keys")
        self.hints_overlay.set_entries(self.keymap.entries())

    # ---- selection state helpers (verbatim Textual ports) ----------------

    def _focused_run(self) -> Optional[Dict[str, Any]]:
        i = self.runs_list.currentRow()
        runs = self.src_index.runs
        if 0 <= i < len(runs):
            return runs[i]
        return None

    def _run_by_path(self, key: str) -> Optional[Dict[str, Any]]:
        for m in self.src_index.runs:
            if m["_path"] == key:
                return m
        return None

    def _run_id_for(self, key: str) -> str:
        m = self._run_by_path(key)
        return str(m["run_id"]) if m else Path(key).stem

    def _resolved_text_from(self, m: Dict[str, Any]) -> Optional[str]:
        return (self.text_from.get(m["_path"])
                or SourceRunIndex.default_text_from(m))

    def _resolved_graph_db(self, m: Dict[str, Any]) -> Optional[str]:
        """Explicit override (flag/state) wins, else the db the run RECORDED
        writing to (e087d059 — provenance-following, never the decomp stack's
        own configured default)."""
        return (self.graph_db_path
                or SourceRunIndex.recorded_graph_db(m, self.graph_capability))

    def _batches(self):
        """The hand-off fold: key = (text_from, graph_db_path, event_propset)
        — everything that applies invocation-wide on the core CLI."""
        picks: List[Tuple[str, Tuple[Optional[str], Optional[str],
                                     Optional[str]]]] = []
        for key in self.picked:
            m = self._run_by_path(key)
            if m is not None:
                picks.append((key, (self._resolved_text_from(m),
                                    self._resolved_graph_db(m),
                                    self._resolved_propset(m)
                                    if self.event_split else None)))
        return group_batches(picks)

    def _reload_indexes(self) -> None:
        self.src_index.load()
        self.dec_index.load()
        self.propset_index.load()
        self.trainrun_index.load()
        self.propose_armed = None  # any reload invalidates an armed confirm
        self.archive_armed = None  # … and an armed Shift+X
        self._decomp_counts = self.dec_index.counts_by_source_manifest()
        # A reload may evict manifests the batch still names; drop those picks
        # rather than hand off paths the core would refuse.
        self.picked = [p for p in self.picked
                       if self._run_by_path(p) is not None]

    def _propsets_for_source(self, s: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Every proposal set over ONE run source, newest first."""
        return self.propset_index.for_source(
            content_hash=str(s.get("content_hash") or "") or None,
            source_path=str(s.get("source_path") or "") or None)

    def _propsets_for(self, m: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
        """Per SOURCE of the run (manifest order): its sets, newest first — a
        multi-source run is N independent source-bound rings (decomp
        manifest 0.2.6: the carve joins each set to its source)."""
        return [self._propsets_for_source(s) for s in (m.get("sources") or [])]

    @staticmethod
    def _source_key(m: Dict[str, Any], i: int) -> str:
        """The Shift+E pick key: run path + source index."""
        return f"{m['_path']}#{i}"

    def _resolved_propsets(self, m: Dict[str, Any]) -> List[Optional[str]]:
        """Per source: CLI pin (single-source runs only — one pointer carves
        one source) > Shift+E override > latest-by-source (DEC ae450551).
        None = that source has no set."""
        srcs = m.get("sources") or []
        if self.propset_pin:
            return ([self.propset_pin] if len(srcs) == 1
                    else [None] * len(srcs))
        out: List[Optional[str]] = []
        for i, sets in enumerate(self._propsets_for(m)):
            pick = self.propset_pick.get(self._source_key(m, i))
            if pick and not any(s["_path"] == pick for s in sets):
                pick = None  # the picked set was archived (Shift+X) or evicted
            out.append(pick or (sets[0]["_path"] if sets else None))
        return out

    def _resolved_propset(self, m: Dict[str, Any]) -> Optional[Tuple[str, ...]]:
        """The run's carve authority for the hand-off: ONE pointer per source
        as a hashable tuple (it joins the group key; batch_argv renders every
        pointer), or None when ANY source lacks a set — the refusal the
        confirm guard names."""
        ps = self._resolved_propsets(m)
        return tuple(ps) if ps and all(ps) else None

    def _propset_chip(self, m: Dict[str, Any]) -> Optional[str]:
        """The run row's ⚡ label: per-source set summaries ('/'-joined for
        a multi-source run); a PARTIAL cover leads with '⚠k/n no set' so
        run_rows paints it red; None = no source has a set."""
        resolved = self._resolved_propsets(m)
        if not resolved or not any(resolved):
            return None
        parts: List[str] = []
        for ps, sets in zip(resolved, self._propsets_for(m)):
            if ps is None:
                parts.append("no set")
                continue
            chosen = next((s for s in sets if s["_path"] == ps), None)
            parts.append(PropsetIndex.summary(chosen) if chosen else tail(ps, 20))
        missing = sum(1 for ps in resolved if ps is None)
        chip = " / ".join(parts)
        return f"⚠{missing}/{len(resolved)} no set · {chip}" if missing else chip

    def _propose_targets(self) -> List[Dict[str, Any]]:
        """Picked single-source runs, one per source — sources that ALREADY
        have a set included: a propose always mints a NEW versioned set
        beside the old ones (work item 53911bef, coexist — the capability
        mints a fresh proposal_set_id per invocation and Shift+E cycles a
        source's sets newest-first), so regenerating with a newer model no
        longer needs a shell delete of the prior set."""
        out: List[Dict[str, Any]] = []
        seen: set = set()
        for p in self.picked:
            m = self._run_by_path(p)
            if m is None:
                continue
            # ONE target per SOURCE (decomp manifest 0.2.6): a multi-source
            # transcription run contributes every source it names — propose
            # is per source and the carve joins each set to its source —
            # deduped by content hash across the picked runs.
            resolved = self._resolved_propsets(m)
            for i, s in enumerate(m.get("sources") or []):
                key = str(s.get("content_hash") or s.get("source_path")
                          or f"{p}#{i}")
                if key in seen:
                    continue
                seen.add(key)
                out.append({"run": m, "index": i,
                            "source_path": str(s.get("source_path") or ""),
                            "content_hash": str(s.get("content_hash") or ""),
                            "has_set": bool(i < len(resolved) and resolved[i])})
        return out

    # ---- gestures (stage-dispatched, busy-gated like the Textual actions) ----

    def move_cursor(self, delta: int) -> None:
        if self.busy:
            return
        lst = self._active_list()
        if lst.count():
            lst.setCurrentRow(max(0, min(lst.count() - 1,
                                         lst.currentRow() + delta)))

    def on_select(self) -> None:
        if self.busy:
            return
        if self.stage == "runs":
            m = self._focused_run()
            if m is None:
                return
            if not SourceRunIndex.transcribers(m):
                # The core would refuse it at run time ("no transcribers");
                # surfacing that at PICK time keeps the hand-off clean.
                self.error = f"{m['run_id']}: manifest lists no transcribers"
                self._refresh_status()
                return
            key = m["_path"]
            if key in self.picked:
                self.picked.remove(key)
            else:
                self.picked.append(key)
            self.error = None
            self._paint_runs()
            self._refresh_status()
        elif self.stage == "results" and self.results_run is None:
            if self.dec_index.runs:
                idx = self.res_list.currentRow()
                if 0 <= idx < len(self.dec_index.runs):
                    self.results_run = idx
                    self._paint_results()
                    self._refresh_status()
        elif self.stage == "results":
            self._open_segments()
        elif self.stage == "vadform":
            self._form_edit()

    def on_cycle_text_from(self) -> None:
        """t: cycle the focused run's authoritative transcriber (per run —
        grouping folds equal picks back into shared invocations)."""
        if self.stage != "runs" or self.busy:
            return
        m = self._focused_run()
        if m is None:
            return
        ts = SourceRunIndex.transcribers(m)
        if len(ts) < 2:
            self.notice = ("single-transcriber run — text-from is its sole "
                           "transcriber")
            self._refresh_status()
            return
        cur = self._resolved_text_from(m)
        nxt = ts[(ts.index(cur) + 1) % len(ts)] if cur in ts else ts[-1]
        self.text_from[m["_path"]] = nxt
        self.notice = None
        self._paint_runs()
        self._refresh_status()

    def on_results(self) -> None:
        """v: the decomp-runs view (re-reads manifests so a batch that just
        finished elsewhere shows without a restart); in the VAD form, v walks
        the last probe's predicted skeleton."""
        if self.stage == "vadform":
            if self.probe_view_display:
                self.notice = None
                self._show_stage("probeview")
            else:
                self.error = "no probe yet — p runs the form's config first"
                self._refresh_status()
            return
        if self.stage != "runs" or self.busy:
            return
        self.notice = None
        self.error = None
        self._reload_indexes()
        self.results_run = None
        self._show_stage("results")

    def on_reload(self) -> None:
        """r: reload the indexes (runs) / play the focused span (segments,
        probeview — the correction-TUI replay key)."""
        if self.stage in ("segments", "probeview"):
            self._play_focused()
            return
        if self.stage != "runs" or self.busy:
            return
        self._reload_indexes()
        self.notice = (f"reloaded: {len(self.src_index.runs)} transcription / "
                       f"{len(self.dec_index.runs)} decomp run(s)")
        self._paint_runs()
        self._refresh_status()

    def on_back(self) -> None:
        if self.busy:
            return
        if self.stage == "probeview":
            self._show_stage("vadform")
        elif self.stage == "vadform":
            self._show_stage("segments")
        elif self.stage == "segments":
            self._stop_player()
            self._show_stage("results")  # seat stays open — b is a browse
        elif self.stage == "results":
            if self.results_run is not None:
                self.results_run = None
                self._paint_results()
                self._refresh_status()
            else:
                self._show_stage("runs")

    def on_toggle_respine(self) -> None:
        """Shift+R: fresh spine under the SAME config (DEC 9241564f) — rides
        the plan into the hand-off argv as --respine."""
        if self.stage != "runs" or self.busy:
            return
        self.respine = not self.respine
        self.notice = ("batch respine ON — confirmed groups run --respine "
                       "(fresh spine, same config; coexists with the old one)"
                       if self.respine else "batch respine off")
        self._paint_runs()
        self._refresh_status()

    def on_toggle_event_split(self) -> None:
        """e: the event-carve stage batch-wide (DEC ae450551). Arming flips
        sentence-split OFF (the ratified respine seam default; s re-enables
        explicitly — stages stay composable per 6cc10fb7)."""
        if self.stage != "runs" or self.busy:
            return
        self.event_split = not self.event_split
        if self.event_split:
            self.sentence_split = False
            self.notice = ("batch event-split ON (sentence-split off — the "
                           "respine seam default; s re-enables) — rows show "
                           "each run's resolved propset")
        else:
            self.notice = "batch event-split off"
        self._paint_runs()
        self._refresh_status()

    def on_cycle_propset(self) -> None:
        """Shift+E: cycle the focused run's proposal set among its source's
        sets (newest first — older sets are earlier model generations)."""
        if self.stage != "runs" or not self.event_split or self.busy:
            return
        m = self._focused_run()
        if m is None:
            return
        if self.propset_pin:
            self.notice = "propset pinned by --event-propset — cycle disabled"
            self._refresh_status()
            return
        # ONE ring over every (source, set) pair of the run (0.2.6): a
        # single-source run cycles its sets as before; a multi-source run
        # walks source 0's sets, then source 1's, ... — each step picks that
        # source's set, so every source keeps its own effective pick.
        rings = self._propsets_for(m)
        pairs = [(i, s) for i, sets in enumerate(rings) for s in sets]
        if len(pairs) < 2:
            self.notice = ("no alternate proposal sets for this source"
                           if pairs else "no proposal set matches this source")
            self._refresh_status()
            return
        resolved = self._resolved_propsets(m)
        cur = self.propset_cursor.get(m["_path"])
        if cur is None:   # park on the first source's effective set
            cur = next((k for k, (i, s) in enumerate(pairs)
                        if i == 0 and s["_path"] == resolved[0]), 0)
        nxt = (cur + 1) % len(pairs)
        i, chosen = pairs[nxt]
        self.propset_cursor[m["_path"]] = nxt
        self.propset_pick[self._source_key(m, i)] = chosen["_path"]
        ring = [s["_path"] for s in rings[i]]
        src = Path(str((m.get("sources") or [{}])[i].get("source_path")
                       or f"source-{i}")).stem
        where = (f"src {i + 1}/{len(rings)} {src[:24]} · "
                 if len(rings) > 1 else "")
        self.notice = (f"{where}propset → {PropsetIndex.summary(chosen)} · model "
                       f"…{str(chosen.get('training_run_id') or '?')[-8:]} "
                       f"({ring.index(chosen['_path']) + 1}/{len(ring)}, "
                       f"newest first)")
        self._paint_runs()
        self._refresh_status()

    def on_toggle_split(self) -> None:
        """s: the BATCH sentence-split flag in RUNS; the probe PREVIEW toggle
        in the form/skeleton views (re-derives in place, no re-probe)."""
        if self.busy:
            return
        if self.stage == "runs":
            self.sentence_split = not self.sentence_split
            self.notice = ("batch sentence-split ON — confirmed groups run "
                           "--sentence-split (parallel spine, new skeleton "
                           "hash)" if self.sentence_split
                           else "batch sentence-split off")
            self._paint_runs()
            self._refresh_status()
            return
        if self.stage not in ("vadform", "probeview"):
            return
        self.probe_split = not self.probe_split
        if self._probe_ctx is not None:
            self._rebuild_probe_view()
            self._paint_stage()
        else:
            self.notice = (f"sentence-split preview "
                           f"{'ON' if self.probe_split else 'off'}")
            self._paint_vadform()
        self._refresh_status()

    # ---- the propose pre-stage (Shift+P, two-step) -----------------------

    def on_propose_missing(self) -> None:
        """Shift+P (DEC 1cfe6d0f): propose over every picked event-armed
        source — a source that already has a set gets a NEW one beside it
        (53911bef coexist; the fresh set becomes that source's default).
        TWO-STEP: the first press names the resolved model and the fresh/
        regenerate split, the second runs it — a wrong model burns a walk
        session (b9717422), so the human stays between resolution and
        inference."""
        if self.stage != "runs" or self.propose_busy or self.busy:
            return
        if not self.event_split:
            self.error = ("arm event-split (e) first — propose serves "
                          "event-armed runs")
            self._refresh_status()
            return
        if self.propose_armed is not None:
            armed, self.propose_armed = self.propose_armed, None
            self.error = None
            self.propose_busy = True
            sources = [t["source_path"] for t in armed["targets"]]
            fut = self.sess.propose_batch(self.event_capability,
                                          armed["run"]["_path"], sources,
                                          progress=self.progress_note.emit)
            fut.add_done_callback(self.propose_done.emit)
            self._armed_run = armed["run"]
            self._armed_targets = list(armed["targets"])
            return
        targets = self._propose_targets()
        if not targets:
            self.error = "pick runs first — propose serves the picked batch"
            self._refresh_status()
            return
        self.trainrun_index.load()
        run, why = self.trainrun_index.resolve_pick(self.training_run_pick,
                                                    self.training_run_pin)
        if run is None:
            self.error = why
            self._refresh_status()
            return
        regen = sum(1 for t in targets if t["has_set"])
        self.propose_armed = {"run": run, "targets": targets}
        self.error = None
        self.notice = (f"Shift+P again: propose {len(targets)} source(s) with "
                       f"{TrainingRunIndex.summary(run)} — {why}"
                       + (f" · {regen} regenerate (a new set beside the old; "
                          f"Shift+E cycles)" if regen else "")
                       + "; m picks another run")
        self._refresh_status()

    def on_pick_model(self) -> None:
        """m (work item a13ebc66): the training-run picker — choose the
        propose model in-app. The pick is session-scoped and outranks the
        workspace marker's pin (resolve_pick: pick > pin > newest); the yaml
        pin demotes to the fallback default and is never rewritten, so its
        model-register comments survive. Any armed Shift+P confirm is
        dropped on pick — the model it named may no longer be the one."""
        if self.stage != "runs" or self.propose_busy or self.busy:
            return
        self.trainrun_index.load()
        if not self.trainrun_index.runs and not self.trainrun_index.archived:
            self.error = (f"no training runs under "
                          f"{self.trainrun_index.training_runs_dir}")
            self._refresh_status()
            return
        self.trainrun_picker.open_for(self.trainrun_index.runs,
                                      self.training_run_pin,
                                      self.training_run_pick,
                                      archived=self.trainrun_index.archived)

    def _on_trainrun_picked(self, run: Dict[str, Any]) -> None:
        self.training_run_pick = run["_path"]
        self.propose_armed = None
        self.error = None
        self.notice = (f"propose model → {TrainingRunIndex.summary(run)} "
                       f"(picked for this sitting; Shift+P proposes with it)")
        self._refresh_status()

    def _on_trainrun_lifecycle(self, run: Dict[str, Any],
                               verb: str) -> Tuple[List[Dict[str, Any]],
                                                   List[Dict[str, Any]], str]:
        """a / x in the m picker (b20cb911, DEC 14678fc9): archive, unarchive
        or delete the cursored run through the substrate lifecycle seam — a
        sidecar beside the manifest, never the manifest itself. Archive is
        reversible visibility (the run leaves the ladder; a pin or pick on
        it turns into a loud miss); delete serves an ARCHIVED run nothing
        under the workspace still names (a propset's training_run_id, the
        yaml pin) — the refusal names each holder. Hands the picker its
        refreshed lists + the note its foot echoes."""
        lc = ArtifactLifecycle(Path(run["_path"]).parent)
        rid = str(run.get("run_id") or "?")[-8:]
        try:
            if verb == "delete":
                root = Path(self.trainrun_index.training_runs_dir).parent
                holders = find_holders(root, artifact_id(lc.dir), exclude_dir=lc.dir)
                lc.delete(holders=holders)
                if self.training_run_pick == run["_path"]:
                    self.training_run_pick = None
                note = f"deleted …{rid}"
            else:
                rec, changed = getattr(lc, verb)(
                    actor="user:decomp-qt",
                    reason=f"{verb}d in the decomp app (m picker)")
                note = f"…{rid} {rec['state']}" + ("" if changed else " (already)")
        except LifecycleRefusal as e:
            note = f"refused: {e}"
        self.trainrun_index.load()
        self.propose_armed = None
        self.notice = note
        self._refresh_status()
        return self.trainrun_index.runs, self.trainrun_index.archived, note

    def on_propset_picker(self) -> None:
        """x (b20cb911's unarchive rung, on the 8d29f0f0 kit shell): the
        focused run's cursor-source proposal sets as a lifecycle picker —
        the Shift+E ring never lists an ARCHIVED set, so unarchive needs
        this listing. The cursor source = the one Shift+E last parked on,
        else source 0 (the on_archive_propset rule)."""
        if self.stage != "runs" or not self.event_split or self.busy:
            return
        m = self._focused_run()
        if m is None:
            return
        if self.propset_pin:
            self.notice = "propset pinned by --event-propset — picker disabled"
            self._refresh_status()
            return
        srcs = m.get("sources") or []
        if not srcs:
            return
        rings = self._propsets_for(m)
        pairs = [(i, s) for i, sets in enumerate(rings) for s in sets]
        cur = self.propset_cursor.get(m["_path"])
        i = pairs[cur][0] if cur is not None and cur < len(pairs) else 0
        i = min(i, len(srcs) - 1)
        self._propset_picker_ctx = (m["_path"], i)
        resolved = self._resolved_propsets(m)
        self.propset_picker.open_for(
            Path(str(srcs[i].get("source_path") or "?")).name,
            rings[i] if i < len(rings) else [],
            archived=self._archived_propsets_for_source(srcs[i]),
            effective=resolved[i] if i < len(resolved) else None)

    def _archived_propsets_for_source(self, s: Dict[str, Any]) -> List[Dict[str, Any]]:
        """The ARCHIVED sets bound to one run source — the for_source join
        applied to the index's archived half (for_source itself reads only
        the active sets, deliberately: rings never see archived)."""
        ch = str(s.get("content_hash") or "") or None
        sp = str(s.get("source_path") or "") or None
        out: List[Dict[str, Any]] = []
        for m in self.propset_index.archived:
            src = m.get("source") or {}
            if ((ch and src.get("content_hash") == ch)
                    or (sp and str(src.get("path") or "") == sp)):
                out.append(m)
        return out

    def _on_propset_picked(self, chosen: Dict[str, Any]) -> None:
        """A pick lands in the Shift+E slot (session-scoped, like the model
        pick): that source's effective set until re-picked or archived."""
        ctx = self._propset_picker_ctx
        if ctx is None:
            return
        run_path, i = ctx
        m = self._run_by_path(run_path)
        if m is None:
            return
        self.propset_pick[self._source_key(m, i)] = chosen["_path"]
        self.propset_cursor.pop(run_path, None)
        self.archive_armed = None
        self.error = None
        self.notice = (f"propset → {PropsetIndex.summary(chosen)} "
                       f"(picked for this sitting)")
        self._paint_runs()
        self._refresh_status()

    def _on_propset_lifecycle(self, chosen: Dict[str, Any],
                              verb: str) -> Tuple[List[Dict[str, Any]],
                                                  List[Dict[str, Any]], str]:
        """a / x in the propset picker: archive, unarchive or delete the
        cursored set through the substrate lifecycle seam (the
        _on_trainrun_lifecycle contract) — hands the picker its refreshed
        source-scoped lists + the note its foot echoes."""
        ctx = self._propset_picker_ctx
        lc = ArtifactLifecycle(Path(str(chosen.get("_path") or "")).parent)
        sid = str(chosen.get("proposal_set_id") or "?")[-8:]
        try:
            if verb == "delete":
                root = Path(self.propset_index.proposals_dir).parent
                holders = find_holders(root, artifact_id(lc.dir),
                                       exclude_dir=lc.dir)
                lc.delete(holders=holders)
                note = f"deleted …{sid}"
            else:
                rec, changed = getattr(lc, verb)(
                    actor="user:decomp-qt",
                    reason=f"{verb}d in the decomp app (propset picker)")
                note = f"…{sid} {rec['state']}" + ("" if changed else " (already)")
        except LifecycleRefusal as e:
            note = f"refused: {e}"
        self._reload_indexes()
        self.archive_armed = None
        sets: List[Dict[str, Any]] = []
        archived: List[Dict[str, Any]] = []
        if ctx is not None:
            m = self._run_by_path(ctx[0])
            srcs = (m.get("sources") or []) if m is not None else []
            if m is not None and ctx[1] < len(srcs):
                sets = self._propsets_for_source(srcs[ctx[1]])
                archived = self._archived_propsets_for_source(srcs[ctx[1]])
        self.notice = note
        self._paint_runs()
        self._refresh_status()
        return sets, archived, note

    def on_archive_propset(self) -> None:
        """Shift+X (b20cb911, DEC 14678fc9): archive the focused run's
        EFFECTIVE proposal set — the source Shift+E last parked on, else
        source 0 — through the lifecycle sidecar: the set leaves every ring
        and the source's default falls to its next-newest set (or 'no set');
        the manifest stays on disk (the lifecycle CLI lists / unarchives it).
        TWO-STEP like Shift+P: the first press names the set."""
        if (self.stage != "runs" or not self.event_split or self.busy
                or self.propose_busy):
            return
        m = self._focused_run()
        if m is None:
            return
        if self.propset_pin:
            self.notice = "propset pinned by --event-propset — archive disabled"
            self._refresh_status()
            return
        rings = self._propsets_for(m)
        resolved = self._resolved_propsets(m)
        pairs = [(i, s) for i, sets in enumerate(rings) for s in sets]
        cur = self.propset_cursor.get(m["_path"])
        i = pairs[cur][0] if cur is not None and cur < len(pairs) else 0
        target = resolved[i] if i < len(resolved) else None
        chosen = (next((s for s in rings[i] if s["_path"] == target), None)
                  if target else None)
        if chosen is None:
            self.archive_armed = None
            self.notice = "no proposal set to archive for this source"
            self._refresh_status()
            return
        label = (f"{PropsetIndex.summary(chosen)} · model "
                 f"…{str(chosen.get('training_run_id') or '?')[-8:]}")
        if self.archive_armed != target:
            self.archive_armed = target
            self.notice = (f"Shift+X again: archive propset {label} — leaves "
                           f"the ring, stays on disk ({len(rings[i]) - 1} other "
                           f"set(s) for this source)")
            self._refresh_status()
            return
        self.archive_armed = None
        ArtifactLifecycle(Path(target).parent).archive(
            actor="user:decomp-qt", reason="archived in the decomp app (Shift+X)")
        self.propset_pick.pop(self._source_key(m, i), None)
        self.propset_cursor.pop(m["_path"], None)
        self._reload_indexes()
        left = len(self._propsets_for(m)[i]) if i < len(rings) else 0
        self.notice = (f"archived propset {label} — {left} set(s) remain for "
                       f"this source; python -m cjm_substrate.utils.lifecycle "
                       f"unarchives")
        self._paint_runs()
        self._refresh_status()

    def _model_chip(self) -> Optional[str]:
        """The strip's model chip: the run Shift+P would propose with and
        why (picked/pin/newest), or the loud miss — only while event-split
        is armed, since propose serves event-armed runs."""
        if not self.event_split:
            return None
        run, why = self.trainrun_index.resolve_pick(self.training_run_pick,
                                                    self.training_run_pin)
        if run is None:
            return "⚠ no model"
        tag = {"picked": "picked", "workspace pin": "pin"}.get(why, "newest")
        return f"{str(run.get('run_id') or '?')[-8:]} ({tag})"

    def _on_propose_done(self, fut) -> None:
        self.propose_busy = False
        try:
            res = fut.result()
        except Exception as e:
            res = {"landed": 0, "error": str(e)}
        if res["error"]:
            self.error = (f"propose failed after {res['landed']} set(s): "
                          f"{res['error']}")
            self.notice = None
        else:
            self.error = None
            self.notice = (f"⚡ {res['landed']} propset(s) landed with "
                           f"{TrainingRunIndex.summary(self._armed_run)}")
        # A regenerated source's fresh set is its new default (53911bef):
        # drop the operator's older Shift+E pick so latest-per-source rules
        # again — an older generation stays one Shift+E away.
        for t in self._armed_targets:
            self.propset_pick.pop(self._source_key(t["run"], t["index"]), None)
        self.propset_cursor.clear()
        self._reload_indexes()
        if self.stage == "runs":
            self._paint_runs()
        self._refresh_status()

    def _on_progress_note(self, text: str) -> None:
        self.notice = text
        self._refresh_status()

    # ---- the segments drill (results enter -> the session seat) ----------

    def _open_segments(self) -> None:
        """Drill the focused results source into its committed fine spine —
        the session opens the READ-ONLY seat against the db this decomp run
        RECORDED (e087d059; explicit --graph-db-path still wins)."""
        m = self.dec_index.runs[self.results_run]
        srcs = m.get("sources") or []
        i = self.res_list.currentRow()
        if not srcs or not (0 <= i < len(srcs)):
            return
        src = srcs[i]
        ids = [str(x) for x in (src.get("segment_ids") or [])]
        if not ids:
            self.error = (f"{m['run_id']}: no segment_ids recorded for this "
                          "source (pre-segment_ids manifest?)")
            self._refresh_status()
            return
        db = (self.graph_db_path
              or SourceRunIndex.recorded_graph_db(m, self.graph_capability))
        if not db or not Path(db).exists():
            self.error = ("no graph db recorded for this decomp run — pass "
                          "--graph-db-path" if not db else
                          f"recorded graph db missing on disk: {tail(db, 40)}")
            self._refresh_status()
            return
        self.busy = f"opening graph (read-only) · {tail(db, 32)} …"
        self.error = None
        self._refresh_status()
        self._seg_pending = (m, src, ids)
        fut = self.sess.open_source(db, ids, str(src.get("source_node_id")
                                                 or ""))
        fut.add_done_callback(self.seg_opened.emit)

    def _on_seg_opened(self, fut) -> None:
        self.busy = None
        m, src, ids = self._seg_pending
        try:
            res = fut.result()
        except Exception as e:
            # SystemExit-shaped library exits resolve as exceptions here — an
            # async failure must paint, never take the window with it.
            self.error = f"graph open failed: {e}"
            self._refresh_status()
            return
        rows = res["rows"]
        self.seg_rows = rows
        self.seg_missing = len(ids) - len(rows)
        self.seg_title = str(src.get("title") or "?")
        self.seg_vad = vad_summary(m)
        self.seg_source_id = str(src.get("source_node_id") or "")
        self.seg_asegs = res["asegs"]
        cfg = m.get("config") or {}
        caps = m.get("capabilities") or {}
        self.seg_vad_id = str(cfg.get("vad_capability")
                              or "cjm-capability-silero-vad")
        self.seg_vad_cfg = dict((caps.get(self.seg_vad_id) or {})
                                .get("config") or {})
        self.seg_fa_id = str(cfg.get("fa_capability")
                             or "cjm-capability-qwen3-forced-aligner")
        self.seg_fa_cfg = dict((caps.get(self.seg_fa_id) or {})
                               .get("config") or {})
        # B.5 probe parity: the split preview segments with the RUN's
        # segmenter when the manifest recorded one, else the pipeline default.
        self.seg_seg_id = str(cfg.get("seg_capability") or "cjm-capability-pysbd")
        self.seg_seg_cfg = dict((caps.get(self.seg_seg_id) or {})
                                .get("config") or {})
        self.seg_text_from = str(cfg.get("text_from") or "")
        self.seg_display = build_display(rows, self.gap_threshold)
        self.notice = None
        self.error = None
        self._show_stage("segments")

    # ---- playback (r on segments/probeview; gaps included) ---------------

    def _stop_player(self) -> None:
        if self.player is not None:
            self.player.stop()

    def on_stop_audio(self) -> None:
        if self.player is not None:
            self.player.stop()
            self.notice = None
            self._refresh_status()

    def nudge_speed(self, delta: float) -> None:
        """[ / ] playback-speed ladder 0.5–3.0 (QMediaPlayer playbackRate)."""
        if self.stage not in ("segments", "probeview"):
            return
        self.speed = max(0.5, min(3.0, self.speed + delta))
        self.notice = f"speed ×{self.speed:g}"
        self._refresh_status()

    def _play_focused(self) -> None:
        """Play the focused entry's audio span — gaps INCLUDED: their audio
        exists in NO fine chunk, but the coarse model-input WAV spans it
        (locate_span resolves the owning WAV + local times)."""
        if self.stage == "segments" and self.seg_display:
            entries, rows, lst = self.seg_display, self.seg_rows, self.seg_list
        elif self.stage == "probeview" and self.probe_view_display:
            entries, rows, lst = (self.probe_view_display,
                                  self.probe_view_rows, self.pv_list)
        else:
            return
        cur = lst.currentRow()
        if not (0 <= cur < len(entries)):
            return
        kind, ri, secs = entries[cur]
        r = rows[ri]
        if kind == "gap":
            s = float(r.get("start_time") or 0.0)
            span = (s - secs, s)
        else:
            if r.get("start_time") is None or r.get("end_time") is None:
                return
            span = (float(r["start_time"]), float(r["end_time"]))
        loc = locate_span(self.seg_asegs, span[0], span[1])
        if loc is None:
            self.error = "no model-input WAV covers this span"
            self._refresh_status()
            return
        wav, ls, le = loc
        if self.player is None:
            self.player = SpanPlayer(self)
        self.player.play_span(wav, ls, le, rate=self.speed)
        err = self.player.error_text()
        if err:
            self.error = f"playback unavailable: {err}"
        else:
            self.error = None
            self.notice = (f"▶ {fmt_span(span)} ×{self.speed:g}")
        self._refresh_status()

    # ---- the VAD probe pair (c form -> p probe -> v skeleton) -------------

    def on_vad_config(self) -> None:
        """c (segments): open the VAD probe form — rows from the capability's
        manifest config_schema (kit ConfigForm), seeded with the RUN's
        recorded config; the probe target freezes to the coarse chunk holding
        the focused entry (gap start for gap rows)."""
        if self.stage != "segments" or self.busy:
            return
        schema = capability_config_schema(self.manifests_dir, self.seg_vad_id)
        form = ConfigForm.from_schema(schema)
        if not form.fields:
            self.error = (f"{self.seg_vad_id}: no config_schema in "
                          f"{self.manifests_dir}")
            self._refresh_status()
            return
        form.apply(self.seg_vad_cfg)
        self.vad_form = form
        self.probe_result = None
        self._probe_ctx = None
        self.probe_view_rows = []
        self.probe_view_display = []
        t = None
        i = self.seg_list.currentRow()
        if 0 <= i < len(self.seg_display):
            kind, ri, secs = self.seg_display[i]
            r = self.seg_rows[ri]
            if r.get("start_time") is not None:
                t = float(r["start_time"]) - (secs if kind == "gap" else 0.0)
        self.probe_aseg = (aseg_index_for(self.seg_asegs, t)
                           if t is not None else None)
        self.error = None
        self._show_stage("vadform")

    def _form_edit(self) -> None:
        """enter on a form row: closed sets cycle in place; open kinds take a
        modal value dialog (parse failures stay in-status, value untouched)."""
        if self.vad_form is None:
            return
        i = self.form_list.currentRow()
        if not (0 <= i < len(self.vad_form.fields)):
            return
        field = self.vad_form.fields[i]
        if field.cycle():
            self.error = None
            self._paint_vadform()
            return
        text, ok = QInputDialog.getText(self, "value",
                                        f"{field.title}:", text=field.render())
        if not ok:
            return
        try:
            field.parse(text)
        except ValueError as e:
            self.error = f"{field.key}: {e}"
            self._refresh_status()
            return
        self.error = None
        self._paint_vadform()
        self._refresh_status()

    def on_probe(self) -> None:
        """p (vadform): run the form's config over the target coarse WAV
        (probe-only, ee4a4b9c) — VAD sweep + FA realign + sentence spans as
        one session Future."""
        if (self.stage != "vadform" or self.vad_form is None
                or self.probe_busy or self.busy):
            return
        if self.probe_aseg is None or not self.seg_asegs:
            self.error = "no probe target — reopen the form from a timed entry"
            self._refresh_status()
            return
        a = self.seg_asegs[self.probe_aseg]
        if not a.get("wav"):
            self.error = "target coarse chunk has no model-input WAV on disk"
            self._refresh_status()
            return
        cfg = {f.key: f.value for f in self.vad_form.fields}
        self.probe_busy = True
        self.error = None
        self._paint_vadform()
        self._refresh_status()
        fut = self.sess.probe_chain(self.seg_vad_id, cfg, a["wav"],
                                    rendition_id=a.get("rendition"),
                                    text_from=self.seg_text_from,
                                    fa_id=self.seg_fa_id,
                                    fa_cfg=self.seg_fa_cfg,
                                    seg_id=self.seg_seg_id,
                                    seg_cfg=self.seg_seg_cfg,
                                    progress=self.progress_note.emit)
        fut.add_done_callback(self.probe_done.emit)

    def _on_probe_done(self, fut) -> None:
        self.probe_busy = False
        try:
            out = fut.result()
        except Exception as e:
            self.error = f"probe failed: {e}"
            if self.stage == "vadform":
                self._paint_vadform()
            self._refresh_status()
            return
        a = self.seg_asegs[self.probe_aseg]
        self._probe_ctx = (a, out["local"], out["fa_text"], out["fa_words"],
                           out["seg_spans"])
        if out["fa_error"]:
            self.notice = (f"FA realign unavailable ({out['fa_error']}) — "
                           "text borrowed")
        elif out["seg_error"]:
            self.notice = (f"sentence-segmentation unavailable "
                           f"({out['seg_error']}) — split preview off")
        self._rebuild_probe_view()
        if self.stage == "vadform":
            self._paint_vadform()
        self._refresh_status()

    def _rebuild_probe_view(self) -> None:
        """Derive the probe readout + walkable predicted rows from the last
        probe (re-derivable when s toggles — the VAD + FA outputs are
        unchanged, only the skeleton refinement differs; the pipeline's OWN
        split stage runs probe-side, DEC f1024568 deliverable d)."""
        if self._probe_ctx is None:
            return
        a, local, fa_text, fa_words, seg_spans = self._probe_ctx
        chunks_local = list(local)
        realigned = None
        n_cuts = 0
        if fa_text and fa_words is not None:
            if self.probe_split and seg_spans is not None:
                chunks_local = split_predicted(fa_text, fa_words, local,
                                               seg_spans)
                n_cuts = len(chunks_local) - len(local)
            realigned = realign_rows(fa_text, fa_words, chunks_local)
        predicted = [(s + a["start"], e + a["start"]) for s, e in chunks_local]
        committed = [(float(r["start_time"]), float(r["end_time"]))
                     for r in self.seg_rows
                     if r.get("start_time") is not None
                     and r.get("end_time") is not None
                     and a["start"] <= float(r["start_time"]) < a["end"]]
        gaps = []
        for kind, ri, secs in self.seg_display:
            if kind != "gap":
                continue
            gs = float(self.seg_rows[ri].get("start_time") or 0.0) - secs
            if a["start"] <= gs < a["end"]:
                gaps.append((gs, gs + secs))
        self.probe_result = probe_compare(committed, predicted, gaps)
        self.probe_view_rows = predicted_rows(predicted, self.seg_rows, gaps,
                                              realigned=realigned)
        if n_cuts > 0:
            # Mark the rows the split stage minted (spans absent from the raw
            # VAD prediction) — the ✂ is the judgment target.
            originals = {(round(s, 3), round(e, 3)) for s, e in local}
            for row, (ls, le) in zip(self.probe_view_rows, chunks_local):
                row["split"] = (round(ls, 3), round(le, 3)) not in originals
        self.probe_view_display = build_display(self.probe_view_rows,
                                                self.gap_threshold)
        if fa_words is not None:
            split_tag = (f" · sentence-split ×{n_cuts}" if self.probe_split
                         else "")
            self.notice = (f"v walks the predicted skeleton "
                           f"(FA-realigned{split_tag})")
        elif self.probe_split:
            self.notice = ("split preview needs FA — realign unavailable, "
                           "raw skeleton shown")
        elif not (self.notice or "").startswith("FA realign unavailable"):
            self.notice = "v walks the predicted skeleton (text borrowed)"

    # ---- confirm + exits -------------------------------------------------

    def on_confirm(self) -> None:
        if self.stage != "runs" or self.busy:
            return
        if not self.picked:
            self.error = "pick at least one transcription run first"
            self._refresh_status()
            return
        for (tf, db, ps), paths in self._batches():
            if db is None:
                # Proceeding without a db is the guaranteed-wrong-graph
                # failure the first drive hit (e087d059) — block, don't warn.
                self.error = (f"no graph db recorded for "
                              f"{self._run_id_for(paths[0])} — pass "
                              "--graph-db-path to override")
                self._refresh_status()
                return
            if self.event_split and ps is None:
                self.error = (f"event-split armed but no proposal set matches "
                              f"{self._run_id_for(paths[0])} — propose over "
                              "its source first (or e disarms)")
                self._refresh_status()
                return
        self.plan = {
            "batches": [{"text_from": tf, "graph_db_path": db,
                         # one pointer per source (0.2.6); a plain list so a
                         # persisted plan round-trips through JSON unchanged
                         "event_propset": (list(ps) if ps else None),
                         "manifests": list(paths)}
                        for (tf, db, ps), paths in self._batches()],
            "runs_dir": str(self.src_index.runs_dir),
            "manifests_dir": self.manifests_dir,
            "sysmon_capability": self.sysmon_capability,
            "graph_capability": self.graph_capability,
            "graph_db_path": self.graph_db_path,
            "sentence_split": self.sentence_split,
            "respine": self.respine,
            "event_split": self.event_split,
        }
        self.close()

    def on_quit(self) -> None:
        self.close()

    def closeEvent(self, event) -> None:
        """ANY exit path: stop audio, tear the seat + loop down blocking (the
        hand-off runs in this terminal right after)."""
        if self.player is not None:
            self.player.close()
            self.player = None
        self.sess.close()
        super().closeEvent(event)


def fmt_span(span: Tuple[float, float]) -> str:
    """Status text for a playing span (source-coordinate seconds)."""
    return f"{fmt_ts(span[0])} → {fmt_ts(span[1])}"
