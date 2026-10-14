# cjm-transcript-decomp-qt

<!-- generated from the context graph by `cjm-context-graph readme` — do not edit by hand; edit the graph (the urge to hand-edit = move it on-graph) -->

Qt shell for the decomp-batch setup app — the second workflow TUI migrated to the PySide6 lane (DEC c4b0d6e5 leg 3, architecture 6c574c89). Imports the pure spine of cjm-transcript-decomp-tui (run/propset/training-run indexes, group_batches, SegmentStack + the fold/probe helpers, sidecar state, discovery) and shares its CLI ladder (build_parser/resolve_split_flags/resolve_settings/hand_off), so Qt-confirmed batches hand off to cjm-transcript-decomp-core byte-identically. The read-only graph seat, the VAD/FA/sentence probe chain, and the P propose pre-stage run under Qt via DecompSession — a kit LoopThreadSession owning the tui SegmentStack, resolving to the shell through queued Signals. Audio = SpanPlayer on QMediaPlayer (position-window span playback for fine segments AND gap rows).

## Modules

- **`cjm_transcript_decomp_qt.__init__`**
- **`cjm_transcript_decomp_qt.app`** — The Qt decomp-batch shell: the same batch queueing over transcription-run
- **`cjm_transcript_decomp_qt.cli`** — Console-script driver for the Qt shell: the SAME argument surface,
- **`cjm_transcript_decomp_qt.panes`** — Pure row/label builders for the decomp Qt shell — the paint logic, Qt-free.
- **`cjm_transcript_decomp_qt.propset_picker`** — Modal proposal-set lifecycle picker (the b20cb911 propset-unarchive
- **`cjm_transcript_decomp_qt.session`** — One read-only decomp inspection seat + the propose pre-stage behind a
- **`cjm_transcript_decomp_qt.trainrun_picker`** — Modal training-run picker for the propose pre-stage (work item a13ebc66):

## API

### `cjm_transcript_decomp_qt.app`

- `DecompWindow` _class_ — Batch-setup window; exit with `plan` set = the CLI hands off headless.
- `fmt_span` _function_ — Status text for a playing span (source-coordinate seconds).

### `cjm_transcript_decomp_qt.cli`

- `main` _function_ — Resolve the shared setup surface, run the Qt batch window, hand off.

### `cjm_transcript_decomp_qt.panes`

- `batch_html` _function_ — The hand-off fold panel: how many core invocations the ordered picks
- `config_diffs` _function_ — The form's deltas from the RUN's recorded config, as key=value chips.
- `decomp_header` _function_
- `decomp_rows` _function_ — One row per decomp run: id, timestamp, source/segment totals, source
- `drill_header_html` _function_ — Drilled decomp run: id + the transcription manifest it consumed.
- `drill_source_rows` _function_ — One row per extended source: title, fine-segment count, Source node.
- `form_rows` _function_ — One row per config field; a value differing from the RUN's recorded
- `probe_detail` _function_ — The focused predicted entry in full, plain text.
- `probe_rows` _function_ — The predicted skeleton painted with the SEGMENTS grammar (direct
- `probe_summary_html` _function_ — The probe readout: predicted vs committed counts + the recovered spans
- `probe_target_html` _function_ — The frozen probe target: which coarse chunk the sweep runs over.
- `probeview_header_html` _function_ — The predicted-skeleton headline: which config produced it, over which
- `run_detail_html` _function_ — Focused-run detail (614dd647): WHAT was transcribed + WHICH graph it
- `run_rows` _function_ — One row per transcription run: ORDERED queue position (the batch is a
- `runs_header` _function_ — The listing header: corpus size + where the manifests come from.
- `segment_detail` _function_ — The focused entry IN FULL as plain text (setPlainText — no truncation,
- `segment_rows` _function_ — The committed fine spine with FOCUSABLE interleaved gap rows — the gap
- `segments_header_html` _function_ — The drilled source's headline: row/gap/missing counts + the recorded
- `trainrun_rows` _function_ — One picker row per training run (work item a13ebc66): the manifest
- `vadform_header_html` _function_ — Form headline + the probe-only doctrine + the split-preview state.

### `cjm_transcript_decomp_qt.propset_picker`

- `PropsetPicker` _class_ — One source's proposal-set ring as a native lifecycle picker.

### `cjm_transcript_decomp_qt.session`

- `DecompSession` _class_ — The loop-thread seat pair for the decomp shell.
- `bench_gestures` _function_ — The decomp shell's canonical gesture for the scale harness (design detail 4389a579): the

### `cjm_transcript_decomp_qt.trainrun_picker`

- `TrainingRunPicker` _class_ — The propose-model picker: a header row (kit modal_header, mouse

## Dependencies

**Depends on:** `PySide6`, `cjm-context-graph-layer`, `cjm-substrate`, `cjm-substrate-qt-kit`, `cjm-transcript-decomp-core`
