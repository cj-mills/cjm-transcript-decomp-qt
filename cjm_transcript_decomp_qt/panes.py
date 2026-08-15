"""Pure row/label builders for the decomp Qt shell — the paint logic, Qt-free.

Each function mirrors a Textual `_paint_*` region of cjm-transcript-decomp-tui's
app.py, re-expressed as row dicts ({"text", "style", "index"}) and small HTML
fragments the shell materializes into QListWidgets, QLabels and
QPlainTextEdits. No windowing (visible_slice) — Qt lists scroll natively — and
detail texts return PLAIN strings for setPlainText, so transcript prose never
passes through markup (the defang problem avoided, the transcription-qt
precedent). Style words match the kit vocabulary (STYLE_COLORS): green/blue/
yellow/red/cyan/magenta/dim + bold; a composed Textual row's span colors
flatten to one row style, the same trade the transcription panes made."""

import html
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from cjm_substrate_tui_kit.viewport import tail
from cjm_transcript_decomp_tui.runs import SourceRunIndex
from cjm_transcript_decomp_tui.segments import fmt_ts


# ---- runs stage ---------------------------------------------------------


def runs_header(n: int, runs_dir: Any) -> str:
    """The listing header: corpus size + where the manifests come from."""
    return (f"<b>Transcription runs ({n})</b> &nbsp;·&nbsp; "
            f"{html.escape(str(runs_dir))}/")


def run_rows(runs: List[Dict[str, Any]],           # SourceRunIndex.runs (newest first)
             picked: List[str],                    # Ordered batch (_path keys)
             text_from: Dict[str, str],            # Operator t-cycle overrides
             decomp_counts: Dict[str, int],        # resolved path -> decomp runs
             event_split: bool,                    # Event-carve armed (⚡ chips paint)
             propset_chip: Callable[[Dict[str, Any]], Optional[str]],
             ) -> List[Dict[str, Any]]:
    """One row per transcription run: ORDERED queue position (the batch is a
    queue, not a set), timestamp, source/segment counts, transcriber chain,
    the resolved text-from chip, the prior-decomp chip, and — event-armed —
    the run's resolved propset (None from `propset_chip` = the missing-set
    refusal the confirm gate will name)."""
    rows: List[Dict[str, Any]] = []
    picked_set = set(picked)
    for i, m in enumerate(runs):
        key = m["_path"]
        mark = (f"[{picked.index(key) + 1}]" if key in picked_set else "[ ]")
        when = time.strftime("%Y-%m-%d %H:%M",
                             time.localtime(float(m.get("created_at") or 0)))
        chunks = [f"{mark} {m['run_id']}", when,
                  f"{len(m.get('sources') or [])} src · "
                  f"{SourceRunIndex.segment_count(m)} seg"]
        names = [t.removeprefix("cjm-capability-")
                 for t in SourceRunIndex.transcribers(m)]
        if names:
            chunks.append("+".join(names))
        tf = text_from.get(key) or SourceRunIndex.default_text_from(m)
        if len(names) > 1 and tf:
            chunks.append(f"tf={tf.removeprefix('cjm-capability-')}")
        n = decomp_counts.get(str(Path(key).resolve()), 0)
        if n:
            chunks.append(f"·decomp×{n}")
        style = "green" if key in picked_set else ""
        if event_split:
            chip = propset_chip(m)
            if chip is None:
                chunks.append("⚡no propset")
                style = "red" if key not in picked_set else "green"
            else:
                chunks.append(f"⚡{chip}")
        rows.append({"text": "  ".join(chunks), "style": style, "index": i})
    if not rows:
        rows.append({"text": "(no transcription run manifests found — "
                             "confirmed transcription runs land here)",
                     "style": "dim", "index": None})
    return rows


def run_detail_html(m: Dict[str, Any],             # The focused run manifest
                    resolved_db: Optional[str],    # Its effective graph db
                    ) -> str:
    """Focused-run detail (614dd647): WHAT was transcribed + WHICH graph it
    landed in, in-pane — identity must not need a transcription-TUI
    round-trip. The db line keeps the Textual warning grammar (no db recorded
    / recorded but missing on disk / ok)."""
    parts: List[str] = []
    srcs = m.get("sources") or []
    for s in srcs[:4]:
        nm = html.escape(Path(str(s.get("source_path") or "?")).stem)
        parts.append(f"&nbsp;· <b>{nm}</b> <span style='color:#8a9299'>"
                     f"{len(s.get('segments') or [])} seg</span>")
    if len(srcs) > 4:
        parts.append(f"<span style='color:#8a9299'>&nbsp;… {len(srcs) - 4} "
                     "more source(s)</span>")
    if resolved_db is None:
        parts.append("<b><span style='color:#c74a3c'>&nbsp;⚠ no graph db "
                     "recorded in this manifest</span></b>")
    elif not Path(resolved_db).exists():
        parts.append(f"<b><span style='color:#c74a3c'>&nbsp;⚠ recorded graph "
                     f"db missing on disk: {html.escape(tail(resolved_db, 40))}"
                     "</span></b>")
    else:
        parts.append(f"<span style='color:#2b8a9d'>&nbsp;graph db: "
                     f"{html.escape(tail(resolved_db, 48))}</span>")
    return "<br>".join(parts)


def batch_html(batches: List[Tuple[Tuple[Optional[str], Optional[str],
                                         Optional[str]], List[str]]],
               run_id_for: Callable[[str], str],   # _path -> display run id
               sentence_split: bool, respine: bool, event_split: bool) -> str:
    """The hand-off fold panel: how many core invocations the ordered picks
    collapse into, one line per (text_from, graph_db, propset) group, with
    the batch-level toggles named the way the Textual panel named them."""
    if not batches:
        return ""
    total = sum(len(paths) for _, paths in batches)
    head = (f"<b>Batch ({total} run(s) -&gt; {len(batches)} "
            f"invocation(s))</b>")
    if sentence_split:
        head += (" <b><span style='color:#9b59b6'>✂ sentence-split ON"
                 "</span></b>")
    if respine:
        head += (" <b><span style='color:#2b8a9d'>⟳ respine ON (fresh spine, "
                 "same config)</span></b>")
    if event_split:
        head += (" <b><span style='color:#9b59b6'>⚡ event-split ON (model "
                 "cuts, ad963c57)</span></b>")
    lines = [head + ":"]
    for bi, ((tf, db, ps), paths) in enumerate(batches):
        short = html.escape((tf or "?").removeprefix("cjm-capability-"))
        line = (f"&nbsp;&nbsp;{bi + 1}. <span style='color:#3f9d55'>"
                f"text-from {short}</span>")
        if db is None:
            line += ("  <b><span style='color:#c74a3c'>⚠ no graph db "
                     "recorded</span></b>")
        elif not Path(db).exists():
            line += (f"  <b><span style='color:#c74a3c'>⚠ db missing: "
                     f"{html.escape(tail(db, 24))}</span></b>")
        else:
            line += (f"  <span style='color:#8a9299'>db "
                     f"{html.escape(tail(db, 24))}</span>")
        if event_split:
            if ps is None:
                line += ("  <b><span style='color:#c74a3c'>⚡ no propset"
                         "</span></b>")
            else:
                line += (f"  <span style='color:#9b59b6'>⚡ "
                         f"{html.escape(Path(ps).parent.name[-12:])}</span>")
        line += (": <span style='color:#8a9299'>"
                 + html.escape(", ".join(run_id_for(p) for p in paths))
                 + "</span>")
        lines.append(line)
    return "<br>".join(lines)


# ---- results stage ------------------------------------------------------


def decomp_header(n: int, runs_dir: Any) -> str:
    return (f"<b>Decomp runs ({n})</b> &nbsp;·&nbsp; "
            f"{html.escape(str(runs_dir))}/")


def decomp_rows(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One row per decomp run: id, timestamp, source/segment totals, source
    titles in-row (614dd647 extended — identity without a drill), text-from."""
    rows: List[Dict[str, Any]] = []
    for i, m in enumerate(runs):
        when = time.strftime("%Y-%m-%d %H:%M",
                             time.localtime(float(m.get("created_at") or 0)))
        srcs = m.get("sources") or []
        segs = sum(int(s.get("segment_count") or 0) for s in srcs)
        chunks = [str(m["run_id"]), when,
                  f"{len(srcs)} source(s) · {segs} seg"]
        names = [str(s.get("title") or "?") for s in srcs]
        if names:
            extra = f" +{len(names) - 2}" if len(names) > 2 else ""
            chunks.append(", ".join(names[:2]) + extra)
        tf = (m.get("config") or {}).get("text_from")
        if tf:
            chunks.append(f"tf={str(tf).removeprefix('cjm-capability-')}")
        rows.append({"text": "  ".join(chunks), "style": "", "index": i})
    if not rows:
        rows.append({"text": "(no decomp manifests found — batch hand-offs "
                             "land here)", "style": "dim", "index": None})
    return rows


def drill_header_html(m: Dict[str, Any]) -> str:
    """Drilled decomp run: id + the transcription manifest it consumed."""
    head = f"<b>{html.escape(str(m['run_id']))}</b>"
    src_manifest = m.get("source_manifest")
    if src_manifest:
        head += (f" <span style='color:#8a9299'>&lt;- "
                 f"{html.escape(tail(str(src_manifest), 48))}</span>")
    return head


def drill_source_rows(m: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One row per extended source: title, fine-segment count, Source node."""
    rows: List[Dict[str, Any]] = []
    for i, s in enumerate(m.get("sources") or []):
        chunks = [str(s.get("title") or "?"),
                  f"{int(s.get('segment_count') or 0)} fine segment(s)"]
        node = str(s.get("source_node_id") or "")
        if node:
            chunks.append(f"Source {node[:8]}…")
        rows.append({"text": "  ".join(chunks), "style": "", "index": i})
    if not rows:
        rows.append({"text": "(no sources extended — the run failed or was "
                             "aborted before its first commit)",
                     "style": "dim", "index": None})
    return rows


# ---- segments stage -----------------------------------------------------


def segments_header_html(title: str, n_rows: int, n_gaps: int, missing: int,
                         threshold_s: float, vad_line: Optional[str]) -> str:
    """The drilled source's headline: row/gap/missing counts + the recorded
    VAD knobs (166dd2b8 — a miss is only interpretable against the threshold
    that produced it)."""
    head = (f"<b>{html.escape(title)}</b> &nbsp;·&nbsp; "
            f"<span style='color:#8a9299'>{n_rows} fine segment(s)</span>")
    if n_gaps:
        head += (f" &nbsp;·&nbsp; <b><span style='color:#b9770e'>{n_gaps} "
                 f"gap(s) ≥{threshold_s:g}s</span></b>")
    if missing:
        head += (f" &nbsp;·&nbsp; <b><span style='color:#c74a3c'>{missing} "
                 "id(s) not in graph</span></b>")
    if vad_line:
        head += (f"<br><span style='color:#2b8a9d'>{html.escape(vad_line)}"
                 "</span>")
    return head


def segment_rows(display: List[Tuple[str, int, float]],  # build_display entries
                 rows: List[Dict[str, Any]],             # Spine-ordered segment rows
                 ) -> List[Dict[str, Any]]:
    """The committed fine spine with FOCUSABLE interleaved gap rows — the gap
    IS the inspection target (de994164), so it selects like a segment."""
    out: List[Dict[str, Any]] = []
    for i, (kind, ri, secs) in enumerate(display):
        r = rows[ri]
        if kind == "gap":
            s = float(r.get("start_time") or 0.0)
            out.append({"text": f"⚠ {secs:.1f}s gap  {fmt_ts(s - secs)} → "
                                f"{fmt_ts(s)}  no segment was cut here",
                        "style": "bold yellow", "index": i})
        else:
            s, e = r.get("start_time"), r.get("end_time")
            span = (f"{fmt_ts(float(s))}–{fmt_ts(float(e))}"
                    if s is not None and e is not None else "?–?")
            text = str(r.get("text") or "").replace("\n", " ")
            out.append({"text": f"{int(r.get('index') or ri):>4}  {span}  "
                                f"{text}",
                        "style": "", "index": i})
    if not out:
        out.append({"text": "(no committed segments found in the graph)",
                    "style": "dim", "index": None})
    return out


def segment_detail(entry: Tuple[str, int, float],  # The focused display entry
                   row: Dict[str, Any],            # Its segment row
                   ) -> str:
    """The focused entry IN FULL as plain text (setPlainText — no truncation,
    no markup): a gap's exact uncovered span, or a segment's timing + text."""
    kind, _ri, secs = entry
    if kind == "gap":
        s = float(row.get("start_time") or 0.0)
        return (f"uncovered span {fmt_ts(s - secs)} → {fmt_ts(s)} "
                f"({secs:.1f}s) — audition the source audio here (r plays it: "
                "the coarse model-input WAV spans what no fine chunk cut)")
    s, e = row.get("start_time"), row.get("end_time")
    head = (f"{fmt_ts(float(s))} → {fmt_ts(float(e))} · "
            f"{float(e) - float(s):.1f}s\n\n"
            if s is not None and e is not None else "")
    return head + str(row.get("text") or "")


# ---- vad-form stage -----------------------------------------------------


def vadform_header_html(vad_id: str, title: str, probe_split: bool) -> str:
    """Form headline + the probe-only doctrine + the split-preview state."""
    return (f"<b>VAD probe · <span style='color:#2b8a9d'>"
            f"{html.escape(vad_id.removeprefix('cjm-capability-'))}</span></b>"
            f" &nbsp;·&nbsp; <span style='color:#8a9299'>{html.escape(title)}"
            "</span><br><span style='color:#8a9299'>probe-only: nothing "
            "commits — a promising sweep argues an explicit re-decomposition"
            "</span><br><span style='color:#8a9299'>sentence-split preview: "
            "</span>"
            + ("<b><span style='color:#9b59b6'>ON</span></b>" if probe_split
               else "<span style='color:#8a9299'>off</span>")
            + " <span style='color:#8a9299'>(s toggles — the pipeline's "
              "post-FA split stage over the predicted skeleton)</span>")


def form_rows(form: Any,                    # kit ConfigForm (or None)
              run_cfg: Dict[str, Any],      # The RUN's recorded config (base)
              ) -> List[Dict[str, Any]]:
    """One row per config field; a value differing from the RUN's recorded
    config paints changed (and names the run's base — the sweep must always
    be readable AGAINST the committed knobs)."""
    rows: List[Dict[str, Any]] = []
    for i, f in enumerate(form.fields if form is not None else []):
        base = run_cfg.get(f.key, f.default)
        changed = f.value != base
        text = f"{f.title:<26}  {f.render()}"
        if changed:
            text += f"   (run: {base})"
        rows.append({"text": text, "style": "yellow" if changed else "",
                     "index": i})
    if not rows:
        rows.append({"text": "(no configurable fields)", "style": "dim",
                     "index": None})
    return rows


def probe_target_html(probe_aseg: Optional[int],
                      asegs: List[Dict[str, Any]]) -> str:
    """The frozen probe target: which coarse chunk the sweep runs over."""
    if probe_aseg is None or not asegs:
        return ("<span style='color:#c74a3c'>no probe target — reopen the "
                "form from a timed entry</span>")
    a = asegs[probe_aseg]
    text = (f"<span style='color:#8a9299'>target: coarse chunk "
            f"{probe_aseg + 1}/{len(asegs)} · {fmt_ts(a['start'])} → "
            f"{fmt_ts(a['end'])}</span>")
    if not a.get("wav"):
        text += (" <b><span style='color:#c74a3c'>⚠ no model-input WAV on "
                 "disk</span></b>")
    return text


def probe_summary_html(result: Optional[Dict[str, Any]], busy: bool) -> str:
    """The probe readout: predicted vs committed counts + the recovered spans
    — the de994164 question answered ('would this config have cut the missed
    audio?')."""
    if busy:
        return "<span style='color:#2b8a9d'>probing …</span>"
    if result is None:
        return ""
    lines = [f"<b>predicted {result['predicted']} chunk(s) vs "
             f"{result['committed']} committed · "
             f"{len(result['recovered'])} recovered span(s)</b>"]
    for s, e in result["recovered"][:8]:
        lines.append(f"<span style='color:#3f9d55'>&nbsp;+ {fmt_ts(s)} → "
                     f"{fmt_ts(e)}  ({e - s:.1f}s) — was uncovered</span>")
    if len(result["recovered"]) > 8:
        lines.append(f"<span style='color:#8a9299'>&nbsp;… "
                     f"{len(result['recovered']) - 8} more</span>")
    return "<br>".join(lines)


def config_diffs(form: Any, run_cfg: Dict[str, Any]) -> List[str]:
    """The form's deltas from the RUN's recorded config, as key=value chips."""
    out: List[str] = []
    for f in (form.fields if form is not None else []):
        base = run_cfg.get(f.key, f.default)
        if f.value != base:
            out.append(f"{f.key}={f.render()}")
    return out


# ---- probe-skeleton stage -----------------------------------------------


def probeview_header_html(diffs: List[str],           # config_diffs output
                          probe_aseg: Optional[int],
                          asegs: List[Dict[str, Any]],
                          result: Optional[Dict[str, Any]],
                          realigned: bool) -> str:
    """The predicted-skeleton headline: which config produced it, over which
    coarse chunk, counts, and whether the text is FA-realigned (what a
    re-decomposition would COMMIT) or borrowed (orientation only)."""
    head = "<b>predicted skeleton</b> &nbsp;·&nbsp; "
    head += (f"<span style='color:#b9770e'>{html.escape('; '.join(diffs))}"
             "</span>" if diffs
             else "<span style='color:#8a9299'>run config</span>")
    if probe_aseg is not None and asegs:
        a = asegs[probe_aseg]
        head += (f" &nbsp;·&nbsp; <span style='color:#8a9299'>chunk "
                 f"{probe_aseg + 1}: {fmt_ts(a['start'])} → "
                 f"{fmt_ts(a['end'])}</span>")
    if result is not None:
        tag = ("text FA-realigned — what a re-decomposition would commit"
               if realigned else
               "text borrowed from committed rows (FA realign unavailable)")
        head += (f"<br><span style='color:#8a9299'>predicted "
                 f"{result['predicted']} vs {result['committed']} committed · "
                 f"{len(result['recovered'])} recovered · {tag}</span>")
    return head


def probe_rows(display: List[Tuple[str, int, float]],  # build_display entries
               rows: List[Dict[str, Any]],             # predicted_rows output
               ) -> List[Dict[str, Any]]:
    """The predicted skeleton painted with the SEGMENTS grammar (direct
    comparison): gap rows = holes STILL uncut at this config, recovered
    chunks green with a + mark, split-minted chunks ✂-marked."""
    out: List[Dict[str, Any]] = []
    for i, (kind, ri, secs) in enumerate(display):
        r = rows[ri]
        if kind == "gap":
            s = float(r.get("start_time") or 0.0)
            out.append({"text": f"⚠ {secs:.1f}s gap  {fmt_ts(s - secs)} → "
                                f"{fmt_ts(s)}  STILL uncut at this config",
                        "style": "bold yellow", "index": i})
            continue
        s, e = r.get("start_time"), r.get("end_time")
        span = (f"{fmt_ts(float(s))}–{fmt_ts(float(e))}"
                if s is not None and e is not None else "?–?")
        marks = ("+ " if r.get("recovered") else "") \
            + ("✂ " if r.get("split") else "")
        text = str(r.get("text") or "").replace("\n", " ")
        if not text:
            text = ("(no words assigned — audition with r)" if r.get("realigned")
                    else "(no committed text — audition with r)")
        out.append({"text": f"{int(r.get('index') or ri):>4}  {span}  "
                            f"{marks}{text}",
                    "style": "green" if r.get("recovered") else "",
                    "index": i})
    if not out:
        out.append({"text": "(probe predicted no chunks)", "style": "dim",
                    "index": None})
    return out


def probe_detail(entry: Tuple[str, int, float],  # The focused display entry
                 row: Dict[str, Any],            # Its predicted row
                 ) -> str:
    """The focused predicted entry in full, plain text."""
    kind, _ri, secs = entry
    if kind == "gap":
        s = float(row.get("start_time") or 0.0)
        return (f"still uncovered at this config: {fmt_ts(s - secs)} → "
                f"{fmt_ts(s)} ({secs:.1f}s)")
    s, e = row.get("start_time"), row.get("end_time")
    head = ""
    if s is not None and e is not None:
        head = (f"{fmt_ts(float(s))} → {fmt_ts(float(e))} · "
                f"{float(e) - float(s):.1f}s"
                + ("  · recovered span" if row.get("recovered") else "")
                + ("  · ✂ split-minted" if row.get("split") else "") + "\n\n")
    return head + str(row.get("text") or "")
