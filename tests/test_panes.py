"""Pure-builder tests for panes.py — the paint logic, no Qt.

Real spine pieces where cheap (build_display / predicted_rows from the tui,
kit ConfigForm over a small schema), plain dicts for the manifests."""

from cjm_substrate_qt_kit.configform import ConfigForm
from cjm_transcript_decomp_core.segments import build_display, predicted_rows

from cjm_transcript_decomp_qt.panes import (batch_html, config_diffs,
                                            decomp_rows, drill_header_html,
                                            drill_source_rows, form_rows,
                                            probe_detail, probe_rows,
                                            probe_summary_html,
                                            probe_target_html, run_detail_html,
                                            run_rows, segment_detail,
                                            segment_rows,
                                            segments_header_html)


def src_manifest(path="/runs/r1.json", run_id="run_1", n_seg=3,
                 transcribers=("cjm-capability-whisper",
                               "cjm-capability-voxtral-hf")):
    return {"_path": path, "run_id": run_id, "created_at": 1755200000.0,
            "config": {"transcriber_capabilities": list(transcribers)},
            "sources": [{"source_path": "/media/talk.mp3",
                         "segments": [{}] * n_seg}]}


# ---- runs stage ---------------------------------------------------------


def test_run_rows_queue_position_and_chips():
    runs = [src_manifest(), src_manifest(path="/runs/r2.json", run_id="run_2")]
    rows = run_rows(runs, picked=["/runs/r2.json"], text_from={},
                    decomp_counts={}, event_split=False,
                    propset_chip=lambda m: None)
    assert rows[0]["text"].startswith("[ ] run_1")
    assert rows[1]["text"].startswith("[1] run_2")   # ordered queue, not [x]
    assert rows[1]["style"] == "green"
    assert "whisper+voxtral-hf" in rows[0]["text"]
    assert "tf=voxtral-hf" in rows[0]["text"]        # accuracy-slot default


def test_run_rows_event_chip_missing_propset_paints_red():
    rows = run_rows([src_manifest()], picked=[], text_from={},
                    decomp_counts={}, event_split=True,
                    propset_chip=lambda m: None)
    assert "⚡no propset" in rows[0]["text"]
    assert rows[0]["style"] == "red"
    rows = run_rows([src_manifest()], picked=[], text_from={},
                    decomp_counts={}, event_split=True,
                    propset_chip=lambda m: "ab12cd34 5+2")
    assert "⚡ab12cd34 5+2" in rows[0]["text"]


def test_run_rows_empty_placeholder():
    rows = run_rows([], [], {}, {}, False, lambda m: None)
    assert rows[0]["index"] is None and rows[0]["style"] == "dim"


def test_run_detail_html_db_warnings(tmp_path):
    m = src_manifest()
    html = run_detail_html(m, None)
    assert "no graph db recorded" in html
    html = run_detail_html(m, "/nope/missing.db")
    assert "missing on disk" in html
    ok = tmp_path / "g.db"
    ok.write_bytes(b"")
    html = run_detail_html(m, str(ok))
    assert "graph db:" in html and "talk" in html


def test_batch_html_folds_and_toggles(tmp_path):
    db = tmp_path / "g.db"
    db.write_bytes(b"")
    batches = [(("cjm-capability-voxtral-hf", str(db), None),
                ["/runs/r1.json", "/runs/r2.json"])]
    html = batch_html(batches, run_id_for=lambda p: p.rsplit("/", 1)[1],
                      sentence_split=True, respine=True, event_split=False)
    assert "2 run(s) -&gt; 1 invocation(s)" in html
    assert "✂ sentence-split ON" in html and "⟳ respine ON" in html
    assert "text-from voxtral-hf" in html
    assert "r1.json, r2.json" in html
    assert batch_html([], lambda p: p, False, False, False) == ""


# ---- results stage ------------------------------------------------------


DEC = {"_path": "/runs/d1.json", "run_id": "dec_1", "created_at": 1755200000.0,
       "source_manifest": "/runs/r1.json",
       "config": {"text_from": "cjm-capability-voxtral-hf"},
       "sources": [{"title": "talk", "segment_count": 42,
                    "source_node_id": "abcdef1234567890"}]}


def test_decomp_rows_and_drill():
    rows = decomp_rows([DEC])
    assert "dec_1" in rows[0]["text"] and "1 source(s) · 42 seg" in rows[0]["text"]
    assert "talk" in rows[0]["text"] and "tf=voxtral-hf" in rows[0]["text"]
    assert "r1.json" in drill_header_html(DEC)
    drows = drill_source_rows(DEC)
    assert "talk  42 fine segment(s)  Source abcdef12…" == drows[0]["text"]
    assert drill_source_rows({"sources": []})[0]["index"] is None


# ---- segments stage -----------------------------------------------------


SEGS = [{"index": 0, "start_time": 0.0, "end_time": 2.0, "text": "hello"},
        {"index": 1, "start_time": 8.0, "end_time": 9.5, "text": "world\nagain"}]


def test_segment_rows_interleave_focusable_gaps():
    display = build_display(SEGS, threshold_s=2.0)
    rows = segment_rows(display, SEGS)
    assert len(rows) == 3                       # gap row between the two segs
    assert "6.0s gap" in rows[1]["text"] and rows[1]["style"] == "bold yellow"
    assert rows[1]["index"] == 1                # focusable, not decoration
    assert "world again" in rows[2]["text"]     # newline flattened in-row


def test_segment_detail_gap_and_seg():
    display = build_display(SEGS, threshold_s=2.0)
    gap_text = segment_detail(display[1], SEGS[1])
    assert "uncovered span 0:02.0 → 0:08.0 (6.0s)" in gap_text
    seg_text = segment_detail(display[2], SEGS[1])
    assert seg_text.startswith("0:08.0 → 0:09.5 · 1.5s")
    assert "world\nagain" in seg_text            # FULL text, newline kept


def test_segments_header_counts():
    html = segments_header_html("talk", 5, 2, 1, 2.0, "silero-vad: thr 0.4")
    assert "2 gap(s) ≥2s" in html and "1 id(s) not in graph" in html
    assert "silero-vad" in html


# ---- vad form + probe view ----------------------------------------------


SCHEMA = {"properties": {
    "threshold": {"type": "number", "default": 0.5, "title": "Threshold"},
    "speech_pad_ms": {"type": "integer", "default": 30}}}


def test_form_rows_paint_run_deltas():
    form = ConfigForm.from_schema(SCHEMA)
    form.apply({"threshold": 0.3})              # the RUN's recorded config
    rows = form_rows(form, {"threshold": 0.3})
    assert rows[0]["style"] == "" and "(run:" not in rows[0]["text"]
    form.fields[0].value = 0.7                  # operator sweep
    rows = form_rows(form, {"threshold": 0.3})
    assert rows[0]["style"] == "yellow" and "(run: 0.3)" in rows[0]["text"]
    assert config_diffs(form, {"threshold": 0.3}) == ["threshold=0.7"]


def test_probe_target_and_summary():
    assert "no probe target" in probe_target_html(None, [])
    asegs = [{"start": 0.0, "end": 30.0, "wav": None}]
    assert "no model-input WAV" in probe_target_html(0, asegs)
    assert probe_summary_html(None, busy=True) == \
        "<span style='color:#2b8a9d'>probing …</span>"
    html = probe_summary_html({"predicted": 4, "committed": 3,
                               "recovered": [(1.0, 2.5)]}, busy=False)
    assert "predicted 4 chunk(s) vs 3 committed" in html
    assert "0:01.0 → 0:02.5  (1.5s) — was uncovered" in html


def test_probe_rows_marks_recovered_and_split():
    rows = predicted_rows([(0.0, 2.0), (8.0, 9.5)], SEGS,
                          gap_spans=[(2.0, 8.0)], realigned=["hello", ""])
    rows[1]["split"] = True
    display = build_display(rows, threshold_s=2.0)
    out = probe_rows(display, rows)
    seg_rows = [r for r in out if "gap" not in r["text"]]
    assert "STILL uncut" in out[1]["text"]      # the hole this config leaves
    assert "✂ (no words assigned — audition with r)" in seg_rows[1]["text"]
    detail = probe_detail(display[0], rows[0])
    assert detail.startswith("0:00.0 → 0:02.0") and "hello" in detail


def test_trainrun_rows_marks_effective_pick_pin_and_newest():
    """Work item a13ebc66: the picker rows mark the EFFECTIVE model ● green
    by the resolve_pick ladder (pick > pin > newest), chip the pin row so
    the standing default stays visible under a pick, chip the head row
    newest, carry the manifest facts describe() reads, and paint a dim
    placeholder for an empty index."""
    from cjm_transcript_decomp_qt.panes import trainrun_rows
    runs = [{"run_id": "trainrun_20260827_001206_b1072a9f", "_path": "/b",
             "classes": ["speech", "inhale"], "dataset_id": "dataset_x_e53b8daa"},
            {"run_id": "trainrun_20260810_113513_f4717dc5", "_path": "/f",
             "classes": ["inhale"]}]
    rows = trainrun_rows(runs, pin=None, pick=None)
    assert [r["style"] for r in rows] == ["green", ""]
    assert rows[0]["text"].startswith("● trainrun_20260827_001206_b1072a9f")
    assert "newest" in rows[0]["text"] and "← pin" not in rows[0]["text"]
    assert "ds …e53b8daa" in rows[0]["text"] and "[inhale]" in rows[1]["text"]
    rows = trainrun_rows(runs, pin="f4717dc5", pick=None)
    assert [r["style"] for r in rows] == ["", "green"] and "← pin" in rows[1]["text"]
    rows = trainrun_rows(runs, pin="f4717dc5", pick="/b")
    assert [r["style"] for r in rows] == ["green", ""]
    assert "← pin" in rows[1]["text"] and rows[1]["text"].startswith("  ")
    assert [r["index"] for r in rows] == [0, 1]
    assert trainrun_rows([], None, None)[0]["style"] == "dim"


def test_run_rows_partial_multi_source_cover_paints_red():
    """Multi-source event carve (decomp manifest 0.2.6): a run whose sources
    are only PARTLY covered by proposal sets chips '⚠k/n no set · …' and
    paints red like the no-set case — the confirm guard's refusal class
    stays visible on the row; a full cover paints the plain chip."""
    rows = run_rows([src_manifest()], picked=[], text_from={},
                    decomp_counts={}, event_split=True,
                    propset_chip=lambda m: "⚠1/2 no set · ab12cd34 5+2 / no set")
    assert rows[0]["style"] == "red" and "⚡⚠1/2 no set" in rows[0]["text"]
    rows = run_rows([src_manifest()], picked=["/runs/r1.json"], text_from={},
                    decomp_counts={}, event_split=True,
                    propset_chip=lambda m: "⚠1/2 no set · ab12cd34 5+2 / no set")
    assert rows[0]["style"] == "green"                    # picked wins, as for no-set
    rows = run_rows([src_manifest()], picked=[], text_from={},
                    decomp_counts={}, event_split=True,
                    propset_chip=lambda m: "ab12cd34 5+2 / ef56ab78 3")
    assert rows[0]["style"] == "" and "⚡ab12cd34 5+2 / ef56ab78 3" in rows[0]["text"]
