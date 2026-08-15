"""Offscreen paint probe: stand the window up over a fabricated runs corpus
and walk the cheap gestures (no capability loads, no graph seat) — the
paint-path verification layer pytest cannot give (67335f7d), Qt edition. Run
from a NEUTRAL cwd:

    QT_QPA_PLATFORM=offscreen python offscreen_probe.py
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from cjm_transcript_decomp_qt.app import DecompWindow  # noqa: E402


def write_manifests(runs: Path) -> None:
    runs.mkdir()
    (runs / "tr1.json").write_text(json.dumps({
        "run_id": "run_tr1", "format": "cjm/transcription-core-run/0.2",
        "created_at": time.time(),
        "config": {"transcriber_capabilities": ["cjm-capability-whisper",
                                                "cjm-capability-voxtral-hf"]},
        "capabilities": {"cjm-capability-graph-sqlite":
                         {"db_path": "/nope/graph.db"}},
        "sources": [{"source_path": "/media/talk.mp3",
                     "segments": [{}, {}, {}]}]}))
    (runs / "dc1.json").write_text(json.dumps({
        "run_id": "run_dc1", "format": "cjm/transcript-decomp-core-run/0.3",
        "created_at": time.time(), "source_manifest": str(runs / "tr1.json"),
        "config": {"text_from": "cjm-capability-voxtral-hf"},
        "sources": [{"title": "talk", "segment_count": 3,
                     "source_node_id": "src-1", "segment_ids": []}]}))


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="decomp-qt-probe-"))
    manifests = tmp / "manifests"
    manifests.mkdir()
    write_manifests(tmp / "runs")
    app = QApplication(sys.argv[:1])
    win = DecompWindow(str(manifests), runs_dir=str(tmp / "runs"),
                       graph_db_path=None,
                       proposals_dir=str(tmp / "proposals"),
                       training_runs_dir=str(tmp / "training-runs"))
    win.show()
    assert win.stage == "runs"
    assert win.runs_list.count() == 1, "one transcription run painted"
    assert "run_tr1" in win.runs_list.item(0).text()
    assert "tf=voxtral-hf" in win.runs_list.item(0).text()
    assert "no graph db" not in win.runs_detail.text()  # recorded (missing on disk)
    assert "missing on disk" in win.runs_detail.text()

    # pick -> ordered queue chip + batch panel with the db warning
    win.on_select()
    assert win.picked and "[1]" in win.runs_list.item(0).text()
    assert "1 invocation(s)" in win.runs_batch.text()
    assert "db missing" in win.runs_batch.text()

    # batch toggles paint
    win.on_toggle_split()
    assert not win.sentence_split
    win.on_toggle_respine()
    assert "respine ON" in win.runs_batch.text()
    win.on_toggle_event_split()
    assert win.event_split and not win.sentence_split
    assert "⚡no propset" in win.runs_list.item(0).text()
    win.on_confirm()      # event-armed with no propset -> block, name it
    assert "no proposal set matches" in (win.error or "")
    win.on_toggle_event_split()

    # confirm passes the db-exists guard? No — db missing on disk is allowed
    # at confirm (the Textual guard blocks only db=None), so the plan lands.
    win.on_confirm()
    assert win.plan is not None and win.plan["batches"][0]["graph_db_path"] \
        == "/nope/graph.db"
    win.plan = None

    # v -> decomp results list + drill; enter on empty segment_ids errors
    win.on_results()
    assert win.stage == "results" and win.res_list.count() == 1
    assert "run_dc1" in win.res_list.item(0).text()
    win.on_select()       # drill
    assert win.results_run == 0
    assert "talk  3 fine segment(s)" in win.res_list.item(0).text()
    win.on_select()       # segments: no segment_ids recorded -> in-status
    assert "no segment_ids recorded" in (win.error or "")
    win.on_back()
    assert win.results_run is None
    win.on_back()
    assert win.stage == "runs"

    # propose gating: needs event-split armed first
    win.on_propose_missing()
    assert "arm event-split" in (win.error or "")

    win.close()
    print("offscreen probe: all assertions passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
