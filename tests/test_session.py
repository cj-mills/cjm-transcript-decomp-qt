"""DecompSession contract tests — headless, no Qt, no real capabilities.

The seam's promises: seat work happens on the loop thread (never the
caller's), the seat is lazy and re-points rather than re-minting, the
composite probe degrades exactly like the Textual shell (FA failure ->
fa_error + no split leg, sentence failure -> seg_error, VAD always resolves),
the propose seat loads off the loop thread and tears down in finally, and
close() runs the seat teardown."""

import threading
from typing import Any, Dict, List, Optional

import pytest

from cjm_transcript_decomp_qt.session import DecompSession


class FakeStack:
    """Stands in for the tui SegmentStack inside the session's loop."""

    instances: List["FakeStack"] = []

    def __init__(self, manifests_dir: str, graph_capability: str):
        self.args = (manifests_dir, graph_capability)
        self.db_path: Optional[str] = None
        self.opens: List[str] = []
        self.threads: List[str] = []
        self.closed = False
        self.fail_fa = False
        self.fail_seg = False
        self.transcript: Optional[str] = "hello world in text"
        FakeStack.instances.append(self)

    async def open(self, db_path: str) -> None:
        self.threads.append(threading.current_thread().name)
        self.opens.append(db_path)
        self.db_path = db_path

    async def read_segments(self, ids: List[str]) -> List[Dict[str, Any]]:
        # id "gone" simulates a spine row the graph no longer holds
        return [{"id": i, "start_time": 1.0, "end_time": 2.0,
                 "rendition_id": f"rend-{i}"} for i in ids if i != "gone"]

    async def read_audio_join(self, source_id: str,
                              rendition_ids: List[str]) -> List[Dict[str, Any]]:
        return [{"start": 0.0, "end": 30.0, "wav": "/tmp/a.wav",
                 "rendition": rendition_ids[0] if rendition_ids else None}]

    async def probe_vad(self, vad_id, cfg, wav_path):
        return [(0.0, 1.0), (2.0, 3.0)]

    async def read_transcript_text(self, rendition_id, transcriber):
        return self.transcript

    async def probe_fa(self, fa_id, cfg, wav_path, text):
        if self.fail_fa:
            raise RuntimeError("no FA capability")
        return [("hello", 0.0, 0.5), ("world", 0.6, 1.0)]

    async def probe_segment(self, seg_id, cfg, text):
        if self.fail_seg:
            raise RuntimeError("no segmenter")
        return [(0, 5), (6, 11)]

    async def close(self) -> None:
        self.closed = True


class FakeManager:
    instances: List["FakeManager"] = []

    def __init__(self, search_paths=None):
        self.unloaded: List[str] = []
        FakeManager.instances.append(self)

    async def execute_capability_task_async(self, capability, task, method,
                                            **kwargs):
        if str(kwargs.get("source", "")).endswith("boom.mp3"):
            raise RuntimeError("propose exploded")
        return {"ok": True}

    def unload_capability(self, iid):
        self.unloaded.append(iid)


class Recorder:
    def __init__(self):
        self.load_threads: List[str] = []
        self.loads: List[List[str]] = []

    def loader(self, manager, directives):
        self.load_threads.append(threading.current_thread().name)
        self.loads.append(list(directives))


@pytest.fixture()
def session():
    FakeStack.instances = []
    FakeManager.instances = []
    rec = Recorder()
    sess = DecompSession("/tmp/manifests", "cjm-capability-graph-sqlite",
                         timeout=10.0, stack_factory=FakeStack,
                         manager_factory=FakeManager, loader=rec.loader)
    sess.start()
    yield sess, rec
    sess.close()


PROBE_KW = dict(rendition_id="rend-a", text_from="cjm-capability-voxtral-hf",
                fa_id="fa", fa_cfg={}, seg_id="seg", seg_cfg={})


def open_source(sess):
    return sess.open_source("/data/graph.db", ["a", "gone", "b"],
                            "src-node").result(10)


def test_open_source_reads_on_loop_thread_and_joins_audio(session):
    sess, _ = session
    res = open_source(sess)
    stack = FakeStack.instances[0]
    assert stack.args == ("/tmp/manifests", "cjm-capability-graph-sqlite")
    assert stack.threads == ["decomp-session"]   # seat work on the loop, only
    assert [r["id"] for r in res["rows"]] == ["a", "b"]  # "gone" dropped
    assert res["asegs"][0]["wav"] == "/tmp/a.wav"


def test_open_source_repoints_one_lazy_seat(session):
    sess, _ = session
    open_source(sess)
    sess.open_source("/data/other.db", ["a"], "src-node").result(10)
    assert len(FakeStack.instances) == 1          # lazy mint, once
    assert FakeStack.instances[0].opens == ["/data/graph.db", "/data/other.db"]


def test_probe_chain_full_legs_and_progress(session):
    sess, _ = session
    open_source(sess)
    notes: List[str] = []
    out = sess.probe_chain("vad", {"threshold": 0.4}, "/tmp/a.wav",
                           progress=notes.append, **PROBE_KW).result(10)
    assert out["local"] == [(0.0, 1.0), (2.0, 3.0)]
    assert out["fa_text"] == "hello world in text"
    assert out["fa_words"] and out["seg_spans"] == [(0, 5), (6, 11)]
    assert out["fa_error"] is None and out["seg_error"] is None
    assert any("aligning (FA" in n for n in notes)


def test_probe_chain_fa_failure_degrades_and_skips_split_leg(session):
    sess, _ = session
    open_source(sess)
    FakeStack.instances[0].fail_fa = True
    out = sess.probe_chain("vad", {}, "/tmp/a.wav", **PROBE_KW).result(10)
    assert out["local"]                       # VAD still resolves
    assert out["fa_text"] is None and "no FA capability" in out["fa_error"]
    assert out["seg_spans"] is None and out["seg_error"] is None


def test_probe_chain_seg_failure_keeps_fa(session):
    sess, _ = session
    open_source(sess)
    FakeStack.instances[0].fail_seg = True
    out = sess.probe_chain("vad", {}, "/tmp/a.wav", **PROBE_KW).result(10)
    assert out["fa_words"] is not None
    assert out["seg_spans"] is None and "no segmenter" in out["seg_error"]


def test_probe_chain_without_rendition_is_vad_only(session):
    sess, _ = session
    open_source(sess)
    out = sess.probe_chain("vad", {}, "/tmp/a.wav",
                           **{**PROBE_KW, "rendition_id": None}).result(10)
    assert out["local"] and out["fa_text"] is None and out["fa_error"] is None


def test_probe_chain_without_seat_raises(session):
    sess, _ = session
    with pytest.raises(RuntimeError, match="seat not open"):
        sess.probe_chain("vad", {}, "/tmp/a.wav", **PROBE_KW).result(10)


def test_propose_batch_loads_off_loop_and_tears_down(session):
    sess, rec = session
    notes: List[str] = []
    res = sess.propose_batch("cjm-capability-pyannote", "/runs/tr/manifest.json",
                             ["/media/a.mp3", "/media/b.mp3"],
                             progress=notes.append).result(10)
    assert res == {"landed": 2, "error": None}
    assert rec.loads == [["cjm-capability-pyannote"]]
    # Blocking load rides to_thread — never the loop thread itself.
    assert rec.load_threads[0] != "decomp-session"
    assert FakeManager.instances[0].unloaded == ["cjm-capability-pyannote"]
    assert notes[0].startswith("propose: opening")
    assert "1/2" in notes[1] and "2/2" in notes[2]


def test_propose_batch_failure_reports_landed_and_still_unloads(session):
    sess, _ = session
    res = sess.propose_batch("cjm-capability-pyannote", "/runs/tr/manifest.json",
                             ["/media/a.mp3", "/media/boom.mp3",
                              "/media/c.mp3"]).result(10)
    assert res["landed"] == 1 and "propose exploded" in res["error"]
    assert FakeManager.instances[0].unloaded == ["cjm-capability-pyannote"]


def test_close_tears_seat_down(session):
    sess, _ = session
    open_source(sess)
    stack = FakeStack.instances[0]
    sess.close()
    assert stack.closed and sess.stack is None
    assert not sess.running
