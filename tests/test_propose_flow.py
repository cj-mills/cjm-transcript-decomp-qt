"""Offscreen tests of the propose pre-stage flow through a real DecompWindow
over tmp manifests: the training-run picker (a13ebc66 — pick > pin >
newest, the model chip, an armed confirm dropped on pick) and propset
coexistence (53911bef — a source that already has a set is a REGENERATE
target, the fresh set becomes its default when the batch lands)."""

import json
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from cjm_transcript_decomp_core.runs import PropsetIndex, TrainingRunIndex
from cjm_transcript_decomp_qt.app import DecompWindow


class _Done:
    """A resolved Future stand-in for the propose_done slot."""

    def __init__(self, res):
        self._res = res

    def result(self):
        return self._res


def _workspace(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "run_a.json").write_text(json.dumps({
        "format": "cjm-transcription-core/run-manifest", "run_id": "run_a",
        "created_at": 2.0,
        "config": {"transcriber_capabilities": ["cjm-capability-voxtral-hf"]},
        "sources": [{"source_path": str(tmp_path / "a.wav"),
                     "content_hash": "sha256:aaaa", "segments": [1, 2]}]}))
    (runs / "run_b.json").write_text(json.dumps({
        "format": "cjm-transcription-core/run-manifest", "run_id": "run_b",
        "created_at": 1.0,
        "config": {"transcriber_capabilities": ["cjm-capability-voxtral-hf"]},
        "sources": [{"source_path": str(tmp_path / "b.wav"),
                     "content_hash": "sha256:bbbb", "segments": [1]}]}))
    props = tmp_path / "proposals"
    (props / "propset_old").mkdir(parents=True)
    (props / "propset_old" / "manifest.json").write_text(json.dumps({
        "format": PropsetIndex.FORMAT, "proposal_set_id": "propset_old_11111111",
        "created_at": 1.0, "training_run_id": "trainrun_1_6f803b12",
        "source": {"content_hash": "sha256:aaaa", "path": str(tmp_path / "a.wav")},
        "counts": {"inhale": 3}}))
    trs = tmp_path / "training-runs"
    for rid in ("trainrun_1_6f803b12", "trainrun_2_b1072a9f"):
        (trs / rid).mkdir(parents=True)
        (trs / rid / "manifest.json").write_text(json.dumps({
            "format": TrainingRunIndex.FORMAT, "run_id": rid,
            "classes": ["speech", "inhale"]}))
    (tmp_path / "manifests").mkdir()
    return runs, props, trs


@pytest.fixture()
def window(tmp_path):
    QApplication.instance() or QApplication(sys.argv[:1])
    runs, props, trs = _workspace(tmp_path)
    w = DecompWindow(str(tmp_path / "manifests"), runs_dir=str(runs),
                     proposals_dir=str(props), training_runs_dir=str(trs),
                     training_run_pin="6f803b12", event_split=True)
    yield w
    w.sess.close()
    w.close()


def test_pick_ladder_chip_and_armed_confirm(window):
    w = window
    assert len(w.src_index.runs) == 2 and len(w.trainrun_index.runs) == 2
    # the pin is the standing default and the chip says so
    assert w._model_chip() == "6f803b12 (pin)"
    w.picked = [w.src_index.runs[1]["_path"]]            # run_b: no propset
    w.on_propose_missing()
    assert w.propose_armed["run"]["run_id"].endswith("6f803b12")
    assert "workspace pin" in w.notice and "regenerate" not in w.notice
    # m opens the picker parked on the pin row; a pick outranks the pin and
    # drops the armed confirm (the model it named may no longer be the one)
    w.on_pick_model()
    p = w.trainrun_picker
    assert p.isVisible() and p.list.count() == 2 and p.list.currentRow() == 1
    p.list.setCurrentRow(0)
    p._pick_current()
    assert not p.isVisible()
    assert w.training_run_pick.endswith("trainrun_2_b1072a9f/manifest.json")
    assert w.propose_armed is None and "picked for this sitting" in w.notice
    assert w._model_chip() == "b1072a9f (picked)"
    w.on_propose_missing()
    assert w.propose_armed["run"]["run_id"].endswith("b1072a9f")
    assert "— picked" in w.notice
    # a vanished pick is a loud miss, never a silent fall-through to the pin
    w.propose_armed = None
    w.training_run_pick = "/nowhere/manifest.json"
    w.on_propose_missing()
    assert w.propose_armed is None and "no longer on disk" in w.error
    assert w._model_chip() == "⚠ no model"
    # event-split off: no model chip at all
    w.event_split = False
    assert w._model_chip() is None


def test_propset_coexist_regenerate_targets_and_fresh_default(window):
    w = window
    run_a = w.src_index.runs[0]                          # has propset_old
    assert w._resolved_propset(run_a) is not None
    w.picked = [run_a["_path"], w.src_index.runs[1]["_path"]]
    # coexist: the source WITH a set is a target too, named as a regenerate
    assert [m["run_id"] for m in w._propose_targets()] == ["run_a", "run_b"]
    w.on_propose_missing()
    assert len(w.propose_armed["targets"]) == 2
    assert "propose 2 source(s)" in w.notice and "1 regenerate" in w.notice
    assert "Shift+E cycles" in w.notice
    # an older Shift+E pick on the regenerated source is dropped when the
    # batch lands, so the fresh set (newest) is that source's default again
    w.propose_armed = None
    w.picked = [run_a["_path"]]
    w.on_propose_missing()
    w.propset_pick[run_a["_path"]] = "/stale/manifest.json"
    w.on_propose_missing()                               # second press: runs
    assert w.propose_busy and w._armed_targets[0]["run_id"] == "run_a"
    w._on_propose_done(_Done({"landed": 1, "error": None}))
    assert not w.propose_busy and run_a["_path"] not in w.propset_pick
    assert w.notice.startswith("⚡ 1 propset(s) landed with 6f803b12")
    # only multi-source picks leave nothing to propose
    w.picked = []
    w.on_propose_missing()
    assert "pick runs first" in w.error
