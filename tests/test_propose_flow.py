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
    assert [(t["run"]["run_id"], t["has_set"]) for t in w._propose_targets()] \
        == [("run_a", True), ("run_b", False)]
    w.on_propose_missing()
    assert len(w.propose_armed["targets"]) == 2
    assert "propose 2 source(s)" in w.notice and "1 regenerate" in w.notice
    assert "Shift+E cycles" in w.notice
    # an older Shift+E pick on the regenerated source is dropped when the
    # batch lands, so the fresh set (newest) is that source's default again
    w.propose_armed = None
    w.picked = [run_a["_path"]]
    w.on_propose_missing()
    key = w._source_key(run_a, 0)
    w.propset_pick[key] = "/stale/manifest.json"
    w.on_propose_missing()                               # second press: runs
    assert w.propose_busy and w._armed_targets[0]["run"]["run_id"] == "run_a"
    w._on_propose_done(_Done({"landed": 1, "error": None}))
    assert not w.propose_busy and key not in w.propset_pick
    assert w.notice.startswith("⚡ 1 propset(s) landed with 6f803b12")
    # no picks: nothing to propose
    w.picked = []
    w.on_propose_missing()
    assert "pick runs first" in w.error


def test_multi_source_run_proposes_per_source_and_carries_one_pointer_each(window, tmp_path):
    """Multi-source transcription runs (decomp manifest 0.2.6): propose is
    per SOURCE — a two-source run contributes both sources as targets, its
    row chips '⚠1/2 no set' (red) until every source has a set, Shift+E
    walks one ring over every (source, set) pair keeping a pick per source,
    and the hand-off authority is ONE pointer per source (a tuple; the plan
    renders it as a list)."""
    w = window
    runs = tmp_path / "runs"
    (runs / "run_ab.json").write_text(json.dumps({
        "format": "cjm-transcription-core/run-manifest", "run_id": "run_ab",
        "created_at": 3.0,
        "config": {"transcriber_capabilities": ["cjm-capability-voxtral-hf"]},
        "sources": [{"source_path": str(tmp_path / "a.wav"),
                     "content_hash": "sha256:aaaa", "segments": [1]},
                    {"source_path": str(tmp_path / "b.wav"),
                     "content_hash": "sha256:bbbb", "segments": [1, 2]}]}))
    w._reload_indexes()
    ab = next(m for m in w.src_index.runs if m["run_id"] == "run_ab")
    # source a has propset_old, source b nothing: partial cover
    assert w._resolved_propsets(ab) == [str(tmp_path / "proposals" / "propset_old" / "manifest.json"), None]
    assert w._resolved_propset(ab) is None
    assert w._propset_chip(ab).startswith("⚠1/2 no set · ")
    w.picked = [ab["_path"]]
    targets = w._propose_targets()
    assert [(t["index"], t["has_set"]) for t in targets] == [(0, True), (1, False)]
    w.on_propose_missing()
    assert "propose 2 source(s)" in w.notice and "1 regenerate" in w.notice
    # a set lands for b (and a second for a): full cover, one pointer per source
    props = tmp_path / "proposals"
    for name, h, path, ts in (("propset_b", "sha256:bbbb", "b.wav", 5.0),
                              ("propset_a2", "sha256:aaaa", "a.wav", 6.0)):
        (props / name).mkdir()
        (props / name / "manifest.json").write_text(json.dumps({
            "format": PropsetIndex.FORMAT, "proposal_set_id": f"{name}_22222222",
            "created_at": ts, "training_run_id": "trainrun_2_b1072a9f",
            "source": {"content_hash": h, "path": str(tmp_path / path)},
            "counts": {"inhale": 2}}))
    w._reload_indexes()
    ab = next(m for m in w.src_index.runs if m["run_id"] == "run_ab")
    ps = w._resolved_propset(ab)
    assert isinstance(ps, tuple) and [p.split("/")[-2] for p in ps] == ["propset_a2", "propset_b"]
    assert "⚠" not in w._propset_chip(ab) and " / " in w._propset_chip(ab)
    # Shift+E: the ring walks (a: a2, old) then (b: b) — a pick per source
    w.runs_list.setCurrentRow(w.src_index.runs.index(ab))
    w.on_cycle_propset()
    assert "src 1/2" in w.notice and w._resolved_propsets(ab)[0].endswith("propset_old/manifest.json")
    assert w._resolved_propsets(ab)[1].endswith("propset_b/manifest.json")   # b untouched
    w.on_cycle_propset()
    assert "src 2/2" in w.notice and w._resolved_propsets(ab)[0].endswith("propset_old/manifest.json")
    w.on_cycle_propset()                                                    # wraps to a: a2
    assert "src 1/2" in w.notice and w._resolved_propsets(ab)[0].endswith("propset_a2/manifest.json")
    # the hand-off fold keys on the tuple; the plan carries a list
    (key, paths), = w._batches()
    assert key[2] == ps and paths == [ab["_path"]]


def test_lifecycle_archive_propset_and_archived_pin_are_loud(window, tmp_path):
    """Work item b20cb911 (DEC 14678fc9): Shift+X archives the focused run's
    effective propset behind a two-step — the set leaves the ring (the
    row's chip goes to None, a Shift+E pick on it is dropped) but stays on
    disk beside a lifecycle.json; the m picker's a/x verbs drive the same
    seam through the window hook; a workspace pin on an ARCHIVED run is a
    loud miss naming the archive, never a silent fall-through; delete
    refuses while a propset still names the run."""
    w = window
    run_a = w.src_index.runs[0]                          # has propset_old
    w.runs_list.setCurrentRow(0)
    assert w._focused_run() is run_a
    assert w._resolved_propset(run_a) is not None
    w.propset_pick[w._source_key(run_a, 0)] = w._propsets_for(run_a)[0][0]["_path"]
    w.on_archive_propset()                               # first press arms
    assert w.archive_armed and "Shift+X again" in w.notice
    assert "11111111" in w.notice and "0 other set(s)" in w.notice
    w.on_archive_propset()                               # second press archives
    assert w.archive_armed is None and w.notice.startswith("archived propset")
    assert "0 set(s) remain" in w.notice
    assert w._resolved_propset(run_a) is None and w._propset_chip(run_a) is None
    assert w._source_key(run_a, 0) not in w.propset_pick
    assert [m["proposal_set_id"] for m in w.propset_index.archived] \
        == ["propset_old_11111111"]
    assert (tmp_path / "proposals" / "propset_old" / "lifecycle.json").exists()
    assert (tmp_path / "proposals" / "propset_old" / "manifest.json").exists()
    w.on_archive_propset()                               # nothing left
    assert "no proposal set to archive" in w.notice
    # the picker hook: archive the PINNED run -> the pin misses LOUD
    pinned = next(m for m in w.trainrun_index.runs if m["run_id"].endswith("6f803b12"))
    runs, archived, note = w._on_trainrun_lifecycle(pinned, "archive")
    assert [m["run_id"][-8:] for m in runs] == ["b1072a9f"]
    assert [m["run_id"][-8:] for m in archived] == ["6f803b12"]
    assert note == "…6f803b12 archived" and w._model_chip() == "⚠ no model"
    w.picked = [w.src_index.runs[1]["_path"]]
    w.on_propose_missing()
    assert w.propose_armed is None and "ARCHIVED" in w.error
    # m still opens (archived runs exist), listing the active run until h
    w.on_pick_model()
    assert w.trainrun_picker.isVisible() and w.trainrun_picker.list.count() == 1
    assert len(w.trainrun_picker.archived) == 1
    w.trainrun_picker.reject()
    # delete refuses while a propset names the run; unarchive restores the pin
    runs, archived, note = w._on_trainrun_lifecycle(archived[0], "delete")
    assert note.startswith("refused:") and "propset_old" in note
    assert len(archived) == 1 and (tmp_path / "training-runs" / "trainrun_1_6f803b12").exists()
    runs, archived, note = w._on_trainrun_lifecycle(archived[0], "unarchive")
    assert note == "…6f803b12 active" and len(runs) == 2 and archived == []
    assert w._model_chip() == "6f803b12 (pin)"
    runs, archived, note = w._on_trainrun_lifecycle(runs[0], "unarchive")
    assert note.endswith("(already)")
