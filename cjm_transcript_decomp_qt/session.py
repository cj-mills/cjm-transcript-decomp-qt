"""One read-only decomp inspection seat + the propose pre-stage behind a
private asyncio loop thread — the decomp lane's jobs seam (DEC 6c574c89, the
CapabilitySession pattern with decomp verbs).

The segments drill and the VAD/FA/sentence probes live in the tui's
SegmentStack (asyncio-native); the P propose pre-stage drives a
CapabilityManager task channel. Under Qt neither may touch the paint thread,
so a DecompSession owns a daemon loop thread where both seats live and
exposes them as concurrent Futures the shell resolves through queued Signals.
Deliberately Qt-free (stdlib + spine imports only): testable headless, seat
factories injectable.

Degradation semantics carried from the Textual shell verbatim: the composite
probe's FA leg failing borrows committed text (never wedges the probe), the
sentence-span leg failing drops the split preview; the propose seat tears
down in finally and reports how many sets landed before an error."""

import asyncio
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from cjm_context_graph_layer.ops import store_covariates
from cjm_substrate.core.manager import CapabilityManager
from cjm_substrate_qt_kit.gesturebench import ladder
from cjm_substrate_qt_kit.loopthread import LoopThreadSession
from cjm_transcript_decomp_core import retire
from cjm_transcript_decomp_core.cli import load_capabilities
from cjm_transcript_decomp_core.segments import SegmentStack


class DecompSession(LoopThreadSession):
    """The loop-thread seat pair for the decomp shell.

    start() spins the loop (kit LoopThreadSession); open_source() re-points
    the lazy READ-ONLY SegmentStack seat and reads one source's committed
    spine back as ONE Future; probe_chain() runs the VAD sweep plus the
    FA-realign and sentence-span legs; propose_batch() is the P pre-stage.
    close() runs on_close() on the loop (seat teardown), then stops the
    thread — the exit path. One instance per window, like CapabilitySession.
    Every verb is a measured gesture (the kit's seam, design detail 4389a579)
    in the timings ledger of the db the seat last opened."""

    thread_name = "decomp-session"
    app = "decomp"

    def __init__(self, manifests_dir: str,   # Capability manifests directory
                 graph_capability: str,      # Graph capability instance id
                 *, timeout: float = 600.0,  # probe models load cold sometimes
                 stack_factory: Callable[..., Any] = SegmentStack,
                 manager_factory: Callable[..., Any] = CapabilityManager,
                 loader: Callable[..., Any] = load_capabilities):
        super().__init__(timeout=timeout)
        self.manifests_dir = manifests_dir
        self.graph_capability = graph_capability
        self._stack_factory = stack_factory
        self._manager_factory = manager_factory
        self._loader = loader
        self.stack: Optional[Any] = None  # the SegmentStack seat (loop-owned)
        self._measured_db: Optional[str] = None  # the db the gesture seam measures into

    async def measure_covariates(self) -> Optional[Dict[str, Any]]:
        """The store's growth covariates beside the gestures (design detail 4389a579): the
        layer's store half over the seat's read-only queue."""
        if self.stack is None or self.stack.queue is None:
            return None
        return await store_covariates(self.stack.queue, self.graph_capability)

    # ---- segments seat ---------------------------------------------------

    def open_source(self, db_path: str, segment_ids: List[str],
                    source_node_id: str) -> Future:
        """Open (or re-point) the read-only seat at the db this decomp run
        RECORDED (e087d059 provenance), read the manifest's segment_ids back
        in spine order, and join the coarse-WAV spans — one Future. Resolves
        to {"rows", "asegs"}; the caller counts the id shortfall (missing
        rows are dropped, never invented)."""
        return self.submit(self._open_source(db_path, segment_ids,
                                             source_node_id))

    async def _seat(self, db_path: str) -> Any:  # The open read-only seat
        """Open (or re-point) the read-only seat at a db; a re-pointed seat moves the gesture
        seam's ledger and covariates with it (design detail 4389a579)."""
        if self.stack is None:
            self.stack = self._stack_factory(self.manifests_dir,
                                             self.graph_capability)
        await self.stack.open(db_path)
        if self._measured_db != db_path:
            self._measured_db = db_path
            await self.measure_into(db_path)
        return self.stack

    async def _open_source(self, db_path, segment_ids, source_node_id):
        await self._seat(db_path)
        rows = await self.stack.read_segments(segment_ids)
        rends = sorted({str(r.get("rendition_id"))
                        for r in rows if r.get("rendition_id")})
        asegs = (await self.stack.read_audio_join(source_node_id, rends)
                 if source_node_id and rends else [])
        return {"rows": rows, "asegs": asegs}

    # ---- the composite probe (VAD -> FA realign -> sentence spans) --------

    def probe_chain(self, vad_id: str, vad_cfg: Dict[str, Any], wav_path: str,
                    *, rendition_id: Optional[str], text_from: str,
                    fa_id: str, fa_cfg: Dict[str, Any],
                    seg_id: str, seg_cfg: Dict[str, Any],
                    progress: Optional[Callable[[str], None]] = None) -> Future:
        """Run a VAD config over one coarse WAV, then the FA-realign and
        capability sentence-span legs (PROBE-ONLY, ee4a4b9c). Degradation is
        the Textual contract: an FA failure lands in "fa_error" (the shell
        borrows committed text), a sentence-span failure in "seg_error" (the
        split preview drops) — the VAD result always resolves. `progress`
        (a plain callable — the shell wires it to a queued Signal) narrates
        the FA leg the way the Textual notice did."""
        return self.submit(self._probe_chain(vad_id, vad_cfg, wav_path,
                                             rendition_id, text_from,
                                             fa_id, fa_cfg, seg_id, seg_cfg,
                                             progress))

    async def _probe_chain(self, vad_id, vad_cfg, wav_path, rendition_id,
                           text_from, fa_id, fa_cfg, seg_id, seg_cfg,
                           progress):
        if self.stack is None:
            raise RuntimeError("segment seat not open — open_source first")
        local = await self.stack.probe_vad(vad_id, dict(vad_cfg), wav_path)
        out: Dict[str, Any] = {"local": local, "fa_text": None,
                               "fa_words": None, "seg_spans": None,
                               "fa_error": None, "seg_error": None}
        if rendition_id and text_from:
            try:
                text = await self.stack.read_transcript_text(rendition_id,
                                                             text_from)
                if text:
                    if progress:
                        progress(f"aligning (FA · {len(local)} chunk(s)) …")
                    out["fa_words"] = await self.stack.probe_fa(
                        fa_id, dict(fa_cfg), wav_path, text)
                    out["fa_text"] = text
            except (Exception, SystemExit) as e:
                out["fa_error"] = str(e)
        if out["fa_text"]:
            try:
                out["seg_spans"] = await self.stack.probe_segment(
                    seg_id, dict(seg_cfg), out["fa_text"])
            except (Exception, SystemExit) as e:
                out["seg_error"] = str(e)
        return out

    # ---- the P propose pre-stage -----------------------------------------

    def propose_batch(self, event_capability: str, training_run_path: str,
                      sources: List[str],
                      progress: Optional[Callable[[str], None]] = None) -> Future:
        """The propose seat (DEC 1cfe6d0f): load the event capability
        (blocking load on a worker thread — the loop keeps serving), run the
        audio_event_detection propose task per source, tear down in finally.
        Resolves to {"landed": n, "error": str|None} — a failure reports how
        many sets landed first, the Textual message's contract. The armed
        two-step confirm stays in the shell."""
        return self.submit(self._propose_batch(event_capability,
                                               training_run_path, sources,
                                               progress))

    async def _propose_batch(self, event_capability, training_run_path,
                             sources, progress):
        manager = None
        landed = 0
        error: Optional[str] = None
        try:
            if progress:
                progress("propose: opening capability seat…")
            manager = self._manager_factory(
                search_paths=[Path(self.manifests_dir)])
            await asyncio.to_thread(self._loader, manager, [event_capability])
            for i, src_path in enumerate(sources, 1):
                if progress:
                    progress(f"propose {i}/{len(sources)}: "
                             f"{Path(src_path).name} … (batch pre-stage)")
                await manager.execute_capability_task_async(
                    event_capability, "audio_event_detection", "propose",
                    training_run=training_run_path, source=src_path)
                landed += 1
        except (Exception, SystemExit) as e:
            # SystemExit included: load_capabilities exits on a missing
            # capability, and a library exit must resolve, not kill the loop.
            error = str(e)
        finally:
            if manager is not None:
                try:
                    await asyncio.to_thread(manager.unload_capability,
                                            event_capability)
                except Exception:
                    pass
        return {"landed": landed, "error": error}

    # ---- teardown --------------------------------------------------------

    async def on_close(self) -> None:
        """Kit close() hook: tear the segments seat down ON the loop."""
        if self.stack is not None:
            try:
                await self.stack.close()
            except Exception:
                pass
            self.stack = None


def bench_gestures(
    params: Dict[str, Any],  # db, manifests_dir; optionally the lane's source
) -> List[str]:  # The gestures run, in order
    """The decomp shell's canonical gesture for the scale harness (design detail 4389a579): the
    segments drill on one graph -- the read-only seat, one source's live spine read back by its
    segment ids, the coarse-WAV join -- measured by the session's own seam. The ids are the
    ones a fresh decomp run would record: the source's default live spine, found by the
    `find_spine` step (a measured read of its own; the lane's `source`, else the first source
    by id with a live spine). Recorded runs are not used: a respine retires the ids they hold."""
    s = DecompSession(params["manifests_dir"], params.get("graph_capability")
                      or "cjm-capability-graph-sqlite",
                      timeout=float(params.get("timeout_s", 3600)))
    found: Dict[str, Any] = {}

    async def find_spine():
        q, gid = s.stack.queue, s.graph_capability
        sids = ([params["source"]] if params.get("source")
                else [sid for sid, _ in sorted(await retire.list_sources(q, gid))])
        for sid in sids:
            rends = await retire.source_rendition_ids(q, gid, sid)
            spine = retire.default_live_spine(await retire.list_spines(q, gid, sid, rends))
            if spine and spine.get("segments"):
                found.update(source=sid, ids=await retire.spine_segment_ids(
                    q, gid, rends, spine.get("skeleton_hash")))
                return
        raise RuntimeError("no source with a live spine")

    s.start()
    try:
        return ladder([("seat", lambda: s.call(s._seat(params["db"]), gesture="seat")),
                       ("find_spine", lambda: s.call(find_spine(), gesture="find_spine")),
                       ("open_source", lambda: s.open_source(
                           params["db"], found["ids"], found["source"]).result(s.timeout))])
    finally:
        s.close()
