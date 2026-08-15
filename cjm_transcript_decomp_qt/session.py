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

from cjm_substrate.core.manager import CapabilityManager
from cjm_substrate_qt_kit.loopthread import LoopThreadSession
from cjm_transcript_decomp_core.cli import load_capabilities
from cjm_transcript_decomp_tui.segments import SegmentStack


class DecompSession(LoopThreadSession):
    """The loop-thread seat pair for the decomp shell.

    start() spins the loop (kit LoopThreadSession); open_source() re-points
    the lazy READ-ONLY SegmentStack seat and reads one source's committed
    spine back as ONE Future; probe_chain() runs the VAD sweep plus the
    FA-realign and sentence-span legs; propose_batch() is the P pre-stage.
    close() runs on_close() on the loop (seat teardown), then stops the
    thread — the exit path. One instance per window, like CapabilitySession."""

    thread_name = "decomp-session"

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

    async def _open_source(self, db_path, segment_ids, source_node_id):
        if self.stack is None:
            self.stack = self._stack_factory(self.manifests_dir,
                                             self.graph_capability)
        await self.stack.open(db_path)
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
