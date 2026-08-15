"""Span playback via QMediaPlayer — the Qt lane's audio answer (DEC dcf8a712)
stretched to the decomp shell's need: the r verb auditions a SPAN of a coarse
model-input WAV (a fine segment's slice, or a GAP's uncovered stretch — gap
audio exists in no fine chunk, only the coarse WAV spans it), at the bracket
ladder's 0.5–3.0× speed. Replaces the kit ChunkPlayer (sounddevice + WSOLA)
the Textual shell used; sounddevice stays retired on the lane.

Mechanics: seeks land only once media is loaded, so play_span defers the
setPosition+play to mediaStatusChanged when the source is fresh (an already-
loaded source replays immediately); positionChanged stops at the span end
(backend-granular — a sub-100ms overshoot, inaudible at these spans). Speed
rides setPlaybackRate (the FFmpeg backend's tempo scaling)."""

from typing import Optional

from PySide6.QtCore import QUrl
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer

_READY = (QMediaPlayer.LoadedMedia, QMediaPlayer.BufferedMedia,
          QMediaPlayer.EndOfMedia)


class SpanPlayer:
    """Play/stop one WAV span at a time; r replays, escape stops."""

    def __init__(self, parent=None):
        self._player = QMediaPlayer(parent)
        self._out = QAudioOutput(parent)
        self._player.setAudioOutput(self._out)
        self._player.positionChanged.connect(self._check_end)
        self._player.mediaStatusChanged.connect(self._on_status)
        self._start_ms = 0
        self._end_ms: Optional[int] = None
        self._pending = False

    @property
    def playing(self) -> bool:
        return self._player.playbackState() == QMediaPlayer.PlayingState

    def play_span(self, path: str, start_s: float, end_s: float,
                  rate: float = 1.0) -> None:
        """Start `path` at start_s, stopping at end_s (WAV-local seconds).

        Stop-then-play always: stale audio under a fresh focus would mismatch
        the row on screen (the SegmentPlayer contract, span edition)."""
        self._player.stop()
        self._start_ms = max(0, int(start_s * 1000))
        self._end_ms = int(end_s * 1000)
        self._player.setPlaybackRate(max(0.1, float(rate)))
        url = QUrl.fromLocalFile(path)
        if (self._player.source() == url
                and self._player.mediaStatus() in _READY):
            self._begin()
        else:
            self._pending = True
            self._player.setSource(url)

    def _on_status(self, status) -> None:
        if self._pending and status in _READY:
            self._pending = False
            self._begin()

    def _begin(self) -> None:
        self._player.setPosition(self._start_ms)
        self._player.play()

    def _check_end(self, pos: int) -> None:
        if self._end_ms is not None and pos >= self._end_ms:
            self.stop()

    def stop(self) -> None:
        self._pending = False
        self._end_ms = None
        self._player.stop()

    def close(self) -> None:
        self.stop()
        self._player.setSource(QUrl())

    def error_text(self) -> Optional[str]:
        """The player's last error string, or None (surfaced in-status)."""
        if self._player.error() == QMediaPlayer.NoError:
            return None
        return self._player.errorString() or "playback error"
