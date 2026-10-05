"""One conservative arXiv metadata request at a time across local processes."""

from __future__ import annotations

from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import time
from typing import Iterator
from uuid import uuid4

from backend.app.file_lock import try_file_lock


MIN_REQUEST_INTERVAL_SECONDS = 3.0
MAX_INLINE_WAIT_SECONDS = 60.0
DEFAULT_GATE_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "runtime" / "arxiv-api.lock"
)
BOOT_TIME_TOLERANCE_SECONDS = 0.25


class ArxivCooldownActive(RuntimeError):
    """A shared server cooldown is too long to wait inside a worker."""

    def __init__(self, remaining_seconds: float) -> None:
        self.remaining_seconds = remaining_seconds
        super().__init__(
            f"arXiv shared cooldown has {remaining_seconds:.0f} seconds remaining."
        )


class RequestSlot:
    """The held request slot, including any server-requested cooldown."""

    def __init__(self, gate: ArxivRequestGate, initial_deadline: float) -> None:
        self._gate = gate
        self._initial_deadline = initial_deadline
        self._defer_seconds = 0.0
        self._active = True

    def defer_for(self, seconds: float) -> None:
        """Persist a server delay before releasing the process lock."""
        if not self._active:
            raise RuntimeError("The arXiv request slot has already closed.")
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Request cooldown must be finite and nonnegative.")
        self._defer_seconds = max(self._defer_seconds, seconds)
        self._gate._write_deadline(
            max(self._initial_deadline, time.monotonic() + self._defer_seconds)
        )


class ArxivRequestGate:
    def __init__(
        self,
        lock_path: Path = DEFAULT_GATE_PATH,
        *,
        minimum_interval_seconds: float = MIN_REQUEST_INTERVAL_SECONDS,
    ) -> None:
        if not math.isfinite(minimum_interval_seconds) or minimum_interval_seconds < 0:
            raise ValueError("minimum_interval_seconds must be finite and nonnegative")
        self.lock_path = lock_path.expanduser().resolve()
        self.state_path = self.lock_path.with_name(self.lock_path.name + ".last")
        self.minimum_interval_seconds = minimum_interval_seconds

    @contextmanager
    def slot(self) -> Iterator[RequestSlot]:
        """Wait for a request slot and hold it through the entire HTTP call."""
        while True:
            with try_file_lock(self.lock_path) as lock_file:
                if lock_file is not None:
                    deadline = self._next_allowed()
                    if deadline is not None:
                        # Sleep may return early; do not release the slot until
                        # the persisted deadline has actually passed.
                        while (wait := deadline - time.monotonic()) > 0:
                            if wait > MAX_INLINE_WAIT_SECONDS:
                                raise ArxivCooldownActive(wait)
                            time.sleep(wait)

                    # Persist before sending, so a process crash still leaves a slot.
                    initial_deadline = time.monotonic() + self.minimum_interval_seconds
                    self._write_deadline(initial_deadline)
                    slot = RequestSlot(self, initial_deadline)
                    try:
                        yield slot
                    finally:
                        # Completion-based spacing is conservative when calls are slow.
                        slot._active = False
                        self._write_deadline(
                            time.monotonic()
                            + max(self.minimum_interval_seconds, slot._defer_seconds)
                        )
                    return
            time.sleep(0.05)

    def _next_allowed(self) -> float | None:
        try:
            raw = self.state_path.read_text(encoding="ascii")
        except FileNotFoundError:
            return None
        try:
            saved = json.loads(raw)
            if isinstance(saved, dict) and saved.get("version") == 3:
                value = saved["next_allowed_unix"]
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ValueError("Invalid wall-clock deadline")
                # Persist wall time so a reboot cannot change the deadline's meaning.
                remaining = max(0.0, float(value) - time.time())
                return time.monotonic() + remaining
            if isinstance(saved, dict) and saved.get("version") == 2:
                value = saved["next_allowed_monotonic"]
            elif type(saved) in (int, float):
                # Before v2, this file held the time of the last request event.
                value = saved + self.minimum_interval_seconds
            else:
                raise ValueError("Invalid state shape")
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid arXiv request gate state: {self.state_path}") from error
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid arXiv request gate state: {self.state_path}")
        # Earlier formats cannot be translated across a reboot: their monotonic
        # origin has changed. The file timestamp tells us when that happened.
        boot_started_at = time.time() - time.monotonic()
        if self.state_path.stat().st_mtime < boot_started_at - BOOT_TIME_TOLERANCE_SECONDS:
            return None
        return float(value)

    def _write_deadline(self, value: float) -> None:
        temporary = self.state_path.with_name(
            f"{self.state_path.name}.{uuid4().hex}.tmp"
        )
        try:
            with temporary.open("w", encoding="ascii") as output:
                json.dump(
                    {
                        "version": 3,
                        "next_allowed_unix": time.time()
                        + max(0.0, value - time.monotonic()),
                    },
                    output,
                    allow_nan=False,
                )
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, self.state_path)
        finally:
            temporary.unlink(missing_ok=True)


DEFAULT_ARXIV_REQUEST_GATE = ArxivRequestGate()
