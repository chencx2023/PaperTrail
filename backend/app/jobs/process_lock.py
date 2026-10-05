"""Keep one Job worker active per database across OS processes."""

from contextlib import contextmanager
import errno
import os
from pathlib import Path
from typing import Iterator

from backend.app.file_lock import try_file_lock


def worker_status(database_path: Path) -> str:
    """Probe the existing OS lock without creating files or starting a worker.

    A held lock means a cooperating worker exists, not that an upstream is healthy.
    """
    database_path = database_path.expanduser().resolve()
    lock_path = database_path.with_name(database_path.name + ".worker.lock")
    try:
        with lock_path.open("r+b") as handle:
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError as error:
                if error.errno in {errno.EACCES, errno.EAGAIN} or getattr(error, "winerror", None) == 33:
                    return "running"
                return "unknown"
    except FileNotFoundError:
        return "stopped"
    except OSError:
        return "unknown"
    return "stopped"


@contextmanager
def worker_singleton(database_path: Path) -> Iterator[bool]:
    """Hold an OS lock until the worker exits; yield False on contention."""
    database_path = database_path.expanduser().resolve()
    lock_path = database_path.with_name(database_path.name + ".worker.lock")
    with try_file_lock(lock_path) as lock_file:
        yield lock_file is not None
