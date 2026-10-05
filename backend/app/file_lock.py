"""Small cross-process file lock shared by worker and arXiv request gates."""

from contextlib import contextmanager
import errno
import os
from pathlib import Path
from typing import BinaryIO, Iterator


@contextmanager
def try_file_lock(path: Path) -> Iterator[BinaryIO | None]:
    """Yield the locked file, or None when another process holds its first byte."""
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    # Leave the file in place: unlinking it would create a second lock target.
    with path.open("a+b") as lock_file:
        lock_file.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            if error.errno in {errno.EACCES, errno.EAGAIN} or getattr(
                error, "winerror", None
            ) == 33:
                yield None
                return
            raise

        try:
            yield lock_file
        finally:
            lock_file.seek(0)
            if os.name == "nt":
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
