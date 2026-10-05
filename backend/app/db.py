"""SQLite connection and initial schema bootstrap."""

from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine

from backend.app.models import Base


def open_database(path: Path, *, create_schema: bool = True) -> Engine:
    database_path = Path(path).resolve()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(
        f"sqlite:///{database_path.as_posix()}",
        connect_args={"timeout": 10},
    )

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(connection, _record):
        cursor = connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    if create_schema:
        Base.metadata.create_all(engine)
    return engine
