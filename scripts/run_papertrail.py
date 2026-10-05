"""Start the API and worker together; preview unless --live or --offline is explicit."""

import argparse
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.db import open_database
from backend.app.jobs.process_lock import worker_status
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError


def child_commands(database, data_root, port, *, offline=False, share=False):
    common = ["--database", str(Path(database).resolve()), "--data-root", str(Path(data_root).resolve())]
    api = [sys.executable, "-u", str(PROJECT_ROOT / "scripts/api_server.py"), *common, "--port", str(port)]
    worker = [sys.executable, "-u", str(PROJECT_ROOT / "scripts" / ("t06_offline_demo_worker.py" if offline else "job_worker.py")), *common]
    worker += ["--watch"] if offline else ["--live"]
    if share:
        api += ["--share"]
    return [api, worker]


def preflight(database, port):
    if not Path(database).is_file():
        raise ValueError("Database missing; initialize it explicitly with api_server.py --init-db.")
    if worker_status(Path(database)) != "stopped":
        raise ValueError("Another worker owns this database, or its lock cannot be checked.")
    with socket.socket() as listener:
        try:
            listener.bind(("127.0.0.1", port))
        except OSError as error:
            raise ValueError(f"API port {port} is already in use.") from error
    engine = open_database(Path(database), create_schema=False)
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1 FROM jobs LIMIT 1"))
    except SQLAlchemyError:
        raise ValueError("Database schema is unavailable; no processes started.") from None
    finally:
        engine.dispose()


def stop_children(children):
    # Only children created by this launcher; never kill another service by port/name.
    for child in children:
        if child.poll() is None:
            child.terminate()
    for child in children:
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=PROJECT_ROOT / "data/runtime/papertrail.sqlite")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--share", action="store_true", help="Use access credentials supplied by share_papertrail.py.")
    parser.add_argument("--stop-file", type=Path, help="Stop owned children when this supervisor signal file appears.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--live", action="store_true", help="Run queued tasks, including configured external services.")
    mode.add_argument("--offline", action="store_true", help="Use fixed fixtures, only under .tooling.")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    commands = child_commands(args.database, args.data_root, args.port, offline=args.offline, share=args.share)
    if not (args.live or args.offline):
        print("[PREVIEW] No processes started. Add --live, or --offline for isolated fixtures.")
        return 0
    if args.offline:
        allowed = (PROJECT_ROOT / ".tooling").resolve()
        if not args.database.resolve().is_relative_to(allowed) or not args.data_root.resolve().is_relative_to(allowed):
            parser.error("Offline fixtures must stay under .tooling.")
    try:
        preflight(args.database, args.port)
    except (ValueError, OSError) as error:
        print(f"[STARTUP FAILED] {error}")
        return 2
    children = []
    environment = dict(os.environ, LITELLM_LOCAL_MODEL_COST_MAP="True", PYTHONUTF8="1")
    try:
        for command in commands:
            children.append(subprocess.Popen(command, cwd=PROJECT_ROOT, env=environment))
        print(f"[STARTED] http://127.0.0.1:{args.port} — API and worker, same database. Ctrl+C stops both.")
        while True:
            if args.stop_file and args.stop_file.exists():
                return 0
            for name, child in zip(("API", "worker"), children):
                code = child.poll()
                if code is not None:
                    print(f"[STOPPED] {name} exited with code {code}; no automatic task retry.")
                    return 1
            time.sleep(0.5)
    except KeyboardInterrupt:
        return 0
    finally:
        stop_children(children)


if __name__ == "__main__":
    raise SystemExit(main())
