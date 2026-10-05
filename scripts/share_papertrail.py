"""Temporarily share the fully operational app through a protected HTTPS tunnel."""

import argparse
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from scripts.run_papertrail import preflight, stop_children


def tunnel_command(provider, cloudflared, port, logs):
    if provider == "cloudflare":
        return [str(cloudflared.resolve()), "tunnel", "--no-autoupdate", "--protocol", "http2", "--url", f"http://127.0.0.1:{port}"]
    ssh = shutil.which("ssh")
    if not ssh:
        raise ValueError("OpenSSH is required for localhost.run.")
    return [ssh, "-T", "-F", "none", "-o", "BatchMode=yes", "-o", "PubkeyAuthentication=no",
            "-o", "PasswordAuthentication=no", "-o", "KbdInteractiveAuthentication=no",
            "-o", "IdentityAgent=none", "-o", "StrictHostKeyChecking=accept-new",
            "-o", f"UserKnownHostsFile={logs.resolve().as_posix()}/known_hosts", "-o", "ServerAliveInterval=30",
            "-o", "ConnectTimeout=15", "-o", "ExitOnForwardFailure=yes",
            "-R", f"80:127.0.0.1:{port}", "nokey@localhost.run"]


def find_public_url(provider, output):
    if provider == "cloudflare":
        pattern = r"(https://[a-z0-9-]+\.trycloudflare\.com)(?=[\s/]|$)"
    else:
        # The SSH banner includes admin.localhost.run before the actual tunnel.
        # Accept only an announced tunnel endpoint, never a documentation link.
        pattern = r"\btunneled with tls termination,\s*(https://[a-z0-9-]+\.(?:lhr\.life|localhost\.run))(?=[\s/]|$)"
    found = re.findall(pattern, output)
    return found[-1] if found else None


def public_access_probe(client, public_url, token):
    """Read our access gate and worker through the real HTTPS endpoint."""
    try:
        denied = client.get(public_url + "/api/health")
        if denied.status_code != 401 or denied.json().get("code") != "PUBLIC_ACCESS_REQUIRED":
            return False
        runtime = client.get(public_url + "/api/runtime", headers={"Authorization": f"Bearer {token}"})
        if runtime.status_code != 200:
            return False
        state = runtime.json()
        return state.get("worker_status") == "running" and state.get("shared_workspace") is True
    except (httpx.HTTPError, ValueError, AttributeError):
        return False


def update_public_availability(access_file, owner_pid, reachable, *, public_url=None):
    saved = json.loads(access_file.read_text(encoding="utf-8"))
    if saved.get("pid") != owner_pid:
        return
    checked_at = datetime.now(timezone.utc).isoformat()
    saved.update(active=reachable, last_checked_at=checked_at)
    if reachable:
        saved["last_verified_at"] = checked_at
        if public_url:
            saved["public_url"] = public_url
            saved["share_url"] = public_url + "/?access=" + saved["access_token"]
    access_file.write_text(json.dumps(saved, indent=2), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=PROJECT_ROOT / "data/public-workspace/runtime.sqlite")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data/public-workspace/data")
    parser.add_argument("--init-db", action="store_true", help="Explicitly initialize a new shared workspace.")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--cloudflared", type=Path, default=PROJECT_ROOT / ".tooling/cloudflared.exe")
    parser.add_argument("--provider", choices=("cloudflare", "localhost.run"), default="cloudflare")
    parser.add_argument("--live", action="store_true", help="Allow queued jobs and configured external services.")
    args = parser.parse_args(argv)
    if not args.live:
        print("[PREVIEW] No public link or worker started. Add --live for full operation.")
        return 0
    if args.provider == "cloudflare" and not args.cloudflared.is_file():
        parser.error("Install the official cloudflared binary at --cloudflared first.")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.init_db:
        from backend.app.db import open_database
        engine = open_database(args.database)
        engine.dispose()
    preflight(args.database, args.port)
    token = secrets.token_urlsafe(36)
    logs = PROJECT_ROOT / ".tooling/share"
    logs.mkdir(parents=True, exist_ok=True)
    access_file = logs / "access.json"
    if access_file.is_file():
        previous = json.loads(access_file.read_text(encoding="utf-8"))
        access_file.write_text(json.dumps({**previous, "active": False}, indent=2), encoding="utf-8")
    stop_file = logs / "stop"
    stop_file.unlink(missing_ok=True)
    environment = dict(os.environ, PAPERTRAIL_ACCESS_TOKEN=token, PYTHONUTF8="1", PYTHONUNBUFFERED="1", LITELLM_LOCAL_MODEL_COST_MAP="True")
    children = []
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    try:
        with (logs / "app.log").open("w", encoding="utf-8") as app_log, (logs / "tunnel.log").open("w", encoding="utf-8") as tunnel_log:
            children.append(subprocess.Popen([sys.executable, "-u", str(PROJECT_ROOT / "scripts/run_papertrail.py"), "--database", str(args.database.resolve()), "--data-root", str(args.data_root.resolve()), "--port", str(args.port), "--stop-file", str(stop_file), "--live", "--share"], cwd=PROJECT_ROOT, env=environment, stdout=app_log, stderr=subprocess.STDOUT, creationflags=flags))
            headers = {"Authorization": f"Bearer {token}"}
            deadline = time.monotonic() + 45
            with httpx.Client(timeout=2, trust_env=False) as client:
                while True:
                    if children[0].poll() is not None:
                        raise RuntimeError("App startup failed; inspect .tooling/share/app.log.")
                    try:
                        runtime = client.get(f"http://127.0.0.1:{args.port}/api/runtime", headers=headers)
                        if runtime.status_code == 200 and runtime.json()["worker_status"] == "running":
                            break
                    except httpx.HTTPError:
                        pass
                    if time.monotonic() > deadline:
                        raise RuntimeError("API/worker readiness timed out; no tunnel opened.")
                    time.sleep(.5)
            children.append(subprocess.Popen(tunnel_command(args.provider, args.cloudflared, args.port, logs), cwd=PROJECT_ROOT, stdout=tunnel_log, stderr=subprocess.STDOUT, creationflags=flags))
            deadline = time.monotonic() + 60
            ready_count = 0
            ready_url = None
            while True:
                if any(child.poll() is not None for child in children):
                    raise RuntimeError("Sharing process exited; inspect .tooling/share logs.")
                found = find_public_url(args.provider, (logs / "tunnel.log").read_text(encoding="utf-8", errors="replace"))
                if found:
                    if found != ready_url:
                        ready_count = 0
                        ready_url = found
                    with httpx.Client(timeout=4, trust_env=False) as client:
                        ready_count = ready_count + 1 if public_access_probe(client, found, token) else 0
                    if ready_count < 2:
                        if time.monotonic() > deadline:
                            raise RuntimeError("Public HTTPS endpoint is unreachable; no active share link published.")
                        time.sleep(3)
                        continue
                    public_url = found
                    access_file.write_text(json.dumps({"public_url": public_url, "access_token": token, "share_url": public_url + "/?access=" + token, "local_port": args.port, "database": str(args.database.resolve()), "data_root": str(args.data_root.resolve()), "provider": args.provider, "mode": "full_operation", "pid": os.getpid(), "active": True}, indent=2), encoding="utf-8")
                    update_public_availability(access_file, os.getpid(), True)
                    print("[PUBLIC ADDRESS READY] Full share link saved in .tooling/share/access.json (private, ignored by Git).", flush=True)
                    break
                if time.monotonic() > deadline:
                    raise RuntimeError("Tunnel registration timed out; no verified public address.")
                time.sleep(.5)
            next_check = time.monotonic() + 30
            reachable = True
            with httpx.Client(timeout=4, trust_env=False) as client:
                while True:
                    if any(child.poll() is not None for child in children):
                        raise RuntimeError("App or tunnel stopped; the share is no longer available.")
                    if time.monotonic() >= next_check:
                        latest = find_public_url(args.provider, (logs / "tunnel.log").read_text(encoding="utf-8", errors="replace"))
                        if latest and latest != public_url and public_access_probe(client, latest, token) and public_access_probe(client, latest, token):
                            public_url = latest
                            checked = True
                            print("[PUBLIC ADDRESS UPDATED] Provider rotated its hostname; the verified new share link is saved in .tooling/share/access.json.", flush=True)
                        else:
                            checked = public_access_probe(client, public_url, token)
                        update_public_availability(access_file, os.getpid(), checked, public_url=public_url)
                        if checked != reachable:
                            print("[PUBLIC REACHABLE]" if checked else "[PUBLIC UNREACHABLE] SSH/app still running, but the public link is unavailable.", flush=True)
                        reachable = checked
                        next_check = time.monotonic() + 30
                    time.sleep(.5)
    except KeyboardInterrupt:
        return 0
    except (RuntimeError, OSError) as error:
        print(f"[SHARE STOPPED] {error}", flush=True)
        return 1
    finally:
        # Stop the supervised app gracefully so it can terminate its API/worker children.
        if children and children[0].poll() is None:
            # A stop file is understood by the launcher even on Windows without a console.
            (logs / "stop").touch()
            try:
                children[0].wait(timeout=12)
            except subprocess.TimeoutExpired:
                pass
        stop_children(children)
        if access_file.is_file():
            saved = json.loads(access_file.read_text(encoding="utf-8"))
            if saved.get("pid") == os.getpid():
                access_file.write_text(json.dumps({**saved, "active": False}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
