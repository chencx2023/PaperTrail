"""Access credentials and small per-session operation budgets for temporary sharing."""

from collections import deque
from contextlib import closing
import hmac
import time
import re
import sqlite3
from urllib.parse import urlsplit

from starlette.responses import JSONResponse, RedirectResponse


class PublicAccess:
    def __init__(self, token: str, *, mutation_limit=120, suggestion_limit=20, per_minute=20):
        if not isinstance(token, str) or len(token) < 32:
            raise ValueError("A public access token of at least 32 characters is required.")
        self.token = token
        self.mutation_limit = mutation_limit
        self.suggestion_limit = suggestion_limit
        self.per_minute = per_minute
        self.mutations = 0
        self.suggestions = 0
        self.recent = deque()
        self.expensive = {"prepare": 0, "answer": 0, "screening": 0}
        self.database_path = None

    def valid(self, value):
        return isinstance(value, str) and hmac.compare_digest(value.encode(), self.token.encode())

    def operation_kind(self, path):
        """Meter user intents once; their worker-managed stages do not pass this gate."""
        if path == "/api/answers" or re.fullmatch(r"/api/researches/[^/]+/answers", path):
            return "answer"
        if (path == "/api/collections"
                or re.fullmatch(r"/api/collections/[^/]+/(imports|revisions)", path)
                or re.fullmatch(r"/api/researches/[^/]+/prepare", path)):
            return "prepare"
        if re.fullmatch(r"/api/searches/[^/]+/screening", path):
            return "screening"
        job_retry = re.fullmatch(r"/api/jobs/([^/]+)/retry", path)
        research_retry = re.fullmatch(r"/api/researches/([^/]+)/retry", path)
        if not (job_retry or research_retry):
            return None
        if self.database_path is None:
            raise ValueError("Retry budgets require the workspace database.")
        with closing(sqlite3.connect(self.database_path.as_uri() + "?mode=ro", uri=True)) as connection:
            if job_retry:
                row = connection.execute("SELECT kind FROM jobs WHERE id = ?",
                    (job_retry.group(1),)).fetchone()
            else:
                row = connection.execute(
                    "SELECT o.kind FROM research_records r JOIN research_operations o "
                    "ON o.id = r.operation_id AND o.research_id = r.id WHERE r.id = ?",
                    (research_retry.group(1),)).fetchone()
        if row is None:
            # The downstream endpoint rejects missing/incompatible retry identities.
            return None
        if row[0] in {"import", "index", "prepare"}:
            return "prepare"
        if row[0] == "answer":
            return "answer"
        if job_retry and row[0] == "search":
            return None
        raise ValueError("The saved retry operation cannot be classified.")

    async def dispatch(self, request, call_next):
        def failure(status, message):
            return JSONResponse({"code": "PUBLIC_ACCESS_LIMIT" if status == 429 else "PUBLIC_ACCESS_REQUIRED", "message": message}, status_code=status)

        bearer = request.headers.get("authorization", "")
        bearer_ok = bearer.startswith("Bearer ") and self.valid(bearer[7:])
        safe = request.method in {"GET", "HEAD", "OPTIONS"}
        bootstrap = safe and self.valid(request.query_params.get("access"))
        cookie_ok = self.valid(request.cookies.get("papertrail_access"))
        if not (bearer_ok or cookie_ok or bootstrap):
            response = failure(401, "需要访问凭证，请使用包含凭证的完整分享链接。")
        elif bootstrap:
            params = request.query_params.multi_items()
            from urllib.parse import urlencode
            query = urlencode([(key, value) for key, value in params if key != "access"])
            response = RedirectResponse(request.url.path + ("?" + query if query else ""), status_code=303)
            # Public links use HTTPS even when the tunnel delivers plain HTTP.
            # Proxy header recognition must not weaken the access cookie.
            response.set_cookie("papertrail_access", self.token, httponly=True, samesite="lax", secure=True, max_age=86400)
        else:
            if not safe:
                origin = request.headers.get("origin")
                if not bearer_ok and (not origin or urlsplit(origin).netloc != request.headers.get("host")):
                    response = failure(403, "操作来源无法确认，请在工作台页面操作或使用 Bearer 凭证。")
                else:
                    now = time.monotonic()
                    while self.recent and self.recent[0] < now - 60:
                        self.recent.popleft()
                    suggestion = request.url.path in ("/api/query-suggestions", "/api/query-expansions")
                    path = request.url.path
                    try:
                        expensive_kind = self.operation_kind(path)
                    except (sqlite3.Error, ValueError, OSError):
                        response = failure(503, "暂时无法检查任务限额，请稍后再试。")
                        response.headers["Referrer-Policy"] = "no-referrer"
                        response.headers["Cache-Control"] = "no-store"
                        return response
                    # Retries consume the same costly-operation budget as initial submissions.
                    if len(self.recent) >= self.per_minute or self.mutations >= self.mutation_limit or (suggestion and self.suggestions >= self.suggestion_limit) or (expensive_kind and self.expensive[expensive_kind] >= 10):
                        response = failure(429, "本次分享的操作限额已达到，请稍后再试或联系工作台主人。")
                    else:
                        # Charge before awaiting downstream; concurrent requests cannot overrun the budget.
                        self.mutations += 1
                        self.suggestions += int(suggestion)
                        if expensive_kind:
                            self.expensive[expensive_kind] += 1
                        self.recent.append(now)
                        response = await call_next(request)
            else:
                response = await call_next(request)
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = "frame-ancestors 'none'"
        return response
