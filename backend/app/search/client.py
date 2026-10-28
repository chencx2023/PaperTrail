"""HTTP boundary and retry policy for arXiv search."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import ExitStack
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import time
from typing import Callable, Protocol, TYPE_CHECKING

import httpx

from backend.app.search.arxiv import (
    ArxivAPIError,
    InvalidArxivFeed,
    ParsedArxivFeed,
    parse_atom_feed,
)
from backend.app.search.request_gate import (
    DEFAULT_ARXIV_REQUEST_GATE,
    MIN_REQUEST_INTERVAL_SECONDS,
    ArxivCooldownActive,
    ArxivRequestGate,
)

if TYPE_CHECKING:
    from backend.app.search.service import SearchQueryPlan


ARXIV_ENDPOINT = "https://export.arxiv.org/api/query"
TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}
MAX_AUTOMATIC_RETRY_WAIT_SECONDS = 60.0
MAX_RECORDED_RETRY_AFTER_SECONDS = 1_000_000_000_000.0
DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS = 120.0


class ArxivTransportError(RuntimeError):
    """Raised when no HTTP response was received."""


class ArxivCooldownError(ArxivTransportError):
    """Raised when a shared cooldown blocks a request before sending."""


@dataclass(frozen=True, slots=True)
class ArxivRequest:
    endpoint: str
    params: tuple[tuple[str, str], ...]
    headers: tuple[tuple[str, str], ...]
    timeout_seconds: float


@dataclass(frozen=True, slots=True)
class TransportResponse:
    status_code: int
    body: bytes
    headers: tuple[tuple[str, str], ...]


class ArxivTransport(Protocol):
    def send(self, request: ArxivRequest) -> TransportResponse: ...


@dataclass(frozen=True, slots=True)
class SearchAttempt:
    attempt: int
    outcome: str
    status_code: int | None
    started_at: str
    finished_at: str
    error_type: str | None
    error_message: str | None


@dataclass(frozen=True, slots=True)
class ArxivFetchResult:
    request: ArxivRequest
    response_body: bytes
    response_headers: tuple[tuple[str, str], ...]
    fetched_at: str
    attempts: tuple[SearchAttempt, ...]
    feed: ParsedArxivFeed


class ArxivSearchError(RuntimeError):
    """A failed search together with the evidence needed to persist it."""

    def __init__(
        self,
        *,
        kind: str,
        message: str,
        request: ArxivRequest,
        attempts: tuple[SearchAttempt, ...],
        response_body: bytes | None = None,
        response_headers: tuple[tuple[str, str], ...] = (),
        fetched_at: str | None = None,
        status_code: int | None = None,
    ) -> None:
        self.kind = kind
        self.request = request
        self.attempts = attempts
        self.response_body = response_body
        self.response_headers = response_headers
        self.fetched_at = fetched_at
        self.status_code = status_code
        super().__init__(message)


def build_arxiv_request(plan: SearchQueryPlan) -> ArxivRequest:
    return ArxivRequest(
        endpoint=ARXIV_ENDPOINT,
        params=(
            ("search_query", plan.search_query),
            ("start", "0"),
            ("max_results", "20"),
            ("sortBy", "relevance"),
            ("sortOrder", "descending"),
        ),
        headers=(("User-Agent", "PaperTrail/0.1"),),
        timeout_seconds=20.0,
    )


class HttpxArxivTransport:
    """HTTPX transport used by the live CLI."""

    def __init__(
        self,
        *,
        http_transport: httpx.BaseTransport | None = None,
        request_gate: ArxivRequestGate | None = DEFAULT_ARXIV_REQUEST_GATE,
        traffic_gate: ArxivRequestGate | None = None,
    ) -> None:
        self._client = httpx.Client(
            transport=http_transport,
            follow_redirects=False,
            limits=httpx.Limits(max_connections=1, max_keepalive_connections=0),
        )
        self._request_gate = request_gate
        self._traffic_gate = traffic_gate

    def send(self, request: ArxivRequest) -> TransportResponse:
        if self._request_gate is None:
            return self._send_under_gate(request)
        try:
            with ExitStack() as stack:
                slot = stack.enter_context(self._request_gate.slot())
                if self._traffic_gate is not None:
                    stack.enter_context(self._traffic_gate.slot())
                result = self._send_under_gate(request)
                retry_after = response_cooldown_seconds(result, datetime.now(timezone.utc))
                if retry_after is not None:
                    slot.defer_for(retry_after)
                return result
        except ArxivCooldownActive as error:
            raise ArxivCooldownError(str(error)) from error

    def _send_under_gate(self, request: ArxivRequest) -> TransportResponse:
        try:
            response = self._client.get(
                request.endpoint,
                params=request.params,
                headers=dict(request.headers),
                timeout=request.timeout_seconds,
            )
        except httpx.RequestError as error:
            raise ArxivTransportError(str(error)) from error
        return TransportResponse(
            status_code=response.status_code,
            body=response.content,
            headers=tuple(response.headers.multi_items()),
        )

    def close(self) -> None:
        self._client.close()


class ArxivClient:
    def __init__(
        self,
        *,
        transport: ArxivTransport | None = None,
        max_attempts: int = 3,
        minimum_interval_seconds: float = MIN_REQUEST_INTERVAL_SECONDS,
        clock: Callable[[], datetime] | None = None,
        monotonic: Callable[[], float] | None = None,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        self.transport = transport or HttpxArxivTransport()
        self.max_attempts = max_attempts
        self.minimum_interval_seconds = minimum_interval_seconds
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.monotonic = monotonic or time.monotonic
        self.sleeper = sleeper or time.sleep
        self._last_request_started: float | None = None
        self._cooldown_until: float | None = None

    def fetch(self, plan: SearchQueryPlan) -> ArxivFetchResult:
        request = build_arxiv_request(plan)
        attempts: list[SearchAttempt] = []
        retry_not_before: float | None = None
        if self._cooldown_until is not None:
            remaining = self._cooldown_until - self.monotonic()
            if remaining > 0:
                raise ArxivSearchError(kind="cooldown",
                    message=f"arXiv request cooldown is active for {remaining:.0f} more seconds.",
                    request=request, attempts=())
            self._cooldown_until = None

        for attempt_number in range(1, self.max_attempts + 1):
            self._wait_for_request_slot(retry_not_before)
            started_at = _format_time(self.clock())
            try:
                response = self.transport.send(request)
            except ArxivCooldownError as error:
                raise ArxivSearchError(
                    kind="cooldown",
                    message=str(error),
                    request=request,
                    attempts=tuple(attempts),
                ) from error
            except ArxivTransportError as error:
                attempts.append(
                    SearchAttempt(
                        attempt=attempt_number,
                        outcome="transport_error",
                        status_code=None,
                        started_at=started_at,
                        finished_at=_format_time(self.clock()),
                        error_type=type(error).__name__,
                        error_message=str(error),
                    )
                )
                if attempt_number < self.max_attempts:
                    retry_not_before = self.monotonic() + self._backoff(attempt_number)
                    continue
                raise ArxivSearchError(
                    kind="transport_error",
                    message=str(error),
                    request=request,
                    attempts=tuple(attempts),
                ) from error

            fetched_at = _format_time(self.clock())
            if not 200 <= response.status_code < 300:
                attempts.append(
                    SearchAttempt(
                        attempt=attempt_number,
                        outcome="http_error",
                        status_code=response.status_code,
                        started_at=started_at,
                        finished_at=fetched_at,
                        error_type="HTTPStatusError",
                        error_message=f"HTTP {response.status_code}",
                    )
                )
                retry_after = response_cooldown_seconds(response, self.clock())
                if retry_after is not None and retry_after > MAX_AUTOMATIC_RETRY_WAIT_SECONDS:
                    # Also protect subsequent queries using an injected transport.
                    self._cooldown_until = self.monotonic() + retry_after
                message = f"arXiv returned HTTP {response.status_code}."
                if (response.status_code == 429
                        and retry_after_seconds(response.headers, self.clock()) is None):
                    message += " No valid Retry-After; local requests paused for 120 seconds."
                if (
                    response.status_code in TRANSIENT_STATUS_CODES
                    and attempt_number < self.max_attempts
                ):
                    delay = max(self._backoff(attempt_number), retry_after or 0.0)
                    if delay <= MAX_AUTOMATIC_RETRY_WAIT_SECONDS:
                        retry_not_before = self.monotonic() + delay
                        continue
                    if retry_after_seconds(response.headers, self.clock()) is not None:
                        message = (
                            f"arXiv returned HTTP {response.status_code}; "
                            "Retry-After exceeds the 60-second automatic retry window."
                        )
                raise ArxivSearchError(
                    kind="http_error",
                    message=message,
                    request=request,
                    attempts=tuple(attempts),
                    response_body=response.body,
                    response_headers=response.headers,
                    fetched_at=fetched_at,
                    status_code=response.status_code,
                )

            try:
                feed = parse_atom_feed(response.body)
            except ArxivAPIError as error:
                attempts.append(
                    SearchAttempt(
                        attempt=attempt_number,
                        outcome="api_error",
                        status_code=response.status_code,
                        started_at=started_at,
                        finished_at=fetched_at,
                        error_type=type(error).__name__,
                        error_message=str(error),
                    )
                )
                raise ArxivSearchError(
                    kind="api_error",
                    message=str(error),
                    request=request,
                    attempts=tuple(attempts),
                    response_body=response.body,
                    response_headers=response.headers,
                    fetched_at=fetched_at,
                    status_code=response.status_code,
                ) from error
            except InvalidArxivFeed as error:
                attempts.append(
                    SearchAttempt(
                        attempt=attempt_number,
                        outcome="invalid_feed",
                        status_code=response.status_code,
                        started_at=started_at,
                        finished_at=fetched_at,
                        error_type=type(error).__name__,
                        error_message=str(error),
                    )
                )
                raise ArxivSearchError(
                    kind="invalid_feed",
                    message=str(error),
                    request=request,
                    attempts=tuple(attempts),
                    response_body=response.body,
                    response_headers=response.headers,
                    fetched_at=fetched_at,
                    status_code=response.status_code,
                ) from error

            attempts.append(
                SearchAttempt(
                    attempt=attempt_number,
                    outcome="success",
                    status_code=response.status_code,
                    started_at=started_at,
                    finished_at=fetched_at,
                    error_type=None,
                    error_message=None,
                )
            )
            return ArxivFetchResult(
                request=request,
                response_body=response.body,
                response_headers=response.headers,
                fetched_at=fetched_at,
                attempts=tuple(attempts),
                feed=feed,
            )

        raise RuntimeError("Unreachable retry state.")

    def _backoff(self, attempt_number: int) -> float:
        return self.minimum_interval_seconds * (2 ** (attempt_number - 1))

    def _wait_for_request_slot(self, retry_not_before: float | None = None) -> None:
        now = self.monotonic()
        earliest = retry_not_before or now
        if self._last_request_started is not None:
            earliest = max(
                earliest,
                self._last_request_started + self.minimum_interval_seconds,
            )
        remaining = earliest - now
        if remaining > 0:
            self.sleeper(remaining)
            now = self.monotonic()
        self._last_request_started = now


def response_cooldown_seconds(response: TransportResponse, now: datetime) -> float | None:
    """Respect Retry-After; an unspecified 429 gets a conservative local pause."""
    delay = retry_after_seconds(response.headers, now)
    if delay is None and response.status_code == 429:
        return DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS
    return delay


def retry_after_seconds(
    headers: tuple[tuple[str, str], ...], now: datetime
) -> float | None:
    """Parse a Retry-After delay from either seconds or an HTTP date."""
    for name, raw in headers:
        if name.casefold() != "retry-after":
            continue
        value = raw.strip()
        if value.isascii() and value.isdigit():
            significant = value.lstrip("0") or "0"
            if len(significant) > 12:
                return MAX_RECORDED_RETRY_AFTER_SECONDS
            return min(float(int(significant)), MAX_RECORDED_RETRY_AFTER_SECONDS)
        try:
            retry_at = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            return None
        seconds = max(0.0, (retry_at - now).total_seconds())
        return min(seconds, MAX_RECORDED_RETRY_AFTER_SECONDS)
    return None


def _format_time(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("Clock values must include a timezone.")
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
