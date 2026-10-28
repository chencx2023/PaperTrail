"""One bounded provider retry for Agent selection, without replaying tools."""
import asyncio
from contextvars import ContextVar

recovery_events = ContextVar("qa_provider_recovery_events", default=None)
RECOVERY_POLICY = {"version":"agent-selection-one-transient-retry-v1",
    "maximum_retries_per_run":1, "http_statuses":[500,502,503,504], "delay_seconds":0.25}


def bounded_selector(selector):
    used, pending = 0, False

    def record(**event):
        events = recovery_events.get()
        if events is not None:
            events.append({"stage":"agent_selection", **event})

    async def call(*args, **kwargs):
        nonlocal used, pending
        while True:
            try:
                result = await selector(*args, **kwargs)
            except Exception as error:
                status = getattr(getattr(error,"response",None),"status_code",None)
                if not isinstance(status,int):
                    status = getattr(error,"status_code",None)
                if not isinstance(status,int) or isinstance(status,bool):
                    raise  # Let the native malformed-message policy handle its own errors.
                temporary = status in RECOVERY_POLICY["http_statuses"]
                retry = temporary and used < RECOVERY_POLICY["maximum_retries_per_run"]
                record(http_status=status, error_kind=type(error).__name__,
                    decision="retry" if retry else "exhausted" if temporary else "not_retryable",
                    retries_used=used)
                if not retry:
                    raise
                used += 1; pending = True
                await asyncio.sleep(RECOVERY_POLICY["delay_seconds"])
                continue
            if pending:
                record(decision="recovered", retries_used=used)
                pending = False
            return result

    return call
