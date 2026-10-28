"""Offline checks of bounded Agent selection recovery; no saved experiments."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from backend.app.qa.recovery import bounded_selector, recovery_events


class ProviderError(Exception):
    def __init__(self, status, adapter_status=None):
        self.status_code = adapter_status or status
        self.response = SimpleNamespace(status_code=status)


class ProviderRecoveryTest(unittest.IsolatedAsyncioTestCase):
    async def test_transient_retry_once_per_run_preserves_selection_arguments(self):
        calls, events = [], []
        outcomes = iter((500, None, 503))

        async def selector(*args, **kwargs):
            calls.append((args, kwargs))
            status = next(outcomes)
            if status:
                raise ProviderError(status)
            return "selected"

        token = recovery_events.set(events)
        try:
            with patch("backend.app.qa.recovery.asyncio.sleep"):
                wrapped = bounded_selector(selector)
                self.assertEqual(await wrapped("history", tools=["tool"]), "selected")
                self.assertEqual(calls[0], calls[1])
                with self.assertRaises(ProviderError):
                    await wrapped("next", tools=["tool"])
        finally:
            recovery_events.reset(token)
        self.assertEqual(len(calls), 3)
        self.assertEqual([e["decision"] for e in events], ["retry", "recovered", "exhausted"])

    async def test_auth_balance_access_and_rate_limit_are_not_retried(self):
        for status in (401, 402, 403, 429):
            with self.subTest(status=status):
                calls, events = [], []

                async def selector():
                    calls.append(1)
                    raise ProviderError(status, adapter_status=500)

                token = recovery_events.set(events)
                try:
                    with self.assertRaises(ProviderError):
                        await bounded_selector(selector)()
                finally:
                    recovery_events.reset(token)
                self.assertEqual(calls, [1])
                self.assertEqual(events[0]["http_status"], status)
                self.assertEqual(events[0]["decision"], "not_retryable")

    async def test_non_http_failure_does_not_trigger_provider_retry(self):
        async def selector():
            raise ValueError("malformed tool response")

        with self.assertRaisesRegex(ValueError, "malformed"):
            await bounded_selector(selector)()


if __name__ == "__main__":
    unittest.main()
