import asyncio
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import httpx

if importlib.util.find_spec("backend.app.search.screening"):
    from backend.app.search import screening
else:
    screening = None

CONFIG = {"DEEPSEEK_MODEL": "test-model", "DEEPSEEK_BASE_URL": "https://example.test", "DEEPSEEK_API_KEY": "secret-never-saved"}


def saved_search():
    return {"search_id": "a" * 32, "status": "succeeded", "question": "coding agent 的发展潜力如何？",
            "query_text": "coding agent", "snapshot_sha256": "b" * 64, "candidates": [
                {"versioned_id": "2501.00001v2", "title": "Repository agent evaluation", "abstract": "We evaluate coding agents on repository tasks and measure failure modes.", "display_rank": 1},
                {"versioned_id": "2501.00002v1", "title": "Advertising agents", "abstract": "We optimize advertising auctions with bidding agents.", "display_rank": 2}]}


def model_output(search=None):
    search = search or saved_search()
    return {"interpretation": "考察编码代理的实际能力、局限与应用条件。", "criteria": ["研究对象是编码代理", "提供能力或局限的具体依据"],
            "papers": [{"paper_id": p["versioned_id"], "label": "high" if i == 0 else "low",
                        "reason": "评测实际任务并分析失败。" if i == 0 else "研究竞价代理，研究对象不同。",
                        "evidence_quote": " ".join(p["abstract"].split()[:8])}
                       for i, p in enumerate(search["candidates"])]}


def span_output(search=None):
    search = search or saved_search()
    from backend.app.search.screening_spans import abstract_spans
    value = model_output(search)
    value['papers'] = [{k: row[k] for k in ('paper_id', 'label', 'reason')} |
        {'evidence_span_id': abstract_spans(p['abstract'])[0]['id'] if p['abstract'] else None}
        for row, p in zip(value['papers'], search['candidates'], strict=True)]
    return value


class AbstractScreeningTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.assertIsNotNone(screening, "摘要筛选模块尚未实现")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.calls = []

    async def generate(self, output=None, search=None, **kwargs):
        search = search or saved_search()
        def upstream(request):
            self.calls.append(json.loads(request.content))
            return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(span_output(search) if output is None else output, ensure_ascii=False)}}],
                                           "usage": {"prompt_tokens": 90, "completion_tokens": 60, "total_tokens": 150}})
        async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as client:
            return await screening.generate_screening(search, self.root, config=CONFIG, client=client, **kwargs)

    async def test_original_question_and_all_abstracts_are_sent_and_success_is_reused(self):
        original = saved_search()
        before = deepcopy(original)
        result = await self.generate(search=original)
        self.assertEqual(original, before)
        user = json.loads(self.calls[0]["messages"][1]["content"])
        self.assertEqual(user["question"], original["question"])
        self.assertEqual([''.join(span['text'] for span in p['abstract_spans']) for p in user['papers']], [p['abstract'] for p in original['candidates']])
        self.assertEqual(result["snapshot_sha256"], original["snapshot_sha256"])
        self.assertEqual(result["usage"]["total_tokens"], 150)
        self.assertIsNone(result["cost_usd"])
        self.assertFalse(result["cache_hit"])
        repeated = await self.generate(search=original)
        self.assertTrue(repeated["cache_hit"])
        self.assertEqual(repeated["screening_id"], result["screening_id"])
        self.assertEqual(len(self.calls), 1)
        self.assertNotIn(CONFIG["DEEPSEEK_API_KEY"], "".join(p.read_text(encoding="utf-8") for p in self.root.rglob("*.json")))

    async def test_missing_duplicate_unknown_ids_and_fabricated_quotes_are_rejected(self):
        cases = []
        for change in ("missing", "duplicate", "unknown", "fake-quote", "empty-high-quote", "bad-label"):
            value = model_output()
            if change == "missing": value["papers"].pop()
            elif change == "duplicate": value["papers"][1]["paper_id"] = value["papers"][0]["paper_id"]
            elif change == "unknown": value["papers"][1]["paper_id"] = "invented"
            elif change == "fake-quote": value["papers"][0]["evidence_quote"] = "Agents already replace all developers."
            elif change == "empty-high-quote": value["papers"][0]["evidence_quote"] = ""
            elif change == "bad-label": value["papers"][0]["label"] = "99% confident"
            cases.append((change, value))
        for name, value in cases:
            with self.subTest(case=name), self.assertRaises(screening.ScreeningError):
                screening._validate_output(value, screening._input(saved_search())[0])
        self.assertIsNone(screening.read_screening(saved_search(), self.root))
        self.assertEqual(self.calls, [], '旧格式严格验证只读，不发起模型修复')

    async def test_changed_question_abstract_or_snapshot_never_reuses_old_result(self):
        result = await self.generate()
        for field in ("question", "snapshot", "abstract"):
            changed = saved_search()
            if field == "question": changed["question"] = "代码评审代理有什么局限？"
            elif field == "snapshot": changed["snapshot_sha256"] = "c" * 64
            else: changed["candidates"][1]["abstract"] += " More detail."
            self.assertIsNone(screening.read_screening(changed, self.root))
        self.assertEqual(screening.read_screening(saved_search(), self.root)["screening_id"], result["screening_id"])

    async def test_verified_quote_over_style_target_is_kept_but_hard_length_limit_is_enforced(self):
        search = saved_search()
        quote = "the agent reliably completes simple tasks but exhibits characteristic failures on complex tasks, suggesting that bridging domain logic and code execution is a key bottleneck to generalizability"
        self.assertEqual(len(quote.split()), 27)
        search["candidates"][0]["abstract"] = quote + "."
        output = model_output(search)
        output["papers"][0]["evidence_quote"] = quote
        result = screening._validate_output(output, screening._input(search)[0])
        self.assertEqual(result["papers"][0]["evidence_quote"], quote)

        search["candidates"][0]["abstract"] = "a" * 401
        output["papers"][0]["evidence_quote"] = search["candidates"][0]["abstract"]
        with self.assertRaises(screening.ScreeningError):
            screening._validate_output(output, screening._input(search)[0])

    async def test_read_only_lookup_creates_nothing_and_corrupt_result_is_not_regenerated(self):
        self.assertIsNone(screening.read_screening(saved_search(), self.root))
        self.assertEqual(list(self.root.iterdir()), [])
        await self.generate()
        path = next((self.root / "abstract-screenings").glob("*.json"))
        path.write_text("not-json", encoding="utf-8")
        with self.assertRaises(screening.ScreeningError):
            screening.read_screening(saved_search(), self.root)
        with self.assertRaises(screening.ScreeningError):
            await self.generate()
        self.assertEqual(len(self.calls), 1)

    async def test_no_high_relevance_and_empty_abstract_are_valid_without_filling_a_quota(self):
        search = saved_search()
        search["candidates"][1]["abstract"] = ""
        value = span_output(search)
        value["papers"][0]["label"] = "low"
        value["papers"][1].update(label="insufficient", evidence_span_id=None, reason="摘要为空，无法判断。")
        result = await self.generate(value, search)
        self.assertEqual([p["label"] for p in result["papers"]], ["low", "insufficient"])

    async def test_busy_lock_prevents_duplicate_request(self):
        from backend.app.file_lock import try_file_lock
        with try_file_lock(self.root / "abstract-screenings/request.lock"):
            with self.assertRaisesRegex(screening.ScreeningError, "正在"):
                await self.generate()
        self.assertEqual(self.calls, [])

    async def test_timeout_and_provider_error_do_not_retry_or_leak_secrets(self):
        calls = []
        async def slow(request):
            calls.append(request)
            await asyncio.sleep(1)
            return httpx.Response(200, json={})
        async with httpx.AsyncClient(transport=httpx.MockTransport(slow)) as client:
            with self.assertRaisesRegex(screening.ScreeningError, "超时"):
                await screening.generate_screening(saved_search(), self.root, config=CONFIG, client=client, timeout_seconds=.01)
        self.assertEqual(len(calls), 1)
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(401, text=CONFIG["DEEPSEEK_API_KEY"]))) as client:
            with self.assertRaises(screening.ScreeningError) as caught:
                await screening.generate_screening(saved_search(), self.root, config=CONFIG, client=client)
        self.assertNotIn(CONFIG["DEEPSEEK_API_KEY"], str(caught.exception))
        self.assertIsNone(screening.read_screening(saved_search(), self.root))

    async def test_over_limit_not_ready_or_invalid_identity_never_calls_model(self):
        for kind in ("limit", "pending", "identity", "input-length"):
            search = saved_search()
            if kind == "limit": search["candidates"] *= 16
            elif kind == "pending": search["status"] = "queued"
            elif kind == "identity": search["search_id"] = "../unsafe"
            else: search["candidates"][0]["abstract"] = "a" * 200_000
            with self.subTest(kind=kind), self.assertRaises(screening.ScreeningError):
                await self.generate(search=search)
        self.assertEqual(self.calls, [])

    async def test_malformed_choice_returns_safe_failure_and_records_unknown_usage(self):
        for choice in (None, "invalid-choice"):
            response = {"choices": [choice], "usage": "invalid-usage"}
            async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=response))) as client:
                try:
                    with self.assertRaises(screening.ScreeningError):
                        await screening.generate_screening(saved_search(), self.root, config=CONFIG, client=client)
                except AttributeError:
                    self.fail("异常模型格式泄漏了内部 AttributeError")
        failures = list((self.root / "abstract-screenings/attempts").glob("*.json"))
        self.assertEqual(len(failures), 2)
        self.assertTrue(all(json.loads(p.read_text(encoding="utf-8"))["usage"] == {} for p in failures))

    async def test_valid_output_with_malformed_usage_saves_result_with_unknown_tokens(self):
        response = {"choices": [{"message": {"content": json.dumps(span_output())}}], "usage": ["invalid-usage"]}
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=response))) as client:
            try:
                result = await screening.generate_screening(saved_search(), self.root, config=CONFIG, client=client)
            except AttributeError:
                self.fail("异常用量字段使有效筛选结果未能保存")
        self.assertEqual(result["usage"], {})
        self.assertEqual(screening.read_screening(saved_search(), self.root)["screening_id"], result["screening_id"])


if __name__ == "__main__":
    unittest.main()
