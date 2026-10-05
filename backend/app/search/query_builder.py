"""Bounded keyword groups; arXiv syntax is constructed here, never by a model."""

from dataclasses import dataclass
from typing import Any

from backend.app.search.service import InvalidSearchQuery, SearchQueryPlan, build_arxiv_query

PLAN_VERSION = 1
GROUP_QUERY_VERSION = 2
MAX_EXPANSIONS = 2


class InvalidQueryPlan(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class KeywordQuery:
    label: str
    groups: tuple[tuple[str, ...], ...]
    exclude: tuple[str, ...]

    def as_data(self):
        return {"label": self.label, "groups": [list(g) for g in self.groups], "exclude": list(self.exclude)}


@dataclass(frozen=True, slots=True)
class MultiQueryPlan:
    baseline_query_text: str
    expansions: tuple[KeywordQuery, ...]

    def as_data(self):
        return {"version": PLAN_VERSION, "baseline_query_text": self.baseline_query_text,
                "expansions": [query.as_data() for query in self.expansions]}


def _phrase(value: Any) -> str:
    if not isinstance(value, str) or len(value) > 100 or not 1 <= len(value.split()) <= 6:
        raise InvalidQueryPlan("每个英文表达限 1–6 个简单词、100 字符。")
    try:
        return build_arxiv_query(value).normalized_query_text
    except InvalidSearchQuery:
        raise InvalidQueryPlan("扩展词不能包含网址、字段、运算符或其他检索语法。") from None


def _identity(groups, exclude):
    return (tuple(sorted({tuple(sorted(p.casefold() for p in group)) for group in groups})),
            tuple(sorted(p.casefold() for p in exclude)))


def build_query_plan(baseline_query_text: str, expansions: Any) -> MultiQueryPlan:
    try:
        baseline = build_arxiv_query(baseline_query_text)
    except (InvalidSearchQuery, TypeError, AttributeError):
        raise InvalidQueryPlan("请先确认有效的单查询英文检索词。") from None
    if not isinstance(expansions, list) or not 1 <= len(expansions) <= MAX_EXPANSIONS:
        raise InvalidQueryPlan("原查询之外只能补充 1–2 组查询。")
    seen = {_identity([(term,) for term in baseline.query_terms], ())}
    queries = []
    for item in expansions:
        if not isinstance(item, dict) or set(item) != {"label", "groups", "exclude"}:
            raise InvalidQueryPlan("扩展查询格式无效，请保留标签、概念组和排除词。")
        label, raw_groups, raw_exclude = item["label"], item["groups"], item["exclude"]
        if not isinstance(label, str) or not label.strip() or len(label) > 80:
            raise InvalidQueryPlan("每组查询需要简短的用途说明。")
        if not isinstance(raw_groups, list) or not 1 <= len(raw_groups) <= 3:
            raise InvalidQueryPlan("每组查询限 1–3 个概念组。")
        groups = []
        for raw_group in raw_groups:
            if not isinstance(raw_group, list) or not 1 <= len(raw_group) <= 3:
                raise InvalidQueryPlan("每个概念组限 1–3 个同义表达。")
            group = tuple(_phrase(p) for p in raw_group)
            if len({p.casefold() for p in group}) != len(group):
                raise InvalidQueryPlan("同一概念组包含重复表达。")
            groups.append(group)
        if len({tuple(sorted(p.casefold() for p in g)) for g in groups}) != len(groups):
            raise InvalidQueryPlan("查询包含重复概念组。")
        if not isinstance(raw_exclude, list) or len(raw_exclude) > 3:
            raise InvalidQueryPlan("每组最多三个排除表达。")
        exclude = tuple(_phrase(p) for p in raw_exclude)
        if len({p.casefold() for p in exclude}) != len(exclude):
            raise InvalidQueryPlan("排除表达重复。")
        identity = _identity(groups, exclude)
        if identity in seen:
            raise InvalidQueryPlan("扩展查询与原查询或另一组重复，请修改后采用。")
        seen.add(identity)
        queries.append(KeywordQuery(label.strip(), tuple(groups), exclude))
    return MultiQueryPlan(baseline.normalized_query_text, tuple(queries))


def plan_from_data(value: Any) -> MultiQueryPlan:
    if (not isinstance(value, dict) or set(value) != {"version", "baseline_query_text", "expansions"}
            or type(value["version"]) is not int or value["version"] != PLAN_VERSION):
        raise InvalidQueryPlan("多查询计划的版本或格式无效，请重新采用。")
    return build_query_plan(value["baseline_query_text"], value["expansions"])


def _field_clause(phrase: str) -> str:
    literal = f'"{phrase}"' if " " in phrase else phrase
    return f"(ti:{literal} OR abs:{literal})"


def query_plans(plan: MultiQueryPlan) -> tuple[SearchQueryPlan, ...]:
    queries = [build_arxiv_query(plan.baseline_query_text)]
    for query in plan.expansions:
        clauses = []
        for group in query.groups:
            clauses.append(_field_clause(group[0]) if len(group) == 1 else
                           "(" + " OR ".join(_field_clause(p) for p in group) + ")")
        expression = " AND ".join(clauses)
        for phrase in query.exclude:
            expression += " ANDNOT " + _field_clause(phrase)
        queries.append(SearchQueryPlan(
            query_text=query.label, normalized_query_text=expression,
            query_terms=tuple(p for g in query.groups for p in g),
            search_query=expression, contract_version=GROUP_QUERY_VERSION,
        ))
    return tuple(queries)
