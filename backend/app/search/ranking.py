"""Deterministic RRF on saved native lists, one vote per work per query."""

from dataclasses import dataclass, replace
from fractions import Fraction

from backend.app.search.arxiv import ArxivCandidate, ArxivRecord, ArxivVersionRef, ParsedArxivFeed

RRF_K = 60


@dataclass(frozen=True, slots=True)
class FusedRanking:
    candidates: tuple[ArxivCandidate, ...]
    records: tuple[ArxivRecord, ...]
    ranking: tuple[dict, ...]
    merge_order: tuple[str, ...]


def fuse_rankings(feeds: list[ParsedArxivFeed | None]) -> FusedRanking:
    scores, matches, representatives, records = {}, {}, {}, {}
    for query_index, feed in enumerate(feeds, 1):
        if feed is None:
            continue
        seen = set()
        for candidate in feed.candidates:
            work = candidate.arxiv_id
            if work in seen:
                continue
            seen.add(work)
            rank = len(seen)
            scores[work] = scores.get(work, Fraction()) + Fraction(1, RRF_K + rank)
            matches.setdefault(work, []).append({"query_index": query_index, "rank": rank,
                                                "source_rank": candidate.source_rank})
            existing = representatives.get(work)
            if existing is None or (candidate.version or 0) > (existing.version or 0):
                representatives[work] = candidate
        for record in feed.records:
            if record.arxiv_id in seen:
                records.setdefault(record.versioned_id, record)
    order = sorted(scores, key=lambda work: (-scores[work], work))
    candidates, ranking = [], []
    for rank, work in enumerate(order, 1):
        versions = tuple(ArxivVersionRef(r.versioned_id, r.version, r.source_rank)
                         for r in records.values() if r.arxiv_id == work)
        candidates.append(replace(representatives[work], display_rank=rank, versions=versions))
        ranking.append({"arxiv_id": work, "rank": rank, "score": float(scores[work]),
                        "matches": matches[work]})
    return FusedRanking(tuple(candidates), tuple(records.values()), tuple(ranking), tuple(scores))
