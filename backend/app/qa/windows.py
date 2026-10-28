"""Join a verified overlapping successor for generation, without changing the index."""

import hashlib

WINDOW_POLICY = {"version": "exact-overlap-successor-v1", "maximum_continuation_characters": 700,
                 "successors_per_original": 1, "ambiguous_successor": "skip",
                 "relevance": "inherited_from_original_not_independently_LLM_scored"}
REPAIR_WINDOW_POLICY = {**WINDOW_POLICY, "version":"exact-overlap-neighbors-v2",
    "maximum_preceding_characters":700, "predecessors_per_original":1,
    "ambiguous_predecessor":"skip"}


def _hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def expand_original_windows(contexts, native_texts, *, overlap, chunk_chars, cutoff,
                            include_predecessor=False):
    """Return new Context/Text copies plus exact source offsets; no model or mutation."""
    if not isinstance(overlap, int) or not 0 < overlap < chunk_chars <= 3000:
        return list(contexts), []
    native = list(native_texts)
    if include_predecessor:
        return _expand_neighbors(contexts, native, overlap=overlap, chunk_chars=chunk_chars, cutoff=cutoff)
    output, records = [], []
    for context in contexts:
        original = context.text.text or ""
        key = str(context.text.doc.dockey)
        if context.score < cutoff or len(original) != chunk_chars:
            output.append(context)
            continue
        # Require a real original chunk in the same loaded fixed document.
        originals = [text for text in native if str(text.doc.dockey) == key and text.text == original]
        successors = [text for text in native if str(text.doc.dockey) == key
                      and len(text.text or "") > overlap and text.text != original
                      and text.text[:overlap] == original[-overlap:]]
        unique = {(text.name, _hash(text.text)): text for text in successors}
        ordered = [text for text in native if str(text.doc.dockey) == key]
        anchors = [i for i, text in enumerate(ordered) if text.text == original and text.name == context.text.name]
        if not originals or len(unique) != 1 or len(anchors) != 1 or anchors[0] + 1 >= len(ordered):
            output.append(context)
            continue
        successor = next(iter(unique.values()))
        if ordered[anchors[0] + 1].name != successor.name or ordered[anchors[0] + 1].text != successor.text:
            output.append(context)
            continue
        end = min(len(successor.text), overlap + WINDOW_POLICY["maximum_continuation_characters"])
        window = original + successor.text[overlap:end]
        identifier = "pqac-w" + _hash(context.id + key + window)[:16]
        text_copy = context.text.model_copy(update={"text": window, "embedding": None,
            "name": f"{context.text.name} + continuation {successor.name}"})
        output.append(context.model_copy(update={"id": identifier, "text": text_copy}))
        records.append({"original_context_id": context.id, "window_context_id": identifier,
            "dockey": key, "original_native_locator": context.text.name,
            "successor_native_locator": successor.name, "overlap_characters": overlap,
            "original_native_document_position": anchors[0], "successor_native_document_position": anchors[0] + 1,
            "original_sha256": _hash(original), "successor_sha256": _hash(successor.text),
            "original_span": [0, len(original)], "successor_span": [overlap, end],
            "window_sha256": _hash(window), "score_basis": WINDOW_POLICY["relevance"]})
    return output, records


def _expand_neighbors(contexts, native, *, overlap, chunk_chars, cutoff):
    """One exact, uniquely adjacent chunk each way; never guess a missing neighbor."""
    output, records = [], []
    for context in contexts:
        original, key = context.text.text or "", str(context.text.doc.dockey)
        ordered = [t for t in native if str(t.doc.dockey) == key]
        anchors = [i for i,t in enumerate(ordered) if t.name == context.text.name and t.text == original]
        if context.score < cutoff or not overlap < len(original) <= chunk_chars or len(anchors) != 1:
            output.append(context); continue
        position = anchors[0]
        prefix = suffix = ""
        record = {"original_context_id":context.id, "dockey":key,
            "original_native_locator":context.text.name, "original_native_document_position":position,
            "original_sha256":_hash(original), "original_span":[0,len(original)],
            "overlap_characters":overlap, "score_basis":REPAIR_WINDOW_POLICY["relevance"]}
        for direction in (-1, 1):
            candidates = [t for t in ordered if t.text != original and len(t.text or "") > overlap
                and (t.text[-overlap:] == original[:overlap] if direction < 0
                     else original[-overlap:] == t.text[:overlap])]
            unique = {(t.name,_hash(t.text)):t for t in candidates}
            adjacent = position + direction
            if len(unique) != 1 or not 0 <= adjacent < len(ordered):
                continue
            neighbor = next(iter(unique.values()))
            if ordered[adjacent].name != neighbor.name or ordered[adjacent].text != neighbor.text:
                continue
            if direction < 0:
                end = len(neighbor.text) - overlap
                start = max(0,end - REPAIR_WINDOW_POLICY["maximum_preceding_characters"])
                prefix = neighbor.text[start:end]
                label = "predecessor"
            else:
                start = overlap
                end = min(len(neighbor.text),start + REPAIR_WINDOW_POLICY["maximum_continuation_characters"])
                suffix = neighbor.text[start:end]
                label = "successor"
            record.update({label+"_native_locator":neighbor.name, label+"_native_document_position":adjacent,
                label+"_sha256":_hash(neighbor.text), label+"_span":[start,end]})
        if not prefix and not suffix:
            output.append(context); continue
        window = prefix + original + suffix
        identifier = "pqac-w" + _hash(context.id + key + window)[:16]
        text_copy = context.text.model_copy(update={"text":window,"embedding":None,
            "name":context.text.name+" + verified adjacent context"})
        output.append(context.model_copy(update={"id":identifier,"text":text_copy}))
        records.append({**record,"window_context_id":identifier,"window_sha256":_hash(window)})
    return output, records
