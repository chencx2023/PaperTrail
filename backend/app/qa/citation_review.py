"""Expose actual citations before draft wording within the existing review call.

Source identity remains program-checked; semantic support remains a model
judgement, independently checked against originals during development review.
"""

SOURCE_BOUND_INSTRUCTIONS = """
Source-first review: each target starts with program-filled cited_original_units,
its ACTUAL current citations. Read these before the draft's original statement.
Start conditions_comparison with a short exact source excerpt, then compare the
literal claim, including EVERY extra modifier, numerical/mathematical operator,
actor and condition. Do not echo a draft modifier as something the source said.
A definition and an example/experimental setting have different scope: shared
notation or an equality does not establish an unstated selection procedure.
If the bound sources support only a narrower statement, replace/narrow it in
the existing correction. Preserve indispensable answer content. An unnecessary
unsupported modifier should be removed, not turned into a new requested gap.
The remaining original units are available for counterconstraints or explicit
source rebinding when needed, not silent support from an unrelated example.
For an incidental modifier not needed to answer the user's requested aspect,
require direct support in its CURRENT cited units. If absent, remove the modifier
with a replacement; do not search the rest of the paper for a matching term to
rescue it. An appendix variant, illustration or separate procedure does not
establish the main definition's unstated premise merely by sharing a symbol or
word. Necessary requested content may be rebound to explicit evidence, but keep
that evidence's variant and operating condition in the replacement itself.
Use only the existing four output fields. Keep comparisons and reasons concise;
do not build a separate audit, copy the whole outline, or repeat a paper survey
for every fact. All existing coverage, protected-answer and gap rules still apply.
"""


def source_first_payload(payload, units):
    by_id = {unit['id']: unit for unit in units}
    fields = ('id', 'context_id', 'text', 'source_scope', 'scope_basis')
    targets = []
    for row in payload['target_facts']:
        packet = [{key: by_id[sid][key] for key in fields} for sid in row['original']['sentence_ids']]
        targets.append({'fact_id': row['fact_id'], 'cited_original_units': packet,
            **{key: value for key, value in row.items() if key not in ('fact_id', 'cited_original_units')}})
    return {'question': payload['question'], 'target_facts': targets,
        **{key: value for key, value in payload.items()
            if key not in ('question', 'target_facts', 'program_filled_original_units')},
        'program_filled_original_units': payload['program_filled_original_units']}
