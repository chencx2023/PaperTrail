"""Read-only preflight for the exact native index an answer may search."""

from dataclasses import dataclass
from pathlib import Path

from paperqa import Docs, Settings
from paperqa.agents.search import get_directory_index
from sqlalchemy.engine import Engine

from backend.app.collections.indexing import get_index_artifact, settings_for_artifact
from backend.app.collections.service import get_collection_revision


LOCAL_ANSWER_TOOLS = ("paper_search", "gather_evidence", "gen_answer", "complete")


@dataclass(frozen=True, slots=True)
class AllowedAnswerDocument:
    document_id: str
    versioned_id: str
    sha256: str


@dataclass(frozen=True, slots=True)
class AnswerScope:
    revision_id: str
    index_artifact_id: str
    documents: tuple[AllowedAnswerDocument, ...]
    settings: Settings


async def prepare_answer_scope(
    engine: Engine,
    revision_id: str,
    index_artifact_id: str,
    *,
    data_root: Path,
    settings: Settings,
) -> AnswerScope:
    """Reject a wrong or changed index before any answer model is called."""

    revision = get_collection_revision(engine, revision_id)
    if revision is None or not revision.members:
        raise ValueError("A nonempty confirmed revision is required for answering.")
    artifact = get_index_artifact(engine, revision_id, settings=settings)
    if artifact is None or artifact.id != index_artifact_id or artifact.status != "ready":
        raise ValueError("The requested ready index does not belong to this revision and configuration.")
    configured = settings_for_artifact(settings, artifact, data_root=data_root)
    configured.agent.agent_type = "ToolSelector"
    configured.agent.tool_names = list(LOCAL_ANSWER_TOOLS)
    configured.agent.search_count = min(3, len(revision.members))

    expected = {member.document_id + ".pdf": member for member in revision.members}
    index = await get_directory_index(configured, build=False)
    if set(await index.index_files) != set(expected) or await index.count != len(expected):
        raise ValueError("The native index contains documents outside the confirmed revision.")
    for file_key, member in expected.items():
        saved = await index.get_saved_object(file_key)
        if (
            not isinstance(saved, Docs)
            or set(saved.docs) != {member.versioned_id}
            or not saved.texts
            or any(str(text.doc.dockey) != member.versioned_id for text in saved.texts)
        ):
            raise ValueError("The native index document identity differs from the confirmed revision.")

    return AnswerScope(
        revision_id=revision.id,
        index_artifact_id=artifact.id,
        documents=tuple(
            AllowedAnswerDocument(member.document_id, member.versioned_id, member.sha256)
            for member in revision.members
        ),
        settings=configured,
    )
