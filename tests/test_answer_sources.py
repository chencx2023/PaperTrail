"""Answer source metadata comes from the fixed version and its verified search snapshot."""
import json
from pathlib import Path
import tempfile
import unittest

from sqlalchemy.orm import Session
from backend.app.db import open_database
from backend.app.models import CollectionRevisionDocument, DocumentVersion, Paper
from tests.test_collection_index import imported_revision
from tests.test_ingestion_selection import saved_search


class AnswerSourcesTest(unittest.IsolatedAsyncioTestCase):
    async def test_mutable_paper_metadata_cannot_replace_the_frozen_source(self):
        from backend.app.evidence.sources import frozen_document_sources
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name); data = root / 'data'; run = saved_search(data)
        engine = open_database(root / 'db.sqlite')
        self.addCleanup(engine.dispose)
        revision = await imported_revision(engine, data, run, '2401.00001v1')
        with Session(engine) as session:
            member = session.query(CollectionRevisionDocument).filter_by(revision_id=revision.id).one()
            doc = session.get(DocumentVersion, member.document_id)
            expected = json.loads(doc.authors_json)
            paper = session.get(Paper, doc.paper_id)
            paper.title = 'Mutable future title'; paper.authors_json = '["Wrong Author"]'; paper.abstract = 'Mutable future abstract'
            session.commit()
            allowed = [{'document_id': doc.id, 'versioned_id': member.versioned_id, 'sha256': member.sha256}]
            result = frozen_document_sources(session, allowed, data_root=data)[doc.id]
            self.assertEqual(result['title'], 'Memory for Agents: First Version')
            self.assertEqual(result['authors'], expected)
            self.assertEqual(result['year'], int(doc.published_at[:4]))
            self.assertEqual(result['abstract_status'], 'saved')
            self.assertNotIn('Mutable', result['abstract'])
            (run / 'response.atom').write_bytes(b'tampered')
            invalid = frozen_document_sources(session, allowed, data_root=data)[doc.id]
            self.assertIsNone(invalid['abstract'])
            self.assertEqual(invalid['abstract_status'], 'unverified')
            self.assertEqual(invalid['authors'], expected)
            allowed[0]['versioned_id'] = '2401.00001v3'
            self.assertEqual(frozen_document_sources(session, allowed, data_root=data), {})
