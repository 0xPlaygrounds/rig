#!/usr/bin/env python3
"""Tests for SQLite vector fixture conversion tools."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


FIXTURE_DIR = Path(__file__).resolve().parent


class FixtureToolTests(unittest.TestCase):
    def test_retrieval_converter_writes_relevance_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            corpus = tmp_path / "corpus.jsonl"
            queries = tmp_path / "queries.jsonl"
            qrels = tmp_path / "qrels.tsv"
            document_embeddings = tmp_path / "document_embeddings.jsonl"
            query_embeddings = tmp_path / "query_embeddings.jsonl"
            output = tmp_path / "fixture.json"

            corpus.write_text(
                "\n".join(
                    [
                        json.dumps({"_id": "doc-a", "title": "A", "text": "alpha"}),
                        json.dumps({"_id": "doc-b", "title": "B", "text": "beta"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            queries.write_text(
                json.dumps({"_id": "query-a", "text": "alpha?"}) + "\n",
                encoding="utf-8",
            )
            qrels.write_text("query-a 0 doc-a 1\nquery-a 0 doc-b 0\n", encoding="utf-8")
            document_embeddings.write_text(
                "\n".join(
                    [
                        json.dumps({"id": "doc-a", "embedding": [1.0, 0.0]}),
                        json.dumps({"id": "doc-b", "embedding": [0.0, 1.0]}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            query_embeddings.write_text(
                json.dumps({"id": "query-a", "embedding": [1.0, 0.0]}) + "\n",
                encoding="utf-8",
            )

            subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(FIXTURE_DIR / "retrieval_to_fixture.py"),
                    "--corpus",
                    str(corpus),
                    "--queries",
                    str(queries),
                    "--qrels",
                    str(qrels),
                    "--document-embeddings",
                    str(document_embeddings),
                    "--query-embeddings",
                    str(query_embeddings),
                    "--output",
                    str(output),
                    "--k",
                    "1",
                    "--min-recall",
                    "1.0",
                ],
                check=True,
            )

            fixture = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(fixture["metric"], "cosine")
            self.assertEqual(fixture["dimensions"], 2)
            self.assertEqual([query["id"] for query in fixture["queries"]], ["query-a"])
            self.assertEqual(fixture["queries"][0]["relevant_ids"], ["doc-a"])
            self.assertEqual(fixture["queries"][0]["min_recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
