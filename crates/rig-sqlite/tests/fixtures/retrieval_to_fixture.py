#!/usr/bin/env python3
"""Convert BEIR/MTEB-style retrieval exports into a Rig SQLite vector fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--qrels", required=True, type=Path)
    parser.add_argument("--document-embeddings", required=True, type=Path)
    parser.add_argument("--query-embeddings", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metric", choices=["cosine", "l2", "l1"], default="cosine")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--min-recall", type=float)
    parser.add_argument("--category", default="retrieval")
    parser.add_argument("--drop-missing-relevance", action="store_true")
    return parser.parse_args()


def record_id(record: dict) -> str:
    value = record.get("_id", record.get("id"))
    if value is None:
        raise ValueError(f"record has no id: {record}")
    return str(value)


def load_jsonl_by_id(path: Path) -> tuple[dict[str, dict], list[str]]:
    records: dict[str, dict] = {}
    order: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            item_id = record_id(record)
            records[item_id] = record
            order.append(item_id)
    return records, order


def load_embeddings(path: Path) -> dict[str, list[float]]:
    embeddings: dict[str, list[float]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            item_id = record_id(record)
            values = record.get("embedding", record.get("vector"))
            if values is None:
                raise ValueError(f"embedding record `{item_id}` has no embedding/vector")
            embeddings[item_id] = [float(value) for value in values]
    return embeddings


def load_qrels(path: Path) -> dict[str, list[str]]:
    qrels: dict[str, list[str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            parts = line.strip().split()
            if parts[0].lower() in {"query-id", "query_id", "qid"}:
                continue
            if len(parts) >= 4 and parts[1] == "0":
                query_id, document_id, score = parts[0], parts[2], parts[3]
            elif len(parts) >= 3:
                query_id, document_id, score = parts[0], parts[1], parts[2]
            else:
                raise ValueError(f"invalid qrels row: {line.rstrip()}")

            if float(score) > 0.0:
                qrels.setdefault(query_id, []).append(document_id)
    return qrels


def text_for_document(record: dict) -> str:
    title = str(record.get("title", "")).strip()
    text = str(record.get("text", "")).strip()
    if title and text:
        return f"{title}\n{text}"
    return title or text


def main() -> None:
    args = parse_args()
    corpus, corpus_order = load_jsonl_by_id(args.corpus)
    queries, query_order = load_jsonl_by_id(args.queries)
    document_embeddings = load_embeddings(args.document_embeddings)
    query_embeddings = load_embeddings(args.query_embeddings)
    qrels = load_qrels(args.qrels)

    document_ids = [
        item_id
        for item_id in corpus_order
        if item_id in corpus and item_id in document_embeddings
    ]
    if args.max_documents is not None:
        document_ids = document_ids[: args.max_documents]
    included_documents = set(document_ids)

    documents = []
    for index, document_id in enumerate(document_ids):
        record = corpus[document_id]
        documents.append(
            {
                "id": document_id,
                "category": args.category,
                "priority": index,
                "rating": 0.0,
                "published": True,
                "title": text_for_document(record),
                "embedding": document_embeddings[document_id],
            }
        )

    fixture_queries = []
    for query_id in query_order:
        if args.max_queries is not None and len(fixture_queries) >= args.max_queries:
            break
        if query_id not in query_embeddings or query_id not in qrels:
            continue

        relevant_ids = [
            document_id for document_id in qrels[query_id] if document_id in included_documents
        ]
        missing_relevance = [
            document_id
            for document_id in qrels[query_id]
            if document_id not in included_documents
        ]
        if missing_relevance and not args.drop_missing_relevance:
            raise SystemExit(
                f"query `{query_id}` has relevant docs missing from fixture documents; "
                "increase --max-documents or pass --drop-missing-relevance"
            )
        if not relevant_ids:
            continue

        query = {
            "id": query_id,
            "text": str(queries[query_id].get("text", "")),
            "embedding": query_embeddings[query_id],
            "k": args.k,
            "relevant_ids": relevant_ids,
        }
        if args.min_recall is not None:
            query["min_recall"] = args.min_recall
        fixture_queries.append(query)

    dimensions = len(documents[0]["embedding"]) if documents else 0
    fixture = {
        "metric": args.metric,
        "dimensions": dimensions,
        "documents": documents,
        "queries": fixture_queries,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
