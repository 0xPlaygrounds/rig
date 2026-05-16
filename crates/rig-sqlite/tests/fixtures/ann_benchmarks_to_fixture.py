#!/usr/bin/env python3
"""Convert an ANN-Benchmarks HDF5 dataset into a Rig SQLite vector fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("hdf5", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--metric", choices=["auto", "cosine", "l2", "l1"], default="auto")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--category", default=None)
    parser.add_argument(
        "--omit-expected-outside-documents",
        action="store_true",
        help="Omit expected_ids when a sliced fixture excludes ANN ground-truth neighbors.",
    )
    return parser.parse_args()


def vector(values) -> list[float]:
    return [float(value) for value in values]


def hdf5_metric(hdf5, override: str) -> str:
    if override != "auto":
        return override

    raw = hdf5.attrs.get("distance") or hdf5.attrs.get("type") or ""
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    name = str(raw).lower()

    if name in {"angular", "cosine"}:
        return "cosine"
    if name in {"euclidean", "l2"}:
        return "l2"
    if name in {"manhattan", "l1"}:
        return "l1"

    raise ValueError(
        "could not infer metric from HDF5 attrs; pass --metric cosine, l2, or l1"
    )


def main() -> None:
    try:
        import h5py
    except ImportError as error:
        raise SystemExit("install h5py to convert ANN-Benchmarks HDF5 files") from error

    args = parse_args()
    with h5py.File(args.hdf5, "r") as hdf5:
        train = hdf5["train"]
        test = hdf5["test"]
        neighbors = hdf5["neighbors"]
        metric = hdf5_metric(hdf5, args.metric)

        document_count = args.max_documents or train.shape[0]
        query_count = args.max_queries or test.shape[0]
        document_count = min(document_count, train.shape[0])
        query_count = min(query_count, test.shape[0])
        k = min(args.k, neighbors.shape[1])
        category = args.category or args.hdf5.stem

        documents = []
        for index in range(document_count):
            documents.append(
                {
                    "id": str(index),
                    "category": category,
                    "priority": index,
                    "rating": 0.0,
                    "published": True,
                    "title": f"{category} document {index}",
                    "embedding": vector(train[index]),
                }
            )

        queries = []
        for index in range(query_count):
            neighbor_ids = [int(neighbor) for neighbor in neighbors[index][:k]]
            outside = [neighbor for neighbor in neighbor_ids if neighbor >= document_count]
            if outside and not args.omit_expected_outside_documents:
                raise SystemExit(
                    "sliced fixture excludes ANN ground-truth neighbors; "
                    "increase --max-documents or pass --omit-expected-outside-documents"
                )

            query = {
                "id": str(index),
                "text": f"{category} query {index}",
                "embedding": vector(test[index]),
                "k": k,
            }
            if not outside:
                query["expected_ids"] = [str(neighbor) for neighbor in neighbor_ids]
            queries.append(query)

    fixture = {
        "metric": metric,
        "dimensions": len(documents[0]["embedding"]) if documents else 0,
        "documents": documents,
        "queries": queries,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
