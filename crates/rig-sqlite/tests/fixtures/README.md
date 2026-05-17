# SQLite Vector Fixtures

`external_vector_fixture_matches_ground_truth` reads a JSON fixture from
`RIG_SQLITE_VECTOR_FIXTURE` and validates SQLite vector search against an exact
Rust oracle. It also supports retrieval relevance assertions through
`relevant_ids` and `min_recall`.

Run the checked-in vector fixture:

```bash
RIG_SQLITE_VECTOR_FIXTURE=crates/rig-sqlite/tests/fixtures/vector_fixture.json \
  cargo test -p rig-sqlite external_vector_fixture_matches_ground_truth -- --ignored
```

Run the checked-in retrieval fixture:

```bash
RIG_SQLITE_VECTOR_FIXTURE=crates/rig-sqlite/tests/fixtures/retrieval_fixture.json \
  cargo test -p rig-sqlite external_vector_fixture_matches_ground_truth -- --ignored
```

Convert a BEIR or MTEB retrieval export with precomputed embeddings:

```bash
python3 crates/rig-sqlite/tests/fixtures/retrieval_to_fixture.py \
  --corpus /path/to/corpus.jsonl \
  --queries /path/to/queries.jsonl \
  --qrels /path/to/qrels/test.tsv \
  --document-embeddings /path/to/document_embeddings.jsonl \
  --query-embeddings /path/to/query_embeddings.jsonl \
  --output /tmp/retrieval_fixture.json \
  --k 10 --min-recall 0.8
```

Embedding JSONL records must contain `id` or `_id` plus `embedding` or `vector`.

Run the local validation suite:

```bash
bash crates/rig-sqlite/tests/fixtures/run_vector_validation.sh --skip-clippy --skip-integration
```
