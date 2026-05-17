# SQLite Vector Fixtures

The checked-in fixture tests validate SQLite vector search against an exact Rust
oracle. They also support retrieval relevance assertions through `relevant_ids`
and `min_recall`.

Run the checked-in vector and retrieval fixtures:

```bash
cargo test -p rig-sqlite checked_in_
```

Run a generated external fixture:

```bash
RIG_SQLITE_VECTOR_FIXTURE=/tmp/retrieval_fixture.json \
  cargo test -p rig-sqlite external_vector_fixture_matches_ground_truth -- --ignored
```
