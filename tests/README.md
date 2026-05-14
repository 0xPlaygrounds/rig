# Test Suites

Rig's root crate uses integration test targets under `tests/`.

- `tests/<provider>.rs` are provider-specific test targets.
- `tests/providers/<provider>/cassette/` contains provider tests backed by committed HTTP cassettes.
- `tests/providers/<provider>/live/` contains provider tests that still require a real service.
- `tests/integrations.rs` is the vector-store and external-service integration target.
- `tests/core.rs` contains provider-agnostic core behavior tests.

Most provider tests are ignored live tests unless they have been migrated to cassettes.

## Core Tests

Run provider-agnostic core tests with:

```bash
cargo test -p rig --test core
```

Run all default non-ignored tests for the root crate with:

```bash
cargo test -p rig
```

Run the same checks with all root crate features enabled:

```bash
cargo test -p rig --all-features
```

## Cassette Provider Tests

Cassette tests replay committed HTTP interactions by default and do not require provider API
keys. Cassette files live under `tests/cassettes/<provider>/...`.

Replay the migrated provider suites with:

```bash
cargo test -p rig --all-features --test openai openai::cassette -- --nocapture --test-threads=1
cargo test -p rig --all-features --test anthropic anthropic::cassette -- --nocapture --test-threads=1
cargo test -p rig --all-features --test gemini gemini::cassette -- --nocapture --test-threads=1
```

Record mode requires the relevant provider API key in the environment and overwrites existing
cassette files:

```bash
RIG_PROVIDER_TEST_MODE=record \
cargo test -p rig --all-features --test openai openai::cassette -- --nocapture --test-threads=1
```

```bash
RIG_PROVIDER_TEST_MODE=record \
cargo test -p rig --all-features --test anthropic anthropic::cassette -- --nocapture --test-threads=1
```

```bash
RIG_PROVIDER_TEST_MODE=record \
cargo test -p rig --all-features --test gemini gemini::cassette -- --nocapture --test-threads=1
```

Run one cassette test by passing a test-name substring:

```bash
cargo test -p rig --all-features --test gemini \
  streaming_tools_smoke \
  -- --nocapture --test-threads=1
```

The test filter after `--test <target>` is a substring match. Use the full module path only when
the shorter test name is ambiguous.

## Cassette Safety

After recording or reviewing cassette changes, run the provider safety checks:

```bash
cargo test -p rig --test openai cassette_safety -- --nocapture
cargo test -p rig --test anthropic cassette_safety -- --nocapture
cargo test -p rig --test gemini cassette_safety -- --nocapture
```

Review cassette diffs for:

- no API keys, bearer tokens, cookies, or provider account identifiers;
- expected request paths, methods, and bodies;
- expected provider responses for the scenario;
- no unrelated cassette churn.

## Live Provider Tests

Live provider tests use real provider APIs, local model servers, or account credentials. They are
ignored by default unless a test file says otherwise.

Run ignored tests for one provider target with:

```bash
cargo test -p rig --all-features --test openrouter -- --ignored --nocapture --test-threads=1
```

Run one ignored provider test with:

```bash
cargo test -p rig --all-features --test openai \
  responses_document_file_id_roundtrip_live \
  -- --ignored --nocapture --test-threads=1
```

Use the provider-specific environment variables named in the ignored test reason or provider
module, such as `OPENROUTER_API_KEY`, `MISTRAL_API_KEY`, `GROQ_API_KEY`, `XAI_API_KEY`,
`HUGGINGFACE_API_KEY`, or local services such as Ollama, llamafile, and llama.cpp.

## Integration Tests

External-service integration tests are collected under the `integrations` target and are gated by
feature flags.

Run all enabled non-ignored integration tests with:

```bash
cargo test -p rig --all-features --test integrations
```

Run one feature-gated integration group with:

```bash
cargo test -p rig --features qdrant --test integrations qdrant -- --nocapture
cargo test -p rig --features mongodb --test integrations mongodb -- --nocapture
cargo test -p rig --features sqlite --test integrations sqlite -- --nocapture
```

Some integration tests start Docker containers through `testcontainers`; Docker must be running.
Other integrations are ignored because they need external credentials or pre-provisioned services.
Run ignored integration tests explicitly:

```bash
cargo test -p rig --features bedrock --test integrations bedrock -- --ignored --nocapture --test-threads=1
cargo test -p rig --features vectorize --test integrations vectorize -- --ignored --nocapture --test-threads=1
```

Check each integration module for required environment variables. For example, Vectorize requires
`VECTORIZE_INDEX_NAME`, and Bedrock tests require AWS credentials plus access to the configured
Bedrock models.
