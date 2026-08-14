# Test Suites

Rig's root crate uses integration test targets under `tests/`.

- `tests/<provider>.rs` are provider-specific test targets.
- `tests/providers/<provider>/cassette/` contains provider tests backed by committed HTTP cassettes.
- `tests/providers/<provider>/live/` contains provider tests that still require a real service.
- `tests/integrations.rs` is the vector-store and external-service integration target.
- `tests/core.rs` contains provider-agnostic core behavior tests.

Most provider tests are ignored live tests unless they have been migrated to cassettes.

## Testing Doctrine

**Recorded provider traffic is the default evidence; provider APIs are the
ultimate judge of whether the code is correct.** Every genuine pre-existing
streaming bug found during the #2258 refactor was found by live recording,
not by the synthetic corpus — a corpus written alongside an abstraction
encodes the team's model of the wire and structurally cannot falsify it.

- **Cassette-first.** New provider behavior gets a cassette-backed test.
  A unit test still earns its place — for internal behavior that is
  definitory rather than observed — but each unit test of provider-facing
  behavior should say (in its doc comment) why it isn't, or can't be, a
  cassette test. Recording needs API keys and isn't trivial; writing the
  cassette test a contributor couldn't is core maintainer work.
- **Record first, derive the assertion, review the derivation.** Prefer
  assertions generated from a replay of real traffic and then reviewed over
  assertions hand-authored from documentation — hand-authored expectations
  encode the same model of the wire the code under test does.
- **Assert on the request boundary too.** A frozen cassette replays the
  provider's *responses*; it cannot by itself catch outbound drift in the
  requests the live code builds. The cassette harness matches each request
  body against the recorded one, so a request-shape regression fails as a
  404 mock miss — treat that as a first-class assertion, and when a change
  intentionally alters a request, update the recorded body deliberately and
  say so. (This is exactly what caught fabricated ids reaching request
  serializers in #2258.)
- **Never weaken a cassette to make it pass.** Update assertions and
  recordings to the new intended behavior, or re-record; a scrubbed value
  must redact real data, never invent data that was not recorded.

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
cargo test -p rig --all-features --test chatgpt chatgpt::cassette -- --nocapture --test-threads=1
cargo test -p rig --all-features --test bedrock bedrock::cassette -- --nocapture --test-threads=1
cargo test -p rig --all-features --test cohere cohere::cassette -- --nocapture --test-threads=1
cargo test -p rig --all-features --test doubleword doubleword::cassette -- --nocapture --test-threads=1
cargo test -p rig --all-features --test venice venice::cassette -- --nocapture --test-threads=1
```

Venice's text-to-speech scenario records through the direct recorder rather than
the httpmock proxy: the proxy exports bodies as strings, so a binary response
(raw audio) is exported with no body at all and replays as zero bytes. Its
transcription scenario stays on the proxy path, where the same limitation drops
the *request's* multipart body — a cassette that recorded no body still matches
a multipart request, and the multipart shape itself is pinned by unit tests
beside the provider.

Bedrock cassette replay does not require AWS credentials. Bedrock record mode uses the AWS
SDK credential provider chain and a direct SigV4-aware recorder, so it requires AWS credentials
with Bedrock model access in `us-east-1` and overwrites existing cassette files. The Bedrock
recorder buffers streaming/event-stream responses and stores non-UTF-8 cassette bodies as base64;
those opaque bodies are intended for replay fidelity, and safety checks also scan their decoded
bytes for credential-shaped material.

Record mode requires the relevant provider credentials in the environment and overwrites existing
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

```bash
RIG_PROVIDER_TEST_MODE=record \
cargo test -p rig --all-features --test cohere cohere::cassette -- --nocapture --test-threads=1
```

```bash
RIG_PROVIDER_TEST_MODE=record \
cargo test -p rig --all-features --test bedrock bedrock::cassette -- --nocapture --test-threads=1
```

```bash
RIG_PROVIDER_TEST_MODE=record \
cargo test -p rig --all-features --test doubleword doubleword::cassette -- --nocapture --test-threads=1
```

```bash
RIG_PROVIDER_TEST_MODE=record \
cargo test -p rig --all-features --test venice venice::cassette -- --nocapture --test-threads=1
```

```bash
CHATGPT_ACCESS_TOKEN=... CHATGPT_ACCOUNT_ID=... RIG_PROVIDER_TEST_MODE=record \
cargo test -p rig --all-features --test chatgpt chatgpt::cassette -- --nocapture --test-threads=1
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

Record mode scrubs and safety-checks cassette contents before writing fixtures.
The committed cassette safety tests enforce the same scrubbed form during normal
test runs.

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

## Local Artifact Model Tests

`rig-candle` has an ignored native model-contract suite. It is not an HTTP
cassette: it loads one pinned Qwen3 GGUF artifact and runs provider-neutral
completion, buffered/raw-streaming parity, parallel and sequential tools,
zero-argument and complex typed arguments, call/result history correlation,
result serialization, invalid-call recovery, hook rewrite chaining, turn-local
request patches, cancellation/max-turn diagnostics, extraction with usage,
tool-choice, protocol hygiene, and synthetic structured-output scenarios
through Rig's agent driver.

```bash
export RIG_CANDLE_TEST_MODEL_DIR="$PWD/crates/rig-candle/test-models/qwen3-4b-q4-k-m"
./crates/rig-candle/tests/download_qwen3.sh
cargo test --release -p rig-candle --test live_conformance \
  -- --ignored --nocapture --test-threads=1
```

The 2.33-GiB model is checksum-verified, cached in an ignored directory, and
loaded once per test binary. The measured ARM64 release run completed in 164.41
seconds; allow at least fifteen minutes for slower CPU hosts and more than twice
the checkpoint size during loading. Use serial execution to bound CPU and memory use.
See `crates/rig-candle/README.md` for revisions, hashes, measured performance,
and the boundary between model-contract and provider-transport tests.

Reusable scenarios and typed validators are exported from
`rig_core::test_utils`. A provider suite should call a complete model-driving
scenario when its cassette records the same prompt and tool definitions. When
wire-specific prompts, schemas, request parameters, or metadata must remain
local, the provider test should retain that transport setup and call the shared
validator on its public Rig result. Do not move authentication, HTTP body, SSE,
hosted-tool, remote-file, or provider-session assertions into the portable
module.

Universal scenarios require only the public completion/agent contract.
Optional capability scenarios—parallel model emission, structured reasoning,
provider-assigned IDs, native constrained decoding, and hosted tools—must be
selected explicitly. A provider that does not expose one optional capability
must not weaken the universal assertions or silently mark the scenario passed.

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
