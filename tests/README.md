# Test Suites

Rig's root crate uses integration test targets under `tests/`.

- `tests/<provider>.rs` are provider-specific test targets.
- `tests/providers/<provider>/cassette/` contains provider tests backed by committed HTTP cassettes.
- `tests/providers/<provider>/live/` contains provider tests that still require a real service.
- `tests/integrations.rs` is the vector-store and external-service integration target.
- `tests/core.rs` contains provider-agnostic core behavior tests.
- `tests/ecs_consumer.rs` exercises the real headless ECS maintenance consumer;
  its [recording, golden and repair workflow](consumer/README.md) is also runnable
  through `cargo run -p rig --example ecs-consumer -- plan`.

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

## Prompt Cache Testing

Provider prompt caching is a **prefix match**: the cache key is derived from the exact request
bytes up to each breakpoint, so any change to an earlier block invalidates everything after it.
The failure that costs real money is therefore not "caching is off" — it is "caching silently
degraded", where a reordered map, a rewritten earlier turn, or a re-advertised tool set moves the
prefix and every request quietly misses while the counters stay non-zero.

Four layers guard this. Each catches something the others structurally cannot.

### Layer 0 — free, key-free, whole-corpus (`tests/cassette_cache_prefix.rs`)

Three checks over cassettes that already exist, costing no provider traffic:

- `recorded_conversations_do_not_move_their_cache_prefix` — for consecutive same-endpoint requests
  in one cassette, turn N-1's canonical blocks must be a prefix of turn N's. The rule lives in
  `tests/common/cache_prefix.rs` and is shared with the per-scenario harness so the two cannot
  drift. It compares cached *content*: `cache_control` markers are stripped first, because
  Anthropic's documented incremental-caching pattern moves the breakpoint forward every turn and
  that is correct behavior, not a prefix move.
- `every_provider_is_covered_by_the_prefix_check` — a per-provider census that makes the check
  **fail closed**. Every recorded request is classified modeled / non-conversational / unmodeled,
  and an unmodeled *conversational* endpoint is a finding rather than a silent skip.
- `*_request_serialization_is_deterministic` — serializes the same `CompletionRequest` eight times
  through each provider's real client and requires byte-identical output. This is the only check
  that can see unstable map or tool ordering: cassette replay compares key-sorted canonical JSON,
  and a `serde_json::Value` round-trip normalizes key order too, so a `HashMap` in a request body
  busts every real cache while all recorded evidence looks identical.
- `every_cassette_provider_has_a_cache_suite` — a provider with cassettes and no cache scenario
  fails unless it is in `NO_CACHE_SUITE` with a reason.

### Layer 1 — the shared harness (`tests/common/cache_conformance.rs`)

One deterministic three-turn probe — warm, byte-identical repeat, then append and repeat —
asserted against a per-provider `CacheSupport` descriptor, so adding a provider is a descriptor
rather than another copy of the assertion logic.

The assertion that carries the weight is `assert_hit_ratio`. `cached_input_tokens > 0` passes
just as happily when 200 of 40,000 prefix tokens are cached as when 39,800 are; the ratio is
taken against turn 1's billed prompt. **The denominator differs per provider and getting it
backwards makes the assertion vacuous**, so it is derived in `CacheAccounting` from each
provider's own usage mapping: Anthropic reports cache tokens *alongside* `input_tokens`, while
OpenAI, Gemini, Cohere, DeepSeek, Mistral, OpenRouter and every `openai::Usage` reuser report
them *inside* it.

Growth is asserted as a ratio, not a monotonic token count: providers cache in coarse blocks, so
the absolute figure drifts a few tokens as block boundaries re-align (Gemini was measured going
3,765 -> 3,760 across a 21-token append) while a genuine prefix move collapses it to zero.

### Layer 2 — per-provider cassettes (`tests/cassettes/<provider>/prompt_caching/`)

Recorded live, replayed key-free. Record one scenario at a time:

```bash
RIG_PROVIDER_TEST_MODE=record \
cargo test -p rig --all-features --test openai openai::cassette::prompt_caching \
  -- --exact --nocapture --test-threads=1
```

### Layer 3 — live economics (`live_cache_economics`, `#[ignore]`d)

A cassette pins what a provider did at record time; only a live run catches the provider changing
its cache semantics under us. Each cell prints a `LIVE-CACHE-ECONOMICS` row, so the economics
table can be regenerated rather than trusted as a transcription:

```bash
cargo test -p rig --all-features --test openai live_cache_economics \
  -- --exact --ignored --nocapture --test-threads=1
```

### Rules for recording cache fixtures

- **Never commit a cache cassette whose recorded turn 2 shows zero reads.** The Layer-1 assertions
  run identically in record mode, so a bad session fails instead of committing a fixture that pins
  a miss. That is the intended outcome — a recorded miss is worse than a failed recording.
- **Some providers need a warm-up pass.** Gemini's implicit cache only serves a prefix once an
  entry for *that exact prefix* exists, so on a cold run the grown turn-3 prefix reads zero. Run
  the scenario twice and keep the second recording. Mistral's cache is intermittent for a
  different reason (routing without cache affinity) and may need several attempts.
- **Padding must be deterministic and committed** — no nonce, no timestamp. A nonce guarantees a
  turn-1 miss, churns the cassette on every re-record, and breaks body matching. The org-pre-warm
  risk it would avoid is already tolerated by `assert_warms`.
- **Pad above the provider's documented minimum** (`min_cacheable_tokens` in the descriptor) or
  the API silently declines to cache. Where a provider's rate limit is tighter than the default
  probe, `CacheProbe::with_padding` shrinks it — Groq's 8,000 TPM tier needs this.
- **All three turns must record back-to-back in one test body**, because cache TTLs are minutes.
  The shared probe does this by construction.
- **Never mark a cache scenario `.unordered()`.** Ordered replay is what lets two byte-identical
  requests replay two different recorded responses, which is the only reason a miss-then-hit pair
  is replayable at all.

### What replay does and does not prove

Replay is not a tautology: the harness *matches request bodies*, so a rig change that perturbs the
outbound prefix fails as a replay miss in CI with no API key. That is what turns a recorded
cassette into a permanent cache regression test.

It has exactly two blind spots, both covered elsewhere rather than papered over: body matching
compares key-sorted canonical JSON, so map reordering is invisible to it (Layer 0's determinism
check exists for that), and a cassette cannot notice the provider changing its behavior after
record time (Layer 3 exists for that).

### Gemini: two caching features, and they behave differently

Gemini is the one provider in the matrix with **two** caching features, and the
suite covers both because they are not interchangeable.

**Implicit caching** is automatic and best-effort. Measured on
`gemini-2.5-flash` over an 18,497-token corpus: five consecutive turns reusing
that corpus read **zero** cached tokens (~92k tokens billed at full price), and
only a sixth request read 99.6%. It keys on a prefix the provider has already
seen, so a fresh conversation starts cold and there is no way to pre-warm it.

**Explicit caching** (`cachedContents`) uploads once and hands back a handle.
Same corpus, same day: **100.0% on turn one**, and 100.0% again from an
unrelated conversation. It bills storage per token-hour, so it pays when one
large fixed payload is reused enough to beat that — and it pays immediately
rather than after a warm-up.

Practical consequences for anyone touching these fixtures:

- **Run every implicit-cache scenario twice and keep the second recording.** A
  cold first pass records a turn-3 miss and fails, which is the intended
  outcome — a recorded miss is worse than a failed recording session.
- **Disable thinking** (`generationConfig.thinkingConfig.thinkingBudget: 0`) on
  every cell that is not specifically about thinking. Gemini 2.5 spends its
  output budget on thoughts first, so a small `max_tokens` yields a response
  with no message at all.
- **Explicit-cache cells create billed server-side resources.** Delete what you
  create, including on the failure path.
- **Cache handles are account-scoped and server-generated.** They ride in
  request bodies, request *paths* and responses, and the generated-token
  scrubber cannot reach them (it stops a token at `/`), so
  `scrub_resource_names` handles them. Never assert a literal handle.
- `below_minimum_does_not_cache` is the cell that gives every other cell's
  padding its meaning. If it ever starts caching, the documented 1,024-token
  minimum is wrong and every probe's padding needs revisiting.

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
`HUGGINGFACE_API_KEY`, or local services such as Ollama and llama.cpp (`llama-server`, which also serves a `.llamafile`).

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
