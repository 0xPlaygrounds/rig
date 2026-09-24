# Test Suites

Run the smallest useful local check for changed behavior and leave comprehensive
execution to CI; see [DEVELOPING.md](../DEVELOPING.md) for check selection,
`cargo xtask verify` modes, review, and publication.

Provider test targets have two owners:

- Cassette-backed suites are targets of `rig-cassette`, beside the corpus they
  replay: `crates/rig-cassette/tests/<provider>.rs` with its modules under
  `crates/rig-cassette/tests/providers/<provider>/cassette/` (and `/live/`
  for the ignored live cells those suites still carry). The shared drivers they
  include live in `crates/rig-cassette/tests/common/`.
- Providers with no recorded corpus keep their live-only suites in the facade:
  `tests/<provider>.rs` with `tests/providers/<provider>/`. These are ignored
  tests that require a real service.
- `tests/core.rs` contains provider-agnostic core behavior tests and the guards
  that scan the source tree, which need the repository root.
- `test-support/service-tests/integrations.rs` runs the vector-store suites from `tests/integrations/` as the unpublished `rig-service-tests` package. `tests/integrations.rs` retains the root Bedrock integrations.
- The [ECS consumer harness](https://github.com/gold-silver-copper/rigcoder/tree/main/crates/rigcoder-verify)
  is owned by rigcoder. Run `cargo run --locked -p rigcoder-verify -- verify`
  there for its maintenance/repair cases, replay and supported resume checks.
  Rig retains its runtime and provider conformance suites.

Cassette suites require a checkout of this repository: `rig-cassette` excludes
its fixtures and integration tests from the published package. The unpublished
`rig-cassette-minimal` runner at
`crates/rig-cassette/tests/minimal/Cargo.toml` executes the same `verify` and
`world_replay` sources, plus the shared effect-log/classic-replay regressions.
It selects only cassette's `agent,ecs` features: no native HTTP engine,
`serde_json/preserve_order` or `serde_json/float_roundtrip`.

```sh
RIG_PROVIDER_TEST_MODE=replay cargo nextest run --locked -p rig-cassette-minimal --all-features --retries 0
```

Keep `RIG_REGENERATE_GOLDEN` unset. This execution complements, rather than
replaces, the unified verification runs through `rig-cassette`.
The default/all-features cassette library still owns the unified library
regressions; `core-all` also retains the migrated classic replay cases.
Minimal verification runs with zero retries; unified verification and both
standalone/default-member parity configurations retain their existing two.
The standalone parity lane explicitly includes the `rig-test-support` ECS
helper regressions. Source ownership includes the library tests shared by
the minimal runner, not only its two verification entrypoints.

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
  serializers in #2258.) Body matching compares key-sorted canonical JSON,
  so map reordering is invisible to it (Layer 0's determinism check below
  exists for that), and a cassette cannot notice the provider changing its
  behavior after record time (Layer 3 exists for that).
- **Never weaken a cassette to make it pass.** Update assertions and
  recordings to the new intended behavior, or re-record; a scrubbed value
  must redact real data, never invent data that was not recorded.

## Core Tests

Start with the owning package and a test-name filter. Confirm that the filter
actually ran the intended tests; a successful command running zero tests is not
verification.

```bash
cargo test --locked -p rig-core --lib <test-name-filter>
cargo test --locked -p rig --test <provider> <test-name-filter>
cargo clippy --locked -p rig-core --all-features --tests -- -D warnings
```

The Clippy command is useful for core test changes involving lint-sensitive
patterns, not a mandatory check for every edit. Use the owning package and
relevant features for other changes. Broader facade test commands:

```bash
cargo test -p rig --test core          # provider-agnostic core tests
cargo test -p rig                      # all default non-ignored root-crate tests
cargo test -p rig --all-features       # same, with all root crate features
```

### Fallible test assertions

Workspace Clippy denies `panic_in_result_fn`: `assert!` and `assert_eq!` inside
a test returning `Result` fail that check even when the test itself passes.
Use `anyhow::ensure!(actual == expected, "value changed")` in such tests,
alongside `?` for fallible setup. Preserve each assertion's condition; do not
add lint suppressions or weaken the test. Unit-returning tests can keep their
existing assertion style.

### Core feature isolation

The core crate's dev-dependency on itself enables `test-utils`, `websocket`,
and its default features. Consequently, a core unit-test run with
`--no-default-features` does not prove those features are absent. For an actual
feature-disabled library check, select core alone without test targets:

```bash
cargo check --locked -p rig-core --no-default-features --lib
```

When the change needs WASM coverage and the target is installed, add
`--target wasm32-unknown-unknown`. Keep runtime test coverage and compilation
coverage distinct in the report.

## Cassette Provider Tests

Cassette tests replay committed HTTP interactions by default and do not require
provider API keys. Fixtures live under `crates/rig-cassette/fixtures/cassettes/<provider>/...`.

Replay one migrated provider suite (`anthropic`, `bedrock`, `chatgpt`, `cohere`,
`copilot`, `deepseek`, `doubleword`, `gemini`, `groq`, `llamacpp`, `mistral`,
`mistralrs`, `ollama`, `openai`, `openrouter`, `perplexity`, `venice`, `xai`)
with:

```bash
cargo test -p rig-cassette --all-features --test <provider> <provider>::cassette -- --nocapture --test-threads=1
```

Record mode requires the relevant provider credentials in the environment and
overwrites existing cassette files:

```bash
RIG_PROVIDER_TEST_MODE=record \
cargo test -p rig-cassette --all-features --test <provider> <provider>::cassette -- --nocapture --test-threads=1
```

Prefer re-recording by fixture, from the repository root:

```bash
cargo xtask cassette record [--cap N] [--pause SECS] [--dry-run] [--no-cleanup] <provider/scenario.yaml>...
```

It runs each fixture's owning test once in record mode and logs every run in
an attempt ledger, before it starts and when it ends. A test stops at six
attempts, or at `--cap` when that is lower. After a failed run it copies every
fixture change aside and restores the provider's fixtures as they were just
before the run. It ends with the cleanup pass below, whatever happened, and
exits non-zero when any test did not record, including one already at its cap.
The other commands:

- `cassette owner <fixture>...` prints every test that records a fixture.
- `cassette spend [--cap-per-wire USD] [--cap-total USD]` prices the attempt
  ledger conservatively. A failed or interrupted attempt costs at least $0.05.
- `cassette scan [--base REF] [<fixture>...]` checks changed fixtures for
  credentials, account ids, emails and home paths. It matches case-sensitively
  on whole tokens, and searches for every exported `*_API_KEY`, `*_TOKEN` and
  `*_SECRET` value literally.
- `cassette goldens [--test TARGET]...` regenerates effect goldens from
  replay and reverts churn that only touches `header.deliveries`.
- `cassette cleanup [ledger.jsonl]` runs the cleanup pass on its own.

The recorder refuses to write a fixture, and panics, in two cases:

- A reply is an account failure the cell did not declare as its subject: a
  refused credential, a spent quota, a rate limit or an empty balance. This
  includes one delivered inside a successful stream. Declare an expected
  failure with `CassetteSpec::expects_account_failure` or, in a wrapper that
  presents a rejected key, `ProviderCassette::expect_account_failure`
  (`bogus_api_key()` declares it too).
- An OpenAI or xAI Responses request stored a response the same session never
  deleted. Send `store: false` unless the cell is a chain that deletes its
  own state.

Fixtures older than the stored-state rule are listed in
`crates/rig-cassette/tests/common/stored_state_grandfathered.txt`. When you
re-record one with `store: false`, delete its line; the safety tests fail
until you do.

A failed or refused recording is written under the attempt root, never over
the fixture. The attempt root is `RIG_CASSETTE_ATTEMPT_DIR`, or by default
`cassette-attempts` in the target directory. As replies arrive, before the
test can panic, the recorder appends every stored response, file, cache,
interaction, conversation or vector store it sees created to `ledger.jsonl`
there. `cargo xtask cassette cleanup` deletes whatever the ledger still holds,
using each provider's API key. A 404, or Gemini's 403 "not found", counts as
already gone.

The recorder normalizes volatile timestamps (`created`, `created_at`, …) at
any depth, so a recording pass sees live values its fixture never holds.
Compare a reply document with its fixture through
`assert_matches_recorded_document`, or one field through
`assert_wire_value_matches`. Both compare such keys by type when recording and
exactly in replay. The cassette-safety tests reject an exact comparison of a
volatile key outside replay.

A handful of committed cassettes hold bytes no provider will return: they are
hand-derived from a live capture, or deliberately corrupted to pin a parser
regression. The test that owns such a fixture marks itself, so the sweep above
abandons it instead of healing it:

```rust
if crate::cassettes::skip_when_recording(
    "cell 3 is hand-derived from cell 2: the tool input carries a control byte",
) {
    return;
}
```

Hand-editing a committed cassette means adding that line, with a reason naming
why the bytes are not obtainable live. Scripted fault families need no marker:
they borrow frames from another scenario's fixture and never open a recording
session of their own.

ChatGPT record mode additionally needs `CHATGPT_ACCESS_TOKEN=... CHATGPT_ACCOUNT_ID=...`.

Bedrock cassette replay does not require AWS credentials. Bedrock record mode uses the AWS
SDK credential provider chain and a direct SigV4-aware recorder, so it requires AWS credentials
with Bedrock model access in `us-east-1`. The Bedrock recorder buffers streaming/event-stream
responses and stores non-UTF-8 cassette bodies as base64; those opaque bodies are intended for
replay fidelity, and safety checks also scan their decoded bytes for credential-shaped material.

Venice's text-to-speech scenario records through the direct recorder rather than
the httpmock proxy: the proxy exports bodies as strings, so a binary response
(raw audio) is exported with no body at all and replays as zero bytes. Its
transcription scenario stays on the proxy path, where the same limitation drops
the *request's* multipart body — a cassette that recorded no body still matches
a multipart request, and the multipart shape itself is pinned by unit tests
beside the provider.

Run one cassette test by passing a test-name substring after the test target;
the filter is a substring match, so use the full module path only when the
shorter name is ambiguous:

```bash
cargo test -p rig-cassette --all-features --test gemini \
  streaming_tools_smoke \
  -- --nocapture --test-threads=1
```

### Cassette Safety

Record mode scrubs and safety-checks cassette contents before writing fixtures,
and the committed cassette safety tests enforce the same scrubbed form during
normal test runs. Scrubbing removes credentials (API keys, bearer tokens,
cookies, OAuth tokens, SigV4 material), AWS account numbers in ARNs and
home-directory paths, drops response headers outside a small allowlist, and
normalizes volatile timestamps. Everything else the provider sent is recorded
verbatim:
thinking signatures, encrypted and redacted reasoning, tool-call and response
ids, cache keys, cursors, generated images. Those values are provider-issued
and carry no secret of ours, and keeping them lets a recording seed a live
call and lets a replay decode the provider's own bytes. Request matching
compares the recorded bytes as they are, so a fixture must hold exactly what
the test sends: a fixture recorded before this policy replays only while its
placeholders are values a response delivered and the test echoes back, and a
fixture whose placeholder stands for a value the test itself mints must be
re-recorded. Still review every cassette diff for:

- no API keys, bearer tokens, cookies, or provider account identifiers;
- expected request paths, methods, and bodies;
- expected provider responses for the scenario;
- no unrelated cassette churn.

### History survival and portability

A provider hands Rig opaque fields only it can interpret (thinking signatures,
encrypted reasoning, redacted reasoning, reasoning item ids, tool-call ids)
and expects them back on the next turn, in the slot that carries them: the
signature on its own reasoning block, the ciphertext on its reasoning item,
the call id on both the call and its result. The rule lives in
`test-support/rig-test-support/src/history_survival.rs`, modeled per dialect,
and is applied twice:

- `crates/rig-cassette/tests/cassette_history_survival.rs` sweeps every
  committed cassette and effect golden at zero provider cost: delivered fields
  must reach every continuation request in their slot, every recorded request
  must pair its tool calls with results (a stored Responses chain answers
  calls held by `previous_response_id`), every native `chat_history` must
  pair too (this includes the requests after cancellations, invalid arguments
  and provider faults), and a census proves each provider and content kind
  was examined. Legacy `REDACTED_<n>` placeholders cannot prove a slot and
  are counted, not judged. Exemptions need a cited provider behavior and are
  reported when stale.
- `history_survival_matrix` cells per provider run three prompts over
  deterministic lookup/verify tools with one transient tool failure, assert
  the same delivered content on unary and streaming transports, then apply
  the rule to the recording they just made. `portability_matrix` cells decode
  another wire's committed reply through Rig's real decoder, continue it on
  the target wire, and assert no foreign reasoning state reached it.

Reasoning records the service that issued it, and a request replays only that
service's reasoning (Claude on Bedrock shares Anthropic's).
Further cell families exercise the same round trip: `stateful_chain_matrix`
(OpenAI `previous_response_id` and file ids, Gemini `cachedContents` and
Interactions), `session_matrix` (history persisted through serde, an ECS
checkpoint restored into a fresh world, or agent memory, then continued),
`adversarial_matrix` (reused call ids, reordered parallel results, long
ciphertext, signed empty reasoning, a history ported across three providers
and back), `image_input_matrix` and `request_identity_matrix`. Chains create
and delete their server-side resources in the recorded session, so a
committed chain replays but its handles cannot seed a live call.

Record a cell with the ordinary record mode and an exact test name. Set
`RIG_LONG_TASK_ATTEMPT_DIR` to keep the exchanges of a failed attempt outside
the tree, and `RIG_HISTORY_SURVIVAL_CENSUS` or `RIG_PORTABILITY_REPORT` to
write the census and the forwarded-field report to a file.

## Prompt Cache Testing

Provider prompt caching is a **prefix match**: the cache key is derived from the exact request
bytes up to each breakpoint, so any change to an earlier block invalidates everything after it.
The failure that costs real money is therefore not "caching is off" — it is "caching silently
degraded", where a reordered map, a rewritten earlier turn, or a re-advertised tool set moves the
prefix and every request quietly misses while the counters stay non-zero.

Four layers guard this. Each catches something the others structurally cannot.

### Layer 0 — free, key-free, whole-corpus (`tests/cassette_cache_prefix.rs`)

Checks over cassettes that already exist, costing no provider traffic:

- `recorded_conversations_do_not_move_their_cache_prefix` — for consecutive same-endpoint requests
  in one cassette, turn N-1's canonical blocks must be a prefix of turn N's. The rule lives in
  `test-support/rig-test-support/src/cache_prefix.rs` and is shared with the per-scenario harness so the two cannot
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

### Layer 1 — the shared harness (`test-support/rig-test-support/src/cache_conformance.rs`)

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

### Layer 2 — per-provider cassettes (`crates/rig-cassette/fixtures/cassettes/<provider>/prompt_caching/`)

Recorded live, replayed key-free. Replay is not a tautology: the harness matches request bodies,
so a rig change that perturbs the outbound prefix fails as a replay miss in CI with no API key.
Record one scenario at a time:

```bash
RIG_PROVIDER_TEST_MODE=record \
cargo test -p rig-cassette --all-features --test openai openai::cassette::prompt_caching \
  -- --exact --nocapture --test-threads=1
```

### Layer 3 — live economics (`live_cache_economics`, `#[ignore]`d)

A cassette pins what a provider did at record time; only a live run catches the provider changing
its cache semantics under us. Each cell prints a `LIVE-CACHE-ECONOMICS` row, so the economics
table can be regenerated rather than trusted as a transcription:

```bash
cargo test -p rig-cassette --all-features --test openai live_cache_economics \
  -- --exact --ignored --nocapture --test-threads=1
```

### Rules for recording cache fixtures

- **Never commit a cache cassette whose recorded turn 2 shows zero reads.** The Layer-1 assertions
  run identically in record mode, so a bad session fails instead of committing a fixture that pins
  a miss. That is the intended outcome — a recorded miss is worse than a failed recording.
- **Some providers need a warm-up pass.** Gemini's implicit cache only serves a prefix once an
  entry for *that exact prefix* exists, so on a cold run the grown turn-3 prefix reads zero. Run
  every implicit-cache scenario twice and keep the second recording. Mistral's cache is
  intermittent for a different reason (routing without cache affinity) and may need several attempts.
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

## Live Provider Tests

Live provider tests use real provider APIs, local model servers, or account credentials. They are
ignored by default unless a test file says otherwise.

```bash
# all ignored tests for one provider target
cargo test -p rig-cassette --all-features --test openrouter -- --ignored --nocapture --test-threads=1
# one ignored provider test
cargo test -p rig-cassette --all-features --test openai \
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
feature flags. Some start Docker containers through `testcontainers`, so Docker must be running;
others are ignored because they need external credentials or pre-provisioned services. Check each
integration module for required environment variables (Vectorize requires `VECTORIZE_INDEX_NAME`;
Bedrock needs AWS credentials plus access to the configured Bedrock models).

```bash
# all enabled non-ignored integration tests
cargo test -p rig-service-tests --all-features --test integrations
# one feature-gated group
cargo test -p rig-service-tests --features qdrant --test integrations qdrant -- --nocapture
cargo test -p rig-service-tests --features mongodb --test integrations mongodb -- --nocapture
cargo test -p rig-service-tests --features sqlite --test integrations sqlite -- --nocapture
# ignored groups
cargo test -p rig --features bedrock --test integrations bedrock -- --ignored --nocapture --test-threads=1
cargo test -p rig-service-tests --features vectorize --test integrations vectorize -- --ignored --nocapture --test-threads=1
```

## Shared test implementation

`test-support/rig-test-support` compiles neutral tools, cassette paths, cache and
stream assertions, golden comparison, and the ECS harness once. Provider binaries
import these modules; their cassette-safety tests remain registered in each binary.
Generic ECS matrix drivers live under `crates/rig-cassette/tests/common/ecs_matrix`: a cell edit
then rebuilds its provider binaries without invalidating unrelated providers.
Their long-loop regression tests stay with those modules. Other shared helper
regressions run in the support crate; its ECS regressions also retain the
standalone root-parity dependency configuration.

Matrix declarations keep literal scenario names and explicit per-cell parameters.
`golden_matrix!` runs a common agent/world cell, `resume_matrix!` preserves each
checkpoint cut and its named oracle, and `case_matrix!` selects a family body.
A `case_matrix!` declaration without a wrapper contains only scripted rows;
it registers tests without claiming cassette scenarios.
Their parsers reject malformed rows and exclude ignored rows from the recording
inventory. Keep a compiled listing when changing declarations: source discovery
alone does not establish that a configuration registers or executes a test.

## Agent/ECS regression scenarios

Native ECS tests execute real provider adapters against the same cassettes as
rig-agent tests. Original and native golden comparisons retain their complete
assertions. The [comparison guide](ecs_parity/README.md) describes shared
boundaries and family-specific limitations. Behavioral obligations live in the
tests and their helpers; current test runs and CI establish which tests pass.
A compiled listing is not proof of execution or an exhaustive functional superset.
Shared-provider tests do not count as native agent migrations.

List registrations or run a native family, for example:

```sh
cargo nextest list --locked -p rig --features bedrock
RIG_PROVIDER_TEST_MODE=replay cargo test --locked -p rig-cassette --test anthropic ecs_outcome -- --nocapture
```

### Stream-fault cells

`tests/providers/{gemini,openai}/cassette/stream_faults.rs` and their
`ecs_stream_faults.rs` twins drive the runner and the native runtime through
the real adapter into a stream that ends badly: the committed error
recordings for a setup failure, the committed text stream dropped by its
consumer, and scripted faults served by `rig::test_utils`'s sequenced
transport — a recording cut before its terminal (`test-support/rig-test-support/src/stream_faults.rs`),
a Gemini refusal, an in-band error after content. Every scripted frame is a
labelled constant with its provenance beside the cell; no cassette is
hand-edited and no new fixture is committed. A scripted cell pins what the
runtime does with the fault (the failure kind, the record, the committed
history, the tools that never ran, the witness's facts); the request shape it
would have sent is pinned by the recording's owning test, not by the cell.
Every HTTP wire threads the witness's `AdapterContext` through its request,
so a native cell reads the adapter's boundary facts (the request, the
status, the provider's verdict, usage and error envelope, the closure) for
Gemini, the OpenAI Chat Completions and Responses wires and Anthropic;
Cohere, Ollama and the Gemini Interactions wire report the transport facts
without a payload projection. Error classification follows the one funnel in
`rig_core::provider_response` (see `AGENTS.md`, Error Handling).

Consumed cassettes and goldens remain in Git; historical execution logs, proof
snapshots, and archive infrastructure are intentionally not maintained.
