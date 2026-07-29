# Data-oriented migration log

Branch `data-oriented-migration`; design authority
`audit/data-oriented-rearchitecture.md` (Revision 2.1). One entry per phase,
updated as the phase progresses; committed with the phase.

## P1 — Normalize the payloads (COMPLETE)

**Done so far**
- Core vocabulary (by hand): concrete `CompletionResponse` (+`FinishReason`,
  `provider: String`, `model`) and constructor/builders in
  `completion/request.rs`; `GetTokenUsage` deleted; `CompletionModel` lost
  `type Response`/`type StreamingResponse`; `CompletionRequestBuilder::send/
  stream` concrete. `streaming.rs`: `StreamFinal` + `StreamFinalKind`;
  `RawStreamingChoice`/`StreamingResult`/`StreamingCompletionResponse`/
  `StreamedAssistantContent` de-genericized; `From<StreamingCompletionResponse>
  for CompletionResponse` fills normalized metadata; inline tests migrated via
  a `mock_final(total_tokens)` helper.
- `internal/openai_chat_completions_compatible.rs` (by hand):
  `CompatibleStreamProfile` — `type Usage` rebound to `Into<Usage>`,
  `type FinalResponse` deleted, `build_final_response -> StreamFinal`;
  `send_compatible_streaming_request` concrete; `record_usage(&Usage)`.
- `openai/completion/` (by hand): Ext trait bounds rebound
  (`StreamingUsage: Into<Usage>`, `Response: TryInto<CompletionResponse>` +
  `ProviderResponseExt<Usage: Into<Usage> + Clone>`); wire→core `TryFrom`
  retargeted with `map_finish_reason` (openai-style vocabulary) and provider
  placeholder `"openai"`; `GenericCompletionModel` stamps
  `Ext::PROVIDER_NAME` post-conversion; wire `StreamingCompletionResponse<U>`
  deleted; profile builds `StreamFinal::new(Ext::PROVIDER_NAME, usage.into())`.
- `test_utils` (agent, complete): `MockResponse` deleted;
  `MockStreamEvent::final_response(usage)` → `StreamFinal::new("mock", ..)`;
  `MockCompletionModel` concrete; telemetry `record_token_usage(&Usage)`;
  mock stream profiles aligned with the trait change.
- rig-agent `run/streamed.rs` production (by hand): `ingest` non-generic,
  `Final` arm reads `final.usage`.

**Fleet migration (eight parallel agents, edits-only/no-cargo; central cargo
triage after each wave)** — all complete:
- anthropic/cohere/xai/chatgpt — per-provider `map_finish_reason`s;
  anthropic-compatible providers keep their own name via post-conversion
  stamping; provider-typed streaming final structs deleted.
- gemini/ollama/copilot — gemini finishReason mapping incl. SAFETY→
  ContentFilter; interactions API documented as having no finish reason;
  copilot rides the responses_api conversion with provider re-stamped.
- openai responses_api (+websocket) — status-based
  `finish_reason_from_status` (the Responses API signals termination via
  status, not a string), Stop→ToolCalls upgrade when output contains
  function calls.
- 16 thin compat files — only 7 needed edits (deepseek, mira, mistral,
  openrouter `From<Usage>` impls + conversions; azure embedding usage;
  groq/deepseek stale alias removal); 9 clean by construction via the
  shared openai wire types.
- rig-agent — `TurnSource` lost `type Raw` entirely (the trait now carries
  no response typing); `MultiTurnStreamItem`/`DriveStream`/`DriveItem`/
  agent `StreamingResult` concrete; `StreamingPrompt<M,R>`→
  `StreamingPrompt<M>`; zero unresolved.
- companions (bedrock/gemini-grpc/candle/vertexai) — mapped SDK enums;
  candle's `infer` returns a (normalized, wire) pair so detail tests keep
  full access; bedrock leaves `model` unset (Converse never echoes it).
- workspace tests/ + examples — cassette-value assertions preserved;
  wire-specific assertions re-expressed by parsing recorded cassette bodies
  into provider wire structs (new `recorded_response_bodies` helper in
  tests/common/cassettes.rs — replay-mode-only, noted in-file).

**Verification (final)**: `cargo check --workspace --all-targets`: 0 errors
(all crates, tests, and example packages). `cargo clippy --workspace
--all-targets`: clean (2 warnings fixed). `cargo test -p rig-core
--all-features --lib`: 1008 passed. `cargo test -p rig-agent
--all-features`: 500 passed. Facade suites (`--test-threads=1`, cassette
replay): **1214 passed, 0 failed**. Cassette YAMLs untouched — request
bytes unchanged by construction (P1 changed no request-building code), and
the `cassette_safety` scenario/file lint passes.

**Triage findings during central verification**
- The tests-sweep agent's `const SCENARIO` hoists violated the
  `cassette_safety` string-literal lint; literals re-inlined at call sites
  (one careless regex pass mispaired five gpt_5_6_reasoning scenarios —
  repaired linewise, lesson recorded).
- Two genuine normalization improvements forced by re-expressed tests:
  the shared openai chat-completions conversion and deepseek's now populate
  `message_id` from the wire `id` (previously only reachable via
  `raw_response`); groq/deepseek roundtrip tests were the detectors.

**Known data losses at the normalized streaming boundary** (accepted;
unary paths keep wire access via provider structs):
- ollama: per-final duration metrics (total_duration etc.).
- copilot/openai responses: terminal reasoning metadata/context.
- candle: throughput metrics (tokens_per_second, TTFT, prefill) — detail
  assertions retained at the `infer` level; `candle_local` example prints
  normalized fields.
- llamacpp live-only test: raw-vs-normalized text equality replaced by
  exact-content assertions on normalized text (the single dropped raw-half
  assertion, commented in-file).

**Deviations so far**
- `StreamFinal` carries `provider`/`model` and `CompletionResponse.provider`
  is `String` — codified in the design doc as Revision 2.1 (not deviations
  against the current doc).
- The `GetTokenUsage` replacement is `Into<crate::completion::Usage>` bounds
  on wire usage types (allowed `From`/`Into` capability machinery), exactly
  as the doc's §3.2 row records.
- Shared wire conversions fill provider placeholder `"openai"`;
  `GenericCompletionModel` overwrites with `Ext::PROVIDER_NAME` (doc §14 P1
  row's transitional pattern).
