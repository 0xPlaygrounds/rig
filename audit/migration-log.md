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

## P2 — Pure protocol layer (COMPLETE)

- `agent/config.rs`: `AgentConfig` — the model-free, serde agent definition
  (17 classic fields minus the four behavior slots) with plain `with_*`
  helpers.
- `agent/prepare.rs`: `ToolCatalog` (definitions + executable names, plain
  data with `retain_names`), `PreparedRequest`, and the pure
  `prepare_request()` — a faithful port of `build_prepared_completion_request`
  minus IO; the provider capability (`composes_native_output_with_tools`)
  becomes a plain flag parameter.
- **The classic driver now runs THROUGH the pure function**:
  `build_prepared_completion_request` is a thin async wrapper (retrieval
  query + tool snapshot → catalog → `prepare_request`), and
  `PreparedCompletionRequest` became **non-generic** (its builder field —
  the only `M`-bearing one — is now the plain `CompletionRequest`; the
  driver sends via `model.completion/stream(request)`). Added
  `CompletionRequest::messages_for_telemetry` to keep input telemetry
  identical (documents inserted after system messages).
- `agent/hook.rs`: serde derives on the decision vocabulary (`RequestPatch`,
  all action enums, `InvalidToolCallContext`); pure composition helpers —
  `fold_completion_actions`, `fold_observation_actions`,
  `fold_invalid_resolutions`, and the chained-event accumulators
  `ToolCallResolution` / `ToolResultResolution` (terminate-with-salvage
  semantics), with parity tests mirroring the HookStack suites.
- Deviation (deferred, not dropped): `ToolResultAction`/`ToolOutput` serde
  waits for P7 — `ToolOutput: Serialize` conflicts with the blanket
  `IntoToolOutput` impl that P7 deletes.

**Verification**: rig-agent 505 tests green; rig-core 1008; full facade
cassette suite **1214 passed, 0 failed** with untouched cassettes — the
pure path provably produces byte-identical requests.

## P3 — Provider pilot: openai as config + free functions (COMPLETE)

- `providers/descriptor.rs` (new): `ProviderDescriptor` — the capability
  sheet as one `const` value per provider — and `ApiKeyLocation`
  (`Env`/`Inline`/`None`) with environment resolution.
- `http_runtime.rs` (new): `HttpRuntime` — the concrete HTTP executor
  replacing the `H` type parameter; transport variation is an enum
  (`reqwest` + a `test-utils` recording arm), never a generic. Non-success
  statuses return as `(status, body)` values for uniform provider-side
  error shaping.
- `providers/openai/functions.rs` (new): serde `Config`, `DESCRIPTOR`,
  pure `build_request_body`/`build_request` (data → HTTP request, no IO),
  pure `parse_response` (bytes → normalized response), async
  `complete`/`open_stream` over `HttpRuntime`. The pure functions
  transitionally delegate to the same typed conversion the generic path
  uses, guaranteeing byte-identical bodies until the generic path retires.
- **Proof**: `completions_api_pure_functions_replay_recorded_request`
  replays the exchange recorded by the classic agent path through
  `functions::complete` — the cassette server only serves on a request
  match, so a green test IS the byte-identity proof. Plus in-crate unit
  tests for body shape, URL/auth assembly, and response normalization.
- **Deviation (deferred, not dropped)**: the sans-IO `SseParser` push
  parser and `HttpModelStream` move to P5, where `ModelStream`'s
  no-`Box<dyn>` shape actually requires them; P3's `open_stream` rides the
  (already concrete) compat streaming machinery via a transport match.

## P4 — Provider fleet: configs + free functions everywhere (COMPLETE)

Every in-core provider and the three non-HTTP companions now carry the
openai-pilot face (`Config` + `DESCRIPTOR` + pure `build_request_body`/
`build_request`/`parse_response` + async `complete`/`open_stream`):

- **16 OpenAI-compatible providers** via shared `pub(crate)` helpers in
  openai/functions.rs (`compatible_request_body`/`compatible_request`/
  `compatible_parse_response::<Ext>`/`compatible_open_stream`) +
  `stream_profile_for<Ext>(ext)`. Azure's Config models its deployment URL
  scheme (endpoint + api_version, `api-key` header); llamafile ships
  `ApiKeyLocation::None` (local endpoint); huggingface keeps its
  sub-provider ext default (documented).
- **Standalone providers** (anthropic, cohere, xai, chatgpt, gemini,
  ollama, copilot): request bodies delegate to the exact typed conversions
  the trait impls use, and the SSE machinery was genuinely EXTRACTED with
  the trait impls rewired through it (`stream_anthropic_sse`,
  `stream_cohere_sse`, gemini `generate_content_stream`, ollama
  `consume_chat_streaming_response`, chatgpt `build_codex_responses_request`
  + `parse_codex_sse_response`) — single source of truth per provider.
- **Companions**: bedrock/gemini-grpc `Config` describes client
  construction (region/profile/endpoint; endpoint/key) with
  `client_from_config` builders; candle's `Config` + `ModelArtifacts`
  preserve the crate's no-fs invariant; all trait impls rewired through
  extracted free functions.
- `ProviderDescriptor` gained const `with_*` builders (its
  `#[non_exhaustive]` blocks companion struct literals); openai's
  `max_embedding_documents` corrected 2048→1024 (the code's actual
  `MAX_DOCUMENTS`).

**Documented deferrals**: gemini Interactions API face; copilot
`/responses` routing + OAuth flows (Config carries a resolved token —
interactive auth cannot be plain data); anthropic caching knobs not yet in
Config; bedrock's `"aws_bedrock"` (telemetry) vs `"bedrock"` (stream-final)
naming discrepancy noted for unification.

**Verification**: workspace check 0 errors; clippy clean; rig-core 1079
tests (68 new provider-function tests); full facade cassette suite green
(totals below); cassettes untouched.

## P5 — Facade runtime: ProviderConfig, Runtime, AgentSession, AgentStream (COMPLETE)

The facade crate (`rig`) gained the data-oriented runtime surface:

- **`src/provider.rs`**: `for_each_builtin_provider!` x-macro (24 rows) →
  `ProviderConfig` enum (deliberately NOT `non_exhaustive`) with cfg-gated
  `Bedrock`/`GeminiGrpc`/`Mock` arms; free `complete`/`open_stream` with
  exhaustive matches; `Runtime` holding `HttpRuntime` plus monomorphic
  config-keyed `BedrockCache`/`GeminiGrpcCache` slots; `MockScript`
  (serde-skipped shared cursor: clone shares, deserialize resets) with
  unary and streamed scripts — scripted streams without a hand-authored
  terminal now inherit a `StreamFinal` from the paired response, matching
  the unary branch.
- **`src/session.rs`**: `AgentSession` — pull-based blocking driver over
  `AgentRun` (`advance()` → exhaustive `SessionEvent`; decision inboxes
  `reply_before_call`/`reply_turn`/`resolve_invalid`/`provide_tool_results`;
  `SessionPolicy` gates surfacing; `resume()` restores from a serialized
  `AgentRun`; `run()` is the no-tools convenience).
- **`src/stream.rs`**: `AgentStream` — the streaming driver;
  `AgentStreamItem` is exhaustive; complete tool calls surface in call
  order *before* `ToolCallsReady` (announce-before-execute), committed
  results surface post-batch; abandoned turns drain usage.
- **`src/extract.rs`**: `extract<T>` — the `Extractor<M, T>` successor as
  a free function over `schemars::JsonSchema + DeserializeOwned` bounds;
  Tool output mode, schema from `schema_for!(T)`, corrective-feedback
  retry loop carrying the previous raw output.
- **`#[derive(ToolRouter)]`** (rig-derive) + `rig_agent::tool::router_support`:
  monomorphic per-field dispatch (no boxing), catalog in field order,
  classic-loop error shaping shared via `parse_tool_args`/
  `tool_result_output`, `dispatch_all` with order-preserving bounded
  concurrency and preresolved passthrough. 8 integration tests.
- **rig-mcp**: new agent-built crate (`McpToolset`/`McpCallOutcome`) wired
  into the facade as the `mcp` feature (`rig::tool::mcp`).

**Documented deferrals**: `ToolOutput`/`ToolResultAction` serde (blocked on
the `IntoToolOutput` blanket impl — P7); sans-IO `ModelStream`/`SseParser`
extraction (P7); candle `ProviderConfig` arm (P7, needs artifact plumbing).

**Verification**: clippy clean incl. `--all-features`; facade suite green
single-threaded (incl. 6 new `agent_session` + 3 new `agent_stream` tests);
rig-agent 492 + rig-core 1079 + rig-derive (8 router) + rig-mcp all green;
cassettes untouched.

## P6 pre-step — facade purity (maintainer direction, 2026-07-29)

The maintainer directed that the `rig` facade must remain a pure re-export
layer: nothing is implemented there. The P5 runtime modules
(`provider.rs`, `session.rs`, `stream.rs`, `extract.rs`) moved from the
facade into `rig-agent` verbatim; the facade re-exports them
(`rig::provider` etc.), so no user-facing path changed. No dependency
cycle: rig-bedrock/rig-gemini-grpc dev-depend on rig-agent, so rig-agent
takes them as optional normal deps behind new `bedrock`/`gemini-grpc`
features, forwarded by the facade's features. The design doc gained a
Revision 2.2 note re-reading every "in the facade" placement accordingly;
P6's classic runtime consequently also stays in `rig-agent`.

## P6 — Classic runtime re-plumb: Agent loses M, stays in rig-agent (COMPLETE)

Per maintainer direction the facade stayed pure (see the P6 pre-step): the
whole classic runtime remains in rig-agent, now riding the in-crate
`provider` module.

- **`Agent<M>` → `Agent`**: holds `provider: ProviderConfig` +
  `rt: Arc<Runtime>`; `AgentBuilder`/`AgentRunner`/`PromptRequest`/
  `StreamingPromptRequest`/`Extractor`/`ExtractorBuilder` and the
  `StreamingPrompt`/`StreamingChat` traits all lost `M`. Model calls go
  through `provider::complete`/`open_stream`; capability checks read
  `provider.descriptor()`. Hooks, memory, and the tool server are unchanged.
- **`ToProviderConfig` bridge** (client.rs): `client.agent(model)` keeps
  working — each classic client surrenders base_url/headers/credentials as
  its `functions::Config`. Uniform macro for the plain compat providers;
  dedicated impls where the classic client has provider-specific state:
  azure (endpoint/api_version), huggingface (sub-provider model prefix),
  anthropic + 4 anthropic-flavored alias clients (version/beta headers →
  config fields), llamafile (`/v1` build_uri), gemini (query-param key →
  `ApiKeyLocation::Inline`), openai responses vs completions (below),
  chatgpt/copilot (cached OAuth context + ext knobs via new accessors).
  `AgentModelExt` deleted (a portable model no longer names a provider).
- **OpenAI responses face**: new
  `openai::responses_api::functions` (Config/DESCRIPTOR/pure builders/
  complete/open_stream, single-sourced with the trait impl) + hand-written
  `ProviderConfig::OpenAiResponses` arm; canonical `openai::Client` bridges
  to it, `CompletionsClient` keeps the chat-completions arm. Fixed ollama's
  descriptor over-claiming `composes_native_output_with_tools`.
- **Copilot `/responses` routing** extracted and shared
  (`build_copilot_responses_request`, `stream_copilot_responses_from_event_source`);
  chatgpt/copilot gained non-interactive `cached_auth_context()`.
- **Test migration**: ~250 inline rig-agent tests moved from mock models to
  `MockScript` shims (runner/prompt_request test_support); MockScript gained
  `.with_errors`, `.requests()` (shared, serde-skipped), stream fidelity
  (call_id, text-block metadata, MessageId raw events, inherited terminals);
  `HttpRuntime` gained a `Sequenced` test transport. Classic tests that
  injected custom HTTP clients now inject them via
  `Runtime::with_http(HttpRuntime::recording/sequenced)`.
- **Candle/vertexai**: no ProviderConfig arm (candle arm still deferred to
  P7); their examples/tests drive `AgentRun` + `prepare_request` + their own
  model calls (tool_vertexai.rs is the canonical out-of-tree example);
  candle live conformance rewritten as a local sans-IO driver.
- Examples/integrations updated across the workspace; `MIGRATING.md` gained
  the "0.41 → 0.42 (unreleased)" chapter.

**Tests deleted (logged)**: 2 runner tests unexpressible over MockScript
(paused-transport pause point; scripted mid-stream error after final) and a
span-safety-net test tied to a custom CompletionModel telemetry seam; 5
telemetry-content assertions on raw provider requests dropped (closed
dispatch, request counts kept).

**Verification**: workspace `cargo check --all-targets` 0 errors; clippy
clean workspace-wide; full facade suite green single-threaded (all 34 test
targets, incl. every cassette suite byte-replaying against the bridged
runtime); rig-agent 490 + doctests; rig-core 1084; cassettes untouched.

## P7 — Cleanup of orphaned plumbing (COMPLETE, rescoped)

- **`CompletionRequestBuilder<M>` deleted** along with the trait's
  `completion_request()` sugar; ~230 call sites migrated to the new
  `CompletionRequest::with_history`/`from_prompt` constructors +
  struct-update syntax, with explicit `model.completion(request)` /
  `provider::complete` where the builder's send/stream sugar was used.
  Cassette byte-fidelity preserved (all suites green).
- **`TurnSource` trait → enum** (Unary/Streaming, exhaustive match);
  `drive_agent` no longer generic over it.
- **`ToolOutput` + the full hook decision vocabulary are serde**
  (`ToolResultAction` included); exposed and fixed a long-standing
  `#[serde(flatten)] Option<Value>` wart — flattened `additional_params`
  now deserialize empty maps to `None` (round-trip equality restored;
  wire bytes unchanged).
- **ConversationMemory test doubles moved to `rig_memory::test_utils`**
  (Counting/Failing/AppendFailing); the trait itself stays in rig-core
  (dependency direction), implementations/doubles live in rig-memory.
- **Bedrock naming unified**: `"bedrock"` → `"aws_bedrock"` in
  CompletionResponse/StreamFinal, matching telemetry and the descriptor.

**Rescoped (logged deviation)**: retiring `CompletionModel`,
`GenericCompletionModel<Ext,H>`, and the `Client<Ext,H>`/`Capabilities`
internals moves to the tail of P8 — those internals still carry the
embedding/transcription/image/audio surfaces that P8 converts to free
functions, and dismantling the layer once (after P8) beats two partial
dismantlings. `wasm_compat` shrink rides the same wave.

**Note**: mid-phase, three subagents were killed by a session limit; their
work was recovered from a stash, reconciled, and verified — no work lost.

**Verification**: workspace check + clippy 0 warnings; full facade suite
green single-threaded (all targets); rig-core 1081 + rig-agent 490 +
rig-memory 65; cassettes untouched.

## P8 — Modalities + store de-genericization (COMPLETE)

- **Embeddings**: per-provider `EmbeddingConfig` + free `embed`/`embed_batches`
  for all 8 in-core embedding providers (pure body/parse helpers extracted,
  trait impls rewired — single source of truth); shared chunk/regroup
  machinery (`embeddings/batching.rs`) honoring
  `DESCRIPTOR.max_embedding_documents`, order-aligned, usage summed.
  `EmbedderConfig` x-macro enum in rig-agent (+ cfg-gated Bedrock/GeminiGrpc
  arms reusing the Runtime client caches, serde `MockEmbedder` with shared
  cursor). fastembed keeps a functions face but no enum arm (local weights
  aren't honest serde config — documented).
- **Minor modalities**: all 19 provider×modality impls (transcription,
  image, audio, rerank) extracted to free functions and rewired;
  `HttpRuntime` gained binary-safe `send_bytes`/`send_multipart`.
- **Vector vocabulary** (rig-core): `VectorSearchRequest` is pre-embedded
  (`OneOrMany<Embedding>`); new `SearchHit`/`StoreRecord`;
  `VectorStoreIndex`/`VectorStoreIndexDyn`/`InsertDocuments` DELETED — no
  shared trait replaces them. `InMemoryVectorStore` collapsed to one
  concrete embedder-free store (LSH preserved).
- **All 12 store crates de-genericized** (lancedb, qdrant, mongodb, neo4j,
  postgres, sqlite, scylladb, surrealdb, milvus, s3vectors, helixdb,
  vectorize): no `EmbeddingModel` parameter, inherent
  `top_n`/`top_n_ids`/`top_n_as<T>`/`insert`/`insert_as<T>`, concrete
  per-store filters kept. Notable: `StoreRecord.id` now round-trips where
  stores previously minted opaque UUIDs (qdrant, s3vectors, vectorize —
  idempotent upserts); lancedb `top_n_ids` distance bug fixed
  (`"distance"` → `"_distance"`).
- **rig-agent**: `dynamic_context`/`retrieved_tools` deleted; passive RAG
  is a documented hook recipe (embed → `top_n` → `RequestPatch::
  extra_context`); dynamic tool retrieval is per-turn
  `RequestPatch::active_tools` — the gemini dynamic_tools cassettes replay
  byte-identically under the new pattern. Tool-server retrieval plumbing
  and `ToolServerError` deleted; `ToolEmbedding` vocabulary kept.
- Examples/READMEs across the workspace rewritten to the pre-embedded
  patterns (examples/custom_vector_store is the canonical template).

**Verification**: workspace check + clippy 0 warnings; 71 test targets
green (rig-core 1096+, rig-agent 484+13 doc, full facade suite incl. all
cassette suites single-threaded, sqlite 50/50, all store crates); cassettes
untouched.
