# Facade features and runtime-import inventory

This inventory is fixed to source revision
`87f3f5b77a3caeffa10d60225c41e386753bf05e`. It closes two distinct questions:
which root-facade features forward to core or optional companions, and which
external examples/tests directly import the current classic runtime through
`rig`, `rig_core`, or either prelude.

## Complete root facade feature map

The root package declares 41 features. Values below come from
`cargo metadata --no-deps --format-version 1`; the declarations begin at
[`Cargo.toml:254`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/Cargo.toml#L254).

| Feature | Exact current forwarding | Boundary class after split |
| --- | --- | --- |
| `default` | `rig-core/default`, `rustls` | facade composition; classic runtime remains the default separately |
| `audio` | `rig-core/audio` | core capability forwarding |
| `bedrock` | `dep:rig-bedrock` | optional integration |
| `candle` | `dep:rig-candle` | optional integration |
| `derive` | `rig-core/derive` | portable derive forwarding; contextual tool macro path must split |
| `discord-bot` | `rig-core/discord-bot` | current classic integration; move forwarding to `rig-agent` |
| `epub` | `rig-core/epub` | core loader forwarding |
| `fastembed` | `dep:rig-fastembed`, `rig-fastembed/hf-hub`, `rig-fastembed/ort-download-binaries` | optional integration bundle |
| `fastembed-hf-hub` | `dep:rig-fastembed`, `rig-fastembed/hf-hub` | optional integration |
| `fastembed-ort-download-binaries` | `dep:rig-fastembed`, `rig-fastembed/ort-download-binaries` | optional integration |
| `gemini-grpc` | `dep:rig-gemini-grpc` | optional integration |
| `helixdb` | `dep:rig-helixdb` | optional integration |
| `image` | `rig-core/image` | core capability forwarding |
| `lancedb` | `dep:rig-lancedb` | optional integration |
| `memory` | `dep:rig-memory` | optional integration layered into the always-present facade memory module |
| `milvus` | `dep:rig-milvus` | optional integration |
| `mongodb` | `dep:rig-mongodb` | optional integration |
| `native-tls` | `rig-core/native-tls` plus optional fastembed/helixdb/lancedb/milvus TLS features | cross-package transport composition |
| `neo4j` | `dep:rig-neo4j` | optional integration |
| `pdf` | `rig-core/pdf` | core loader forwarding |
| `postgres` | `dep:rig-postgres` | optional integration |
| `qdrant` | `dep:rig-qdrant` | optional integration |
| `rayon` | `rig-core/rayon` | core implementation forwarding |
| `reqwest` | `rig-core/reqwest` | core transport forwarding |
| `reqwest-middleware` | `rig-core/reqwest-middleware` | core transport forwarding |
| `reqwest-middleware-native-tls` | `rig-core/reqwest-middleware-native-tls` | core transport forwarding |
| `reqwest-middleware-rustls` | `rig-core/reqwest-middleware-rustls` | core transport forwarding |
| `rmcp` | `rig-core/rmcp` | mixed tool integration; portable protocol values and runtime dispatch must be split |
| `rustls` | `rig-core/rustls` plus optional fastembed/helixdb/lancedb/milvus TLS features | cross-package transport composition |
| `s3vectors` | `dep:rig-s3vectors` | optional integration |
| `scylladb` | `dep:rig-scylladb` | optional integration |
| `socks` | `rig-core/socks` | core transport forwarding |
| `sqlite` | `dep:rig-sqlite` | optional integration |
| `surrealdb` | `dep:rig-surrealdb` | optional integration |
| `test-utils` | `rig-core/test-utils` | test-only core/runtime support that must split by test layer |
| `vectorize` | `dep:rig-vectorize` | optional integration |
| `vertexai` | `dep:rig-vertexai` | optional integration |
| `wasm` | `rig-core/wasm` | core target forwarding; must not select Bevy |
| `websocket` | `rig-core/websocket` | core transport forwarding |
| `websocket-native-tls` | `rig-core/websocket-native-tls` | core transport forwarding |
| `websocket-rustls` | `rig-core/websocket-rustls` | core transport forwarding |

The 18 companion crates are selected by 20 feature names: one per companion
except `rig-fastembed`, which has three entry features. The `memory` feature
augments an always-present facade module. The two TLS features also forward into
whichever optional companion is enabled. These are public feature dependencies,
not private implementation imports. After the split, runtime-free transport and
integration features must continue to avoid selecting either runtime.

## Runtime-import scan

The scan considers external Rust files under `examples/**`, `tests/**`,
`crates/*/examples/**`, and `crates/*/tests/**`. A file is selected when either
a `use` statement rooted at `rig`/`rig_core` or a fully qualified
`rig`/`rig_core` path names `agent`, `extractor`, either prelude, or one of
these current runtime-bearing symbols:

```text
Agent AgentBuilder AgentRunner AgentRun AgentRunStep AgentHook HookStack
RequestPatch OutputMode Prompt Chat TypedPrompt PromptError
StructuredOutputError StreamingPrompt StreamingChat PromptRequest
StreamingPromptRequest Extractor ExtractorBuilder ToolContext ToolSet
ToolSetBuilder ToolServer ToolServerHandle
```

The scan is deliberately conservative: importing a prelude counts even if that
file happens to use only a portable item, because the prelude currently imports
classic runtime traits and types into method resolution. Portable-only direct
imports such as `CompletionModel`, `Message`, `EmbeddingModel`, or `Tool` do not
count unless the same file also references a runtime-bearing surface. Fully
qualified annotations count; for example,
`crates/rig-bedrock/examples/common/mod.rs` imports portable `Tool` but directly
names `rig_core::tool::ToolContext` in its method signature.

| Runtime reference site | Files | Public dependency exposed |
| --- | ---: | --- |
| Workspace example packages | 55 | predominantly root `rig`; runtime selection is currently implicit in the default prelude |
| Per-crate examples | 15 | a mix of direct `rig_core`, root `rig`, and companion APIs |
| Root provider tests | 234 | classic prompt/agent/hook/tool behavior mixed into provider test targets |
| Other root tests and fixtures | 12 | root/core runtime behavior and integration coverage |
| `rig-derive` integration/UI tests | 10 | generated core `Tool`/`ToolContext` paths |
| **Total** | **326** | every path below or in the exhaustive provider-group table |

This list is about public import coupling. Unit tests embedded in
`crates/rig-core/src/**` commonly use private `crate::...` imports instead and
are classified by the module/symbol ownership and conformance matrices.

## Complete example paths

These 70 files directly import a runtime-bearing public surface.

```text
crates/rig-bedrock/examples/agent_with_bedrock.rs
crates/rig-bedrock/examples/common/mod.rs
crates/rig-bedrock/examples/document_with_bedrock.rs
crates/rig-bedrock/examples/image_with_bedrock.rs
crates/rig-bedrock/examples/rag_with_bedrock.rs
crates/rig-bedrock/examples/streaming_with_bedrock.rs
crates/rig-bedrock/examples/streaming_with_bedrock_and_tools.rs
crates/rig-derive/examples/rig_tool/async_tool.rs
crates/rig-derive/examples/rig_tool/full.rs
crates/rig-derive/examples/rig_tool/simple.rs
crates/rig-derive/examples/rig_tool/with_description.rs
crates/rig-gemini-grpc/examples/gemini_grpc_agent.rs
crates/rig-lancedb/examples/vector_search_local_ann_agent.rs
crates/rig-memory/examples/agent_with_memory_policies.rs
crates/rig-vertexai/examples/tool_vertexai.rs
examples/agent/src/main.rs
examples/agent_autonomous/src/main.rs
examples/agent_evaluator_optimizer/src/main.rs
examples/agent_orchestrator/src/main.rs
examples/agent_parallelization/src/main.rs
examples/agent_prompt_chaining/src/main.rs
examples/agent_routing/src/main.rs
examples/agent_run_stepping/src/main.rs
examples/agent_stream_chat/src/main.rs
examples/agent_with_agent_tool/src/main.rs
examples/agent_with_approval_policy/src/main.rs
examples/agent_with_context/src/main.rs
examples/agent_with_default_max_turns/src/main.rs
examples/agent_with_durable_approval/src/main.rs
examples/agent_with_echochambers/src/main.rs
examples/agent_with_human_in_the_loop/src/main.rs
examples/agent_with_loaders/src/main.rs
examples/agent_with_memory/src/main.rs
examples/agent_with_memory_streaming/src/main.rs
examples/agent_with_retry_hook/src/main.rs
examples/agent_with_tools/src/main.rs
examples/agent_with_tools_otel/src/main.rs
examples/calculator_chatbot/src/main.rs
examples/candle_wasm_chat/src/lib.rs
examples/chain/src/main.rs
examples/complex_agentic_loop_claude/src/main.rs
examples/debate/src/main.rs
examples/discord_bot/src/main.rs
examples/enum_dispatch/src/main.rs
examples/force_tool_first_turn/src/main.rs
examples/gemini_deep_research/src/main.rs
examples/gemini_default_api_recovery/src/main.rs
examples/gemini_extractor_with_rag/src/main.rs
examples/gemini_video_understanding/src/main.rs
examples/manual_tool_calls/src/main.rs
examples/multi_agent/src/main.rs
examples/multi_turn_agent/src/main.rs
examples/multi_turn_agent_extended/src/main.rs
examples/openai_agent_completions_api_otel/src/main.rs
examples/openai_streaming_per_call_usage/src/main.rs
examples/openai_streaming_with_tools_otel/src/main.rs
examples/pdf_agent/src/main.rs
examples/rag/src/main.rs
examples/rag_dynamic_tools/src/main.rs
examples/rag_dynamic_tools_multi_turn/src/main.rs
examples/rag_ollama/src/main.rs
examples/reasoning_loop/src/main.rs
examples/request_hook/src/main.rs
examples/reqwest_middleware/src/main.rs
examples/rmcp/src/main.rs
examples/sentiment_classifier/src/main.rs
examples/tool_result_outcomes/src/main.rs
examples/transcription/src/main.rs
examples/vector_search/src/main.rs
examples/vector_search_ollama/src/main.rs
```

Migration PR 7 must classify each as classic, Bevy, or portable-only after its
imports are narrowed. It must not mechanically replace every `rig::prelude`
with a Bevy prelude.

## Complete test import groups

The 234 provider-test import sites are exhaustively partitioned here. The path
pattern for every row is `tests/providers/<provider>/**/*.rs`.

| Provider group | Files | Provider group | Files | Provider group | Files |
| --- | ---: | --- | ---: | --- | ---: |
| `anthropic` | 19 | `azure` | 1 | `bedrock` | 3 |
| `chatgpt` | 13 | `cohere` | 4 | `copilot` | 12 |
| `deepseek` | 9 | `doubleword` | 6 | `gemini` | 30 |
| `groq` | 13 | `huggingface` | 7 | `hyperbolic` | 1 |
| `llamacpp` | 9 | `llamafile` | 15 | `minimax` | 2 |
| `mira` | 2 | `mistral` | 9 | `mistralrs` | 3 |
| `moonshot` | 4 | `ollama` | 11 | `openai` | 19 |
| `openrouter` | 15 | `perplexity` | 5 | `together` | 5 |
| `xai` | 12 | `xiaomimimo` | 2 | `zai` | 3 |

The 12 non-provider root test/fixture paths are:

```text
tests/common/reasoning.rs
tests/common/support.rs
tests/core/prompt_response_messages.rs
tests/core/reasoning_stream_stats.rs
tests/data/loaders/agent_with_loaders.rs
tests/integrations/bedrock/adaptive_thinking.rs
tests/integrations/bedrock/agent.rs
tests/integrations/bedrock/documents.rs
tests/integrations/bedrock/image_generation.rs
tests/integrations/bedrock/image_prompt.rs
tests/integrations/bedrock/streaming.rs
tests/integrations/lancedb/mod.rs
```

The 10 `rig-derive` test paths are:

```text
crates/rig-derive/tests/calculator.rs
crates/rig-derive/tests/schemars_schema.rs
crates/rig-derive/tests/tool_context.rs
crates/rig-derive/tests/typed_error.rs
crates/rig-derive/tests/ui/tool_context/fail_context_in_params.rs
crates/rig-derive/tests/ui/tool_context/fail_context_in_required.rs
crates/rig-derive/tests/ui/tool_context/fail_immutable_context.rs
crates/rig-derive/tests/ui/tool_context/fail_multiple_contexts.rs
crates/rig-derive/tests/ui/tool_context/fail_owned_context.rs
crates/rig-derive/tests/visibility.rs
```

Provider targets are currently a mixed test layer: many validate provider wire
mapping, while others exercise classic agent, hook, tool, extraction, and
streaming behavior. Migration must keep the wire/cassette assertions with the
provider mapping and move runtime behavior to `rig-agent` or shared conformance.
The file counts above are an import inventory, not a claim that all tests in a
selected file have the same owner.
