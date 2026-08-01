# Rig examples

Each example is its own package. Run one with:

```sh
cargo run -p <example-name>
```

Most examples expect provider API keys in the environment (e.g. `OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `COHERE_API_KEY`). See each example's source for specifics.

## The construction story

Normal applications use a concrete, monomorphic provider client. It owns
reusable connection data and transport; selecting a model materializes plain
provider configuration behind the fluent facade:

```rust
use rig::prelude::*;
use rig::providers::openai;

let client = openai::Client::from_env()?;
let agent = client
    .agent(openai::GPT_5_2)
    .preamble("You are a helpful assistant.")
    .build();
let answer = agent.prompt("Entertain me!").await?;

let response = client
    .completion_model(openai::GPT_5_2)
    .completion_request("Summarize the answer")
    .temperature(0.2)
    .send()
    .await?;
```

The data-oriented layer remains public for request conversion, exact-wire
tests, and custom execution. Use a provider `functions::Config`,
`CompletionRequest::builder`, `HttpRuntime`, and the provider's free functions:

```rust
let cfg = openai::responses_api::functions::Config::from_env(openai::GPT_5_2)?;
let rt = HttpRuntime::new();
let request = CompletionRequest::builder("Who are you?").temperature(0.2).build();
let response = openai::responses_api::functions::complete(&cfg, &rt, request).await?;
```

Embeddings follow the same shape — an `EmbeddingConfig` plus `functions::embed`.
`EmbeddingsBuilder` is gone; `embed_documents` batches a `Vec` of `Embed`
documents through any provider's `embed`, and vector stores are pre-embedded
(they never embed text themselves):

```rust
use rig::embeddings::default_concurrency;
use rig::prelude::*;
use rig::providers::openai;

let ecfg = openai::functions::EmbeddingConfig::from_env(openai::TEXT_EMBEDDING_ADA_002)?;
let rt = HttpRuntime::new();
let max_documents = openai::functions::DESCRIPTOR
    .max_embedding_documents
    .unwrap_or(usize::MAX);

let embeddings = embed_documents(
    documents,
    max_documents,
    default_concurrency(max_documents),
    |texts| openai::functions::embed(&ecfg, &rt, texts),
)
.await?;

let store = InMemoryVectorStore::from_documents_with_id_f(embeddings, |doc| doc.id.clone())?;
let query = openai::functions::embed(&ecfg, &rt, vec!["…".to_string()])
    .await?
    .embeddings
    .into_iter()
    .next()
    .expect("one embedding per input text");
let hits = store.top_n(VectorSearchRequest::new(OneOrMany::one(query), 1)).await?;
```

Non-chat modalities are free functions on plain configs. Completion-capable
clients expose config factories such as `embedding_config(model)` so every
operation shares the same connection data:
`functions::{transcribe, generate_image, generate_audio, rerank}`. A custom HTTP
stack goes through `HttpRuntime::from_reqwest(client)`, and for agents through
`Runtime::with_http(..)` passed to `AgentBuilder::runtime(Arc::new(..))`.

| Example | Description |
| --- | --- |
| `agent_autonomous` | Demonstrates an autonomous extraction loop that keeps feeding its own output back in. |
| `agent_evaluator_optimizer` | Generator/evaluator loop: a classic agent plus a structured-extraction judge. |
| `agent_orchestrator` | Orchestrator/worker/judge, all three built on structured extraction. |
| `agent_parallelization` | See source. |
| `agent_prompt_chaining` | Demonstrates prompt chaining with two agents in sequence. |
| `agent_routing` | Demonstrates routing one prompt into different follow-up prompts. |
| `agent_run_stepping` | Drives the agent loop by hand with the sans-IO [`AgentRun`] state machine. |
| `agent_stream_chat` | Demonstrates `stream_chat` with prior conversation history. |
| `agent_with_agent_tool` | See source. |
| `agent_with_approval_policy` | Demonstrates a non-interactive, policy-based HITL gate: a `HookEntry` closure auto-approves an allow-list, denies the rest (fail-closed), and applies an arg-based rule (mirrors `needs_approval`/`interrupt_on` predicates). |
| `agent_with_context` | Demonstrates adding small context documents directly to an agent. |
| `agent_with_default_max_turns` | Demonstrates extending the default agent loop budget for tool-heavy prompts. |
| `agent_with_durable_approval` | Demonstrates **durable** HITL: the hand-driven `AgentRun` is serialized while tool calls are pending and resumed from JSON (as another process would), so approval can happen out-of-process / later. |
| `agent_with_echochambers` | See source. |
| `agent_with_human_in_the_loop` | Demonstrates human-in-the-loop tool-call approval: a `HookEntry` closure pauses on each tool call so a human can approve/deny/edit/abort, mapped onto typed `ToolCallAction` values (`Run`/`Skip`/`Rewrite`/`Stop`). |
| `agent_with_loaders` | Demonstrates loading real example files into agent context. |
| `agent_with_memory_streaming` | Demonstrates Rig-managed conversation memory with streaming. |
| `agent_with_memory` | Demonstrates Rig-managed conversation memory with an in-memory backend. |
| `agent_with_tools_otel` | Agent multi-turn with tools, but with a tracing subscriber that sends all logs/traces to an OTel collector. |
| `agent_with_tools` | Demonstrates registering runtime-defined tools on an agent. |
| `agent` | Demonstrates the smallest useful agent setup with OpenAI. |
| `calculator_chatbot` | See source. |
| `chain` | Demonstrates a retrieval-augmented pipeline with `parallel!` and `lookup`. |
| `complex_agentic_loop_claude` | See source. |
| `custom_vector_store` | Example: Implementing a custom vector store backend |
| `debate` | See source. |
| `discord_bot` | See source. |
| `enum_dispatch` | See source. |
| `extractor` | Demonstrates typed extraction and extraction with usage metadata. |
| `force_tool_first_turn` | Demonstrates a per-turn `RequestPatch` footgun and its fix: forcing `tool_choice = Required` on *every* turn loops until `max_turns`, so a `HookEntry` closure gates the patch on the event's `turn == 1` to force the tool only up front. |
| `gemini_deep_research` | See source. |
| `gemini_default_api_recovery` | Demonstrates recovering from Gemini emitting a legacy `default_api` tool name. |
| `gemini_extractor_with_rag` | RAG-backed structured extraction (retrieval runs up front, into the agent config's static context). |
| `gemini_nanobanana_image_generation` | See source. |
| `gemini_stream_kill_token_count` | Live Gemini example: obtaining a token-count estimate when a streaming |
| `gemini_video_understanding` | Demonstrates Gemini video understanding with provider-specific request parameters. |
| `manual_tool_calls` | Demonstrates manual tool-call handling with an explicit bound completion request. |
| `multi_agent` | See source. |
| `multi_extract` | Demonstrates fan-out structured extraction with `try_parallel!`. |
| `multi_turn_agent_extended` | See source. |
| `multi_turn_agent` | See source. |
| `openai_agent_completions_api_otel` | This example shows how you can use OpenAI's Completions API. |
| `openai_streaming_per_call_usage` | Shows how to inspect per-completion-call usage in an agent stream. |
| `openai_streaming_with_tools_otel` | See source. |
| `pdf_agent` | See source. |
| `rag_dynamic_tools_multi_turn` | Dynamic tool selection as a hook recipe, over a multi-turn run: tool docs are embedded with `embed_documents` and the hook narrows `RequestPatch::active_tools` each turn. |
| `rag_dynamic_tools` | Dynamic tool selection as a hook recipe: every candidate tool is registered, tool docs are embedded into a store, and a hook narrows `RequestPatch::active_tools` per prompt. |
| `rag_ollama` | Passive RAG entirely locally: Ollama embedding + completion configs, an in-memory store, and a retrieval hook. |
| `rag` | Passive RAG as a hook: documents embedded with `embed_documents`, retrieved per model call, injected as per-turn context. |
| `reasoning_loop` | A reasoning agent: extract the chain of thought with `Agent::extractor`, then execute it with a tool-carrying agent. |
| `request_hook` | Demonstrates observing prompt/response/tool lifecycle events by stacking several attach-and-forget `HookEntry` records with `add_hook` (delta events additionally require `.observing_deltas()`). |
| `reqwest_middleware` | Demonstrates driving an agent over a caller-supplied `reqwest::Client` through `HttpRuntime::from_reqwest` and `Runtime::with_http`. |
| `rmcp_example` | An example of how you can use `rmcp` with Rig to create an MCP friendly agent. |
| `sentiment_classifier` | Demonstrates the smallest structured extraction for classification. |
| `transcription` | Transcribes one audio file with every transcription provider via `functions::transcribe` (no `TranscriptionModel`). |
| `tool_result_outcomes` | Demonstrates structured disk (`Other`/`EIO`) and network (`Network`/`ENETUNREACH`) tool failures, a host-owned run ledger, and ordered recorder/policy hooks that terminate fatal failures while returning recoverable feedback to the model. Run `cargo run -p tool_result_outcomes -- --help` for credential-free usage. |
| `vector_search_cohere` | Demonstrates vector search with separate Cohere document and query embedding configs (`with_input_type`). |
| `vector_search_ollama` | Demonstrates vector search against a local Ollama embedding config. |
| `vector_search` | Demonstrates embedding documents with `embed_documents` and querying a pre-embedded in-memory vector store. |
