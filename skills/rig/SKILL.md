---
name: rig
description: >
  Build LLM-powered applications with Rig, the Rust AI framework. Use when
  creating agents, RAG pipelines, tool-calling workflows, structured extraction,
  or streaming completions. Covers all providers with a unified API.
argument-hint: "[what-to-build]"
allowed-tools:
  - Read
  - Glob
  - Grep
  - Edit
  - Write
  - Bash
---

# Building with Rig

Rig is a Rust library for building LLM-powered applications with a provider-agnostic API.
All patterns use the builder pattern and async/await via tokio.

## Quick Start

```rust
use rig::completion::Prompt;
use rig::providers::openai;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let client = openai::Client::from_env();

    let agent = client
        .agent(openai::GPT_4O)
        .preamble("You are a helpful assistant.")
        .build();

    let response = agent.prompt("Hello!").await?;
    println!("{}", response);
    Ok(())
}
```

## Core Patterns

### 1. Simple Agent
```rust
let agent = client.agent(openai::GPT_4O)
    .preamble("System prompt")
    .temperature(0.7)
    .max_tokens(2000)
    .build();

let response = agent.prompt("Your question").await?;
```

### 2. Agent with Tools
Define a tool by implementing the `Tool` trait, then attach it:
```rust
let agent = client.agent(openai::GPT_4O)
    .preamble("You can use tools.")
    .tool(MyTool)
    .build();
```
See `references/tools.md` for the full `Tool` trait signature.

### 3. RAG (Retrieval-Augmented Generation)
```rust
let embedding_model = client.embedding_model(openai::TEXT_EMBEDDING_ADA_002);
let index = vector_store.index(embedding_model);

let agent = client.agent(openai::GPT_4O)
    .preamble("Answer using the provided context.")
    .dynamic_context(5, index)  // top-5 similar docs per query
    .build();
```
See `references/rag.md` for vector store setup and the `Embed` derive macro.

### 4. Streaming
```rust
use futures::StreamExt;
use rig::streaming::StreamedAssistantContent;
use rig::agent::prompt_request::streaming::MultiTurnStreamItem;

let mut stream = agent.stream_prompt("Tell me a story").await?;

while let Some(chunk) = stream.next().await {
    match chunk? {
        MultiTurnStreamItem::StreamAssistantItem(
            StreamedAssistantContent::Text(text)
        ) => print!("{}", text.text),
        MultiTurnStreamItem::FinalResponse(resp) => {
            println!("\n{}", resp.response());
        }
        _ => {}
    }
}
```

### 5. Structured Extraction
```rust
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize, JsonSchema)]
struct Person {
    pub name: Option<String>,
    pub age: Option<u8>,
}

let extractor = client.extractor::<Person>(openai::GPT_4O).build();
let person = extractor.extract("John is 30 years old.").await?;
```

### 6. Chat with History
```rust
use rig::completion::Chat;

let history = vec![
    Message::from("Hi, I'm Alice."),
    // ...previous messages
];
let response = agent.chat("What's my name?", history).await?;
```

## Agent Builder Methods

| Method | Description |
|--------|-------------|
| `.preamble(str)` | Set system prompt |
| `.context(str)` | Add static context document |
| `.dynamic_context(n, index)` | Add RAG with top-n retrieval |
| `.tool(impl Tool)` | Attach a callable tool |
| `.tools(Vec<Box<dyn ToolDyn>>)` | Attach multiple tools |
| `.temperature(f64)` | Set temperature (0.0-1.0) |
| `.max_tokens(u64)` | Set max output tokens |
| `.additional_params(json!{...})` | Provider-specific params |
| `.tool_choice(ToolChoice)` | Control tool usage |
| `.build()` | Build the agent |

## Available Providers

Create a client with `ProviderName::Client::from_env()` or `ProviderName::Client::new("key")`.

| Provider | Module | Example Model Constant |
|----------|--------|----------------------|
| OpenAI | `openai` | `GPT_4O`, `GPT_4O_MINI` |
| Anthropic | `anthropic` | `CLAUDE_4_OPUS`, `CLAUDE_4_SONNET` |
| Cohere | `cohere` | `COMMAND_R_PLUS` |
| Mistral | `mistral` | `MISTRAL_LARGE` |
| Gemini | `gemini` | model string |
| Groq | `groq` | model string |
| Ollama | `ollama` | model string |
| DeepSeek | `deepseek` | model string |
| xAI | `xai` | model string |
| Together | `together` | model string |
| Perplexity | `perplexity` | model string |
| OpenRouter | `openrouter` | model string |
| HuggingFace | `huggingface` | model string |
| Azure | `azure` | deployment string |
| Hyperbolic | `hyperbolic` | model string |
| Galadriel | `galadriel` | model string |
| Moonshot | `moonshot` | model string |
| Mira | `mira` | model string |
| Voyage AI | `voyageai` | embeddings only |

## Vector Store Crates

| Backend | Crate |
|---------|-------|
| In-memory | `rig-core` (built-in) |
| MongoDB | `rig-mongodb` |
| LanceDB | `rig-lancedb` |
| Qdrant | `rig-qdrant` |
| SQLite | `rig-sqlite` |
| Neo4j | `rig-neo4j` |
| Milvus | `rig-milvus` |
| SurrealDB | `rig-surrealdb` |

## Key Rules

- All async code runs on tokio.
- Use `WasmCompatSend` / `WasmCompatSync` instead of raw `Send` / `Sync` for WASM compatibility.
- Use proper error types with `thiserror` — never `Result<(), String>`.
- Avoid `.unwrap()` — use `?` operator.

## Further Reference

Detailed API documentation (available when installed via Claude Code skills):
- **tools** — Tool trait, ToolDefinition, ToolEmbedding, attachment patterns
- **rag** — Vector stores, Embed derive, EmbeddingsBuilder, search requests
- **providers** — Provider-specific initialization, model constants, env vars
- **patterns** — Multi-agent, hooks, streaming details, chaining, extraction

For the full reference, see the Rig examples at `rig-core/examples/` or https://docs.rig.rs
