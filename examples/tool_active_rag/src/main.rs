//! Active RAG — retrieval as an ordinary tool the model chooses to call.
//!
//! This is the canonical RAG pattern across frameworks that keep vector search
//! out of the agent core (Pydantic AI's `@agent.tool retrieve`, the OpenAI
//! Agents SDK's `FileSearchTool`, LangChain's `create_retriever_tool`): the
//! model writes the query as a tool argument, and the retrieved text comes back
//! as a tool-result message. Rig ships no vector-store abstraction — the tool's
//! `call` owns retrieval, so swap the trivial lexical scorer below for
//! `EmbeddingModel::embed_text` + your own store in production.
//!
//! Requires `OPENAI_API_KEY`. Run with: `cargo run -p tool_active_rag`

use anyhow::Result;
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::{Prompt, ToolDefinition};
use rig::providers::openai;
use rig::tool::Tool;
use serde::Deserialize;
use serde_json::json;

#[derive(Deserialize)]
struct RetrieveArgs {
    query: String,
}

#[derive(Debug, thiserror::Error)]
#[error("retrieval failed")]
struct RetrieveError;

/// A tiny in-process knowledge base exposed as a tool. No vector store, no
/// embeddings — a `Vec` plus a trivial lexical scorer.
struct Retrieve {
    docs: Vec<(&'static str, &'static str)>,
}

impl Tool for Retrieve {
    const NAME: &'static str = "retrieve";
    type Error = RetrieveError;
    type Args = RetrieveArgs;
    type Output = String;

    async fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Search the knowledge base for documents relevant to a query.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "What to search for" }
                },
                "required": ["query"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let query = args.query.to_lowercase();
        let hits: Vec<String> = self
            .docs
            .iter()
            .filter(|(_, text)| {
                query
                    .split_whitespace()
                    .any(|word| text.to_lowercase().contains(word))
            })
            .map(|(id, text)| format!("[{id}] {text}"))
            .collect();
        Ok(if hits.is_empty() {
            "No relevant documents found.".to_string()
        } else {
            hits.join("\n")
        })
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let docs = vec![
        (
            "doc0",
            "A flurbo is a green alien that lives on cold planets.",
        ),
        (
            "doc1",
            "A glarb-glarb is an ancient tool used to farm the land.",
        ),
        ("doc2", "A linglingdong is a rare mystical instrument."),
    ];

    let agent = openai::Client::from_env()?
        .agent(openai::GPT_4O)
        .preamble(
            "You are a dictionary assistant. When asked about an unfamiliar term, call the \
             `retrieve` tool to look it up before answering.",
        )
        .tool(Retrieve { docs })
        .build();

    // `max_turns` lets the model call `retrieve`, read the result, then answer.
    let answer = agent
        .prompt("What does \"glarb-glarb\" mean?")
        .max_turns(5)
        .await?;
    println!("{answer}");
    Ok(())
}
