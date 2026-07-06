//! Passive RAG — retrieve *without* the model choosing to, by injecting context
//! before each model call from an `AgentHook`.
//!
//! This is the "always-on" retrieval pattern (the analog of Vercel AI SDK's
//! `transformParams` RAG middleware and Semantic Kernel's `AIContextProvider`).
//! On the first turn the hook reads the prompt text, retrieves, and injects the
//! hits via `RequestPatch::extra_context` — which lands in the request's context
//! documents *before* anything is sent to the provider. Works identically on
//! `prompt()` and `stream()` (both drive the runner); note raw
//! `Agent::completion()` / `Extractor` bypass hooks, so retrieve manually there.
//!
//! No vector store, no embeddings, and no core helper: the query is read from the
//! prompt inline with the public `Message`/`UserContent`/`Text` API. Swap the
//! lexical scorer for `EmbeddingModel::embed_text` + your own store in production.
//!
//! Requires `OPENAI_API_KEY`. Run with: `cargo run -p hook_passive_rag`

use std::collections::HashMap;

use anyhow::Result;
use rig::agent::{AgentHook, Flow, HookContext, RequestPatch, StepEvent};
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::{CompletionModel, Document, Message, Prompt};
use rig::message::UserContent;
use rig::providers::openai;

/// Read the user's query text from a prompt message using existing public APIs
/// (`Message`/`UserContent` variants are public; `Text::text()` is a getter).
fn user_query_text(msg: &Message) -> Option<&str> {
    let Message::User { content } = msg else {
        return None;
    };
    content.iter().find_map(|item| match item {
        UserContent::Text(text) => Some(text.text()),
        _ => None,
    })
}

/// A tiny in-process knowledge base — a `Vec` plus a trivial lexical scorer.
struct KnowledgeBase {
    docs: Vec<(&'static str, &'static str)>,
    top_k: usize,
}

impl KnowledgeBase {
    fn search(&self, query: &str) -> Vec<Document> {
        let words: Vec<String> = query
            .to_lowercase()
            .split_whitespace()
            .map(str::to_owned)
            .collect();
        let mut scored: Vec<(usize, &(&str, &str))> = self
            .docs
            .iter()
            .map(|doc| {
                let text = doc.1.to_lowercase();
                let score = words.iter().filter(|w| text.contains(w.as_str())).count();
                (score, doc)
            })
            .filter(|(score, _)| *score > 0)
            .collect();
        scored.sort_by(|a, b| b.0.cmp(&a.0));
        scored
            .into_iter()
            .take(self.top_k)
            .map(|(_, (id, text))| Document {
                id: (*id).to_string(),
                text: (*text).to_string(),
                additional_props: HashMap::new(),
            })
            .collect()
    }
}

/// Injects retrieved documents on the first model call, before the provider request.
struct PassiveRagHook {
    kb: KnowledgeBase,
}

impl<M: CompletionModel> AgentHook<M> for PassiveRagHook {
    async fn on_event(&self, _ctx: &HookContext, event: StepEvent<'_, M>) -> Flow {
        if let StepEvent::CompletionCall { prompt, turn, .. } = event
            && turn == 1
        {
            // Turn 1: the prompt is the user's query. (Drop the `turn == 1` guard
            // to re-retrieve every turn — cost vs. freshness.)
            let Some(query) = user_query_text(prompt) else {
                return Flow::cont(); // e.g. an image-only turn: nothing to query on.
            };
            let docs = self.kb.search(query);
            if docs.is_empty() {
                return Flow::cont();
            }
            return Flow::patch_request(RequestPatch::new().extra_context(docs));
        }
        Flow::cont()
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let kb = KnowledgeBase {
        docs: vec![
            (
                "doc0",
                "A flurbo is a green alien that lives on cold planets.",
            ),
            (
                "doc1",
                "A glarb-glarb is an ancient tool used to farm the land.",
            ),
            ("doc2", "A linglingdong is a rare mystical instrument."),
        ],
        top_k: 1,
    };

    let agent = openai::Client::from_env()?
        .agent(openai::GPT_4O)
        .preamble("You are a dictionary assistant. Use the provided context documents.")
        .add_hook(PassiveRagHook { kb })
        .build();

    let answer = agent.prompt("What does \"glarb-glarb\" mean?").await?;
    println!("{answer}");
    Ok(())
}
