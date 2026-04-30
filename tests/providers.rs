#![allow(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::unwrap_used,
    clippy::unreachable
)]

#[path = "common/reasoning.rs"]
mod reasoning;
#[path = "common/support.rs"]
mod support;

#[path = "providers/anthropic/mod.rs"]
mod anthropic;
#[path = "providers/azure/mod.rs"]
mod azure;
#[path = "providers/chatgpt/mod.rs"]
mod chatgpt;
#[path = "providers/cohere/mod.rs"]
mod cohere;
#[path = "providers/copilot/mod.rs"]
mod copilot;
#[path = "providers/deepseek/mod.rs"]
mod deepseek;
#[path = "providers/galadriel/mod.rs"]
mod galadriel;
#[path = "providers/gemini/mod.rs"]
mod gemini;
#[path = "providers/groq/mod.rs"]
mod groq;
#[path = "providers/huggingface/mod.rs"]
mod huggingface;
#[path = "providers/hyperbolic/mod.rs"]
mod hyperbolic;
#[path = "providers/llamacpp/mod.rs"]
mod llamacpp;
#[path = "providers/llamafile/mod.rs"]
mod llamafile;
#[path = "providers/minimax/mod.rs"]
mod minimax;
#[path = "providers/mira/mod.rs"]
mod mira;
#[path = "providers/mistral/mod.rs"]
mod mistral;
#[path = "providers/moonshot/mod.rs"]
mod moonshot;
#[path = "providers/ollama/mod.rs"]
mod ollama;
#[path = "providers/openai/mod.rs"]
mod openai;
#[path = "providers/openrouter/mod.rs"]
mod openrouter;
#[path = "providers/perplexity/mod.rs"]
mod perplexity;
#[path = "providers/together/mod.rs"]
mod together;
#[path = "providers/voyageai/mod.rs"]
mod voyageai;
#[path = "providers/xai/mod.rs"]
mod xai;
#[path = "providers/xiaomimimo/mod.rs"]
mod xiaomimimo;
#[path = "providers/zai/mod.rs"]
mod zai;
