//! llama.cpp completion models.
//!
//! Completions run through the shared OpenAI-compatible
//! [`GenericCompletionModel`](openai::completion::GenericCompletionModel); the
//! dialect is declared by the `OpenAICompatibleProvider` impl on
//! [`LlamacppExt`](super::client::LlamacppExt) in `client.rs`.

use crate::providers::openai;

// ================================================================
// llama.cpp Completion Models
// ================================================================
/// The model identifier `llamafile` reported, kept as a convenience constant.
///
/// `llama-server` **ignores the request's `model` field entirely** — it serves
/// whichever GGUF it was started with and echoes that file's path back in the
/// response — so any string works here and none of them selects anything.
/// Measured on b10499-6d05498: a request naming a model the server has never
/// heard of returns 200 with a normal completion, not a 404. The multi-model
/// router (`llama-server --models-dir`) is the exception, and there the
/// identifier is whatever `GET /v1/models` lists.
pub const LLAMA_CPP: &str = "LLaMA_CPP";

/// llama.cpp completion model, driven by the shared OpenAI Chat Completions
/// path.
pub type CompletionModel<H = reqwest::Client> =
    openai::completion::GenericCompletionModel<super::client::LlamacppExt, H>;
