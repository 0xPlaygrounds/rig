//! OpenAI: the client and its configuration, the Responses and Chat
//! Completions wires, and every OpenAI-shaped dialect.
//!
//! ```no_run
//! use rig_core::providers::openai;
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let provider = openai::OpenAI::from_env()?;
//!
//! let gpt_5_2 = provider.responses(openai::GPT_5_2);
//! let chat = provider.chat(openai::GPT_5_2);
//! let embeddings = provider.embedding(openai::TEXT_EMBEDDING_3_SMALL, None);
//! # let _ = (gpt_5_2, chat, embeddings);
//! # Ok(())
//! # }
//! ```
//!
//! Each model pairs a wire, which says what to send and how to read the reply,
//! with the client's transport. A vendor that speaks this format builds its
//! client from its own module, such as [`crate::providers::deepseek::from_env`].

pub mod completion;
pub mod embedding;
mod options;
pub mod responses_api;

/// The OpenAI wires: the configuration, the chat-completions wire and one
/// `Dialect` constant per OpenAI-shaped provider.
pub mod wire;

pub use crate::client::openai::OpenAI;
pub use wire::{OpenAIConfig, Route};

#[cfg(feature = "audio")]
#[cfg_attr(docsrs, doc(cfg(feature = "audio")))]
pub mod audio_generation;

#[cfg(feature = "image")]
#[cfg_attr(docsrs, doc(cfg(feature = "image")))]
pub mod image_generation;
#[cfg(feature = "image")]
pub use image_generation::*;

pub mod transcription;

pub use completion::*;
pub use embedding::*;

/// Sanitize nested schema definitions, properties, items, and combinators.
/// Require all properties, supply missing object properties and
/// `additionalProperties: false`, remove `$ref` siblings, and merge `oneOf`
/// into `anyOf`.
pub(crate) fn sanitize_schema(schema: &mut serde_json::Value) {
    crate::providers::internal::schema::sanitize_schema(
        schema,
        crate::providers::internal::schema::SanitizeOptions {
            strip_ref_siblings: true,
            inject_empty_properties: true,
            strip_numeric_constraints: false,
        },
    );
}

/// Return the schema title, or `response_schema` when absent, and a sanitized
/// schema for either structured-output endpoint.
pub(crate) fn structured_output_schema(schema: schemars::Schema) -> (String, serde_json::Value) {
    let name = schema
        .as_object()
        .and_then(|object| object.get("title"))
        .and_then(|title| title.as_str())
        .unwrap_or("response_schema")
        .to_string();
    let mut value = schema.to_value();
    sanitize_schema(&mut value);
    (name, value)
}

#[cfg(feature = "audio")]
pub use audio_generation::{TTS_1, TTS_1_HD};

pub use transcription::*;

/// Whether the OpenAI `model` reads images: every model but GPT-3.5, the
/// text-only GPT-4 snapshots, o1-mini, o1-preview, o3-mini and GPT-5.3 Codex
/// Spark. Chat and Responses both read this one list.
pub(crate) fn reads_images(model: &str) -> bool {
    const TEXT_ONLY: [&str; 9] = [
        "gpt-3.5",
        "gpt-4-32k",
        "gpt-4-0125-preview",
        "gpt-4-1106-preview",
        "gpt-4-turbo-preview",
        "o1-mini",
        "o1-preview",
        "o3-mini",
        "gpt-5.3-codex-spark",
    ];
    let model = model.to_ascii_lowercase();
    !(matches!(model.as_str(), "gpt-4" | "gpt-4-0613" | "gpt-4-0314")
        || TEXT_ONLY.iter().any(|prefix| model.starts_with(prefix)))
}

#[cfg(test)]
mod tests;
