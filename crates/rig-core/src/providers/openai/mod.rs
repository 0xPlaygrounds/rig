//! OpenAI: one configuration, the Responses and Chat Completions wires, and
//! every OpenAI-shaped dialect.
//!
//! ```no_run
//! use rig_core::providers::openai;
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let provider = openai::OpenAI::from_env()?;
//!
//! let gpt_5_2 = provider.responses(openai::GPT_5_2);
//! let chat = provider.chat(openai::GPT_5_2);
//! let embeddings = provider.embeddings(openai::TEXT_EMBEDDING_3_SMALL, None);
//! # Ok(())
//! # }
//! ```
//!
//! A wire says what to send and how to read the reply; `.bind(transport)`
//! joins it to a socket and yields the [`Bound`](crate::driver::Bound) that
//! implements the consumer-facing model traits.

pub mod completion;
pub mod embedding;
pub mod responses_api;

/// The OpenAI wires: the configuration, the chat-completions wire and one
/// `Dialect` constant per OpenAI-shaped provider.
pub mod wire;

pub use wire::{OpenAI, Route};

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

#[cfg(test)]
mod tests;
