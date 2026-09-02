//! OpenAI API client and Rig integration
//!
//! # Example
//! ```ignore
//! use rig_core::{client::CompletionClient, providers::openai};
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let client = openai::Client::new("YOUR_API_KEY")?;
//!
//! let model = client.completion_model(openai::GPT_5_2);
//! # Ok(())
//! # }
//! ```
pub mod client;
pub mod completion;
pub mod embedding;
pub mod model_listing;
pub mod responses_api;

#[cfg(feature = "audio")]
#[cfg_attr(docsrs, doc(cfg(feature = "audio")))]
pub mod audio_generation;

#[cfg(feature = "image")]
#[cfg_attr(docsrs, doc(cfg(feature = "image")))]
pub mod image_generation;
#[cfg(feature = "image")]
pub use image_generation::*;

pub mod transcription;

pub use client::*;
pub use completion::*;
pub use embedding::*;
pub use model_listing::*;
pub use responses_api::ResponsesCompletionModel;

/// Recursively ensures all object schemas in a JSON schema respect OpenAI structured output restrictions.
/// Nested arrays, schema $defs, object properties and enums should be handled through this method
///
/// Sources:
/// - <https://platform.openai.com/docs/guides/structured-outputs#additionalproperties-false-must-always-be-set-in-objects>
/// - <https://platform.openai.com/docs/guides/structured-outputs#all-fields-must-be-required>
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

/// The `(name, schema)` pair OpenAI's structured-output configs need from a
/// request's output schema: the schema's `title` (falling back to
/// `response_schema`, which OpenAI requires a name for) and the schema
/// sanitized for the strict subset.
///
/// Derived once for both API surfaces — Chat Completions' `response_format`
/// and Responses' `text.format` — so a turn's structured output is named and
/// sanitized identically whichever endpoint serves it.
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
pub use audio_generation::{
    AudioGenerationModel, CompletionsAudioGenerationModel, TTS_1, TTS_1_HD,
};

pub use streaming::*;
pub use transcription::*;

#[cfg(test)]
mod tests;
