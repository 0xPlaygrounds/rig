//! Shared OpenAI-compatible wire types, request conversion, and model drivers.
//! Concrete OpenAI clients are enabled separately by the `openai` feature.

#[cfg(feature = "openai")]
pub(crate) use super::openai::client;
pub mod completion;
pub mod embedding;

#[cfg(any(
    feature = "azure",
    feature = "groq",
    feature = "huggingface",
    feature = "openai",
    feature = "venice"
))]
pub mod transcription;

mod observation;
#[cfg(any(
    feature = "chatgpt",
    feature = "copilot",
    feature = "openai",
    feature = "xai"
))]
pub mod responses_api;

#[cfg(all(feature = "image", any(feature = "openai", feature = "azure")))]
#[cfg_attr(docsrs, doc(cfg(feature = "image")))]
pub mod image_generation;
#[cfg(all(feature = "image", any(feature = "openai", feature = "azure")))]
pub use image_generation::*;

#[cfg(feature = "openai")]
pub(crate) use client::{OpenAICompletions, OpenAIResponses};
pub use completion::*;
pub use embedding::*;

#[cfg(feature = "openai")]
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

pub use streaming::*;

#[cfg(test)]
mod tests;
