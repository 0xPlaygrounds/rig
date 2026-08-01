//! OpenAI provider: config + pure functions.
//!
//! The canonical OpenAI face is the Responses API
//! ([`responses_api::functions`]); the Chat Completions face lives in
//! [`functions`] and is also the shared dialect every OpenAI-compatible
//! provider builds on.
//!
//! # Example
//! ```no_run
//! use rig_core::providers::openai;
//!
//! # async fn run(request: rig_core::completion::CompletionRequest)
//! # -> Result<(), Box<dyn std::error::Error>> {
//! let cfg = openai::responses_api::functions::Config::from_env(openai::GPT_5_2)?;
//! let rt = rig_core::http_runtime::HttpRuntime::new();
//! let response = openai::responses_api::functions::complete(&cfg, &rt, request).await?;
//! # Ok(())
//! # }
//! ```
//!
//! The Chat Completions face is the same shape:
//! ```no_run
//! use rig_core::providers::openai;
//!
//! # async fn run(request: rig_core::completion::CompletionRequest)
//! # -> Result<(), Box<dyn std::error::Error>> {
//! let cfg = openai::functions::Config::from_env(openai::GPT_4O)?;
//! let rt = rig_core::http_runtime::HttpRuntime::new();
//! let response = openai::functions::complete(&cfg, &rt, request).await?;
//! # Ok(())
//! # }
//! ```
pub mod api;
pub mod completion;
pub mod embedding;
pub mod functions;
pub mod model_listing;
pub mod responses_api;

mod responses_client {
    crate::providers::client::define_http_client! {
        config = super::responses_api::functions::Config,
        default_base_url = super::responses_api::functions::DEFAULT_BASE_URL,
        api_key_required = true,
    }
}

mod completions_client {
    crate::providers::client::define_http_client! {
        config = super::functions::Config,
        default_base_url = super::functions::DEFAULT_BASE_URL,
        api_key_required = true,
    }
}

pub use completions_client::{
    Client as CompletionsClient, ClientBuilder as CompletionsClientBuilder,
};
pub use responses_client::{Client, ClientBuilder};

crate::providers::client::impl_http_embedding_config_factory!(Client, functions::EmbeddingConfig);
crate::providers::client::impl_http_embedding_config_factory!(
    CompletionsClient,
    functions::EmbeddingConfig
);

impl Client {
    /// Materialize canonical Responses API configuration for `model`.
    pub fn responses_config(&self, model: impl Into<String>) -> responses_api::functions::Config {
        self.config(model)
    }

    /// Materialize Chat Completions configuration for `model`.
    pub fn completions_config(&self, model: impl Into<String>) -> functions::Config {
        self.completions_api().config(model)
    }

    /// Select the Chat Completions API while preserving this connection and runtime.
    pub fn completions_api(&self) -> CompletionsClient {
        CompletionsClient::from_connection(self.connection_config().clone(), self.http_runtime())
    }

    /// Select the canonical Responses API.
    pub fn responses_api(&self) -> Self {
        self.clone()
    }

    /// Materialize transcription configuration sharing this connection.
    pub fn transcription_config(&self, model: impl Into<String>) -> functions::Config {
        self.completions_config(model)
    }

    /// Materialize image-generation configuration sharing this connection.
    #[cfg(feature = "image")]
    pub fn image_generation_config(&self, model: impl Into<String>) -> functions::Config {
        self.completions_config(model)
    }

    /// Materialize audio-generation configuration sharing this connection.
    #[cfg(feature = "audio")]
    pub fn audio_generation_config(&self, model: impl Into<String>) -> functions::Config {
        self.completions_config(model)
    }
}

impl CompletionsClient {
    /// Materialize Chat Completions configuration for `model`.
    pub fn completions_config(&self, model: impl Into<String>) -> functions::Config {
        self.config(model)
    }

    /// Materialize Responses API configuration for `model`.
    pub fn responses_config(&self, model: impl Into<String>) -> responses_api::functions::Config {
        self.responses_api().config(model)
    }

    /// Select the canonical Responses API while preserving this connection and runtime.
    pub fn responses_api(&self) -> Client {
        Client::from_connection(self.connection_config().clone(), self.http_runtime())
    }

    /// Keep the Chat Completions API selected.
    pub fn completions_api(&self) -> Self {
        self.clone()
    }

    /// Materialize transcription configuration sharing this connection.
    pub fn transcription_config(&self, model: impl Into<String>) -> functions::Config {
        self.config(model)
    }

    /// Materialize image-generation configuration sharing this connection.
    #[cfg(feature = "image")]
    pub fn image_generation_config(&self, model: impl Into<String>) -> functions::Config {
        self.config(model)
    }

    /// Materialize audio-generation configuration sharing this connection.
    #[cfg(feature = "audio")]
    pub fn audio_generation_config(&self, model: impl Into<String>) -> functions::Config {
        self.config(model)
    }
}

#[cfg(feature = "audio")]
#[cfg_attr(docsrs, doc(cfg(feature = "audio")))]
pub mod audio_generation;

#[cfg(feature = "image")]
#[cfg_attr(docsrs, doc(cfg(feature = "image")))]
pub mod image_generation;
#[cfg(feature = "image")]
pub use image_generation::*;

pub mod transcription;

pub use api::*;
pub use completion::*;
pub use embedding::*;

/// Recursively ensures all object schemas in a JSON schema respect OpenAI structured output restrictions.
/// Nested arrays, schema $defs, object properties and enums should be handled through this method
pub(crate) fn sanitize_schema(schema: &mut serde_json::Value) {
    use serde_json::Value;

    if let Value::Object(obj) = schema {
        // OpenAI does not allow sibling keywords next to $ref (e.g. "description").
        // Strip everything except $ref so the reference is the sole key.
        if obj.contains_key("$ref") {
            obj.retain(|k, _| k == "$ref");
            return;
        }

        let is_object_schema = obj.get("type") == Some(&Value::String("object".to_string()))
            || obj.contains_key("properties");

        // OpenAI requires "properties" on all object schemas, even empty ones.
        if is_object_schema && !obj.contains_key("properties") {
            obj.insert("properties".to_string(), Value::Object(Default::default()));
        }

        // This is required by OpenAI's Responses API when using strict mode.
        // Source: https://platform.openai.com/docs/guides/structured-outputs#additionalproperties-false-must-always-be-set-in-objects
        if is_object_schema && !obj.contains_key("additionalProperties") {
            obj.insert("additionalProperties".to_string(), Value::Bool(false));
        }

        // This is also required by OpenAI's Responses API
        // Source: https://platform.openai.com/docs/guides/structured-outputs#all-fields-must-be-required
        if let Some(Value::Object(properties)) = obj.get("properties") {
            let prop_keys = properties.keys().cloned().map(Value::String).collect();
            obj.insert("required".to_string(), Value::Array(prop_keys));
        }

        if let Some(defs) = obj.get_mut("$defs")
            && let Value::Object(defs_obj) = defs
        {
            for (_, def_schema) in defs_obj.iter_mut() {
                sanitize_schema(def_schema);
            }
        }

        if let Some(properties) = obj.get_mut("properties")
            && let Value::Object(props) = properties
        {
            for (_, prop_value) in props.iter_mut() {
                sanitize_schema(prop_value);
            }
        }

        if let Some(items) = obj.get_mut("items") {
            sanitize_schema(items);
        }

        // OpenAI doesn't support oneOf so we need to switch this to anyOf
        if let Some(one_of) = obj.remove("oneOf") {
            // If `anyOf` already exists, merge arrays. If not, insert new.
            match obj.get_mut("anyOf") {
                Some(Value::Array(existing)) => {
                    if let Value::Array(mut incoming) = one_of {
                        existing.append(&mut incoming);
                    }
                }
                _ => {
                    obj.insert("anyOf".to_string(), one_of);
                }
            }
        }

        // should handle Enums (anyOf/oneOf)
        for key in ["anyOf", "oneOf", "allOf"] {
            if let Some(variants) = obj.get_mut(key)
                && let Value::Array(variants_array) = variants
            {
                for variant in variants_array.iter_mut() {
                    sanitize_schema(variant);
                }
            }
        }
    }
}

#[cfg(feature = "audio")]
pub use audio_generation::{TTS_1, TTS_1_HD};

pub use streaming::*;
pub use transcription::*;

#[cfg(test)]
mod tests {
    use super::sanitize_schema;
    use serde_json::json;

    #[test]
    fn test_sanitize_strips_ref_sibling_keywords() {
        let mut schema = json!({
            "type": "object",
            "properties": {
                "location": {
                    "$ref": "#/$defs/Location",
                    "description": "The user's location"
                }
            },
            "$defs": {
                "Location": {
                    "type": "object",
                    "properties": {
                        "city": { "type": "string" },
                        "state": { "type": "string" }
                    }
                }
            }
        });

        sanitize_schema(&mut schema);

        // $ref node should only contain "$ref", no "description"
        let location = &schema["properties"]["location"];
        assert_eq!(location, &json!({ "$ref": "#/$defs/Location" }));

        // The referenced $def should still be fully sanitized
        let location_def = &schema["$defs"]["Location"];
        assert_eq!(location_def["additionalProperties"], json!(false));
        assert!(location_def["required"].as_array().is_some());
    }

    #[test]
    fn test_sanitize_adds_additional_properties_false() {
        let mut schema = json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" }
            }
        });

        sanitize_schema(&mut schema);

        assert_eq!(schema["additionalProperties"], json!(false));
    }

    #[test]
    fn test_sanitize_marks_all_properties_required() {
        let mut schema = json!({
            "type": "object",
            "properties": {
                "a": { "type": "string" },
                "b": { "type": "number" }
            }
        });

        sanitize_schema(&mut schema);

        let required = schema["required"].as_array().unwrap();
        assert!(required.contains(&json!("a")));
        assert!(required.contains(&json!("b")));
        assert_eq!(required.len(), 2);
    }

    #[test]
    fn test_sanitize_converts_one_of_to_any_of() {
        let mut schema = json!({
            "oneOf": [
                { "type": "string" },
                { "type": "number" }
            ]
        });

        sanitize_schema(&mut schema);

        assert!(schema.get("oneOf").is_none());
        assert!(schema["anyOf"].as_array().is_some());
    }

    #[test]
    fn test_sanitize_recurses_into_nested_objects() {
        let mut schema = json!({
            "type": "object",
            "properties": {
                "inner": {
                    "type": "object",
                    "properties": {
                        "value": { "type": "string" }
                    }
                }
            }
        });

        sanitize_schema(&mut schema);

        assert_eq!(
            schema["properties"]["inner"]["additionalProperties"],
            json!(false)
        );
        let inner_required = schema["properties"]["inner"]["required"]
            .as_array()
            .unwrap();
        assert!(inner_required.contains(&json!("value")));
    }
}
