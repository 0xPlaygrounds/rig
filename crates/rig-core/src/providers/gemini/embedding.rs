//! Batch text embeddings through the [Gemini API](https://ai.google.dev/api/embeddings).
//!
//! ```no_run
//! use rig_core::providers::gemini::{Gemini, embedding::EMBEDDING_001};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let model = Gemini::from_env()?.embedding(EMBEDDING_001, None);
//! # Ok(())
//! # }
//! ```

use serde_json::{Map, Value, json};

use crate::embeddings;
use crate::error::EncodeError;
use crate::providers::internal::wire::classify_marker_keyed_frame;
use crate::wire::{Body, Capabilities, Descriptor, DocumentDecoder, Encoded, Framing, Mode, Wire};

/// `gemini-embedding-001` embedding model (3072 dimensions by default)
pub const EMBEDDING_001: &str = "gemini-embedding-001";
/// `text-embedding-004` embedding model (768 dimensions by default)
pub const EMBEDDING_004: &str = "text-embedding-004";

/// The width the Gemini API embeds `model` at when no width is asked for:
/// 3072 for [`EMBEDDING_001`], and 768 for [`EMBEDDING_004`] and any model
/// rig does not know. Every Gemini embedding wire defaults to it.
///
/// See <https://ai.google.dev/gemini-api/docs/models#gemini-embedding>
pub fn default_ndims(model: &str) -> usize {
    match model {
        EMBEDDING_001 => 3072,
        _ => 768,
    }
}

/// Gemini's batch embedding endpoint.
///
/// `POST /v1beta/models/{model}:batchEmbedContents`, authenticated by the
/// `key` query parameter the GenerateContent family uses.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Embeddings {
    /// The provider this wire speaks to.
    pub provider: super::GeminiConfig,
    /// The embedding model, as the path names it.
    pub model: String,
    /// The `output_dimensionality` every document in the batch asks for.
    pub ndims: usize,
}

impl Embeddings {
    /// Build a wire for `model` with the requested output dimensions, or
    /// else [`default_ndims`].
    pub fn new(
        provider: super::GeminiConfig,
        model: impl Into<String>,
        ndims: Option<usize>,
    ) -> Self {
        let model = model.into();
        let ndims = ndims.unwrap_or_else(|| default_ndims(&model));
        Self {
            provider,
            model,
            ndims,
        }
    }
}

impl Wire for Embeddings {
    type Op = crate::operation::Embedding;
    type Payload = crate::wire::Encoded;
    type Frame = crate::wire::WireFrame;
    type Event = gemini_api_types::EmbeddingResponse;
    type Decoder<'id> = DocumentDecoder<Self::Op, Self::Event>;
    type Reassembler = crate::wire::document::Unreassembled;

    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new(super::PROVIDER_NAME)
            .model(self.model.as_str())
            .capabilities(Capabilities::embedding(1024, self.ndims))
    }

    fn encode(&self, request: Vec<String>, _mode: Mode) -> Result<Encoded, EncodeError> {
        let model = format!("models/{}", self.model);
        let requests = request.iter().map(|doc| {
            json!({
                "model": model,
                "content": { "parts": [{ "text": doc }] },
                "output_dimensionality": self.ndims,
            })
        });
        let body = Value::Object(Map::from_iter([(
            "requests".to_owned(),
            Value::Array(requests.collect()),
        )]));

        // Avoid allocating formatted batch JSON unless tracing consumes it.
        if tracing::enabled!(target: "rig::embedding", tracing::Level::TRACE)
            && let Ok(pretty_body) = serde_json::to_string_pretty(&body)
        {
            tracing::trace!(
                target: "rig::embedding",
                "Sending embedding request to Gemini API {pretty_body}"
            );
        }

        let request = http::Request::post(format!(
            "{}/v1beta/models/{}:batchEmbedContents?key={}",
            self.provider.base_url,
            self.model,
            self.provider.api_key.expose()
        ))
        .header(http::header::CONTENT_TYPE, "application/json")
        .body(Body::Bytes(serde_json::to_vec(&body)?))?;
        // `batchEmbedContents` has no streaming variant, so a streamed call
        // sends the same bytes and reads the same whole reply.
        Ok(Encoded::new(request, Framing::Whole))
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        DocumentDecoder::new(
            |body| classify_marker_keyed_frame(body, &["embeddings"]),
            |reply| {
                let vectors = reply
                    .embeddings
                    .into_iter()
                    .map(|embedding| embedding.values);
                // Gemini supplies no usage or response id; the driver attaches the raw body.
                Ok(embeddings::EmbeddingResponse::from_vectors(vectors))
            },
        )
    }
}

/// Request and response types for [Gemini embeddings](https://ai.google.dev/api/embeddings).
///
/// ```
/// use rig_core::providers::gemini::embedding::gemini_api_types::EmbeddingValues;
///
/// let vector = EmbeddingValues { values: vec![1.into(), 2.into()] };
/// ```
pub mod gemini_api_types {
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct EmbeddingResponse {
        pub embeddings: Vec<EmbeddingValues>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct EmbeddingValues {
        #[serde(default)]
        pub values: Vec<f64>,
    }
}

impl super::GeminiConfig {
    /// The `batchEmbedContents` embedding wire. `ndims` defaults from the
    /// model identifier.
    pub(crate) fn embedding(&self, model: impl Into<String>, ndims: Option<usize>) -> Embeddings {
        Embeddings::new(self.clone(), model, ndims)
    }
}

#[cfg(test)]
mod tests;
