//! Batch text embeddings through the [Gemini API](https://ai.google.dev/api/embeddings).
//!
//! ```no_run
//! use rig_core::providers::gemini::{Gemini, embedding::EMBEDDING_001};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let model = Gemini::from_env()?.embedding(EMBEDDING_001);
//! # Ok(())
//! # }
//! ```

use crate::error::ProviderError;
use crate::wire::Flow;
use serde_json::json;

use crate::embeddings;
use crate::error::EncodeError;
use crate::providers::internal::wire::classify_marker_keyed_frame;
use crate::wire::{
    Body, Capabilities, Decoder, Descriptor, Encoded, Framing, Mode, Out, Wire, WireEvent,
    WireFrame,
};

/// `gemini-embedding-001` embedding model (3072 dimensions by default)
pub const EMBEDDING_001: &str = "gemini-embedding-001";
/// `text-embedding-004` embedding model (768 dimensions by default)
pub const EMBEDDING_004: &str = "text-embedding-004";

/// Returns the default output dimensionality for known Gemini embedding models.
///
/// See <https://ai.google.dev/gemini-api/docs/models#gemini-embedding>
fn model_default_ndims(model: &str) -> Option<usize> {
    match model {
        EMBEDDING_001 => Some(3072),
        EMBEDDING_004 => Some(768),
        _ => None,
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
    /// The width the caller asked for with [`Self::with_ndims`]. `None`
    /// takes the model's known width.
    pub ndims: Option<usize>,
}

impl Embeddings {
    /// The embedding wire for `model`, at the model's known width.
    pub fn new(provider: super::GeminiConfig, model: impl Into<String>) -> Self {
        Self {
            provider,
            model: model.into(),
            ndims: None,
        }
    }

    /// Ask for `ndims`-wide vectors (`output_dimensionality`). A reply of
    /// another width is [`ProviderError::MismatchedDimensions`].
    pub fn with_ndims(mut self, ndims: usize) -> Self {
        self.ndims = Some(ndims);
        self
    }

    /// The caller's width, else the model's known width. `None` for a model
    /// this build does not know, whose width is neither guessed nor sent.
    fn resolved_ndims(&self) -> Option<usize> {
        self.ndims.or_else(|| model_default_ndims(&self.model))
    }
}

impl Wire for Embeddings {
    type Op = crate::operation::Embedding;
    type Payload = crate::wire::Encoded;
    type Frame = crate::wire::WireFrame;
    type Decoder<'id> = EmbeddingsDecoder;

    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new(super::PROVIDER_NAME)
            .model(self.model.as_str())
            .capabilities(
                Capabilities::embedding(1024, self.resolved_ndims().unwrap_or_default())
                    .declaring(self.ndims),
            )
    }

    fn encode(&self, request: Vec<String>, _mode: Mode) -> Result<Encoded, EncodeError> {
        let ndims = self.resolved_ndims();
        let requests: Vec<_> = request
            .iter()
            .map(|doc| {
                let mut request = serde_json::Map::new();
                request.insert("model".to_owned(), json!(format!("models/{}", self.model)));
                request.insert("content".to_owned(), json!({ "parts": [{ "text": doc }] }));
                if let Some(ndims) = ndims {
                    request.insert("output_dimensionality".to_owned(), json!(ndims));
                }
                request
            })
            .collect();

        let body = json!({ "requests": requests  });

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
        EmbeddingsDecoder
    }
}

/// Decode a `batchEmbedContents` reply containing vectors without usage or identity.
#[derive(Default)]
pub struct EmbeddingsDecoder;

impl<'id> Decoder<'id, crate::operation::Embedding> for EmbeddingsDecoder {
    type Event = gemini_api_types::EmbeddingResponse;

    fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event> {
        classify_marker_keyed_frame(&frame.as_str(), &["embeddings"])
    }

    fn decode(
        &mut self,
        event: Self::Event,
        out: Out<'id, crate::operation::Embedding>,
    ) -> Result<Flow, ProviderError> {
        let vectors = event
            .embeddings
            .into_iter()
            .map(|embedding| embeddings::Embedding {
                // The document each vector belongs to is not on this wire;
                // the operation's fold pairs the batch back on by position.
                document: String::new(),
                vec: embedding
                    .values
                    .into_iter()
                    .filter_map(|value| value.as_f64())
                    .collect(),
            })
            .collect();
        // Gemini supplies no usage or response id; the driver attaches the raw body.
        Ok(out.end(embeddings::EmbeddingResponse::new(vectors)))
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
        pub values: Vec<serde_json::Number>,
    }
}

#[cfg(test)]
mod tests;
