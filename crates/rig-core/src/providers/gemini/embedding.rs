//! Batch text embeddings through the [Gemini API](https://ai.google.dev/api/embeddings).
//!
//! ```no_run
//! use rig_core::providers::gemini::{Gemini, embedding::{Embeddings, EMBEDDING_001}};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let wire = Embeddings::new(Gemini::from_env()?, EMBEDDING_001, None);
//! # Ok(())
//! # }
//! ```

use serde_json::json;

use crate::embeddings;
use crate::error::EncodeError;
use crate::operation::EmbeddingCapabilities;
use crate::providers::internal::wire::classify_marker_keyed_frame;
use crate::wire::{
    Body, Decoder, Encoded, Framing, Mode, Output, Sink, Wire, WireEvent, WireFrame,
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
    pub provider: super::Gemini,
    /// The embedding model, as the path names it.
    pub model: String,
    /// The `output_dimensionality` every document in the batch asks for.
    pub ndims: usize,
}

impl Embeddings {
    /// Build a wire for `model` with the requested output dimensions.
    /// If `ndims` is absent, use the model default or 768 for unknown models.
    pub fn new(provider: super::Gemini, model: impl Into<String>, ndims: Option<usize>) -> Self {
        let model = model.into();
        let ndims = ndims.or_else(|| model_default_ndims(&model)).unwrap_or(768);
        Self {
            provider,
            model,
            ndims,
        }
    }
}

impl Wire for Embeddings {
    type Op = crate::operation::Embedding;
    type Decoder = EmbeddingsDecoder;

    fn name(&self) -> &str {
        super::PROVIDER_NAME
    }

    fn model(&self) -> Option<&str> {
        Some(&self.model)
    }

    fn encode(&self, request: Vec<String>, _mode: Mode) -> Result<Encoded, EncodeError> {
        let requests: Vec<_> = request
            .iter()
            .map(|doc| {
                json!({
                    "model": format!("models/{}", self.model),
                    "content": json!({
                        "parts": [json!({
                            "text": doc
                        })]
                    }),
                    "output_dimensionality": self.ndims,
                })
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

    fn decoder(&self, _mode: Mode) -> Self::Decoder {
        EmbeddingsDecoder
    }

    fn capabilities(&self) -> EmbeddingCapabilities {
        EmbeddingCapabilities::new(1024, self.ndims)
    }
}

/// Decode a `batchEmbedContents` reply containing vectors without usage or identity.
#[derive(Default)]
pub struct EmbeddingsDecoder;

impl Decoder<crate::operation::Embedding> for EmbeddingsDecoder {
    type Event = gemini_api_types::EmbeddingResponse;

    fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event> {
        classify_marker_keyed_frame(&frame.as_str(), &["embeddings"])
    }

    fn interpret(&mut self, event: Self::Event, out: &mut Output<crate::operation::Embedding>) {
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
        out.push(Ok(embeddings::EmbeddingResponse::new(
            vectors,
            super::PROVIDER_NAME,
        )));
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
    use crate::error::ProviderError;
    use serde::{Deserialize, Serialize};

    use crate::embeddings::{self, NormalizeEmbeddingResponse};

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct EmbeddingResponse {
        pub embeddings: Vec<EmbeddingValues>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct EmbeddingValues {
        #[serde(default)]
        pub values: Vec<serde_json::Number>,
    }

    impl NormalizeEmbeddingResponse for EmbeddingResponse {
        fn normalize(
            self,
            provider: &str,
            documents: Vec<String>,
        ) -> Result<embeddings::EmbeddingResponse, ProviderError> {
            if self.embeddings.len() != documents.len() {
                return Err(ProviderError::Response(
                    "Number of returned embeddings does not match input".into(),
                ));
            }
            let docs = documents
                .into_iter()
                .zip(self.embeddings)
                .map(|(document, embedding)| embeddings::Embedding {
                    document,
                    vec: embedding
                        .values
                        .into_iter()
                        .filter_map(|n| n.as_f64())
                        .collect(),
                })
                .collect();
            // batchEmbedContents reports neither usage nor a response id.
            Ok(embeddings::EmbeddingResponse::new(docs, provider))
        }
    }
}

#[cfg(test)]
mod tests;
