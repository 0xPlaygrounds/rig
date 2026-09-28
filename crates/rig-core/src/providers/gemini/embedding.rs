//! Batch text embeddings through the [Gemini API](https://ai.google.dev/api/embeddings).
//!
//! ```no_run
//! use rig_core::providers::gemini::{Gemini, embedding::EMBEDDING_001};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let model = Gemini::from_env()?.embedding(EMBEDDING_001, 3072);
//! # Ok(())
//! # }
//! ```

use crate::error::ProviderError;
use crate::wire::Flow;

use super::api;

use crate::embeddings;
use crate::error::EncodeError;
use crate::providers::internal::wire::classify_marker_keyed_frame;
use crate::wire::{
    Body, Capabilities, Decoder, Descriptor, Encoded, Framing, Mode, Out, Wire, WireEvent,
    WireFrame,
};

/// `gemini-embedding-001` embedding model.
pub const EMBEDDING_001: &str = "gemini-embedding-001";
/// `gemini-embedding-2` multimodal embedding model.
pub const EMBEDDING_2: &str = "gemini-embedding-2";

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
    /// A wire for `model` asking for `ndims` output dimensions.
    pub fn new(provider: super::GeminiConfig, model: impl Into<String>, ndims: usize) -> Self {
        Self {
            provider,
            model: model.into(),
            ndims,
        }
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
            .capabilities(Capabilities::embedding(1024, self.ndims))
    }

    fn encode(&self, request: Vec<String>, _mode: Mode) -> Result<Encoded, EncodeError> {
        let ndims = i32::try_from(self.ndims)
            .map_err(|_| EncodeError::request("ndims is larger than outputDimensionality"))?;
        let body = api::BatchEmbedContentsRequest {
            requests: request
                .into_iter()
                .map(|text| api::EmbedContentRequest {
                    model: Some(format!("models/{}", self.model)),
                    content: Some(api::Content {
                        parts: vec![api::Part {
                            text: Some(text),
                            ..Default::default()
                        }],
                        ..Default::default()
                    }),
                    output_dimensionality: Some(ndims),
                    ..Default::default()
                })
                .collect(),
            ..Default::default()
        };

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
    type Event = api::BatchEmbedContentsResponse;

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
                vec: embedding.values,
            })
            .collect();
        // An embedding response has no usage field; the raw body keeps Gemini's.
        Ok(out.end(embeddings::EmbeddingResponse::new(
            vectors,
            super::PROVIDER_NAME,
        )))
    }
}

#[cfg(test)]
mod tests;
