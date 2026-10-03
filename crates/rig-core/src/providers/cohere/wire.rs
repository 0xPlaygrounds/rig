//! Cohere configuration and its text-embedding and image-embedding wires.
//!
//! ```no_run
//! use rig_core::providers::cohere::{Cohere, EMBED_V4};
//! let wire = Cohere::from_env()?.embedding(EMBED_V4, None);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::client::env::{self, EnvError};
use crate::error::EncodeError;
use crate::error::ProviderError;
use crate::operation::{Embedding, ImageEmbedding};
use crate::providers::openai::wire::{COHERE, Chat, OpenAIConfig};
use crate::wire::Flow;
use crate::wire::{
    Body, Capabilities, Decoder, Descriptor, Encoded, Framing, Mode, Out, Secret, Wire, WireEvent,
    WireFrame,
};
use serde::{Deserialize, Serialize};

use super::PROVIDER_NAME;

use super::embeddings::{
    EmbeddingResponse as CohereEmbeddingResponse,
    ImageEmbeddingResponse as CohereImageEmbeddingResponse, image_data_url, validate_image,
};
use crate::providers::internal::wire::classify_reply_or_message_envelope;

/// Cohere's API root.
const BASE_URL: &str = "https://api.cohere.ai";

/// Where Cohere's OpenAI Compatibility API sits under the API root.
const COMPATIBILITY_PATH: &str = "/compatibility/v1";

/// The environment variable carrying the API key.
const API_KEY_ENV: &str = "COHERE_API_KEY";

/// The settings of a Cohere provider: serializable, and the credential is
/// never serialized. [`connect`](Self::connect) puts it on a transport as a
/// [`Cohere`](super::Cohere) client.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CohereConfig {
    /// The API key, sent as `Authorization: Bearer`.
    pub api_key: Secret,
    /// The API root, without a trailing slash.
    pub base_url: String,
}

impl CohereConfig {
    /// Cohere with default settings.
    pub fn new(api_key: impl Into<Secret>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: BASE_URL.to_owned(),
        }
    }

    /// Cohere from `COHERE_API_KEY`.
    pub fn from_env() -> Result<Self, EnvError> {
        Ok(Self::new(env::required(API_KEY_ENV)?))
    }

    /// Point the wires at another API root.
    pub fn with_base_url(mut self, base_url: impl AsRef<str>) -> Self {
        self.base_url = base_url.as_ref().trim_end_matches('/').to_owned();
        self
    }

    /// The Chat Completions wire for `model`, on Cohere's OpenAI
    /// Compatibility API under this API root.
    pub(crate) fn completion(&self, model: impl Into<String>) -> Chat {
        OpenAIConfig::with_key(&COHERE, self.api_key.clone())
            .with_base_url(format!("{}{COMPATIBILITY_PATH}", self.base_url))
            .chat(model)
    }

    /// Build a text-embedding wire reporting the supplied or known model width,
    /// or zero if unknown. This width is metadata, not a request parameter.
    pub(crate) fn embedding(&self, model: impl Into<String>, ndims: Option<usize>) -> Embeddings {
        let model = model.into();
        let ndims = ndims
            .or_else(|| super::model_dimensions_from_identifier(&model))
            .unwrap_or_default();
        Embeddings {
            provider: self.clone(),
            model,
            ndims,
            input_type: DEFAULT_INPUT_TYPE.to_owned(),
        }
    }

    /// The image-embedding wire.
    ///
    /// Cohere Embed v3 embeds images with one fixed model, so this wire
    /// names no model.
    pub(crate) fn image_embedding(&self) -> ImageEmbeddings {
        ImageEmbeddings {
            provider: self.clone(),
        }
    }

    /// One request, authenticated and typed as JSON.
    fn post(&self, path: &str) -> http::request::Builder {
        http::Request::post(format!("{}{path}", self.base_url))
            .header(http::header::CONTENT_TYPE, "application/json")
            .header(
                http::header::AUTHORIZATION,
                format!("Bearer {}", self.api_key.expose()),
            )
    }
}

/// Default retrieval role for embeddings of stored document chunks.
const DEFAULT_INPUT_TYPE: &str = "search_document";

/// The most texts Cohere embeds in one `/v1/embed` call.
const MAX_DOCUMENTS: usize = 96;

/// The width Cohere's image embeddings come back at.
const IMAGE_NDIMS: usize = 1_024;

/// The text-embedding wire: `POST /v1/embed`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Embeddings {
    /// The provider this wire speaks to.
    pub provider: CohereConfig,
    /// The model to address.
    pub model: String,
    /// The width this wire reports, from the caller or the model's published
    /// dimensions. `0` means neither named one.
    pub ndims: usize,
    /// Cohere's retrieval prompt: `search_document` for stored chunks,
    /// `search_query` for queries, `classification`, `clustering`.
    pub input_type: String,
}

impl Embeddings {
    /// Embed for a different purpose than storing chunks.
    pub fn with_input_type(mut self, input_type: impl Into<String>) -> Self {
        self.input_type = input_type.into();
        self
    }
}

impl Wire for Embeddings {
    type Op = Embedding;
    type Payload = crate::wire::Encoded;
    type Frame = crate::wire::WireFrame;
    type Decoder<'id> = EmbeddingsDecoder;

    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new(PROVIDER_NAME)
            .model(self.model.as_str())
            .capabilities(Capabilities::embedding(MAX_DOCUMENTS, self.ndims))
    }

    fn encode(&self, texts: Vec<String>, _mode: Mode) -> Result<Encoded, EncodeError> {
        let body = serde_json::json!({
            "model": self.model,
            "texts": texts,
            "input_type": self.input_type,
        });
        let request = self
            .provider
            .post("/v1/embed")
            .body(Body::Bytes(serde_json::to_vec(&body)?))?;
        Ok(Encoded::new(request, Framing::Whole))
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        EmbeddingsDecoder
    }
}

/// Decodes one `/v1/embed` reply for texts.
pub struct EmbeddingsDecoder;

impl<'id> Decoder<'id, Embedding> for EmbeddingsDecoder {
    /// The vectors, or the error envelope Cohere can answer a **200** with.
    type Event = Result<CohereEmbeddingResponse, String>;

    fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event> {
        classify_reply_or_message_envelope(&frame.as_str(), "embeddings")
    }

    fn decode(
        &mut self,
        reply: Self::Event,
        out: Out<'id, Embedding>,
    ) -> Result<Flow, ProviderError> {
        let reply = reply.map_err(ProviderError::from_provider_body)?;
        let usage = reply
            .meta
            .as_ref()
            .map(|meta| meta.billed_units.to_usage())
            .unwrap_or_default();
        let vectors = reply
            .embeddings
            .into_iter()
            .map(|vector| vector.into_iter().filter_map(|n| n.as_f64()).collect());
        // Cohere's `/v1/embed` reply names no model.
        Ok(out.end(crate::embeddings::EmbeddingResponse {
            response_id: Some(reply.id),
            usage,
            ..crate::embeddings::EmbeddingResponse::from_vectors(vectors)
        }))
    }
}

/// The image-embedding wire: `POST /v1/embed`, one image per request.
///
/// Cohere Embed v3 accepts a single image per call, so the wire's batch limit
/// is one image.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageEmbeddings {
    /// The provider this wire speaks to.
    pub provider: CohereConfig,
}

impl Wire for ImageEmbeddings {
    type Op = ImageEmbedding;
    type Payload = crate::wire::Encoded;
    type Frame = crate::wire::WireFrame;
    type Decoder<'id> = ImageEmbeddingsDecoder;

    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new(PROVIDER_NAME)
            .model(super::EMBED_ENGLISH_V3)
            .capabilities(Capabilities::embedding(1, IMAGE_NDIMS))
    }

    fn encode(&self, images: Vec<Vec<u8>>, _mode: Mode) -> Result<Encoded, EncodeError> {
        // The wire's batch limit is one image: a caller splits larger
        // batches into one call per image.
        let [image] = images.as_slice() else {
            return Err(EncodeError::request(format!(
                "Cohere embeds one image per request, not {}",
                images.len()
            )));
        };
        let media_type = validate_image(image)?;
        let body = serde_json::json!({
            "model": super::EMBED_ENGLISH_V3,
            "images": [image_data_url(image, media_type)],
            "input_type": "image",
            "embedding_types": ["float"],
        });
        let request = self
            .provider
            .post("/v1/embed")
            .body(Body::Bytes(serde_json::to_vec(&body)?))?;
        Ok(Encoded::new(request, Framing::Whole))
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        ImageEmbeddingsDecoder
    }
}

/// Decodes one `/v1/embed` reply for a single image.
pub struct ImageEmbeddingsDecoder;

impl<'id> Decoder<'id, ImageEmbedding> for ImageEmbeddingsDecoder {
    /// The vector, or the error envelope Cohere can answer a **200** with.
    type Event = Result<CohereImageEmbeddingResponse, String>;

    fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event> {
        classify_reply_or_message_envelope(&frame.as_str(), "embeddings")
    }

    fn decode(
        &mut self,
        reply: Self::Event,
        out: Out<'id, ImageEmbedding>,
    ) -> Result<Flow, ProviderError> {
        let reply = reply.map_err(ProviderError::from_provider_body)?;
        // Each request carries one image, so any other vector count is invalid.
        let [vector] = reply.embeddings.values.as_slice() else {
            return Err(ProviderError::Response(format!(
                "Expected 1 image embedding, got {}",
                reply.embeddings.values.len()
            )));
        };
        let usage = reply
            .meta
            .as_ref()
            .map(|meta| meta.billed_units.to_usage())
            .unwrap_or_default();
        // The fold names the input: an image has no text, and its bytes
        // must never travel back in a response.
        let vector = vector.iter().filter_map(|n| n.as_f64()).collect();
        Ok(out.end(crate::embeddings::EmbeddingResponse {
            usage,
            response_id: reply.id,
            ..crate::embeddings::EmbeddingResponse::from_vectors([vector])
        }))
    }
}

#[cfg(test)]
mod tests;
