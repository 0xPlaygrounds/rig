//! Cohere's text-embedding and image-embedding wires (`POST /v1/embed`),
//! their reply types, and image-input validation.
//!
//! ```
//! use rig_core::providers::cohere::embeddings::FloatEmbeddings;
//! let vectors: FloatEmbeddings = serde_json::from_str(r#"{"float": [[0.5]]}"#)?;
//! assert_eq!(vectors.values.len(), 1);
//! # Ok::<(), serde_json::Error>(())
//! ```

use crate::error::{EncodeError, ProviderError};
use crate::operation::{Embedding, ImageEmbedding};
use crate::providers::internal::wire::classify_reply_or_message_envelope;
use crate::wire::{
    Body, Capabilities, Decoder, Descriptor, Encoded, Flow, Framing, Mode, Out, Wire, WireEvent,
    WireFrame,
};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde::{Deserialize, Serialize};

use super::{CohereConfig, PROVIDER_NAME};

/// `embed-v4.0` embedding model
pub const EMBED_V4: &str = "embed-v4.0";
/// `embed-english-v3.0` embedding model
pub const EMBED_ENGLISH_V3: &str = "embed-english-v3.0";
/// `embed-english-light-v3.0` embedding model
pub const EMBED_ENGLISH_LIGHT_V3: &str = "embed-english-light-v3.0";
/// `embed-multilingual-v3.0` embedding model
pub const EMBED_MULTILINGUAL_V3: &str = "embed-multilingual-v3.0";
/// `embed-multilingual-light-v3.0` embedding model
pub const EMBED_MULTILINGUAL_LIGHT_V3: &str = "embed-multilingual-light-v3.0";

pub(crate) fn model_dimensions_from_identifier(identifier: &str) -> Option<usize> {
    match identifier {
        EMBED_V4 => Some(1_536),
        EMBED_ENGLISH_V3 | EMBED_MULTILINGUAL_V3 => Some(1_024),
        EMBED_ENGLISH_LIGHT_V3 | EMBED_MULTILINGUAL_LIGHT_V3 => Some(384),
        _ => None,
    }
}

impl CohereConfig {
    /// Build a text-embedding wire reporting the supplied or known model width,
    /// or zero if unknown. This width is metadata, not a request parameter.
    pub(crate) fn embedding(&self, model: impl Into<String>, ndims: Option<usize>) -> Embeddings {
        let model = model.into();
        let ndims = ndims
            .or_else(|| model_dimensions_from_identifier(&model))
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
    pub(super) fn post(&self, path: &str) -> http::request::Builder {
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
    type Event = Result<EmbeddingResponse, String>;

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
            .model(EMBED_ENGLISH_V3)
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
            "model": EMBED_ENGLISH_V3,
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
    type Event = Result<ImageEmbeddingResponse, String>;

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

const MAX_IMAGE_BYTES: usize = 5_000_000;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    #[serde(default)]
    pub response_type: Option<String>,
    pub id: String,
    pub embeddings: Vec<Vec<serde_json::Number>>,
    pub texts: Vec<String>,
    #[serde(default)]
    pub meta: Option<Meta>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Meta {
    pub api_version: ApiVersion,
    pub billed_units: BilledUnits,
    #[serde(default)]
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiVersion {
    pub version: String,
    #[serde(default)]
    pub is_deprecated: Option<bool>,
    #[serde(default)]
    pub is_experimental: Option<bool>,
}

/// Cohere's `meta.billed_units`. Token counters are absent when the request
/// was not billed in tokens (image embeds bill `images`), so they are
/// `Option`; the non-token counters default to zero.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BilledUnits {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u32>,
    #[serde(default)]
    pub search_units: u32,
    #[serde(default)]
    pub classifications: u32,
    #[serde(default)]
    pub images: u32,
}

impl std::fmt::Display for BilledUnits {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Input tokens: {}\nOutput tokens: {}\nSearch units: {}\nClassifications: {}",
            self.input_tokens.unwrap_or(0),
            self.output_tokens.unwrap_or(0),
            self.search_units,
            self.classifications
        )?;
        if self.images > 0 {
            write!(f, "\nImages: {}", self.images)?;
        }
        Ok(())
    }
}

/// One Cohere `/v1/embed` answer for a single image, one per input image:
/// Cohere Embed v3 accepts a single image per call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageEmbeddingResponse {
    #[serde(default)]
    pub id: Option<String>,
    pub embeddings: FloatEmbeddings,
    #[serde(default)]
    pub meta: Option<Meta>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FloatEmbeddings {
    #[serde(rename = "float")]
    pub values: Vec<Vec<serde_json::Number>>,
}

impl BilledUnits {
    /// Maps the billed token counters straight through; `total_tokens` is
    /// the sum of whichever counters Cohere sent (an embed bills input only,
    /// so it reports `input_tokens` and `total_tokens`, no `output_tokens`).
    pub(super) fn to_usage(&self) -> crate::completion::Usage {
        let input_tokens = self.input_tokens.map(u64::from);
        let output_tokens = self.output_tokens.map(u64::from);
        let total_tokens = match (input_tokens, output_tokens) {
            (None, None) => None,
            (input, output) => Some(input.unwrap_or(0) + output.unwrap_or(0)),
        };
        crate::completion::Usage {
            input_tokens,
            output_tokens,
            total_tokens,
            ..Default::default()
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub(super) enum ImageInputError {
    #[error("Cohere image embeddings support PNG, JPEG, WebP, or GIF file bytes")]
    UnsupportedFormat,
    #[error("Cohere image embeddings accept at most 5 MB per image; received {actual_bytes} bytes")]
    TooLarge { actual_bytes: usize },
}

/// Detect an accepted image media type. Returns a document error for unsupported
/// formats or inputs exceeding 5,000,000 bytes.
pub(super) fn validate_image(bytes: &[u8]) -> Result<&'static str, EncodeError> {
    if bytes.len() > MAX_IMAGE_BYTES {
        return Err(EncodeError::request(ImageInputError::TooLarge {
            actual_bytes: bytes.len(),
        }));
    }

    crate::embeddings::image_media_type(bytes)
        .ok_or_else(|| EncodeError::request(ImageInputError::UnsupportedFormat))
}

pub(super) fn image_data_url(bytes: &[u8], media_type: &str) -> String {
    format!("data:{media_type};base64,{}", STANDARD.encode(bytes))
}

#[cfg(test)]
mod tests;
