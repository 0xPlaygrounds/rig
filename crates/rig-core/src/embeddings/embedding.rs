//! Text and image embedding models, responses, and input identifiers.
//!
//! ```no_run
//! use rig_core::embeddings::EmbeddingModel;
//!
//! # async fn example(model: impl EmbeddingModel) -> Result<(), Box<dyn std::error::Error>> {
//! let embedding = model.embed_text("A document").await?;
//! # let _ = embedding;
//! # Ok(())
//! # }
//! ```

use crate::error::ProviderError;
use crate::{
    completion::{ResponseIdentity, Usage},
    wasm_compat::{WasmCompatSend, WasmCompatSync},
};
use serde::{Deserialize, Serialize};

/// Trait for embedding models that can generate embeddings for documents.
pub trait EmbeddingModel: WasmCompatSend + WasmCompatSync {
    /// The maximum number of documents accepted in a single request.
    fn max_documents(&self) -> usize;

    /// The number of dimensions in the embedding vector.
    fn ndims(&self) -> usize;

    /// Embed multiple text documents in a single request and return the full
    /// normalized response: embeddings, usage, provider, and identity.
    ///
    /// Implementations must preserve input order in the returned embeddings.
    fn embed_texts_response(
        &self,
        texts: impl IntoIterator<Item = String> + WasmCompatSend,
    ) -> impl std::future::Future<Output = Result<EmbeddingResponse, ProviderError>> + WasmCompatSend;

    /// Embed multiple text documents in a single request.
    ///
    /// The convenience form of [`EmbeddingModel::embed_texts_response`] for
    /// callers who want only the vectors.
    fn embed_texts(
        &self,
        texts: impl IntoIterator<Item = String> + WasmCompatSend,
    ) -> impl std::future::Future<Output = Result<Vec<Embedding>, ProviderError>> + WasmCompatSend
    {
        async { Ok(self.embed_texts_response(texts).await?.embeddings) }
    }

    /// Embeds one text, returning the last vector or an error if none is returned.
    fn embed_text(
        &self,
        text: &str,
    ) -> impl std::future::Future<Output = Result<Embedding, ProviderError>> + WasmCompatSend {
        async {
            let mut embeddings = self.embed_texts(vec![text.to_string()]).await?;
            embeddings.pop().ok_or_else(|| {
                ProviderError::Response(
                    "embedding provider returned an empty response for embed_text".to_string(),
                )
            })
        }
    }

    /// Embeds one text and returns the normalized response, rejecting an empty
    /// embedding list.
    fn embed_text_response(
        &self,
        text: &str,
    ) -> impl std::future::Future<Output = Result<EmbeddingResponse, ProviderError>> + WasmCompatSend
    {
        async {
            let response = self.embed_texts_response(vec![text.to_string()]).await?;
            if response.embeddings.is_empty() {
                return Err(ProviderError::Response(
                    "embedding provider returned an empty response for embed_text_response"
                        .to_string(),
                ));
            }
            Ok(response)
        }
    }
}

/// Text embeddings and normalized provider metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    /// The embeddings returned by the provider, one per input text, in input order.
    pub embeddings: Vec<Embedding>,
    /// Token usage for this request; every counter is `None` when the
    /// provider reported none (see [`Usage`]).
    #[serde(default)]
    pub usage: Usage,
    /// Stable descriptor name of the provider that produced this response,
    /// for example `"openai"`. Always populated.
    pub provider: String,
    /// Provider-reported model identifier, when the wire response named one.
    #[serde(default)]
    pub model: Option<String>,
    /// Provider-assigned response-scoped identifier, when reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_id: Option<String>,
    /// Transport request identifier from HTTP response headers, or `None` when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_request_id: Option<String>,
    /// Provider response payload, or null when no raw payload was attached.
    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub raw: serde_json::Value,
}

impl EmbeddingResponse {
    /// Create a response from its required parts; optional metadata starts
    /// unset and is filled in with the `with_*` helpers.
    pub fn new(embeddings: Vec<Embedding>, provider: impl Into<String>) -> Self {
        Self {
            embeddings,
            usage: Usage::default(),
            provider: provider.into(),
            model: None,
            response_id: None,
            provider_request_id: None,
            raw: serde_json::Value::Null,
        }
    }

    /// This response's identity metadata as one [`ResponseIdentity`] carrier.
    /// `message_id` is always `None`: nothing here is replayed as an
    /// assistant message.
    pub fn identity(&self) -> ResponseIdentity {
        ResponseIdentity {
            message_id: None,
            response_id: self.response_id.clone(),
            provider_request_id: self.provider_request_id.clone(),
        }
    }
}

crate::provider_response::modality_response_metadata_setters!(EmbeddingResponse);

/// Normalizes embedding payloads using the supplied provider name and input documents.
pub trait NormalizeEmbeddingResponse {
    /// Normalize this payload, attributing it to `provider`. `documents` are
    /// the inputs in request order, for [`Embedding::document`].
    fn normalize(
        self,
        provider: &str,
        documents: Vec<String>,
    ) -> Result<EmbeddingResponse, ProviderError>;
}

/// Image embeddings and normalized provider metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageEmbeddingResponse {
    /// The embeddings returned by the provider, one per input image, in input order.
    pub embeddings: Vec<Embedding>,
    /// Token usage for this request; every counter is `None` when the
    /// provider reported none (see [`Usage`]).
    #[serde(default)]
    pub usage: Usage,
    /// Stable descriptor name of the provider that produced this response,
    /// for example `"openai"`. Always populated.
    pub provider: String,
    /// Provider-reported model identifier, when the wire response named one.
    #[serde(default)]
    pub model: Option<String>,
    /// Provider-assigned response-scoped identifier, when reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_id: Option<String>,
    /// Transport request identifier from HTTP response headers, or `None` when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_request_id: Option<String>,
    /// Provider response payload, or null when no raw payload was attached.
    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub raw: serde_json::Value,
}

impl ImageEmbeddingResponse {
    /// Create a response from its required parts; optional metadata starts
    /// unset and is filled in with the `with_*` helpers.
    pub fn new(embeddings: Vec<Embedding>, provider: impl Into<String>) -> Self {
        Self {
            embeddings,
            usage: Usage::default(),
            provider: provider.into(),
            model: None,
            response_id: None,
            provider_request_id: None,
            raw: serde_json::Value::Null,
        }
    }

    /// This response's identity metadata as one [`ResponseIdentity`] carrier.
    /// `message_id` is always `None`: nothing here is replayed as an
    /// assistant message.
    pub fn identity(&self) -> ResponseIdentity {
        ResponseIdentity {
            message_id: None,
            response_id: self.response_id.clone(),
            provider_request_id: self.provider_request_id.clone(),
        }
    }
}

crate::provider_response::modality_response_metadata_setters!(ImageEmbeddingResponse);

/// Trait for embedding models that can generate embeddings for images.
pub trait ImageEmbeddingModel: WasmCompatSend + WasmCompatSync {
    /// The maximum number of images the provider accepts in one request.
    fn max_documents(&self) -> usize;

    /// The number of dimensions in the embedding vector.
    fn ndims(&self) -> usize;

    /// Embed a batch of images from their encoded file bytes and return the
    /// full normalized response. This is the method a provider implements;
    /// [`ImageEmbeddingModel::embed_images`] and
    /// [`ImageEmbeddingModel::embed_image`] derive from it.
    ///
    /// Implementations must preserve input order in the returned embeddings.
    /// The returned [`Embedding::document`] should identify the input without
    /// retaining the raw image or a reversible encoding of it.
    fn embed_images_response(
        &self,
        images: impl IntoIterator<Item = Vec<u8>> + WasmCompatSend,
    ) -> impl std::future::Future<Output = Result<ImageEmbeddingResponse, ProviderError>> + WasmCompatSend;

    /// Embed a batch of images from their encoded file bytes.
    fn embed_images(
        &self,
        images: impl IntoIterator<Item = Vec<u8>> + WasmCompatSend,
    ) -> impl std::future::Future<Output = Result<Vec<Embedding>, ProviderError>> + WasmCompatSend
    {
        async { Ok(self.embed_images_response(images).await?.embeddings) }
    }

    /// Embeds one encoded image, returning the last vector or an error if none
    /// is returned.
    fn embed_image(
        &self,
        bytes: &[u8],
    ) -> impl std::future::Future<Output = Result<Embedding, ProviderError>> + WasmCompatSend {
        async move {
            let mut embeddings = self.embed_images(vec![bytes.to_owned()]).await?;
            embeddings.pop().ok_or_else(|| {
                ProviderError::Response(
                    "embedding provider returned an empty response for embed_image".to_string(),
                )
            })
        }
    }
}

/// A document identifier and its vector. Equality compares only the document,
/// not vector values.
#[derive(Clone, Default, Deserialize, Serialize, Debug)]
pub struct Embedding {
    /// The text that was embedded, or a non-sensitive input identifier for
    /// non-text embeddings. Used for debugging and equality.
    pub document: String,
    /// The embedding vector
    pub vec: Vec<f64>,
}

impl PartialEq for Embedding {
    fn eq(&self, other: &Self) -> bool {
        self.document == other.document
    }
}

impl Eq for Embedding {}

#[cfg(test)]
mod provider_response_tests;

/// The media type of an encoded image, sniffed from its magic bytes.
///
/// The image-embedding wires need it twice: once to reject a format the
/// provider does not accept, and once to name the vector's input.
pub fn image_media_type(bytes: &[u8]) -> Option<&'static str> {
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        Some("image/png")
    } else if bytes.starts_with(b"\xff\xd8\xff") {
        Some("image/jpeg")
    } else if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        Some("image/gif")
    } else if bytes.starts_with(b"RIFF") && bytes.get(8..12) == Some(b"WEBP".as_slice()) {
        Some("image/webp")
    } else {
        None
    }
}

/// Identifies image bytes by media type and a URL-safe, unpadded SHA-256 digest,
/// without retaining the image or a reversible encoding. Unknown formats use
/// `application/octet-stream`; this function does not validate provider support.
pub fn image_document(bytes: &[u8]) -> String {
    use base64::Engine as _;
    use sha2::Digest as _;
    let media_type = image_media_type(bytes).unwrap_or("application/octet-stream");
    let digest = sha2::Sha256::digest(bytes);
    format!(
        "{media_type};sha256={}",
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(digest)
    )
}
