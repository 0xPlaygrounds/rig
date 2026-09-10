//! The module defines the [EmbeddingModel] and [ImageEmbeddingModel] traits, which represent
//! embedding models that can generate embeddings for text documents and images.
//!
//! The module also defines the [Embedding] struct, which represents a single document embedding.
//!
//! Finally, the module defines the [EmbeddingError] enum, which represents various errors that
//! can occur during embedding generation or processing.

use crate::{
    completion::{ResponseIdentity, Usage},
    wasm_compat::{WasmCompatSend, WasmCompatSync},
};
use serde::{Deserialize, Serialize};

crate::provider_response::provider_error_enum!(
    EmbeddingError, "embedding" {
    /// URL construction or parsing failed while preparing a provider request.
    #[error("UrlError: {0}")]
    UrlError(#[from] url::ParseError),

    #[cfg(not(target_family = "wasm"))]
    /// Error processing the document for embedding
    #[error("DocumentError: {0}")]
    DocumentError(Box<dyn std::error::Error + Send + Sync + 'static>),

    #[cfg(target_family = "wasm")]
    /// Error processing the document for embedding
    #[error("DocumentError: {0}")]
    DocumentError(Box<dyn std::error::Error + 'static>),
    } {
    /// The provider does not support an embedding request parameter configured on the model.
    #[error("{provider} embeddings do not support the `{parameter}` parameter")]
    UnsupportedParameter {
        /// Provider whose embedding API rejected the parameter.
        provider: &'static str,
        /// Unsupported request parameter.
        parameter: &'static str,
    },

    /// A provider request parameter was configured with a value outside the
    /// provider's supported range.
    #[error("{provider} embeddings require `{parameter}` {requirement}")]
    InvalidParameterValue {
        /// Provider whose embedding API constrains the parameter.
        provider: &'static str,
        /// Request parameter with the invalid value.
        parameter: &'static str,
        /// Concise description of the accepted values.
        requirement: &'static str,
    },

    /// Rig cannot decode the requested provider response encoding.
    #[error("Rig cannot decode {provider} embedding responses encoded as `{encoding_format}`")]
    UnsupportedResponseEncoding {
        /// Provider whose response encoding was requested.
        provider: &'static str,
        /// Response encoding that Rig cannot decode.
        encoding_format: &'static str,
    },

    /// A provider that guarantees embedding usage omitted it from the response.
    #[error("{provider} embedding response omitted required usage")]
    MissingUsage {
        /// Provider whose response omitted usage.
        provider: &'static str,
    },

    /// The provider returned vectors of a width other than the one the caller
    /// declared through
    /// [`embedding_model_with_ndims`](crate::client::EmbeddingsClient::embedding_model_with_ndims).
    ///
    /// Raised only when the width was set *explicitly*: a model handle built
    /// without one reports whatever the provider's own table says and has
    /// nothing to disagree with.
    ///
    /// The failure this prevents is silent and expensive. `ndims()` is what a
    /// vector store sizes its index from, so a model reporting one width while
    /// returning another builds an index that cannot hold its own vectors —
    /// and nothing on the request path can catch it, because the providers
    /// where it happens are exactly the ones that *ignore* the `dimensions`
    /// field instead of rejecting it. Measured on `llama-server`
    /// b10499-6d05498, whose embeddings handler reads no such field at all: a
    /// request for 128 dimensions answers 200 with 1024-wide vectors.
    #[error(
        "{provider} embedding response returned {returned}-dimension vectors, but the model was \
         created with {requested} dimensions; this provider does not resize embeddings"
    )]
    MismatchedDimensions {
        /// Provider whose response disagreed with the declared width.
        provider: &'static str,
        /// Width the caller declared.
        requested: usize,
        /// Width the provider actually returned.
        returned: usize,
    },
    }
);

/// Trait for embedding models that can generate embeddings for documents.
pub trait EmbeddingModel: WasmCompatSend + WasmCompatSync {
    /// The maximum number of documents that can be embedded in a single
    /// request.
    ///
    /// A method rather than an associated constant so the value survives type
    /// erasure: [`EmbeddingModelHandle`](super::EmbeddingModelHandle) captures
    /// it by value at construction.
    fn max_documents(&self) -> usize;

    /// The number of dimensions in the embedding vector.
    fn ndims(&self) -> usize;

    /// Embed multiple text documents in a single request and return the full
    /// normalized response: embeddings, usage, provider, and identity.
    ///
    /// This is the method a provider implements; [`EmbeddingModel::embed_texts`],
    /// [`EmbeddingModel::embed_text`] and [`EmbeddingModel::embed_text_response`]
    /// derive from it. It cannot be the other way round: a default that
    /// forwarded to `embed_texts` would have to invent the provider name.
    ///
    /// Implementations must preserve input order in the returned embeddings.
    fn embed_texts_response(
        &self,
        texts: impl IntoIterator<Item = String> + WasmCompatSend,
    ) -> impl std::future::Future<Output = Result<EmbeddingResponse, EmbeddingError>> + WasmCompatSend;

    /// Embed multiple text documents in a single request.
    ///
    /// The convenience form of [`EmbeddingModel::embed_texts_response`] for
    /// callers who want only the vectors.
    fn embed_texts(
        &self,
        texts: impl IntoIterator<Item = String> + WasmCompatSend,
    ) -> impl std::future::Future<Output = Result<Vec<Embedding>, EmbeddingError>> + WasmCompatSend
    {
        async { Ok(self.embed_texts_response(texts).await?.embeddings) }
    }

    /// Embed a single text document.
    fn embed_text(
        &self,
        text: &str,
    ) -> impl std::future::Future<Output = Result<Embedding, EmbeddingError>> + WasmCompatSend {
        async {
            let mut embeddings = self.embed_texts(vec![text.to_string()]).await?;
            embeddings.pop().ok_or_else(|| {
                EmbeddingError::ResponseError(
                    "embedding provider returned an empty response for embed_text".to_string(),
                )
            })
        }
    }

    /// Embed a single text document and return the full normalized response.
    fn embed_text_response(
        &self,
        text: &str,
    ) -> impl std::future::Future<Output = Result<EmbeddingResponse, EmbeddingError>> + WasmCompatSend
    {
        async {
            let response = self.embed_texts_response(vec![text.to_string()]).await?;
            if response.embeddings.is_empty() {
                return Err(EmbeddingError::ResponseError(
                    "embedding provider returned an empty response for embed_text_response"
                        .to_string(),
                ));
            }
            Ok(response)
        }
    }
}

/// The normalized embedding response: the embeddings plus the metadata every
/// provider can report, attributed to the provider that produced it.
///
/// Concrete and provider-neutral, so it survives type erasure through a
/// handle unchanged. The provider's own payload stays reachable through a
/// model's inherent `raw_embed_texts` method, which performs the same request and
/// returns the provider's native type, and through [`Self::raw`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    /// The embeddings returned by the provider, one per input text, in input order.
    pub embeddings: Vec<Embedding>,
    /// Token usage for this request. Zero-valued when the provider reported
    /// none — the sentinel [`Usage`] documents.
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
    /// The provider's transport-level request identifier, taken from the HTTP
    /// response headers — the id provider support asks for. `None` means the
    /// provider reported none; that is a documented outcome, never an error.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_request_id: Option<String>,
    /// The provider's own response for this call: the value the model's
    /// inherent `raw_embed_texts` would have returned, serialized. Every provider
    /// seam populates it. `Value::Null` means the value was built without a
    /// provider behind it (a test double), never that the provider sent
    /// nothing.
    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub raw: serde_json::Value,
}

impl EmbeddingResponse {
    /// Create a response from its required parts; optional metadata starts
    /// unset and is filled in with the `with_*` helpers.
    pub fn new(embeddings: Vec<Embedding>, provider: impl Into<String>) -> Self {
        Self {
            embeddings,
            usage: Usage::new(),
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

/// Convert a provider's own embedding payload into the normalized [`EmbeddingResponse`].
///
/// The provider descriptor name is an *input*, never something the conversion
/// knows — several providers share one wire shape, and a hardcoded name would
/// mislabel every provider but one. A trait rather than `TryFrom<(&str, T)>`
/// so that out-of-tree provider extensions can implement it on their own
/// response type without tripping the orphan rule.
pub trait NormalizeEmbeddingResponse {
    /// Normalize this payload, attributing it to `provider`. `documents` are
    /// the inputs in request order, for [`Embedding::document`].
    fn normalize(
        self,
        provider: &str,
        documents: Vec<String>,
    ) -> Result<EmbeddingResponse, EmbeddingError>;
}

/// The normalized image embedding response: the embeddings plus the metadata every
/// provider can report, attributed to the provider that produced it.
///
/// Concrete and provider-neutral, so it survives type erasure through a
/// handle unchanged. The provider's own payload stays reachable through a
/// model's inherent `raw_embed_images` method, which performs the same request and
/// returns the provider's native type, and through [`Self::raw`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageEmbeddingResponse {
    /// The embeddings returned by the provider, one per input image, in input order.
    pub embeddings: Vec<Embedding>,
    /// Token usage for this request. Zero-valued when the provider reported
    /// none — the sentinel [`Usage`] documents.
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
    /// The provider's transport-level request identifier, taken from the HTTP
    /// response headers — the id provider support asks for. `None` means the
    /// provider reported none; that is a documented outcome, never an error.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_request_id: Option<String>,
    /// The provider's own response for this call: the value the model's
    /// inherent `raw_embed_images` would have returned, serialized. Every provider
    /// seam populates it. `Value::Null` means the value was built without a
    /// provider behind it (a test double), never that the provider sent
    /// nothing.
    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub raw: serde_json::Value,
}

impl ImageEmbeddingResponse {
    /// Create a response from its required parts; optional metadata starts
    /// unset and is filled in with the `with_*` helpers.
    pub fn new(embeddings: Vec<Embedding>, provider: impl Into<String>) -> Self {
        Self {
            embeddings,
            usage: Usage::new(),
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
    ///
    /// A method rather than an associated constant so the value survives type
    /// erasure: [`ImageEmbeddingModelHandle`](super::ImageEmbeddingModelHandle)
    /// captures it by value at construction.
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
    ) -> impl std::future::Future<Output = Result<ImageEmbeddingResponse, EmbeddingError>> + WasmCompatSend;

    /// Embed a batch of images from their encoded file bytes.
    fn embed_images(
        &self,
        images: impl IntoIterator<Item = Vec<u8>> + WasmCompatSend,
    ) -> impl std::future::Future<Output = Result<Vec<Embedding>, EmbeddingError>> + WasmCompatSend
    {
        async { Ok(self.embed_images_response(images).await?.embeddings) }
    }

    /// Embed a single image from its encoded file bytes.
    fn embed_image(
        &self,
        bytes: &[u8],
    ) -> impl std::future::Future<Output = Result<Embedding, EmbeddingError>> + WasmCompatSend {
        async move {
            let mut embeddings = self.embed_images(vec![bytes.to_owned()]).await?;
            embeddings.pop().ok_or_else(|| {
                EmbeddingError::ResponseError(
                    "embedding provider returned an empty response for embed_image".to_string(),
                )
            })
        }
    }
}

/// Struct that holds a single document and its embedding.
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
