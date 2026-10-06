//! Voyage AI configuration and embedding and reranking endpoint wires.
//!
//! ```no_run
//! use rig_core::providers::voyageai::{VoyageAi, VOYAGE_3_5};
//! let wire = VoyageAi::from_env()?.embedding(VOYAGE_3_5, None);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::client::env::{self, EnvError};
use crate::error::EncodeError;
use crate::error::ProviderError;
use crate::operation::{Embedding, Rerank as RerankOp, RerankRequest};
use crate::rerank::{RerankResponse, RerankResult};
use crate::wire::Flow;
use crate::wire::{
    Body, Capabilities, Decoder, Descriptor, Encoded, Framing, Mode, Out, Secret, Wire, WireEvent,
    WireFrame,
};
use serde::{Deserialize, Serialize};

use super::{
    EmbeddingResponse as VoyageEmbeddingResponse, RerankApiResponse, VOYAGEAI_API_BASE_URL,
    model_dimensions_from_identifier,
};

/// The provider descriptor name, as records and telemetry spell it.
const PROVIDER_NAME: &str = "voyageai";

/// The environment variable carrying the API key.
const API_KEY_ENV: &str = "VOYAGE_API_KEY";

/// The most texts `POST /embeddings` accepts in one call.
const MAX_DOCUMENTS: usize = 1024;

/// The most documents `POST /rerank` orders in one call.
const MAX_RERANK_DOCUMENTS: usize = 1000;

/// The settings of a Voyage AI provider: serializable, and the credential
/// is never serialized. [`connect`](Self::connect) puts it on a transport as
/// a [`VoyageAi`](super::VoyageAi) client.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VoyageAiConfig {
    /// The API key, sent as `Authorization: Bearer`.
    pub api_key: Secret,
    /// The API root, without a trailing slash.
    pub base_url: String,
}

impl VoyageAiConfig {
    /// Voyage AI with default settings.
    pub fn new(api_key: impl Into<Secret>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: VOYAGEAI_API_BASE_URL.to_owned(),
        }
    }

    /// Voyage AI from `VOYAGE_API_KEY`.
    pub fn from_env() -> Result<Self, EnvError> {
        Ok(Self::new(env::required(API_KEY_ENV)?))
    }

    /// Point the wires at another API root.
    pub fn with_base_url(mut self, base_url: impl AsRef<str>) -> Self {
        self.base_url = base_url.as_ref().trim_end_matches('/').to_owned();
        self
    }

    /// Build an embedding wire reporting `ndims`, or the known model width,
    /// or zero if unknown. This does not send an output-dimension override;
    /// use [`Embeddings::with_output_dimension`] for that.
    pub(crate) fn embedding(&self, model: impl Into<String>, ndims: Option<usize>) -> Embeddings {
        let model = model.into();
        let ndims = ndims
            .or_else(|| model_dimensions_from_identifier(&model))
            .unwrap_or_default();
        Embeddings {
            provider: self.clone(),
            model,
            ndims,
            input_type: None,
            truncation: None,
            output_dimension: None,
        }
    }

    /// The rerank wire for `model`.
    pub(crate) fn rerank(&self, model: impl Into<String>) -> Rerank {
        Rerank {
            provider: self.clone(),
            model: model.into(),
            top_k: None,
            return_documents: false,
            truncation: None,
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

/// The embedding wire: `POST /embeddings`.
///
/// Every option defaults to `None`, which is Voyage's own server default:
/// no retrieval prompt, truncation enabled, and the model's default output
/// dimension.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Embeddings {
    /// The provider this wire speaks to.
    pub provider: VoyageAiConfig,
    /// The model to address.
    pub model: String,
    /// The width this wire reports, from the caller or the model's published
    /// dimensions. `0` means neither named one.
    pub ndims: usize,
    /// Prepends a retrieval prompt to the input text: `"document"` when
    /// embedding stored chunks, `"query"` when embedding search queries.
    /// Embeddings produced with and without it are compatible.
    pub input_type: Option<String>,
    /// Whether to truncate inputs longer than the model's context. Voyage's
    /// server default is `true`.
    pub truncation: Option<bool>,
    /// Dimensionality of the returned vectors, when overriding the model's
    /// default output dimension.
    pub output_dimension: Option<usize>,
}

impl Embeddings {
    /// Prepend Voyage's retrieval prompt for this input's role.
    pub fn with_input_type(mut self, input_type: impl Into<String>) -> Self {
        self.input_type = Some(input_type.into());
        self
    }

    /// Set whether overlong inputs are truncated (`true`) or rejected (`false`).
    pub fn with_truncation(mut self, truncation: bool) -> Self {
        self.truncation = Some(truncation);
        self
    }

    /// Ask Voyage for vectors of this width.
    ///
    /// This is the width the request carries; [`Self::ndims`] is the width
    /// the wire reports to a vector store, so they are set together.
    pub fn with_output_dimension(mut self, output_dimension: usize) -> Self {
        self.output_dimension = Some(output_dimension);
        self.ndims = output_dimension;
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
        let mut body = serde_json::Map::new();
        body.insert("model".to_owned(), serde_json::json!(self.model));
        body.insert("input".to_owned(), serde_json::json!(texts));
        if let Some(input_type) = &self.input_type {
            body.insert("input_type".to_owned(), serde_json::json!(input_type));
        }
        if let Some(truncation) = self.truncation {
            body.insert("truncation".to_owned(), serde_json::json!(truncation));
        }
        if let Some(output_dimension) = self.output_dimension {
            body.insert(
                "output_dimension".to_owned(),
                serde_json::json!(output_dimension),
            );
        }
        let request = self
            .provider
            .post("/embeddings")
            .body(Body::Bytes(serde_json::to_vec(&body)?))?;
        Ok(Encoded::new(request, Framing::Whole))
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        EmbeddingsDecoder
    }
}

/// Decodes one `/embeddings` reply.
pub struct EmbeddingsDecoder;

impl<'id> Decoder<'id, Embedding> for EmbeddingsDecoder {
    type Event = VoyageEmbeddingResponse;

    fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event> {
        crate::providers::internal::wire::classify_marker_keyed_frame(&frame.as_str(), &["data"])
    }

    fn decode(
        &mut self,
        reply: Self::Event,
        out: Out<'id, Embedding>,
    ) -> Result<Flow, ProviderError> {
        // Voyage reports one count: an embedding bills input only, so output is
        // zero.
        let usage = crate::completion::Usage {
            input_tokens: Some(reply.usage.total_tokens as u64),
            output_tokens: Some(0),
            total_tokens: Some(reply.usage.total_tokens as u64),
            ..Default::default()
        };
        let vectors = reply.data.into_iter().map(|embedding| embedding.embedding);
        Ok(out.end(crate::embeddings::EmbeddingResponse {
            model: Some(reply.model),
            usage,
            ..crate::embeddings::EmbeddingResponse::from_vectors(vectors)
        }))
    }
}

/// The rerank wire: `POST /rerank`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Rerank {
    /// The provider this wire speaks to.
    pub provider: VoyageAiConfig,
    /// The model to address.
    pub model: String,
    /// Return only the `top_k` most relevant documents. `None` returns them
    /// all.
    pub top_k: Option<usize>,
    /// Whether the reply echoes each document's text back.
    pub return_documents: bool,
    /// Whether to truncate documents longer than the model's context.
    /// Voyage's server default is `true`.
    pub truncation: Option<bool>,
}

impl Rerank {
    /// Order only the `top_k` most relevant documents.
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = Some(top_k);
        self
    }

    /// Ask Voyage to echo each ordered document's text back.
    pub fn with_return_documents(mut self, return_documents: bool) -> Self {
        self.return_documents = return_documents;
        self
    }

    /// Set whether overlong documents are truncated (`true`) or rejected (`false`).
    pub fn with_truncation(mut self, truncation: bool) -> Self {
        self.truncation = Some(truncation);
        self
    }
}

impl Wire for Rerank {
    type Op = RerankOp;
    type Payload = crate::wire::Encoded;
    type Frame = crate::wire::WireFrame;
    type Decoder<'id> = RerankDecoder;

    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new(PROVIDER_NAME)
            .model(self.model.as_str())
            .capabilities(Capabilities::rerank(MAX_RERANK_DOCUMENTS))
    }

    fn encode(&self, request: RerankRequest, _mode: Mode) -> Result<Encoded, EncodeError> {
        let mut body = serde_json::Map::new();
        body.insert("query".to_owned(), serde_json::json!(request.query));
        body.insert("documents".to_owned(), serde_json::json!(request.documents));
        body.insert("model".to_owned(), serde_json::json!(self.model));
        if let Some(top_k) = self.top_k {
            body.insert("top_k".to_owned(), serde_json::json!(top_k));
        }
        body.insert(
            "return_documents".to_owned(),
            serde_json::json!(self.return_documents),
        );
        if let Some(truncation) = self.truncation {
            body.insert("truncation".to_owned(), serde_json::json!(truncation));
        }
        let request = self
            .provider
            .post("/rerank")
            .body(Body::Bytes(serde_json::to_vec(&body)?))?;
        Ok(Encoded::new(request, Framing::Whole))
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        RerankDecoder
    }
}

/// Decodes one `/rerank` reply.
pub struct RerankDecoder;

impl<'id> Decoder<'id, RerankOp> for RerankDecoder {
    /// The ordering, or the error envelope Voyage can answer a **200** with.
    type Event = Result<RerankApiResponse, String>;

    fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event> {
        crate::providers::internal::wire::classify_reply_or_message_envelope(
            &frame.as_str(),
            "data",
        )
    }

    fn decode(
        &mut self,
        reply: Self::Event,
        out: Out<'id, RerankOp>,
    ) -> Result<Flow, ProviderError> {
        let reply = reply.map_err(ProviderError::from_provider_body)?;
        // Voyage reports one count: a rerank bills input only, so output is
        // zero.
        let usage = crate::completion::Usage {
            input_tokens: Some(reply.usage.total_tokens as u64),
            output_tokens: Some(0),
            total_tokens: Some(reply.usage.total_tokens as u64),
            ..Default::default()
        };
        let results = reply
            .data
            .into_iter()
            .map(|result| RerankResult {
                index: result.index,
                document: result.document,
                relevance_score: result.relevance_score,
            })
            .collect();
        Ok(out.end(RerankResponse {
            model: Some(reply.model),
            usage,
            ..RerankResponse::new(results)
        }))
    }
}

#[cfg(test)]
mod tests;
