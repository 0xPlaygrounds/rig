//! Voyage AI configuration and embedding and reranking endpoint wires.
//!
//! ```no_run
//! use rig_core::providers::voyageai::{VoyageAi, VOYAGE_3_5};
//! let wire = VoyageAi::from_env()?.embedding(VOYAGE_3_5);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::client::env::{self, EnvError};
use crate::embeddings::Embedding as Vector;
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
    EmbeddingResponse as VoyageEmbeddingResponse, RerankApiResponse, RerankErrorEnvelope,
    VOYAGEAI_API_BASE_URL, model_dimensions_from_identifier,
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

    /// The embedding wire for `model`, at the model's known width.
    pub(crate) fn embedding(&self, model: impl Into<String>) -> Embeddings {
        Embeddings {
            provider: self.clone(),
            model: model.into(),
            ndims: None,
            input_type: None,
            truncation: None,
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
/// the model's default width, no retrieval prompt, and truncation enabled.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Embeddings {
    /// The provider this wire speaks to.
    pub provider: VoyageAiConfig,
    /// The model to address.
    pub model: String,
    /// The width the caller asked for with [`Self::with_ndims`], sent as
    /// `output_dimension`. `None` takes the model's known width.
    pub ndims: Option<usize>,
    /// Prepends a retrieval prompt to the input text: `"document"` when
    /// embedding stored chunks, `"query"` when embedding search queries.
    /// Embeddings produced with and without it are compatible.
    pub input_type: Option<String>,
    /// Whether to truncate inputs longer than the model's context. Voyage's
    /// server default is `true`.
    pub truncation: Option<bool>,
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

    /// Ask for `ndims`-wide vectors (`output_dimension`). A reply of
    /// another width is [`ProviderError::MismatchedDimensions`].
    pub fn with_ndims(mut self, ndims: usize) -> Self {
        self.ndims = Some(ndims);
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
            .capabilities(
                Capabilities::embedding(
                    MAX_DOCUMENTS,
                    self.ndims
                        .or_else(|| model_dimensions_from_identifier(&self.model))
                        .unwrap_or_default(),
                )
                .declaring(self.ndims),
            )
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
        if let Some(ndims) = self.ndims {
            body.insert("output_dimension".to_owned(), serde_json::json!(ndims));
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
        // Voyage reports one count; every token of an embedding is input.
        let usage = crate::completion::Usage {
            input_tokens: Some(reply.usage.total_tokens as u64),
            total_tokens: Some(reply.usage.total_tokens as u64),
            ..Default::default()
        };
        // The vectors only; the operation's fold pairs them with the texts
        // that were sent, which `/embeddings` does not echo back.
        let vectors = reply
            .data
            .into_iter()
            .map(|embedding| Vector {
                document: String::new(),
                vec: embedding.embedding,
            })
            .collect();
        Ok(out.end(crate::embeddings::EmbeddingResponse {
            model: Some(reply.model),
            usage,
            ..crate::embeddings::EmbeddingResponse::new(vectors)
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

/// Ranking and error-envelope markers. Recognizing `message` permits fallback
/// decoding of provider errors returned with successful HTTP status.
const RERANK_REPLY_MARKERS: &[&str] = &["data", "message"];

/// The key that recognizes the error envelope on its own.
const RERANK_ERROR_MARKERS: &[&str] = &["message"];

/// One `/rerank` reply: the ordering, or the error envelope Voyage can
/// answer a **200** with instead.
pub enum RerankReply {
    /// The ordering Voyage returned.
    Reply(RerankApiResponse),
    /// The provider's error envelope, verbatim.
    Failure(String),
}

/// Decodes one `/rerank` reply.
pub struct RerankDecoder;

impl<'id> Decoder<'id, RerankOp> for RerankDecoder {
    type Event = RerankReply;

    /// Decode a ranking, then an error envelope if ranking decoding fails.
    /// Retain the ranking diagnostic when neither shape decodes.
    fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event> {
        let data = frame.as_str();
        crate::providers::internal::wire::classify_or(
            &data,
            |data| {
                crate::providers::internal::wire::classify_marker_keyed_frame::<RerankApiResponse>(
                    data,
                    RERANK_REPLY_MARKERS,
                )
                .map(RerankReply::Reply)
            },
            |data| {
                crate::providers::internal::wire::classify_marker_keyed_frame::<RerankErrorEnvelope>(
                    data,
                    RERANK_ERROR_MARKERS,
                )
                .map(|_| RerankReply::Failure(data.to_owned()))
            },
        )
    }

    fn decode(
        &mut self,
        reply: Self::Event,
        out: Out<'id, RerankOp>,
    ) -> Result<Flow, ProviderError> {
        let reply = match reply {
            RerankReply::Reply(reply) => reply,
            // Preserve the provider body; the driver adds the actual HTTP status.
            RerankReply::Failure(body) => {
                return Err(ProviderError::from_provider_body(body));
            }
        };
        // Voyage reports one count; every token of a rerank is input.
        let usage = crate::completion::Usage {
            input_tokens: Some(reply.usage.total_tokens as u64),
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
