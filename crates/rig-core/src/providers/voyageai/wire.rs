//! Voyage AI configuration and embedding and reranking endpoint wires.
//!
//! ```no_run
//! use rig_core::providers::voyageai::{VoyageAi, VOYAGE_3_5};
//! let wire = VoyageAi::from_env()?.embeddings(VOYAGE_3_5, None);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::client::env::{self, EnvError};
use crate::driver::{HasEmbedding, HasRerank};
use crate::embeddings::Embedding as Vector;
use crate::error::EncodeError;
use crate::error::ProviderError;
use crate::operation::{Embedding, EmbeddingCapabilities, Rerank as RerankOp, RerankRequest};
use crate::rerank::{RerankResponse, RerankResult};
use crate::wire::{
    Body, Decoder, Encoded, Framing, Mode, Output, Secret, Sink, Wire, WireEvent, WireFrame,
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

/// The shared configuration of a Voyage AI provider.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VoyageAi {
    /// The API key, sent as `Authorization: Bearer`.
    pub api_key: Secret,
    /// The API root, without a trailing slash.
    pub base_url: String,
}

impl VoyageAi {
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
    pub fn embeddings(&self, model: impl Into<String>, ndims: Option<usize>) -> Embeddings {
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
    pub fn rerank(&self, model: impl Into<String>) -> Rerank {
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
    pub provider: VoyageAi,
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
    type Decoder = EmbeddingsDecoder;

    fn name(&self) -> &str {
        PROVIDER_NAME
    }

    fn model(&self) -> Option<&str> {
        Some(&self.model)
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

    fn decoder(&self, _mode: Mode) -> Self::Decoder {
        EmbeddingsDecoder
    }

    fn capabilities(&self) -> EmbeddingCapabilities {
        EmbeddingCapabilities::new(MAX_DOCUMENTS, self.ndims)
    }
}

/// Decodes one `/embeddings` reply.
pub struct EmbeddingsDecoder;

impl Decoder<Embedding> for EmbeddingsDecoder {
    type Event = VoyageEmbeddingResponse;

    fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event> {
        crate::providers::internal::wire::classify_marker_keyed_frame(&frame.as_str(), &["data"])
    }

    fn interpret(&mut self, reply: Self::Event, out: &mut Output<Embedding>) {
        let raw = match serde_json::to_value(&reply) {
            Ok(raw) => raw,
            Err(error) => {
                out.push(Err(error.into()));
                return;
            }
        };
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
        out.push(Ok(crate::embeddings::EmbeddingResponse::new(
            vectors,
            PROVIDER_NAME,
        )
        .with_model(reply.model)
        .with_usage(usage)
        .with_raw(raw)));
    }
}

/// The rerank wire: `POST /rerank`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Rerank {
    /// The provider this wire speaks to.
    pub provider: VoyageAi,
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
    type Decoder = RerankDecoder;

    fn name(&self) -> &str {
        PROVIDER_NAME
    }

    fn model(&self) -> Option<&str> {
        Some(&self.model)
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

    fn decoder(&self, _mode: Mode) -> Self::Decoder {
        RerankDecoder
    }

    fn capabilities(&self) -> usize {
        MAX_RERANK_DOCUMENTS
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

impl Decoder<RerankOp> for RerankDecoder {
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

    fn interpret(&mut self, reply: Self::Event, out: &mut Output<RerankOp>) {
        let reply = match reply {
            RerankReply::Reply(reply) => reply,
            // Preserve the provider body; the driver adds the actual HTTP status.
            RerankReply::Failure(body) => {
                out.push(Err(ProviderError::from_provider_body(body)));
                return;
            }
        };
        let raw = match serde_json::to_value(&reply) {
            Ok(raw) => raw,
            Err(error) => {
                out.push(Err(error.into()));
                return;
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
        out.push(Ok(RerankResponse::new(results, PROVIDER_NAME)
            .with_model(reply.model)
            .with_usage(usage)
            .with_raw(raw)));
    }
}

impl HasEmbedding for VoyageAi {
    type Wire = Embeddings;

    fn embedding(&self, model: impl Into<String>, ndims: Option<usize>) -> Embeddings {
        self.embeddings(model, ndims)
    }
}

impl HasRerank for VoyageAi {
    type Wire = Rerank;

    fn rerank(&self, model: impl Into<String>) -> Rerank {
        VoyageAi::rerank(self, model)
    }
}

#[cfg(test)]
mod tests;
