//! Voyage AI as data: one config struct, two wires.
//!
//! Wires: [`Embeddings`] (`Embedding`) and [`Rerank`] (`Rerank`) — the two
//! rows Voyage serves in this crate. Voyage has one dialect, so its
//! defaults are the config's defaults rather than a table of constants.

use crate::client::env::{self, EnvError};
use crate::driver::{HasEmbedding, HasRerank};
use crate::embeddings::{Embedding as Vector, EmbeddingError};
use crate::operation::{Embedding, EmbeddingCapabilities, Rerank as RerankOp, RerankRequest};
use crate::rerank::{RerankError, RerankResponse, RerankResult};
use crate::wire::{
    Body, Decoder, Encoded, Framing, Mode, Output, Secret, Sink, Wire, WireError, WireEvent,
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

    /// The embedding wire for `model`, at `ndims` dimensions when the caller
    /// named one rather than taking the model's published width.
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

    /// One JSON request, authenticated and framed whole.
    ///
    /// Both Voyage routes are the same request — one JSON object POSTed to
    /// a path under this root, answered by one whole reply — so the two
    /// wires differ only in the path and the object they hand here, which
    /// travels in the order they built it.
    ///
    /// Voyage authenticates both routes, so the credential is read through
    /// [`Secret::require`]: a wire reloaded from a scene or a config file
    /// carries no key by contract, and an error naming `VOYAGE_API_KEY`
    /// beats sending `Bearer ` for a provider 401.
    fn post<E: WireError>(
        &self,
        path: &str,
        body: &serde_json::Map<String, serde_json::Value>,
    ) -> Result<Encoded, E> {
        let request = http::Request::post(format!("{}{path}", self.base_url))
            .header(http::header::CONTENT_TYPE, "application/json")
            .header(
                http::header::AUTHORIZATION,
                format!("Bearer {}", self.api_key.require::<E>(API_KEY_ENV)?),
            )
            .body(Body::Bytes(serde_json::to_vec(body).map_err(E::json)?))
            .map_err(|error| E::transport(error.into()))?;
        Ok(Encoded::new(request, Framing::Whole))
    }
}

/// What both Voyage replies carry besides their payload: the body verbatim,
/// for the fields this crate does not model, and the usage Voyage reports —
/// one count on either route, and every token of an embedding or an
/// ordering is an input token.
fn raw_and_usage<T: Serialize>(
    reply: &T,
    total_tokens: usize,
) -> Result<(serde_json::Value, crate::completion::Usage), serde_json::Error> {
    Ok((
        serde_json::to_value(reply)?,
        crate::completion::Usage {
            input_tokens: Some(total_tokens as u64),
            total_tokens: Some(total_tokens as u64),
            ..Default::default()
        },
    ))
}

/// Carry `key` only when the wire names a value: every option both routes
/// take is a key Voyage supplies its own default for, and a request that
/// omits it is how that default is asked for.
fn insert_some(
    body: &mut serde_json::Map<String, serde_json::Value>,
    key: &str,
    value: Option<impl Into<serde_json::Value>>,
) {
    if let Some(value) = value {
        body.insert(key.to_owned(), value.into());
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

    /// Reject, rather than truncate, an input longer than the model's
    /// context.
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

    fn encode(&self, texts: Vec<String>, _mode: Mode) -> Result<Encoded, EmbeddingError> {
        let mut body = serde_json::Map::new();
        body.insert("model".to_owned(), serde_json::json!(self.model));
        body.insert("input".to_owned(), serde_json::json!(texts));
        insert_some(&mut body, "input_type", self.input_type.as_deref());
        insert_some(&mut body, "truncation", self.truncation);
        insert_some(&mut body, "output_dimension", self.output_dimension);
        self.provider.post("/embeddings", &body)
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
        let (raw, usage) = match raw_and_usage(&reply, reply.usage.total_tokens) {
            Ok(parts) => parts,
            Err(error) => {
                out.push(Err(error.into()));
                return;
            }
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

    /// Reject, rather than truncate, a document longer than the model's
    /// context.
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

    fn encode(&self, request: RerankRequest, _mode: Mode) -> Result<Encoded, RerankError> {
        let mut body = serde_json::Map::new();
        body.insert("query".to_owned(), serde_json::json!(request.query));
        body.insert("documents".to_owned(), serde_json::json!(request.documents));
        body.insert("model".to_owned(), serde_json::json!(self.model));
        insert_some(&mut body, "top_k", self.top_k);
        body.insert(
            "return_documents".to_owned(),
            serde_json::json!(self.return_documents),
        );
        insert_some(&mut body, "truncation", self.truncation);
        self.provider.post("/rerank", &body)
    }

    fn decoder(&self, _mode: Mode) -> Self::Decoder {
        RerankDecoder
    }

    fn capabilities(&self) -> usize {
        MAX_RERANK_DOCUMENTS
    }
}

/// The top-level keys that recognize a `/rerank` reply: `data`, the field
/// every ordering carries, and `message` — a 200 whose body is nothing but
/// Voyage's error envelope is a reply too, and recognizing it here is what
/// makes the ordering's typed decode fail and hands the frame to the
/// envelope classifier.
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

    /// Two shapes on one decoder, composed through the classify layer's own
    /// combinator so no triage verdict is read here: the ordering is the
    /// shape the wire mostly sends and stays the diagnostic when neither
    /// decodes, and the envelope is tried exactly when the ordering's
    /// decode fails.
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
            // A 200 that carried the error envelope instead of an ordering:
            // the provider's body verbatim, and the driver stamps the status
            // it arrived under. Without this the frame read as an unmodeled
            // event, the driver warn-skipped it, and the fold failed with a
            // `RerankError` naming neither.
            RerankReply::Failure(body) => {
                out.push(Err(RerankError::from_provider_body(body)));
                return;
            }
        };
        let (raw, usage) = match raw_and_usage(&reply, reply.usage.total_tokens) {
            Ok(parts) => parts,
            Err(error) => {
                out.push(Err(error.into()));
                return;
            }
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
