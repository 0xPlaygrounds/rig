//! Shared request plumbing for OpenAI-adjacent `/rerank` endpoints.
//!
//! There is no reranking endpoint in the OpenAI API, so "OpenAI-compatible"
//! servers that offer one converged on Jina's shape instead: a JSON body of
//! `{model, query, documents, top_n}` answered with
//! `{model, object:"list", usage:{…}, results:[{index, relevance_score}]}`.
//! `llama-server` serves exactly that on `/rerank`, `/reranking`,
//! `/v1/rerank` and `/v1/reranking` (one handler, four aliases).
//!
//! Rig already had [`RerankModel`](crate::rerank::RerankModel) and exactly one
//! implementation of it — Voyage AI's, written against Voyage's own wire
//! (`top_k`, `return_documents`, `truncation`, results under `data`). This
//! module is the reusable half for the Jina-shaped wire, so the next provider
//! that speaks it declares a slot instead of copying a request builder.

use serde::{Deserialize, Serialize};

use crate::client::Client;
use crate::http_client::HttpClientExt;
use crate::rerank::{NormalizeRerankResponse, RerankError, RerankResponse, RerankResult};
use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};

/// Contract for provider extensions that speak the Jina-shaped rerank wire
/// through [`GenericRerankModel`].
#[doc(hidden)]
pub trait JinaCompatibleRerank: crate::client::Provider {
    /// Provider name used in rerank request and response errors.
    const PROVIDER_NAME: &'static str;

    /// The provider's transport request-id response header, when it has one.
    const REQUEST_ID_HEADER: Option<&'static str> = None;

    /// Most documents the provider accepts in one rerank request.
    const MAX_DOCUMENTS: usize;

    /// Whether the model is sent as a `model` field in the request body.
    const SENDS_MODEL_FIELD: bool = true;

    /// The request path for reranking, resolved against the client base URL.
    fn rerank_path(&self) -> String {
        "/rerank".to_string()
    }
}

#[derive(Debug, Serialize)]
struct JinaRerankRequest<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<&'a str>,
    query: &'a str,
    documents: &'a [String],
    #[serde(skip_serializing_if = "Option::is_none")]
    top_n: Option<usize>,
}

/// One scored document.
///
/// The score key is `relevance_score` on the Jina/OpenAI-shaped path and
/// `score` on the text-embeddings-inference path that the same llama.cpp
/// handler switches to; both are accepted so a server answering either shape
/// decodes rather than silently scoring every document zero.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JinaRerankResult {
    pub index: usize,
    #[serde(alias = "score")]
    pub relevance_score: f64,
    /// Present only on servers that echo the document back. llama.cpp does
    /// not on this path, so this is normally absent.
    #[serde(default, alias = "text")]
    pub document: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct JinaRerankUsage {
    #[serde(default)]
    pub prompt_tokens: u64,
    #[serde(default)]
    pub total_tokens: u64,
}

/// The Jina-shaped rerank wire response: what
/// [`GenericRerankModel::raw_rerank`] returns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JinaRerankResponse {
    #[serde(default)]
    pub model: Option<String>,
    pub results: Vec<JinaRerankResult>,
    #[serde(default)]
    pub usage: Option<JinaRerankUsage>,
}

impl NormalizeRerankResponse for JinaRerankResponse {
    fn normalize(self, provider: &str) -> Result<RerankResponse, RerankError> {
        let usage = self.usage.unwrap_or_default();
        Ok(RerankResponse::new(
            self.results
                .into_iter()
                .map(|result| RerankResult {
                    index: result.index,
                    document: result.document,
                    relevance_score: result.relevance_score,
                })
                .collect(),
            provider,
        )
        // A server that omits `model` still produced a ranking; `None` is
        // the honest report.
        .with_optional_model(self.model)
        .with_usage(crate::completion::Usage {
            input_tokens: usage.prompt_tokens,
            total_tokens: usage.total_tokens,
            ..crate::completion::Usage::new()
        }))
    }
}

/// A rerank model on a Jina-shaped `/rerank` endpoint.
#[derive(Clone)]
pub struct GenericRerankModel<Ext, H = crate::http_client::BoxedHttpClient> {
    client: Client<Ext, H>,
    /// Identifier the request carries in its `model` field.
    pub model: String,
    top_n: Option<usize>,
}

impl<Ext, H> GenericRerankModel<Ext, H> {
    /// Create a rerank model handle.
    pub fn new(client: Client<Ext, H>, model: impl Into<String>) -> Self {
        Self {
            client,
            model: model.into(),
            top_n: None,
        }
    }

    /// Ask the server to return only the `top_n` highest-scoring documents.
    ///
    /// Omitted by default, which the wire reads as "all of them". A value
    /// larger than the document count is not an error on llama.cpp — the
    /// server clamps it to the list length.
    pub fn top_n(mut self, top_n: usize) -> Self {
        self.top_n = Some(top_n);
        self
    }
}

impl<Ext, H> GenericRerankModel<Ext, H>
where
    Client<Ext, H>: HttpClientExt + Clone + WasmCompatSend + WasmCompatSync + 'static,
    Ext: JinaCompatibleRerank + Clone + WasmCompatSend + WasmCompatSync + 'static,
    H: WasmCompatSend + WasmCompatSync,
{
    /// Perform the request and return the provider's native Jina-shaped
    /// response instead of the normalized [`RerankResponse`]. Same request,
    /// transport, parser, and error path as
    /// [`crate::rerank::RerankModel::rerank`].
    pub async fn raw_rerank(
        &self,
        query: &str,
        documents: Vec<String>,
    ) -> Result<JinaRerankResponse, RerankError> {
        self.raw_rerank_with_request_id(query, documents)
            .await
            .map(|(response, _)| response)
    }

    /// [`Self::raw_rerank`] plus the transport request id from the provider's
    /// request-id response header, when it carries one.
    pub async fn raw_rerank_with_request_id(
        &self,
        query: &str,
        documents: Vec<String>,
    ) -> Result<(JinaRerankResponse, Option<String>), RerankError> {
        let body = serde_json::to_vec(&JinaRerankRequest {
            model: Ext::SENDS_MODEL_FIELD.then_some(self.model.as_str()),
            query,
            documents: &documents,
            top_n: self.top_n,
        })?;

        let req = self
            .client
            .post(self.client.ext().rerank_path())?
            .body(body)
            .map_err(|error| RerankError::HttpError(error.into()))?;

        let response = self.client.send(req).await?;
        let (parts, body) = response.into_parts();
        let status = parts.status;
        let provider_request_id =
            super::transcription::request_id_from_headers(&parts.headers, Ext::REQUEST_ID_HEADER);
        let response_body: Vec<u8> = body.await?;
        if !status.is_success() {
            return Err(RerankError::from_http_response(
                status,
                String::from_utf8_lossy(&response_body).into_owned(),
            )
            .with_response_headers(Some(Box::new(parts.headers))));
        }

        // Named rather than bare `?`: a 200 whose body is not a rerank
        // payload is indistinguishable from a serde bug without knowing which
        // server produced it, and this driver is shared.
        let parsed: JinaRerankResponse =
            serde_json::from_slice(&response_body).map_err(|error| {
                RerankError::ResponseError(format!(
                    "{}: rerank response was not a Jina-shaped payload: {error}",
                    Ext::PROVIDER_NAME
                ))
            })?;

        Ok((parsed, provider_request_id))
    }
}

impl<Ext, H> crate::rerank::RerankModel for GenericRerankModel<Ext, H>
where
    Client<Ext, H>: HttpClientExt + Clone + WasmCompatSend + WasmCompatSync + 'static,
    Ext: JinaCompatibleRerank + Clone + WasmCompatSend + WasmCompatSync + 'static,
    H: WasmCompatSend + WasmCompatSync,
{
    fn max_documents(&self) -> usize {
        Ext::MAX_DOCUMENTS
    }

    async fn rerank(
        &self,
        query: &str,
        documents: Vec<String>,
    ) -> Result<RerankResponse, RerankError> {
        crate::telemetry::instrument_modality(
            Ext::PROVIDER_NAME,
            &self.model,
            crate::telemetry::ModalityOperation::Rerank,
            async {
                let (response, provider_request_id) =
                    self.raw_rerank_with_request_id(query, documents).await?;
                let captured = serde_json::to_value(&response)?;
                Ok(response
                    .normalize(Ext::PROVIDER_NAME)?
                    .with_optional_provider_request_id(provider_request_id)
                    .with_raw(captured))
            },
        )
        .await
    }
}

impl<Ext, H> crate::client::ConstructRerankModel<Client<Ext, H>> for GenericRerankModel<Ext, H>
where
    Client<Ext, H>: HttpClientExt + Clone + WasmCompatSend + WasmCompatSync + 'static,
    Ext: JinaCompatibleRerank + Clone + WasmCompatSend + WasmCompatSync + 'static,
    H: WasmCompatSend + WasmCompatSync,
{
    fn construct(client: &Client<Ext, H>, model: String) -> Self {
        Self::new(client.clone(), model)
    }
}
