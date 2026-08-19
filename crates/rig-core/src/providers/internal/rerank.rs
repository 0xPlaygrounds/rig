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
use crate::http_client::{self, HttpClientExt};
use crate::rerank::{RerankError, RerankResponse, RerankResult};
use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};

/// Contract for provider extensions that speak the Jina-shaped rerank wire
/// through [`GenericRerankModel`].
pub(crate) trait JinaCompatibleRerank: crate::client::Provider {
    /// Provider name used in rerank request and response errors.
    const PROVIDER_NAME: &'static str;

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
#[derive(Debug, Deserialize)]
struct JinaRerankResult {
    index: usize,
    #[serde(alias = "score")]
    relevance_score: f64,
    /// Present only on servers that echo the document back. llama.cpp does
    /// not on this path, so this is normally absent.
    #[serde(default, alias = "text")]
    document: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
struct JinaRerankUsage {
    #[serde(default)]
    prompt_tokens: u64,
    #[serde(default)]
    total_tokens: u64,
}

#[derive(Debug, Deserialize)]
struct JinaRerankResponse {
    #[serde(default)]
    model: Option<String>,
    results: Vec<JinaRerankResult>,
    #[serde(default)]
    usage: Option<JinaRerankUsage>,
}

/// A rerank model on a Jina-shaped `/rerank` endpoint.
#[derive(Clone)]
pub struct GenericRerankModel<Ext, H = reqwest::Client> {
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

impl<Ext, H> crate::rerank::RerankModel for GenericRerankModel<Ext, H>
where
    Client<Ext, H>: HttpClientExt + Clone + WasmCompatSend + WasmCompatSync + 'static,
    Ext: JinaCompatibleRerank + Clone + WasmCompatSend + WasmCompatSync + 'static,
    H: WasmCompatSend + WasmCompatSync,
{
    const MAX_DOCUMENTS: usize = Ext::MAX_DOCUMENTS;

    async fn rerank(
        &self,
        query: &str,
        documents: Vec<String>,
    ) -> Result<RerankResponse, RerankError> {
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
        let status = response.status();
        if !status.is_success() {
            let text = http_client::text(response).await?;
            return Err(RerankError::from_http_response(status, text));
        }

        let response_body: Vec<u8> = response.into_body().await?;
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

        let usage = parsed.usage.unwrap_or_default();
        Ok(RerankResponse {
            results: parsed
                .results
                .into_iter()
                .map(|result| RerankResult {
                    index: result.index,
                    document: result.document,
                    relevance_score: result.relevance_score,
                })
                .collect(),
            // A server that omits `model` still produced a ranking; report
            // the identifier the request asked for rather than failing.
            model: parsed.model.unwrap_or_else(|| self.model.clone()),
            usage: crate::completion::Usage {
                input_tokens: usage.prompt_tokens,
                output_tokens: 0,
                total_tokens: usage.total_tokens,
                cached_input_tokens: 0,
                cache_creation_input_tokens: 0,
                reasoning_tokens: 0,
                tool_use_prompt_tokens: 0,
            },
        })
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
