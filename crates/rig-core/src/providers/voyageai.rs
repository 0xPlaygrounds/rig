//! Voyage AI (embeddings + rerank) integration.
//!
//! # Example
//! ```no_run
//! use rig_core::providers::voyageai;
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let cfg = voyageai::functions::EmbeddingConfig::from_env(voyageai::VOYAGE_3_5)?;
//! let rt = rig_core::http_runtime::HttpRuntime::new();
//! let response = voyageai::functions::embed(&cfg, &rt, vec!["hello".to_string()]).await?;
//! # Ok(())
//! # }
//! ```
use crate::embeddings;
use crate::embeddings::EmbeddingError;
use serde::Deserialize;
use serde_json::json;

// ================================================================
// Voyage AI Embedding API
// ================================================================

/// `voyage-3-large` embedding model (Voyage AI)
pub const VOYAGE_3_LARGE: &str = "voyage-3-large";
/// `voyage-3.5` embedding model (Voyage AI)
pub const VOYAGE_3_5: &str = "voyage-3.5";
/// `voyage-3.5-lite` embedding model (Voyage AI)
pub const VOYAGE_3_5_LITE: &str = "voyage.3-5.lite";
/// `voyage-code-3` embedding model (Voyage AI)
pub const VOYAGE_CODE_3: &str = "voyage-code-3";
/// `voyage-finance-2` embedding model (Voyage AI)
pub const VOYAGE_FINANCE_2: &str = "voyage-finance-2";
/// `voyage-law-2` embedding model (Voyage AI)
pub const VOYAGE_LAW_2: &str = "voyage-law-2";
/// `voyage-code-2` embedding model (Voyage AI)
pub const VOYAGE_CODE_2: &str = "voyage-code-2";

pub fn model_dimensions_from_identifier(model_identifier: &str) -> Option<usize> {
    match model_identifier {
        "voyage-code-2" => Some(1536),
        "voyage-3-large" | "voyage-3.5" | "voyage.3-5.lite" | "voyage-code-3"
        | "voyage-finance-2" | "voyage-law-2" => Some(1024),
        _ => None,
    }
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingResponse {
    pub object: String,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: Usage,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Usage {
    pub total_tokens: usize,
}

#[derive(Debug, Deserialize)]
pub struct ApiErrorResponse {
    pub(crate) message: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub(crate) enum ApiResponse<T> {
    Ok(T),
    Err(ApiErrorResponse),
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingData {
    pub object: String,
    pub embedding: Vec<f64>,
    pub index: usize,
}

/// Build the serialized `/embeddings` request body. Pure; used by
/// [`functions::embed`].
pub(crate) fn build_embedding_body(
    model: &str,
    texts: &[String],
) -> Result<Vec<u8>, EmbeddingError> {
    Ok(serde_json::to_vec(&json!({
        "model": model,
        "input": texts,
    }))?)
}

/// Parse an `/embeddings` response into the normalized
/// [`embeddings::EmbeddingResponse`], zipping vectors back onto
/// `documents`. Pure; used by [`functions::embed`].
pub(crate) fn parse_embedding_response(
    status: http::StatusCode,
    body: &str,
    documents: Vec<String>,
) -> Result<embeddings::EmbeddingResponse, EmbeddingError> {
    if !status.is_success() {
        return Err(EmbeddingError::from_http_response(status, body.to_string()));
    }
    match serde_json::from_str::<ApiResponse<EmbeddingResponse>>(body)? {
        ApiResponse::Ok(response) => {
            tracing::info!(target: "rig",
                "VoyageAI embedding token usage: {}",
                response.usage.total_tokens
            );

            if response.data.len() != documents.len() {
                return Err(EmbeddingError::ResponseError(
                    "Response data length does not match input length".into(),
                ));
            }

            let usage = crate::completion::Usage {
                input_tokens: response.usage.total_tokens as u64,
                output_tokens: 0,
                total_tokens: response.usage.total_tokens as u64,
                cached_input_tokens: 0,
                cache_creation_input_tokens: 0,
                tool_use_prompt_tokens: 0,
                reasoning_tokens: 0,
            };

            let embeddings = response
                .data
                .into_iter()
                .zip(documents)
                .map(|(embedding, document)| embeddings::Embedding {
                    document,
                    vec: embedding.embedding,
                })
                .collect();

            Ok(embeddings::EmbeddingResponse { embeddings, usage })
        }
        ApiResponse::Err(err) => {
            tracing::warn!(message = %err.message, "provider returned an error response");
            Err(EmbeddingError::from_http_response(status, body.to_string()))
        }
    }
}

pub mod functions {
    //! Voyage AI embeddings as config + pure functions.
    //!
    //! Voyage AI is an embeddings/rerank provider with no completion
    //! surface, so unlike its siblings this `functions` module carries only
    //! the embedding face: a serde [`EmbeddingConfig`], a [`DESCRIPTOR`]
    //! capability sheet, pure [`build_embedding_request`] /
    //! [`parse_embedding_response`] free functions, and the async
    //! [`embed`]/[`embed_batches`] wrappers over
    //! [`HttpRuntime`].

    use http::header::{AUTHORIZATION, CONTENT_TYPE};
    use serde::{Deserialize, Serialize};

    use crate::embeddings::EmbeddingError;
    use crate::http_runtime::HttpRuntime;
    use crate::providers::descriptor::{
        ApiKeyLocation, ConfigError, ProviderDescriptor, required_env_var,
    };

    /// Default Voyage AI API base URL.
    pub const DEFAULT_BASE_URL: &str = "https://api.voyageai.com/v1";

    /// Voyage AI's capability sheet (embeddings only; the completion flags
    /// stay at their `named` defaults because there is no chat surface).
    pub const DESCRIPTOR: ProviderDescriptor =
        ProviderDescriptor::named("voyageai").with_max_embedding_documents(1024);

    /// Plain-data Voyage AI embeddings configuration.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    #[non_exhaustive]
    pub struct EmbeddingConfig {
        /// API base URL (defaults to [`DEFAULT_BASE_URL`]).
        pub base_url: String,
        /// Credential location.
        pub api_key: ApiKeyLocation,
        /// Embedding model identifier requests are built for.
        pub model: String,
        /// Dimensionality of the vectors this model returns.
        ///
        /// The data form of the deleted `EmbeddingModel::ndims()`, which the
        /// classic model took at construction
        /// (`Client::embedding_model_with_ndims`) and reported to callers
        /// sizing a vector-store index. Voyage AI's `/embeddings` request has
        /// no dimensionality parameter, so — exactly as before — this never
        /// reaches the wire; `build_embedding_body`
        /// sends only `model` and `input`.
        ///
        /// [`new`](Self::new) seeds it from
        /// [`model_dimensions_from_identifier`](super::model_dimensions_from_identifier),
        /// the same lookup the classic `make` used for a known model.
        pub ndims: Option<usize>,
        /// Extra headers attached to every request.
        pub extra_headers: Vec<(String, String)>,
    }

    impl EmbeddingConfig {
        /// Config for `model` reading `VOYAGE_API_KEY` from the environment.
        pub fn new(model: impl Into<String>) -> Self {
            let model = model.into();
            let ndims = super::model_dimensions_from_identifier(&model);
            Self {
                base_url: DEFAULT_BASE_URL.to_string(),
                api_key: ApiKeyLocation::Env("VOYAGE_API_KEY".to_string()),
                model,
                ndims,
                extra_headers: Vec::new(),
            }
        }

        /// Declare the dimensionality of the vectors this model returns.
        ///
        /// The replacement for `Client::embedding_model_with_ndims`, for
        /// models the built-in lookup does not know.
        pub fn with_ndims(mut self, ndims: usize) -> Self {
            self.ndims = Some(ndims);
            self
        }

        /// Config for `model` built from the process environment.
        ///
        /// Reads `VOYAGE_API_KEY` (required) — the same variable the deleted
        /// `voyageai::Client::from_env` read. There is no base-URL override: the
        /// classic client always targeted [`DEFAULT_BASE_URL`]. The credential is
        /// validated eagerly but stored as [`ApiKeyLocation::Env`], so the secret
        /// is read at request time rather than held inside the config.
        ///
        /// # Errors
        /// [`ConfigError`] when a required variable is missing or invalid.
        pub fn from_env(model: impl Into<String>) -> Result<Self, ConfigError> {
            let cfg = Self::new(model);
            required_env_var("VOYAGE_API_KEY")?;
            Ok(cfg)
        }

        /// Config for `model` with an explicit API key.
        pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
            self.api_key = ApiKeyLocation::Inline(key.into());
            self
        }

        /// Override the API base URL.
        pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
            self.base_url = base_url.into();
            self
        }
    }

    /// Build the complete HTTP `/embeddings` request for one chunk of
    /// `texts`.
    ///
    /// Pure except for credential resolution.
    pub fn build_embedding_request(
        cfg: &EmbeddingConfig,
        texts: &[String],
    ) -> Result<http::Request<Vec<u8>>, EmbeddingError> {
        let body = super::build_embedding_body(&cfg.model, texts)?;
        let url = format!("{}/embeddings", cfg.base_url.trim_end_matches('/'));
        let mut builder = http::Request::post(url).header(CONTENT_TYPE, "application/json");
        if let Some(key) = cfg
            .api_key
            .resolve()
            .map_err(|e| EmbeddingError::ProviderError(e.to_string()))?
        {
            builder = builder.header(AUTHORIZATION, format!("Bearer {key}"));
        }
        for (name, value) in &cfg.extra_headers {
            builder = builder.header(name.as_str(), value.as_str());
        }
        builder
            .body(body)
            .map_err(|e| EmbeddingError::ProviderError(e.to_string()))
    }

    /// Parse an `/embeddings` response into the normalized
    /// [`crate::embeddings::EmbeddingResponse`]. Pure.
    pub fn parse_embedding_response(
        status: http::StatusCode,
        body: &str,
        documents: Vec<String>,
    ) -> Result<crate::embeddings::EmbeddingResponse, EmbeddingError> {
        super::parse_embedding_response(status, body, documents)
    }

    /// Embed `texts`, chunking to honor [`DESCRIPTOR`]'s
    /// `max_embedding_documents`; embeddings are returned in input order.
    pub async fn embed(
        cfg: &EmbeddingConfig,
        rt: &HttpRuntime,
        texts: Vec<String>,
    ) -> Result<crate::embeddings::EmbeddingResponse, EmbeddingError> {
        crate::embeddings::batching::embed_chunked(
            rt,
            texts,
            DESCRIPTOR.max_embedding_documents,
            |chunk| build_embedding_request(cfg, chunk),
            parse_embedding_response,
        )
        .await
    }

    /// Embed caller-defined batches, returning one order-aligned
    /// [`OneOrMany`](crate::OneOrMany) group per input batch plus summed
    /// usage.
    pub async fn embed_batches(
        cfg: &EmbeddingConfig,
        rt: &HttpRuntime,
        texts: Vec<Vec<String>>,
    ) -> Result<
        (
            Vec<crate::OneOrMany<crate::embeddings::Embedding>>,
            crate::completion::Usage,
        ),
        EmbeddingError,
    > {
        let (counts, flat) = crate::embeddings::batching::split_batches(texts);
        let response = embed(cfg, rt, flat).await?;
        let groups = crate::embeddings::batching::group_batches(&counts, response.embeddings)?;
        Ok((groups, response.usage))
    }

    // ================================================================
    // Rerank
    // ================================================================

    /// Plain-data Voyage AI rerank configuration: model + rerank options +
    /// connection fields (the rerank sibling of [`EmbeddingConfig`]).
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    #[non_exhaustive]
    pub struct RerankConfig {
        /// API base URL (defaults to [`DEFAULT_BASE_URL`]).
        pub base_url: String,
        /// Credential location.
        pub api_key: ApiKeyLocation,
        /// Reranker model identifier requests are built for.
        pub model: String,
        /// Extra headers attached to every request.
        pub extra_headers: Vec<(String, String)>,
        /// Number of top results to return (provider default when `None`).
        pub top_k: Option<usize>,
        /// Whether reranked documents ride back in the response.
        pub return_documents: bool,
        /// Provider-side input truncation toggle.
        pub truncation: Option<bool>,
    }

    impl RerankConfig {
        /// Config for `model` reading `VOYAGE_API_KEY` from the environment.
        pub fn new(model: impl Into<String>) -> Self {
            Self {
                base_url: DEFAULT_BASE_URL.to_string(),
                api_key: ApiKeyLocation::Env("VOYAGE_API_KEY".to_string()),
                model: model.into(),
                extra_headers: Vec::new(),
                top_k: None,
                return_documents: false,
                truncation: None,
            }
        }

        /// Config for `model` built from the process environment.
        ///
        /// Same variable as [`EmbeddingConfig::from_env`]: `VOYAGE_API_KEY`
        /// (required).
        ///
        /// # Errors
        /// [`ConfigError`] when a required variable is missing or invalid.
        pub fn from_env(model: impl Into<String>) -> Result<Self, ConfigError> {
            let cfg = Self::new(model);
            required_env_var("VOYAGE_API_KEY")?;
            Ok(cfg)
        }

        /// Config with an explicit API key.
        pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
            self.api_key = ApiKeyLocation::Inline(key.into());
            self
        }

        /// Override the API base URL.
        pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
            self.base_url = base_url.into();
            self
        }
    }

    /// Build the serialized rerank request body. Pure.
    pub fn build_rerank_body(
        model: &str,
        top_k: Option<usize>,
        return_documents: bool,
        truncation: Option<bool>,
        query: &str,
        documents: &[String],
    ) -> Result<Vec<u8>, crate::rerank::RerankError> {
        use serde_json::json;

        let mut body = json!({
            "query": query,
            "documents": documents,
            "model": model,
        });

        let body_obj = body.as_object_mut().ok_or_else(|| {
            crate::rerank::RerankError::ResponseError(
                "rerank request body must be a JSON object".into(),
            )
        })?;

        if let Some(top_k) = top_k {
            body_obj.insert("top_k".to_owned(), json!(top_k));
        }

        body_obj.insert("return_documents".to_owned(), json!(return_documents));

        if let Some(truncation) = truncation {
            body_obj.insert("truncation".to_owned(), json!(truncation));
        }

        Ok(serde_json::to_vec(&body)?)
    }

    /// Parse a rerank response body into the normalized
    /// [`crate::rerank::RerankResponse`]. Pure.
    pub fn parse_rerank_response(
        status: http::StatusCode,
        body: &[u8],
    ) -> Result<crate::rerank::RerankResponse, crate::rerank::RerankError> {
        use crate::rerank::RerankError;

        if !status.is_success() {
            return Err(RerankError::from_http_response(
                status,
                String::from_utf8_lossy(body),
            ));
        }

        match serde_json::from_slice::<super::ApiResponse<super::RerankApiResponse>>(body)? {
            super::ApiResponse::Ok(response) => {
                tracing::info!(target: "rig",
                    "VoyageAI rerank token usage: {}",
                    response.usage.total_tokens
                );

                let usage = crate::completion::Usage {
                    input_tokens: response.usage.total_tokens as u64,
                    output_tokens: 0,
                    total_tokens: response.usage.total_tokens as u64,
                    cached_input_tokens: 0,
                    cache_creation_input_tokens: 0,
                    reasoning_tokens: 0,
                    tool_use_prompt_tokens: 0,
                };

                let results = response
                    .data
                    .into_iter()
                    .map(|d| crate::rerank::RerankResult {
                        index: d.index,
                        document: d.document,
                        relevance_score: d.relevance_score,
                    })
                    .collect();

                Ok(crate::rerank::RerankResponse {
                    results,
                    model: response.model,
                    usage,
                })
            }
            super::ApiResponse::Err(err) => {
                tracing::warn!(message = %err.message, "provider returned an error response");
                Err(RerankError::from_http_response(
                    status,
                    String::from_utf8_lossy(body),
                ))
            }
        }
    }

    /// Rerank `documents` against `query` with Voyage AI's `/rerank`
    /// endpoint.
    ///
    /// The query and documents ride as arguments; the remaining knobs live
    /// on [`RerankConfig`].
    pub async fn rerank(
        cfg: &RerankConfig,
        rt: &HttpRuntime,
        query: &str,
        documents: Vec<String>,
    ) -> Result<crate::rerank::RerankResponse, crate::rerank::RerankError> {
        let body = build_rerank_body(
            &cfg.model,
            cfg.top_k,
            cfg.return_documents,
            cfg.truncation,
            query,
            &documents,
        )?;
        let url = format!("{}/rerank", cfg.base_url.trim_end_matches('/'));
        let req = crate::providers::openai::functions::bearer_post(
            url,
            &cfg.api_key,
            &cfg.extra_headers,
            true,
        )?
        .body(body)
        .map_err(crate::http_client::Error::from)?;
        let (status, body) = rt.send_bytes(req).await?;
        parse_rerank_response(status, &body)
    }
    /// Credential verification is not available for this provider.
    ///
    /// The deleted client declared `const VERIFY_PATH: &'static str = ""`, so the
    /// classic `verify()` issued a bare `GET` of the base URL — a request that
    /// checked no credential. [`DESCRIPTOR`] therefore carries no `verify_path`
    /// and this reports the fact rather than repeating the empty check.
    ///
    /// # Errors
    /// Always [`VerifyError::Unsupported`](crate::providers::verify::VerifyError::Unsupported).
    pub async fn verify(
        cfg: &EmbeddingConfig,
        rt: &HttpRuntime,
    ) -> Result<(), crate::providers::verify::VerifyError> {
        let _ = (cfg, rt);
        Err(crate::providers::verify::VerifyError::Unsupported {
            provider: DESCRIPTOR.name,
        })
    }
}

// ================================================================
// Voyage AI Rerank API
// ================================================================

/// `rerank-2.5` reranker model (Voyage AI)
pub const RERANK_2_5: &str = "rerank-2.5";
/// `rerank-2.5-lite` reranker model (Voyage AI)
pub const RERANK_2_5_LITE: &str = "rerank-2.5-lite";
/// `rerank-2` reranker model (Voyage AI)
pub const RERANK_2: &str = "rerank-2";
/// `rerank-2-lite` reranker model (Voyage AI)
pub const RERANK_2_LITE: &str = "rerank-2-lite";
/// `rerank-1` reranker model (Voyage AI)
pub const RERANK_1: &str = "rerank-1";
/// `rerank-lite-1` reranker model (Voyage AI)
pub const RERANK_LITE_1: &str = "rerank-lite-1";

#[derive(Debug, Deserialize)]
pub struct RerankApiResponse {
    pub data: Vec<RerankApiData>,
    pub model: String,
    pub usage: RerankApiUsage,
}

#[derive(Debug, Deserialize)]
pub struct RerankApiUsage {
    pub total_tokens: usize,
}

#[derive(Debug, Deserialize)]
pub struct RerankApiData {
    pub index: usize,
    pub relevance_score: f64,
    #[serde(default)]
    pub document: Option<String>,
}
#[cfg(test)]
mod tests {
    #[test]
    fn rerank_body_carries_query_documents_and_options() {
        let body = super::functions::build_rerank_body(
            super::RERANK_2_5,
            Some(3),
            true,
            Some(false),
            "best pizza",
            &["doc a".to_string(), "doc b".to_string()],
        )
        .expect("build");
        let value: serde_json::Value = serde_json::from_slice(&body).expect("json");
        assert_eq!(value["model"], super::RERANK_2_5);
        assert_eq!(value["query"], "best pizza");
        assert_eq!(value["documents"], serde_json::json!(["doc a", "doc b"]));
        assert_eq!(value["top_k"], 3);
        assert_eq!(value["return_documents"], true);
        assert_eq!(value["truncation"], false);
    }

    #[test]
    fn rerank_body_omits_unset_options() {
        let body =
            super::functions::build_rerank_body(super::RERANK_2, None, false, None, "q", &[])
                .expect("build");
        let value: serde_json::Value = serde_json::from_slice(&body).expect("json");
        assert!(value.get("top_k").is_none());
        assert_eq!(value["return_documents"], false);
        assert!(value.get("truncation").is_none());
    }

    #[tokio::test]
    async fn rerank_non_success_preserves_status_and_body() {
        use crate::http_runtime::HttpRuntime;
        use crate::rerank::RerankError;
        use crate::test_utils::RecordingHttpClient;

        let body = r#"{"error":{"message":"boom"}}"#;
        let rt = HttpRuntime::recording(RecordingHttpClient::with_error_response(
            http::StatusCode::SERVICE_UNAVAILABLE,
            body,
        ));
        let cfg = super::functions::RerankConfig::new(super::RERANK_2_5).with_api_key("test-key");

        let error = super::functions::rerank(
            &cfg,
            &rt,
            "query",
            vec!["doc one".to_string(), "doc two".to_string()],
        )
        .await
        .expect_err("rerank should fail with non-success status");

        assert!(matches!(error, RerankError::HttpError(_)));
        assert_eq!(
            error.provider_response_status(),
            Some(http::StatusCode::SERVICE_UNAVAILABLE)
        );
        assert_eq!(error.provider_response_body(), Some(body));
    }

    #[tokio::test]
    async fn rerank_2xx_error_envelope_preserves_status_and_body() {
        use crate::http_runtime::HttpRuntime;
        use crate::rerank::RerankError;
        use crate::test_utils::RecordingHttpClient;

        let body = r#"{"message":"boom"}"#;
        let rt = HttpRuntime::recording(RecordingHttpClient::new(body)); // 200 OK
        let cfg = super::functions::RerankConfig::new(super::RERANK_2_5).with_api_key("test-key");

        let error = super::functions::rerank(
            &cfg,
            &rt,
            "query",
            vec!["doc one".to_string(), "doc two".to_string()],
        )
        .await
        .expect_err("rerank should fail with provider error envelope");

        match &error {
            RerankError::ProviderResponse(stored) => {
                assert_eq!(stored.body, body);
                assert_eq!(stored.status, Some(http::StatusCode::OK));
            }
            other => panic!("expected ProviderResponse, got {other:?}"),
        }
    }
}

#[cfg(test)]
mod ndims_tests {
    use super::functions::EmbeddingConfig;
    use super::{VOYAGE_3_LARGE, VOYAGE_CODE_2, model_dimensions_from_identifier};

    /// The deleted `EmbeddingModel` carried an `ndims` and reported it through
    /// `EmbeddingModel::ndims()`; `EmbeddingConfig` now carries the same
    /// value, seeded from the same lookup table the classic `make` used.
    #[test]
    fn embedding_config_carries_ndims_for_known_models() {
        assert_eq!(
            EmbeddingConfig::new(VOYAGE_3_LARGE).ndims,
            model_dimensions_from_identifier(VOYAGE_3_LARGE)
        );
        assert_eq!(EmbeddingConfig::new(VOYAGE_3_LARGE).ndims, Some(1024));
        assert_eq!(EmbeddingConfig::new(VOYAGE_CODE_2).ndims, Some(1536));
        assert_eq!(EmbeddingConfig::new("some-future-model").ndims, None);
        assert_eq!(
            EmbeddingConfig::new("some-future-model")
                .with_ndims(2048)
                .ndims,
            Some(2048)
        );
    }

    /// Voyage AI's `/embeddings` request has no dimensionality parameter, and
    /// the classic model never sent one either — `ndims` is carried, not
    /// serialized.
    #[test]
    fn ndims_does_not_reach_the_request_body() {
        let cfg = EmbeddingConfig::new(VOYAGE_3_LARGE).with_api_key("secret");
        let req =
            super::functions::build_embedding_request(&cfg, &["hello".to_string()]).expect("build");
        let value: serde_json::Value = serde_json::from_slice(req.body()).expect("json");
        assert_eq!(value["model"], VOYAGE_3_LARGE);
        assert_eq!(value["input"], serde_json::json!(["hello"]));
        assert!(value.get("ndims").is_none());
        assert!(value.get("dimensions").is_none());
        assert!(value.get("output_dimension").is_none());
    }
}
