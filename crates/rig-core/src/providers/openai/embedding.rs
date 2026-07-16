use super::{client::ApiResponse, completion::Usage};
use crate::embeddings::EmbeddingError;
use crate::http_client::HttpClientExt;
use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};
use crate::{embeddings, http_client};
use serde::{Deserialize, Serialize};
use serde_json::json;

// ================================================================
// OpenAI Embedding API
// ================================================================
/// `text-embedding-3-large` embedding model
pub const TEXT_EMBEDDING_3_LARGE: &str = "text-embedding-3-large";
/// `text-embedding-3-small` embedding model
pub const TEXT_EMBEDDING_3_SMALL: &str = "text-embedding-3-small";
/// `text-embedding-ada-002` embedding model
pub const TEXT_EMBEDDING_ADA_002: &str = "text-embedding-ada-002";

#[derive(Debug, Deserialize)]
pub struct EmbeddingResponse {
    pub object: String,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: Usage,
}

#[derive(Debug, Deserialize)]
struct CompatibleEmbeddingResponse {
    #[serde(rename = "object")]
    _object: String,
    pub data: Vec<EmbeddingData>,
    #[serde(rename = "model")]
    _model: String,
    #[serde(default)]
    pub usage: Option<Usage>,
}

/// Contract for provider extensions that speak an OpenAI-compatible embeddings
/// wire format through [`GenericEmbeddingModel`].
#[doc(hidden)]
pub trait OpenAIEmbeddingsCompatible: crate::client::Provider {
    /// Provider name used in embedding request and response errors.
    const PROVIDER_NAME: &'static str;

    /// Whether successful responses from this provider must include usage.
    const REQUIRES_USAGE: bool = true;

    /// The request path for embeddings, resolved against the client base URL.
    fn embeddings_path(&self) -> String {
        "/embeddings".to_string()
    }

    /// Adjust the serialized embedding request immediately before it is sent.
    fn finalize_embeddings_request(
        &self,
        body: &mut serde_json::Value,
    ) -> Result<(), EmbeddingError> {
        let _ = body;
        Ok(())
    }
}

impl OpenAIEmbeddingsCompatible for super::OpenAIResponsesExt {
    const PROVIDER_NAME: &'static str = "openai";
}

impl OpenAIEmbeddingsCompatible for super::OpenAICompletionsExt {
    const PROVIDER_NAME: &'static str = "openai";
}

#[derive(Debug, Deserialize, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EncodingFormat {
    Float,
    Base64,
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingData {
    pub object: String,
    pub embedding: Vec<serde_json::Number>,
    pub index: usize,
}

#[doc(hidden)]
#[derive(Clone)]
pub struct GenericEmbeddingModel<Ext = super::OpenAIResponsesExt, H = reqwest::Client> {
    client: crate::client::Client<Ext, H>,
    pub model: String,
    pub encoding_format: Option<EncodingFormat>,
    pub user: Option<String>,
    ndims: usize,
}

/// The embedding model struct for OpenAI's Embeddings API.
///
/// This preserves the historical public generic shape where the first generic
/// parameter is the HTTP client type.
pub type EmbeddingModel<H = reqwest::Client> = GenericEmbeddingModel<super::OpenAIResponsesExt, H>;

fn model_dimensions_from_identifier(identifier: &str) -> Option<usize> {
    match identifier {
        TEXT_EMBEDDING_3_LARGE => Some(3_072),
        TEXT_EMBEDDING_3_SMALL | TEXT_EMBEDDING_ADA_002 => Some(1_536),
        _ => None,
    }
}

impl<Ext, H> embeddings::EmbeddingModel for GenericEmbeddingModel<Ext, H>
where
    crate::client::Client<Ext, H>:
        HttpClientExt + Clone + WasmCompatSend + WasmCompatSync + 'static,
    Ext: OpenAIEmbeddingsCompatible + Clone + 'static,
{
    const MAX_DOCUMENTS: usize = 1024;

    type Client = crate::client::Client<Ext, H>;

    fn make(client: &Self::Client, model: impl Into<String>, ndims: Option<usize>) -> Self {
        let model = model.into();
        let dims = ndims
            .or(model_dimensions_from_identifier(&model))
            .unwrap_or_default();

        Self::new(client.clone(), model, dims)
    }

    fn ndims(&self) -> usize {
        self.ndims
    }

    async fn embed_texts(
        &self,
        documents: impl IntoIterator<Item = String>,
    ) -> Result<Vec<embeddings::Embedding>, EmbeddingError> {
        let documents: Vec<String> = documents.into_iter().collect();
        let response = self.embed_texts_with_usage(documents).await?;
        Ok(response.embeddings)
    }

    async fn embed_texts_with_usage(
        &self,
        documents: impl IntoIterator<Item = String>,
    ) -> Result<embeddings::EmbeddingResponse, EmbeddingError> {
        let documents: Vec<String> = documents.into_iter().collect();

        let mut body = json!({
            "model": self.model,
            "input": documents,
        });

        let body_object = body
            .as_object_mut()
            .ok_or(EmbeddingError::InvalidRequestBody)?;

        if self.ndims > 0 && self.model.as_str() != TEXT_EMBEDDING_ADA_002 {
            body_object.insert("dimensions".to_owned(), json!(self.ndims));
        }

        if let Some(encoding_format) = &self.encoding_format {
            body_object.insert("encoding_format".to_owned(), json!(encoding_format));
        }

        if let Some(user) = &self.user {
            body_object.insert("user".to_owned(), json!(user));
        }

        self.client.ext().finalize_embeddings_request(&mut body)?;

        let body = serde_json::to_vec(&body)?;

        let req = self
            .client
            .post(self.client.ext().embeddings_path())?
            .body(body)
            .map_err(|e| EmbeddingError::HttpError(e.into()))?;

        let response = self.client.send(req).await?;

        let status = response.status();
        if status.is_success() {
            let response_body: Vec<u8> = response.into_body().await?;
            let parsed: ApiResponse<CompatibleEmbeddingResponse> =
                serde_json::from_slice(&response_body)?;

            match parsed {
                ApiResponse::Ok(response) => {
                    tracing::info!(target: "rig",
                        "embedding token usage: {:?}",
                        response.usage
                    );

                    if response.data.len() != documents.len() {
                        return Err(EmbeddingError::ResponseError(
                            "Response data length does not match input length".into(),
                        ));
                    }

                    let usage = match response.usage {
                        Some(usage) => crate::completion::Usage {
                            input_tokens: usage.prompt_tokens as u64,
                            output_tokens: 0,
                            total_tokens: usage.total_tokens as u64,
                            cached_input_tokens: usage
                                .prompt_tokens_details
                                .as_ref()
                                .map_or(0, |details| details.cached_tokens as u64),
                            cache_creation_input_tokens: 0,
                            tool_use_prompt_tokens: 0,
                            reasoning_tokens: 0,
                        },
                        None if Ext::REQUIRES_USAGE => {
                            return Err(EmbeddingError::MissingUsage {
                                provider: Ext::PROVIDER_NAME,
                            });
                        }
                        None => crate::completion::Usage::new(),
                    };

                    let embeddings = response
                        .data
                        .into_iter()
                        .zip(documents.into_iter())
                        .map(|(embedding, document)| embeddings::Embedding {
                            document,
                            vec: embedding
                                .embedding
                                .into_iter()
                                .filter_map(|n| n.as_f64())
                                .collect(),
                        })
                        .collect();

                    Ok(embeddings::EmbeddingResponse { embeddings, usage })
                }
                ApiResponse::Err(err) => {
                    tracing::warn!(message = %err.message, "provider returned an error response");
                    Err(EmbeddingError::from_http_response(
                        status,
                        String::from_utf8_lossy(&response_body).into_owned(),
                    ))
                }
            }
        } else {
            let text = http_client::text(response).await?;
            Err(EmbeddingError::from_http_response(status, text))
        }
    }
}

impl<Ext, H> GenericEmbeddingModel<Ext, H>
where
    Ext: crate::client::Provider,
{
    pub fn new(
        client: crate::client::Client<Ext, H>,
        model: impl Into<String>,
        ndims: usize,
    ) -> Self {
        Self {
            client,
            model: model.into(),
            encoding_format: None,
            ndims,
            user: None,
        }
    }

    pub fn with_model(client: crate::client::Client<Ext, H>, model: &str, ndims: usize) -> Self {
        Self {
            client,
            model: model.into(),
            encoding_format: None,
            ndims,
            user: None,
        }
    }

    pub fn with_encoding_format(
        client: crate::client::Client<Ext, H>,
        model: &str,
        ndims: usize,
        encoding_format: EncodingFormat,
    ) -> Self {
        Self {
            client,
            model: model.into(),
            encoding_format: Some(encoding_format),
            ndims,
            user: None,
        }
    }

    pub fn encoding_format(mut self, encoding_format: EncodingFormat) -> Self {
        self.encoding_format = Some(encoding_format);
        self
    }

    pub fn user(mut self, user: impl Into<String>) -> Self {
        self.user = Some(user.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::EmbeddingsClient;
    use crate::embeddings::EmbeddingModel as _;
    use crate::http_client::{LazyBody, MultipartForm, Request, Response, StreamingResponse};
    use crate::providers::openai::CompletionsClient;
    use crate::test_utils::RecordingHttpClient;
    use bytes::Bytes;
    use std::future::{self, Future};

    #[derive(Clone)]
    struct CustomHttpClient;

    impl HttpClientExt for CustomHttpClient {
        fn send<T, U>(
            &self,
            _req: Request<T>,
        ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
        where
            T: Into<Bytes> + WasmCompatSend,
            U: From<Bytes> + WasmCompatSend + 'static,
        {
            future::ready(Err(http_client::Error::StreamEnded))
        }

        fn send_multipart<U>(
            &self,
            _req: Request<MultipartForm>,
        ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
        where
            U: From<Bytes> + WasmCompatSend + 'static,
        {
            future::ready(Err(http_client::Error::StreamEnded))
        }

        fn send_streaming<T>(
            &self,
            _req: Request<T>,
        ) -> impl Future<Output = http_client::Result<StreamingResponse>> + WasmCompatSend
        where
            T: Into<Bytes> + WasmCompatSend,
        {
            future::ready(Err(http_client::Error::StreamEnded))
        }
    }

    const RESPONSE_BODY: &str = r#"{
        "object": "list",
        "model": "text-embedding-3-small",
        "usage": { "prompt_tokens": 4, "total_tokens": 4 },
        "data": [{ "object": "embedding", "index": 0, "embedding": [0.1, 0.2] }]
    }"#;

    #[test]
    fn embedding_model_accepts_backend_without_default_or_debug() {
        let client = CompletionsClient::builder()
            .api_key("test-key")
            .http_client(CustomHttpClient)
            .build()
            .expect("build client");

        let model = client.embedding_model(TEXT_EMBEDDING_3_SMALL);

        assert_eq!(model.ndims(), 1_536);
    }

    #[tokio::test]
    async fn openai_embeddings_preserve_path_parameters_and_usage() {
        let http_client = RecordingHttpClient::new(RESPONSE_BODY);
        let client = CompletionsClient::builder()
            .api_key("test-key")
            .http_client(http_client.clone())
            .build()
            .expect("build client");
        let model = client
            .embedding_model(TEXT_EMBEDDING_3_SMALL)
            .encoding_format(EncodingFormat::Float)
            .user("user-123");

        let response = model
            .embed_texts_with_usage(["hello".to_string()])
            .await
            .expect("embedding should succeed");

        assert_eq!(response.usage.input_tokens, 4);
        assert_eq!(response.usage.total_tokens, 4);
        let requests = http_client.requests();
        assert_eq!(requests[0].uri, "https://api.openai.com/v1/embeddings");
        let body: serde_json::Value =
            serde_json::from_slice(&requests[0].body).expect("request body should be JSON");
        assert_eq!(body["dimensions"], serde_json::json!(1_536));
        assert_eq!(body["encoding_format"], serde_json::json!("float"));
        assert_eq!(body["user"], serde_json::json!("user-123"));
    }

    #[test]
    fn public_openai_embedding_response_requires_usage() {
        let body = r#"{
            "object": "list",
            "model": "text-embedding-3-small",
            "data": [{ "object": "embedding", "index": 0, "embedding": [0.1] }]
        }"#;

        assert!(serde_json::from_str::<EmbeddingResponse>(body).is_err());
    }

    #[tokio::test]
    async fn embedding_preserves_raw_provider_error_json_on_api_error_envelope() {
        let body = r#"{"message":"embedding quota exceeded","type":"insufficient_quota"}"#;
        let http_client =
            RecordingHttpClient::with_error_response(http::StatusCode::ACCEPTED, body);
        let client = CompletionsClient::builder()
            .api_key("test-key")
            .http_client(http_client)
            .build()
            .expect("build client");
        let model = client.embedding_model("text-embedding-3-small");

        let error = model
            .embed_texts(["hello".to_string()])
            .await
            .expect_err("embedding should fail with provider error envelope");

        match &error {
            EmbeddingError::ProviderResponse(stored) => {
                assert_eq!(stored.body, body);
                assert_eq!(stored.status, Some(http::StatusCode::ACCEPTED));
                assert_eq!(error.provider_response_body(), Some(body));
                let json = error
                    .provider_response_json()
                    .expect("raw body should be valid JSON")
                    .expect("parsed JSON should be present");
                assert_eq!(json["type"], "insufficient_quota");
            }
            other => panic!("expected ProviderResponse, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn embedding_http_non_success_preserves_status_and_body() {
        let body = r#"{"error":{"message":"invalid api key","type":"invalid_request_error"}}"#;
        let http_client =
            RecordingHttpClient::with_error_response(http::StatusCode::UNAUTHORIZED, body);
        let client = CompletionsClient::builder()
            .api_key("test-key")
            .http_client(http_client)
            .build()
            .expect("build client");
        let model = client.embedding_model("text-embedding-3-small");

        let error = model
            .embed_texts(["hello".to_string()])
            .await
            .expect_err("embedding should fail with non-success status");

        assert!(matches!(error, EmbeddingError::HttpError(_)));
        assert_eq!(
            error.provider_response_status(),
            Some(http::StatusCode::UNAUTHORIZED)
        );
        assert_eq!(error.provider_response_body(), Some(body));
    }
}
