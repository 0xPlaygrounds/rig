use super::{client::ApiResponse, client::Client};
use crate::{
    embeddings::{self, EmbeddingError},
    http_client::HttpClientExt,
    wasm_compat::*,
};
use serde::Deserialize;
use serde_json::json;

#[derive(Deserialize)]
pub struct EmbeddingResponse {
    #[serde(default)]
    pub response_type: Option<String>,
    pub id: String,
    pub embeddings: Vec<Vec<serde_json::Number>>,
    pub texts: Vec<String>,
    #[serde(default)]
    pub meta: Option<Meta>,
}

#[derive(Deserialize)]
pub struct Meta {
    pub api_version: ApiVersion,
    pub billed_units: BilledUnits,
    #[serde(default)]
    pub warnings: Vec<String>,
}

#[derive(Deserialize)]
pub struct ApiVersion {
    pub version: String,
    #[serde(default)]
    pub is_deprecated: Option<bool>,
    #[serde(default)]
    pub is_experimental: Option<bool>,
}

#[derive(Deserialize, Debug)]
pub struct BilledUnits {
    #[serde(default)]
    pub input_tokens: u32,
    #[serde(default)]
    pub output_tokens: u32,
    #[serde(default)]
    pub search_units: u32,
    #[serde(default)]
    pub classifications: u32,
}

impl std::fmt::Display for BilledUnits {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Input tokens: {}\nOutput tokens: {}\nSearch units: {}\nClassifications: {}",
            self.input_tokens, self.output_tokens, self.search_units, self.classifications
        )
    }
}

#[derive(Clone)]
pub struct EmbeddingModel<T = reqwest::Client> {
    client: Client<T>,
    pub model: String,
    pub input_type: String,
    ndims: usize,
}

impl<T> embeddings::EmbeddingModel for EmbeddingModel<T>
where
    T: HttpClientExt + Clone + WasmCompatSend + WasmCompatSync + 'static,
{
    const MAX_DOCUMENTS: usize = 96;
    type Client = Client<T>;

    fn make(client: &Self::Client, model: impl Into<String>, dims: Option<usize>) -> Self {
        let model = model.into();
        let dims = dims
            .or(super::model_dimensions_from_identifier(&model))
            .unwrap_or_default();

        Self::new(client.clone(), model, "search_document", dims)
    }

    fn ndims(&self) -> usize {
        self.ndims
    }

    async fn embed_texts(
        &self,
        documents: impl IntoIterator<Item = String>,
    ) -> Result<Vec<embeddings::Embedding>, EmbeddingError> {
        let documents = documents.into_iter().collect::<Vec<_>>();

        let body = json!({
            "model": self.model.to_string(),
            "texts": documents,
            "input_type": self.input_type
        });

        let body = serde_json::to_vec(&body)?;

        let req = self
            .client
            .post("/v1/embed")?
            .body(body)
            .map_err(|e| EmbeddingError::HttpError(e.into()))?;

        let response = self
            .client
            .send::<_, Vec<u8>>(req)
            .await
            .map_err(EmbeddingError::HttpError)?;

        let status = response.status();
        let raw_body = response.into_body().await?;

        if status.is_success() {
            let body: ApiResponse<EmbeddingResponse> = serde_json::from_slice(raw_body.as_slice())?;

            match body {
                ApiResponse::Ok(response) => {
                    match response.meta {
                        Some(meta) => tracing::info!(target: "rig",
                            "Cohere embeddings billed units: {}",
                            meta.billed_units,
                        ),
                        None => tracing::info!(target: "rig",
                            "Cohere embeddings billed units: n/a",
                        ),
                    };

                    if response.embeddings.len() != documents.len() {
                        return Err(EmbeddingError::DocumentError(
                            format!(
                                "Expected {} embeddings, got {}",
                                documents.len(),
                                response.embeddings.len()
                            )
                            .into(),
                        ));
                    }

                    Ok(response
                        .embeddings
                        .into_iter()
                        .zip(documents.into_iter())
                        .map(|(embedding, document)| embeddings::Embedding {
                            document,
                            vec: embedding.into_iter().filter_map(|n| n.as_f64()).collect(),
                        })
                        .collect())
                }
                ApiResponse::Err(error) => {
                    tracing::warn!(
                        message = %error.message,
                        "Cohere returned an error response"
                    );
                    Err(EmbeddingError::from_http_response(
                        status,
                        String::from_utf8_lossy(&raw_body),
                    ))
                }
            }
        } else {
            Err(EmbeddingError::from_http_response(
                status,
                String::from_utf8_lossy(&raw_body),
            ))
        }
    }
}

impl<T> EmbeddingModel<T> {
    pub fn new(
        client: Client<T>,
        model: impl Into<String>,
        input_type: &str,
        ndims: usize,
    ) -> Self {
        Self {
            client,
            model: model.into(),
            input_type: input_type.to_string(),
            ndims,
        }
    }

    pub fn with_model(client: Client<T>, model: &str, input_type: &str, ndims: usize) -> Self {
        Self {
            client,
            model: model.into(),
            input_type: input_type.into(),
            ndims,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn embeddings_non_success_preserves_status_and_body() {
        use crate::embeddings::EmbeddingModel as _;
        use crate::test_utils::RecordingHttpClient;

        let body = r#"{"error":{"message":"boom"}}"#;
        let http_client =
            RecordingHttpClient::with_error_response(http::StatusCode::SERVICE_UNAVAILABLE, body);
        let client = crate::providers::cohere::Client::builder()
            .api_key("test-key")
            .http_client(http_client)
            .build()
            .expect("build client");
        let model = client.embedding_model(
            crate::providers::cohere::EMBED_ENGLISH_V3,
            "search_document",
        );

        let error = model
            .embed_texts(["hello".to_string()])
            .await
            .expect_err("should fail with non-success status");

        assert!(matches!(error, EmbeddingError::HttpError(_)));
        assert_eq!(
            error.provider_response_status(),
            Some(http::StatusCode::SERVICE_UNAVAILABLE)
        );
        assert_eq!(error.provider_response_body(), Some(body));
    }

    #[tokio::test]
    async fn embeddings_2xx_error_envelope_preserves_status_and_body() {
        use crate::embeddings::EmbeddingModel as _;
        use crate::test_utils::RecordingHttpClient;

        // Deserializes to `ApiResponse::Err(ApiErrorResponse { message })` on a 200 OK.
        let body = r#"{"message":"boom"}"#;
        let http_client = RecordingHttpClient::new(body);
        let client = crate::providers::cohere::Client::builder()
            .api_key("test-key")
            .http_client(http_client)
            .build()
            .expect("build client");
        let model = client.embedding_model(
            crate::providers::cohere::EMBED_ENGLISH_V3,
            "search_document",
        );

        let error = model
            .embed_texts(["hello".to_string()])
            .await
            .expect_err("should fail with provider error envelope");

        match &error {
            EmbeddingError::ProviderResponse(stored) => {
                assert_eq!(stored.body, body);
                assert_eq!(stored.status, Some(http::StatusCode::OK));
            }
            other => panic!("expected ProviderResponse, got {other:?}"),
        }
    }
}
