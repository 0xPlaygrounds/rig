use super::client::OpenRouterExt;
use crate::providers::openai::embedding::GenericEmbeddingModel;

pub type EmbeddingModel<H = reqwest::Client> = GenericEmbeddingModel<OpenRouterExt, H>;

#[cfg(test)]
mod tests {
    use crate::client::EmbeddingsClient;
    use crate::embeddings::{EmbeddingError, EmbeddingModel as _};
    use crate::providers::{openai::embedding::EncodingFormat, openrouter};
    use crate::test_utils::RecordingHttpClient;

    const RESPONSE_BODY: &str = r#"{
        "id": "gen-1",
        "object": "list",
        "model": "openai/text-embedding-3-small",
        "data": [{ "object": "embedding", "index": 0, "embedding": [0.5, 0.6] }]
    }"#;

    fn client(http_client: RecordingHttpClient) -> openrouter::Client<RecordingHttpClient> {
        openrouter::Client::builder()
            .api_key("dummy-key")
            .http_client(http_client)
            .build()
            .expect("client should build")
    }

    #[tokio::test]
    async fn openrouter_embeddings_preserve_supported_parameters_and_zero_absent_usage() {
        let http_client = RecordingHttpClient::new(RESPONSE_BODY);
        let model = client(http_client.clone())
            .embedding_model_with_ndims("openai/text-embedding-3-small", 2)
            .encoding_format(EncodingFormat::Float)
            .user("user-123");

        let response = model
            .embed_texts_with_usage(["hello".to_string()])
            .await
            .expect("embedding request should succeed");

        assert_eq!(response.embeddings.len(), 1);
        assert_eq!(response.usage.total_tokens, 0);
        let requests = http_client.requests();
        assert_eq!(requests[0].uri, "https://openrouter.ai/api/v1/embeddings");
        let body: serde_json::Value =
            serde_json::from_slice(&requests[0].body).expect("request body should be JSON");
        assert_eq!(body["dimensions"], serde_json::json!(2));
        assert_eq!(body["encoding_format"], serde_json::json!("float"));
        assert_eq!(body["user"], serde_json::json!("user-123"));
    }

    #[tokio::test]
    async fn openrouter_rejects_response_length_mismatch() {
        let http_client = RecordingHttpClient::new(RESPONSE_BODY);
        let model = client(http_client).embedding_model("openai/text-embedding-3-small");

        let error = model
            .embed_texts(["one".to_string(), "two".to_string()])
            .await
            .expect_err("response length mismatch should fail");

        assert!(matches!(error, EmbeddingError::ResponseError(_)));
    }
}
