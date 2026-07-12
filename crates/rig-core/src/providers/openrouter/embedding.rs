use super::client::OpenRouterExt;
use crate::providers::openai::embedding::GenericEmbeddingModel;

pub type EmbeddingModel<H = reqwest::Client> = GenericEmbeddingModel<OpenRouterExt, H>;

#[cfg(test)]
mod tests {
    use crate::client::EmbeddingsClient;
    use crate::embeddings::EmbeddingModel as _;
    use crate::providers::openrouter;
    use crate::test_utils::RecordingHttpClient;

    const RESPONSE_BODY: &str = r#"{
        "id": "gen-1",
        "object": "list",
        "model": "openai/text-embedding-3-small",
        "data": [{ "object": "embedding", "index": 0, "embedding": [0.5, 0.6] }]
    }"#;

    #[tokio::test]
    async fn openrouter_embeddings_use_default_path_and_zero_absent_usage() {
        let http_client = RecordingHttpClient::new(RESPONSE_BODY);
        let client = openrouter::Client::builder()
            .api_key("dummy-key")
            .http_client(http_client.clone())
            .build()
            .expect("client should build");

        let model = client.embedding_model("openai/text-embedding-3-small");
        let response = model
            .embed_texts_with_usage(["hello".to_string()])
            .await
            .expect("embedding request should succeed");

        assert_eq!(response.embeddings.len(), 1);
        assert_eq!(response.usage.total_tokens, 0);

        let requests = http_client.requests();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].uri, "https://openrouter.ai/api/v1/embeddings");
    }
}
