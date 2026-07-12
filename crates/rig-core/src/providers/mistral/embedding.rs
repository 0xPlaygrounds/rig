use super::client::MistralExt;
use crate::providers::openai::embedding::GenericEmbeddingModel;

pub const MISTRAL_EMBED: &str = "mistral-embed";

pub const MAX_DOCUMENTS: usize = 1024;

pub type EmbeddingModel<H = reqwest::Client> = GenericEmbeddingModel<MistralExt, H>;

#[cfg(test)]
mod tests {
    use super::MISTRAL_EMBED;
    use crate::client::EmbeddingsClient;
    use crate::embeddings::EmbeddingModel as _;
    use crate::providers::mistral;
    use crate::test_utils::RecordingHttpClient;

    const RESPONSE_BODY: &str = r#"{
        "id": "emb-1",
        "object": "list",
        "model": "mistral-embed",
        "usage": { "prompt_tokens": 5, "total_tokens": 5 },
        "data": [{ "object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3] }]
    }"#;

    #[tokio::test]
    async fn mistral_embeddings_use_v1_path_and_map_usage() {
        let http_client = RecordingHttpClient::new(RESPONSE_BODY);
        let client = mistral::Client::builder()
            .api_key("dummy-key")
            .http_client(http_client.clone())
            .build()
            .expect("client should build");

        let model = client.embedding_model(MISTRAL_EMBED);
        let response = model
            .embed_texts_with_usage(["hello".to_string()])
            .await
            .expect("embedding request should succeed");

        assert_eq!(response.embeddings.len(), 1);
        assert_eq!(response.embeddings[0].vec, vec![0.1, 0.2, 0.3]);
        assert_eq!(response.usage.input_tokens, 5);
        assert_eq!(response.usage.total_tokens, 5);

        let requests = http_client.requests();
        assert_eq!(requests.len(), 1);
        assert!(
            requests[0].uri.ends_with("/v1/embeddings"),
            "expected Mistral to POST /v1/embeddings, got {}",
            requests[0].uri
        );
    }
}
