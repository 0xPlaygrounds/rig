// ================================================================
//! Together AI Embeddings Integration
//! From [Together AI Reference](https://docs.together.ai/docs/embeddings-overview)
// ================================================================

//! Together's embeddings endpoint is OpenAI-compatible, so [`EmbeddingModel`] is
//! a thin alias over the shared [`GenericEmbeddingModel`]. The `/v1/embeddings`
//! path is supplied by [`TogetherExt`](super::client::TogetherExt)'s
//! [`OpenAIEmbeddingsCompatible`](crate::providers::openai::embedding::OpenAIEmbeddingsCompatible)
//! implementation in `client.rs`.

use super::client::TogetherExt;
use crate::providers::openai::embedding::GenericEmbeddingModel;

// ================================================================
// Together AI Embedding API
// ================================================================
pub const BGE_BASE_EN_V1_5: &str = "BAAI/bge-base-en-v1.5";
pub const BGE_LARGE_EN_V1_5: &str = "BAAI/bge-large-en-v1.5";
pub const BERT_BASE_UNCASED: &str = "bert-base-uncased";
pub const M2_BERT_2K_RETRIEVAL_ENCODER_V1: &str = "hazyresearch/M2-BERT-2k-Retrieval-Encoder-V1";
pub const M2_BERT_80M_32K_RETRIEVAL: &str = "togethercomputer/m2-bert-80M-32k-retrieval";
pub const M2_BERT_80M_2K_RETRIEVAL: &str = "togethercomputer/m2-bert-80M-2k-retrieval";
pub const M2_BERT_80M_8K_RETRIEVAL: &str = "togethercomputer/m2-bert-80M-8k-retrieval";
pub const SENTENCE_BERT: &str = "sentence-transformers/msmarco-bert-base-dot-v5";
pub const UAE_LARGE_V1: &str = "WhereIsAI/UAE-Large-V1";

/// Together AI embedding model, backed by the shared OpenAI-compatible
/// [`GenericEmbeddingModel`]. Unlike the previous hand-rolled implementation
/// this honors `ndims` (sent as `dimensions`), `encoding_format`, and `user`.
pub type EmbeddingModel<H = reqwest::Client> = GenericEmbeddingModel<TogetherExt, H>;

#[cfg(test)]
mod tests {
    use super::BGE_BASE_EN_V1_5;
    use crate::client::EmbeddingsClient;
    use crate::embeddings::EmbeddingModel as _;
    use crate::providers::together;
    use crate::test_utils::RecordingHttpClient;

    // Minimal OpenAI-compatible embeddings response. `usage` is intentionally
    // omitted to exercise the optional-usage path the generic model now allows.
    const RESPONSE_BODY: &str = r#"{
        "object": "list",
        "model": "BAAI/bge-base-en-v1.5",
        "data": [{ "object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3] }]
    }"#;

    #[tokio::test]
    async fn together_embeddings_send_dimensions_to_v1_path() {
        let http_client = RecordingHttpClient::new(RESPONSE_BODY);
        let client = together::Client::builder()
            .api_key("dummy-key")
            .http_client(http_client.clone())
            .build()
            .expect("client should build");

        // With `ndims` set, `dimensions` must be sent — the previous hand-rolled
        // Together model silently dropped it (the bug this migration fixes).
        let model = client.embedding_model_with_ndims(BGE_BASE_EN_V1_5, 3);
        model
            .embed_texts(["hello".to_string()])
            .await
            .expect("embedding request should succeed");

        let requests = http_client.requests();
        assert_eq!(requests.len(), 1);
        assert!(
            requests[0].uri.ends_with("/v1/embeddings"),
            "expected Together to POST /v1/embeddings, got {}",
            requests[0].uri
        );
        let body: serde_json::Value =
            serde_json::from_slice(&requests[0].body).expect("request body should be JSON");
        assert_eq!(body["dimensions"], serde_json::json!(3));
        assert_eq!(body["model"], BGE_BASE_EN_V1_5);
    }
}
