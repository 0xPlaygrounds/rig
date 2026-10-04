//! AWS Bedrock embeddings replay smoke test.

use rig::bedrock;

use super::super::support::with_bedrock_cassette;

const EMBEDDING_INPUT: &str = "Rust cassette replay keeps Bedrock tests deterministic.";

#[tokio::test]
async fn embeddings_smoke() {
    with_bedrock_cassette("embeddings/embeddings_smoke", |client| async move {
        let model = client.embedding(bedrock::embedding::AMAZON_TITAN_EMBED_TEXT_V2_0, Some(256));

        let embeddings = model
            .call(vec![EMBEDDING_INPUT.to_string()])
            .await
            .map(|response| response.embeddings)
            .expect("embedding request should succeed");

        assert_eq!(embeddings.len(), 1);
        let embedding = &embeddings[0];
        assert_eq!(embedding.document, EMBEDDING_INPUT);
        assert!(
            !embedding.vec.is_empty(),
            "expected embedding vector to be non-empty"
        );
    })
    .await;
}
