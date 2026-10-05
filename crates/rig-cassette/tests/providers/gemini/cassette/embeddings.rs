//! Gemini embeddings smoke test.

use rig::providers::gemini;

use crate::support::{EMBEDDING_INPUTS, assert_embeddings_nonempty_and_consistent};

#[tokio::test]
async fn embeddings_smoke() {
    super::super::support::with_gemini_cassette(
        "embeddings/embeddings_smoke",
        |client| async move {
            let model = client.embedding(gemini::embedding::EMBEDDING_001, None);

            let response = model
                .call(
                    EMBEDDING_INPUTS
                        .iter()
                        .map(|input| (*input).to_string())
                        .collect::<Vec<_>>(),
                )
                .await
                .expect("embedding request should succeed");

            assert_embeddings_nonempty_and_consistent(&response.embeddings, EMBEDDING_INPUTS.len());
            // The normalized response names its provider, and `raw` is the
            // provider's own payload, serialized — it round-trips to the type
            // `raw_embed_texts` returns.
            assert_eq!(response.provider, "gcp.gemini");
            let raw: gemini::embedding::gemini_api_types::EmbeddingResponse =
                serde_json::from_value(response.raw)
                    .expect("raw payload should round-trip to Gemini's own type");
            assert_eq!(raw.embeddings.len(), EMBEDDING_INPUTS.len());
        },
    )
    .await;
}
