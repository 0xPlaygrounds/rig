//! Llamafile embeddings smoke test.
//!
//! Replays by default; set `RIG_PROVIDER_TEST_MODE=record` to record against a
//! local OpenAI-compatible llama.cpp-family server (see `cassette_support`).

use rig::client::EmbeddingsClient;
use rig::embeddings::EmbeddingModel;

use super::super::cassette_support::{CASSETTE_EMBEDDING_MODEL, with_llamafile_cassette};
use crate::support::{EMBEDDING_INPUTS, assert_embeddings_nonempty_and_consistent};

#[tokio::test]
async fn embeddings_smoke() {
    with_llamafile_cassette("embeddings/embeddings_smoke", |client| async move {
        let model = client.embedding_model(CASSETTE_EMBEDDING_MODEL);

        let embeddings = model
            .embed_texts(EMBEDDING_INPUTS.iter().map(|input| (*input).to_string()))
            .await
            .expect("embedding request should succeed");

        assert_embeddings_nonempty_and_consistent(&embeddings, EMBEDDING_INPUTS.len());
    })
    .await;
}
