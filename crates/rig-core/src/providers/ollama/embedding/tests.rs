use super::*;
use crate::test_utils::RecordingHttpClient;

/// `POST /api/embed`'s reply shape, with two-element vectors in place of the
/// recorded 384-element ones.
const EMBED_BODY: &str = r#"{"model":"all-minilm","embeddings":[[0.5,-0.25],[0.125,0.0]],"total_duration":1000,"load_duration":10,"prompt_eval_count":6}"#;

#[tokio::test]
async fn an_embedding_reply_pairs_its_vectors_with_the_texts_that_were_sent() {
    let response = crate::driver::Model::new(
        OllamaConfig::new().embedding("all-minilm", None),
        RecordingHttpClient::new(EMBED_BODY),
    )
    .call(vec!["first".to_owned(), "second".to_owned()])
    .await
    .expect("the reply decodes");

    assert_eq!(
        response
            .embeddings
            .iter()
            .map(|embedding| (embedding.document.as_str(), embedding.vec.as_slice()))
            .collect::<Vec<_>>(),
        vec![
            ("first", [0.5, -0.25].as_slice()),
            ("second", [0.125, 0.0].as_slice()),
        ]
    );
    assert_eq!(response.model.as_deref(), Some("all-minilm"));
    // Every token of an embedding is input; Ollama reports one counter.
    assert_eq!(response.usage.input_tokens, Some(6));
    assert_eq!(response.usage.total_tokens, Some(6));
    assert_eq!(response.usage.output_tokens, None);
}

#[test]
fn an_embedding_wire_reports_the_models_published_width() {
    assert_eq!(
        OllamaConfig::new()
            .embedding("all-minilm", None)
            .describe()
            .capabilities,
        Capabilities::embedding(1024, 384)
    );
    assert_eq!(
        OllamaConfig::new()
            .embedding("qwen3-embedding", Some(2048))
            .describe()
            .capabilities,
        Capabilities::embedding(1024, 2048),
        "a family whose width varies by size takes the caller's"
    );
}
