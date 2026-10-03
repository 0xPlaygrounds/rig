use super::*;
use crate::completion::CompletionRequest;
use crate::test_utils::RecordingHttpClient;
use crate::wire::secret::tests::a_config_reloads_without_its_credential;

/// A config is data a host persists, so what survives the round trip is the
/// part that is not a credential: a reloaded wire addresses the same daemon
/// with the same model and gets its credential from the environment again,
/// never from the file.
#[test]
fn a_serialized_config_round_trips_everything_but_the_credential() {
    let wire = OllamaConfig::new()
        .with_base_url("http://ollama.internal:11434")
        .completion("qwen3:4b");
    let serialized = serde_json::to_string(&wire).expect("the wire serializes");
    let restored: crate::providers::openai::wire::Chat =
        serde_json::from_str(&serialized).expect("the wire deserializes");

    assert_eq!(restored.model, wire.model);
    assert_eq!(
        restored.provider.base_url,
        "http://ollama.internal:11434/v1"
    );

    // A proxied daemon does take a credential, and that one never travels.
    a_config_reloads_without_its_credential(
        &OllamaConfig::new().with_api_key("ollama-proxy-key"),
        "ollama-proxy-key",
        |ollama| &ollama.api_key,
    );
}

#[test]
fn a_local_daemon_sends_no_authorization_header() {
    let encoded = OllamaConfig::new()
        .completion("qwen3:4b")
        .encode(CompletionRequest::new("hi"), Mode::Unary)
        .expect("the request encodes");
    let request = &encoded.request;
    assert_eq!(request.uri(), "http://localhost:11434/v1/chat/completions");
    assert!(!request.headers().contains_key(http::header::AUTHORIZATION));

    let encoded = OllamaConfig::new()
        .with_api_key("ollama-proxy-key")
        .completion("qwen3:4b")
        .encode(CompletionRequest::new("hi"), Mode::Unary)
        .expect("the request encodes");
    let request = &encoded.request;
    assert_eq!(
        request
            .headers()
            .get(http::header::AUTHORIZATION)
            .and_then(|value| value.to_str().ok()),
        Some("Bearer ollama-proxy-key")
    );
}

/// The reply shape of `crates/rig-cassette/fixtures/cassettes/ollama/models/list_models_smoke.yaml`
/// (`GET /api/tags`), with two of its entries.
const MODELS_BODY: &str = r#"{"models":[{"name":"all-minilm:latest","model":"all-minilm:latest","modified_at":"2026-06-19T17:15:40.188240254-07:00","size":45960996},{"name":"qwen3:4b","model":"qwen3:4b","modified_at":"2026-06-19T16:26:52.429441648-07:00","size":2497293931}]}"#;

#[tokio::test]
async fn the_model_listing_reads_every_installed_model() {
    let models = crate::driver::Model::new(
        OllamaConfig::new().models(),
        RecordingHttpClient::new(MODELS_BODY),
    )
    .list()
    .await
    .expect("the recorded reply decodes");

    assert_eq!(
        models
            .data
            .iter()
            .map(|model| model.id.as_str())
            .collect::<Vec<_>>(),
        vec!["all-minilm:latest", "qwen3:4b"]
    );
}

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

/// `raw` is the whole reply body, so a field the reply type does not model
/// survives in it.
#[tokio::test]
async fn an_embedding_reply_keeps_its_whole_body_as_raw() {
    let body = r#"{"model":"all-minilm","embeddings":[[0.5,-0.25],[0.125,0.0]],"prompt_eval_count":6,"unmodeled":"kept"}"#;
    let response = crate::driver::Model::new(
        OllamaConfig::new().embedding("all-minilm", None),
        RecordingHttpClient::new(body),
    )
    .call(vec!["first".to_owned(), "second".to_owned()])
    .await
    .expect("the reply decodes");

    assert_eq!(
        response.raw,
        serde_json::from_str::<serde_json::Value>(body).expect("the body is JSON")
    );
    assert_eq!(response.raw["unmodeled"], "kept");
}
