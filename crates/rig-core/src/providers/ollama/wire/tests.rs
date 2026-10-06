use super::*;
use crate::completion::CompletionRequest;
use crate::wire::secret::tests::a_config_reloads_without_its_credential;
use crate::wire::{Mode, Wire};

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

    let native = OllamaConfig::new()
        .with_base_url("http://ollama.internal:11434")
        .native_completion("qwen3:4b");
    let serialized = serde_json::to_string(&native).expect("the wire serializes");
    let restored: crate::providers::ollama::Chat =
        serde_json::from_str(&serialized).expect("the wire deserializes");
    assert_eq!(restored, native);

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
        .native_completion("qwen3:4b")
        .encode(CompletionRequest::new("hi"), Mode::Unary)
        .expect("the request encodes");
    let request = &encoded.request;
    assert_eq!(request.uri(), "http://localhost:11434/api/chat");
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
