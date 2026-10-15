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

/// The Messages route sends a key as a bearer token, which Ollama's cloud
/// requires, and nothing for a local daemon.
#[test]
fn the_messages_route_sends_a_key_as_a_bearer_token() {
    for (config, authorization) in [
        (OllamaConfig::new(), None),
        (
            OllamaConfig::new().with_api_key("ollama-key"),
            Some("Bearer ollama-key"),
        ),
    ] {
        let encoded = config
            .connect(crate::test_utils::RecordingHttpClient::new("{}"))
            .anthropic_completion("qwen3:4b")
            .wire
            .encode(CompletionRequest::new("hi"), Mode::Unary)
            .expect("the request encodes");
        let request = &encoded.request;
        assert_eq!(request.uri(), "http://localhost:11434/v1/messages");
        assert!(!request.headers().contains_key("x-api-key"));
        assert_eq!(
            request
                .headers()
                .get(http::header::AUTHORIZATION)
                .and_then(|value| value.to_str().ok()),
            authorization
        );
    }
}

/// The Messages route reads images only as data and documents only as
/// text, so the adapter replaces any other form before a request is sent.
#[test]
fn the_messages_route_takes_only_inline_media() {
    use crate::completion::{Media, Place, ReplayTarget};
    use crate::message::{
        Document, DocumentData, DocumentMediaType, DocumentSourceKind as Source, Image,
        ImageMediaType,
    };
    let wire = OllamaConfig::new().anthropic_completion("qwen3:4b");
    let image = |data: Source| Image {
        data,
        media_type: Some(ImageMediaType::PNG),
        ..Image::default()
    };
    let document = |data: DocumentData, media_type| Document {
        data,
        media_type: Some(media_type),
        additional_params: None,
    };
    let data = image(Source::base64("iVBORw0KGgo="));
    let url = image(Source::url("https://example.com/a.png"));
    let text = document(DocumentData::Text("plain".into()), DocumentMediaType::TXT);
    let pdf = document(Source::base64("JVBERi0=").into(), DocumentMediaType::PDF);
    assert!(wire.encodes("qwen3:4b", Media::Image(&data, Place::User)));
    assert!(!wire.encodes("qwen3:4b", Media::Image(&url, Place::User)));
    assert!(wire.encodes("qwen3:4b", Media::Document(&text)));
    assert!(!wire.encodes("qwen3:4b", Media::Document(&pdf)));
}

/// The Messages route reads its key from the environment when one is set,
/// and a missing one is no error: a local daemon takes none.
#[test]
fn the_messages_route_reads_an_optional_key_from_the_environment() {
    let config = crate::providers::anthropic::AnthropicConfig::from_env_with(
        &crate::providers::anthropic::wire::OLLAMA,
    );
    assert!(config.is_ok(), "{config:?}");
}
