use crate::completion::CompletionRequest;
use crate::providers::anthropic::{self, AnthropicConfig};
use crate::providers::openai::{self, OpenAIConfig};
use crate::wire::{Body, Mode, Wire};

/// The `x-opencode-session` header of `request`, if any.
fn session(request: &http::Request<Body>) -> Option<&str> {
    request
        .headers()
        .get(super::SESSION_HEADER)
        .and_then(|value| value.to_str().ok())
}

/// A client picks Chat Completions or Responses for each model by the API
/// the catalog lists for it, at the prefix each API sits under.
#[test]
fn a_client_reaches_each_model_on_the_api_its_catalog_row_names() {
    let zen = OpenAIConfig::with_key(&openai::wire::OPENCODE, "sk-test");
    for (model, sent) in [
        (
            "gpt-6-luna",
            "https://opencode.ai/inference/openai/v1/responses",
        ),
        (
            "glm-5.3",
            "https://opencode.ai/inference/openai/v1/chat/completions",
        ),
    ] {
        let encoded = zen
            .completion(model)
            .encode(CompletionRequest::new("hi"), Mode::Unary)
            .expect("encodes");
        assert_eq!(encoded.request.uri(), sent, "{model}");
        assert_eq!(session(&encoded.request), None, "Zen takes no session");
    }

    // A configured route outranks the catalog's.
    let encoded = zen
        .with_route(openai::Route::Chat)
        .completion("gpt-6-luna")
        .encode(CompletionRequest::new("hi"), Mode::Unary)
        .expect("encodes");
    assert_eq!(
        encoded.request.uri(),
        "https://opencode.ai/inference/openai/v1/chat/completions"
    );
}

/// Go refuses a request without `x-opencode-session`: every wire sends the
/// request's cache key there, so a conversation's turns share one session,
/// or a fresh id without one.
#[test]
fn go_names_the_session_on_every_wire() {
    let mut keyed = CompletionRequest::new("hi");
    keyed.options.cache_key = Some("conversation-1".to_owned());

    let go = OpenAIConfig::with_key(&openai::wire::OPENCODE_GO, "sk-test");
    let messages = AnthropicConfig::with_key(&anthropic::wire::OPENCODE_GO, "sk-test")
        .completion("qwen3.8-flash");
    let sent = [
        go.chat("glm-5.3").encode(keyed.clone(), Mode::Streaming),
        go.responses("gpt-6-luna")
            .encode(keyed.clone(), Mode::Streaming),
        messages.encode(keyed, Mode::Streaming),
    ]
    .map(|encoded| encoded.expect("encodes"));
    assert_eq!(
        sent.each_ref()
            .map(|encoded| encoded.request.uri().to_string()),
        [
            "https://opencode.ai/inference/go/openai/v1/chat/completions",
            "https://opencode.ai/inference/go/openai/v1/responses",
            "https://opencode.ai/inference/go/anthropic/v1/messages",
        ]
    );
    for encoded in &sent {
        assert_eq!(session(&encoded.request), Some("conversation-1"));
    }

    let encoded = messages
        .encode(CompletionRequest::new("hi"), Mode::Unary)
        .expect("encodes");
    assert!(session(&encoded.request).is_some_and(|id| !id.is_empty()));
}
