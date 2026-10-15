use super::*;
use crate::completion::CompletionRequest;
use crate::operation::Completion;
use crate::providers::anthropic::{self, AnthropicConfig};
use crate::providers::openai::wire::{OPENCODE, OPENCODE_GO};
use crate::wire::{Encoded, Mode, Wire};

/// The URI and session header of the request `wire` sends for `request`.
fn sent<W>(wire: &W, request: CompletionRequest) -> (String, Option<String>)
where
    W: Wire<Op = Completion, Payload = Encoded>,
{
    let encoded = wire
        .encode(request, Mode::Unary)
        .expect("the request encodes");
    let session = encoded
        .request
        .headers()
        .get(SESSION_HEADER)
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned);
    (encoded.request.uri().to_string(), session)
}

fn keyed(cache_key: &str) -> CompletionRequest {
    let mut request = CompletionRequest::new("hi");
    request.options.cache_key = Some(cache_key.to_owned());
    request
}

/// Each gateway sends a model to the endpoint its table lists it under.
#[test]
fn a_model_reaches_the_endpoint_that_serves_it() {
    let zen = OpenAIConfig::with_key(&OPENCODE, "key");
    let go = OpenAIConfig::with_key(&OPENCODE_GO, "key");
    for (config, model, uri) in [
        (
            &zen,
            "gpt-5.5",
            "https://opencode.ai/inference/v1/responses",
        ),
        (
            &zen,
            "grok-4.7",
            "https://opencode.ai/inference/v1/responses",
        ),
        (
            &zen,
            "kimi-k3",
            "https://opencode.ai/inference/v1/chat/completions",
        ),
        (
            &go,
            "muse-spark-1.3-contributor",
            "https://opencode.ai/inference/go/v1/responses",
        ),
        (
            &go,
            "glm-5.3",
            "https://opencode.ai/inference/go/v1/chat/completions",
        ),
    ] {
        let (sent, _) = sent(&config.completion(model), keyed("conversation"));
        assert_eq!(sent, uri, "{model}");
    }
    for (dialect, uri) in [
        (
            &anthropic::wire::OPENCODE,
            "https://opencode.ai/inference/v1/messages",
        ),
        (
            &anthropic::wire::OPENCODE_GO,
            "https://opencode.ai/inference/go/v1/messages",
        ),
    ] {
        let messages = AnthropicConfig::with_key(dialect, "key").completion("claude-haiku-5-5");
        assert_eq!(sent(&messages, keyed("conversation")).0, uri);
    }
}

/// Go names the conversation on every completion route, by the request's
/// cache key or else a fresh id; Zen names none.
#[test]
fn go_names_the_conversation_and_zen_does_not() {
    let go = OpenAIConfig::with_key(&OPENCODE_GO, "key");
    let messages =
        AnthropicConfig::with_key(&anthropic::wire::OPENCODE_GO, "key").completion("qwen3.8-max");
    for request in [keyed("conversation"), CompletionRequest::new("hi")] {
        let expected = request.options.cache_key.clone();
        for (_, session) in [
            sent(&go.completion("gpt-6-luna"), request.clone()),
            sent(&go.completion("kimi-k3"), request.clone()),
            sent(&messages, request.clone()),
        ] {
            let session = session.expect("Go names a session");
            assert!(!session.is_empty());
            if let Some(expected) = &expected {
                assert_eq!(&session, expected);
            }
        }
    }
    let zen = OpenAIConfig::with_key(&OPENCODE, "key");
    assert_eq!(
        sent(&zen.completion("kimi-k3"), keyed("conversation")).1,
        None
    );
}
