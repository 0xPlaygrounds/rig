use super::*;
use crate::completion::CompletionRequest;
use crate::error::ProviderError;
use crate::test_utils::json_body;
use crate::wire::secret::tests::a_config_reloads_without_its_credential;
use crate::wire::{Mode, Wire, WireFrame};

fn cohere() -> CohereConfig {
    CohereConfig::new("cohere-test-key")
}

/// Chat goes to the Compatibility API under the configured root, with the
/// key as a bearer token.
#[test]
fn chat_addresses_the_compatibility_api() {
    let wire = cohere()
        .with_base_url("http://127.0.0.1:9/")
        .completion("command-a-03-2025");
    let encoded = wire
        .encode(crate::completion::CompletionRequest::new("hi"), Mode::Unary)
        .expect("the request encodes");
    assert_eq!(
        encoded.request.uri().to_string(),
        "http://127.0.0.1:9/compatibility/v1/chat/completions"
    );
    assert_eq!(
        encoded.request.headers()[http::header::AUTHORIZATION],
        "Bearer cohere-test-key"
    );
    assert_eq!(json_body(&encoded.request)["model"], "command-a-03-2025");
}

#[test]
fn a_serialized_config_carries_no_key_material() {
    a_config_reloads_without_its_credential(&cohere(), "cohere-test-key", |cohere| &cohere.api_key);

    let wire = cohere().completion("command-a-03-2025");
    let serialized = serde_json::to_string(&wire).expect("the wire serializes");
    assert!(
        !serialized.contains("cohere-test-key"),
        "a wire a host may persist must not carry the credential: {serialized}"
    );
}

/// The response `frames` fold into on the routing wire, streamed.
fn streamed(frames: &[&str]) -> Result<crate::completion::CompletionResponse, ProviderError> {
    let wire = cohere().completion("command-a-03-2025");
    crate::test_utils::decode_reply(
        &wire,
        &CompletionRequest::new("hi"),
        Mode::Streaming,
        frames
            .iter()
            .map(|frame| WireFrame::Text((*frame).to_owned())),
        serde_json::Value::Null,
    )
}

/// The routing decoder reads each frame by its shape: Compatibility API
/// chunks and its `[DONE]`, and native events.
#[test]
fn the_decoder_reads_either_api_by_its_frames() {
    let compatibility = streamed(&[
        r#"{"id":"c1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":"hi"}}]}"#,
        r#"{"id":"c1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
        "[DONE]",
    ])
    .expect("the Compatibility API stream folds");
    assert_eq!(
        compatibility.choice,
        vec![crate::message::AssistantContent::text("hi")]
    );
    let native = streamed(&[
        r#"{"id":"n1","type":"message-start","delta":{"message":{"role":"assistant"}}}"#,
        r#"{"type":"content-start","index":0,"delta":{"message":{"content":{"type":"text","text":""}}}}"#,
        r#"{"type":"content-delta","index":0,"delta":{"message":{"content":{"text":"hi"}}}}"#,
        r#"{"type":"content-end","index":0}"#,
        r#"{"type":"message-end","delta":{"finish_reason":"COMPLETE"}}"#,
    ])
    .expect("the native stream folds");
    assert_eq!(native.choice.len(), 1);
    assert_eq!(native.response_id(), Some("n1"));
}

/// Frames that run out before either API ends its reply are truncated.
#[test]
fn a_reply_cut_short_on_either_api_is_truncated() {
    for frames in [
        vec![r#"{"id":"n1","type":"message-start","delta":{"message":{"role":"assistant"}}}"#],
        vec![
            r#"{"id":"c1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"hi"}}]}"#,
        ],
        vec!["not json"],
    ] {
        assert!(streamed(&frames).is_err(), "{frames:?}");
    }
}

/// Cohere's error body, a `message` string, sent with a success status fails
/// the turn with its text on either mode, and is not read as a native reply.
#[test]
fn an_error_body_with_a_success_status_fails_the_turn() {
    let body = r#"{"id":"x","message":"internal error"}"#;
    for mode in [Mode::Unary, Mode::Streaming] {
        let error = crate::test_utils::decode_reply(
            &cohere().completion("command-a-03-2025"),
            &CompletionRequest::new("hi"),
            mode,
            [WireFrame::Text(body.to_owned())],
            serde_json::Value::Null,
        )
        .expect_err("an error body fails the turn");
        assert!(error.to_string().contains("internal error"), "{error}");
    }
}

/// The facts the Cohere wire (both of its APIs) is given are the facts it answers with: its
/// [`ReplayTarget::facts`](crate::completion::ReplayTarget::facts), which
/// the shared option rules and the cost read, and its descriptor's spec,
/// which `DynModel::spec` returns.
#[test]
fn the_wire_answers_from_the_facts_it_is_given() {
    use crate::catalog::{ModelFacts, ModelSpec};
    use crate::completion::ReplayTarget as _;
    use crate::wire::Wire as _;

    let vendor = crate::providers::registry::ProviderId::catalog("cohere").expect("a vendor");
    let spec = ModelSpec::new(vendor, "command-a-03-2025").with_max_output_tokens(1_234);
    let wire = cohere()
        .completion("command-a-03-2025")
        .with_facts(ModelFacts::new(spec));
    let bound = wire.facts().and_then(ModelFacts::spec);
    assert_eq!(bound.and_then(|spec| spec.max_output_tokens), Some(1_234));
    assert_eq!(
        wire.describe()
            .spec()
            .and_then(|spec| spec.max_output_tokens),
        Some(1_234)
    );
}
