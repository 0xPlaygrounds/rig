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

/// The response `frames` fold into on `wire`, in `mode`.
fn folded<W: Wire<Op = crate::operation::Completion, Frame = WireFrame>>(
    wire: &W,
    mode: Mode,
    frames: &[&str],
) -> Result<crate::completion::CompletionResponse, ProviderError> {
    crate::test_utils::decode_reply(
        wire,
        &CompletionRequest::new("hi"),
        mode,
        frames
            .iter()
            .map(|frame| WireFrame::Text((*frame).to_owned())),
        serde_json::Value::Null,
    )
}

/// Each wire reads its own API's frames: Compatibility API chunks and its
/// `[DONE]`, and native events.
#[test]
fn each_wire_reads_its_own_apis_frames() {
    let compatibility = folded(
        &cohere().completion("command-a-03-2025"),
        Mode::Streaming,
        &[
            r#"{"id":"c1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":"hi"}}]}"#,
            r#"{"id":"c1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
            "[DONE]",
        ],
    )
    .expect("the Compatibility API stream folds");
    assert_eq!(
        compatibility.choice,
        vec![crate::message::AssistantContent::text("hi")]
    );
    let native = folded(
        &cohere().native_completion("command-a-03-2025"),
        Mode::Streaming,
        &[
            r#"{"id":"n1","type":"message-start","delta":{"message":{"role":"assistant"}}}"#,
            r#"{"type":"content-start","index":0,"delta":{"message":{"content":{"type":"text","text":""}}}}"#,
            r#"{"type":"content-delta","index":0,"delta":{"message":{"content":{"text":"hi"}}}}"#,
            r#"{"type":"content-end","index":0}"#,
            r#"{"type":"message-end","delta":{"finish_reason":"COMPLETE"}}"#,
        ],
    )
    .expect("the native stream folds");
    assert_eq!(native.choice.len(), 1);
    assert_eq!(native.response_id(), Some("n1"));
}

/// Frames that run out before the API ends its reply are truncated.
#[test]
fn a_reply_cut_short_on_either_api_is_truncated() {
    let compatibility = cohere().completion("command-a-03-2025");
    let native = cohere().native_completion("command-a-03-2025");
    let chunk = r#"{"id":"c1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"hi"}}]}"#;
    let start = r#"{"id":"n1","type":"message-start","delta":{"message":{"role":"assistant"}}}"#;
    for frames in [vec![chunk], vec!["not json"]] {
        assert!(
            folded(&compatibility, Mode::Streaming, &frames).is_err(),
            "{frames:?}"
        );
    }
    for frames in [vec![start], vec!["not json"]] {
        assert!(
            folded(&native, Mode::Streaming, &frames).is_err(),
            "{frames:?}"
        );
    }
}

/// Cohere's error body, a `message` string, sent with a success status fails
/// the turn in either mode: with the body as sent on the native wire, and as
/// a reply that never ended on the Compatibility wire, whose chunks it is not.
#[test]
fn an_error_body_with_a_success_status_fails_the_turn() {
    let sent = r#"{ "message": "internal error", "id": "x" }"#;
    let body = [sent];
    for mode in [Mode::Unary, Mode::Streaming] {
        let native = folded(
            &cohere().native_completion("command-a-03-2025"),
            mode,
            &body,
        )
        .expect_err("an error body fails the native turn");
        assert_eq!(
            native.to_string(),
            format!("ProviderResponseError: {sent}"),
            "{mode:?}"
        );
        let compatibility = folded(&cohere().completion("command-a-03-2025"), mode, &body);
        assert!(
            matches!(compatibility, Err(ProviderError::Truncated)),
            "{mode:?}: {compatibility:?}"
        );
    }
}

/// The facts the Compatibility wire is given are the facts it answers with: its
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
