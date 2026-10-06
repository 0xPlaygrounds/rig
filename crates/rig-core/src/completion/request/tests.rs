use super::{CompletionResponse, FinishReason, Usage};
use crate::completion::CompletionRequest;
use crate::error::ProviderError;
use crate::message::AssistantContent;
use crate::{http_client, provider_response};

/// An empty conversation or message parses, and the request boundary
/// rejects it by role and index. An empty tool result is the adapter's to
/// fill, as pi says `(no tool output)`.
mod empty_lists_parse_and_are_rejected_when_sent {
    use crate::message::Message;
    use crate::test_utils::{MockCompletionModel, MockTurn};
    use serde_json::json;

    #[test]
    fn an_empty_history_parses() {
        let mut request =
            serde_json::to_value(super::CompletionRequest::new("hello")).expect("serializes");
        request["chat_history"] = json!([]);
        let request = serde_json::from_value::<super::CompletionRequest>(request)
            .expect("an empty history parses");
        assert!(request.chat_history.is_empty());
    }

    /// Deserializing `[]` is not where the rule lives: the message reads and
    /// writes the same JSON, and sending it is what fails.
    #[tokio::test]
    async fn an_empty_content_list_round_trips_and_is_rejected_when_sent() {
        let json = json!({"role": "user", "content": []});
        let message = serde_json::from_value::<Message>(json.clone()).expect("parses");
        assert_eq!(serde_json::to_value(&message).expect("serializes"), json);

        let model = MockCompletionModel::from_turns([MockTurn::text("unreachable")]);
        let error = model
            .call(super::CompletionRequest::new(message))
            .await
            .expect_err("sending it is rejected");
        assert!(
            error
                .to_string()
                .contains("user message at index 0 has no content"),
            "{error}"
        );
        assert_eq!(model.request_count(), 0);
    }
}

/// The request-boundary check accepts what providers accept.
mod message_content {}

fn tool_call_choice() -> Vec<AssistantContent> {
    vec![AssistantContent::tool_call(
        "call_1",
        crate::message::ToolName::new("lookup").expect("tool name"),
        serde_json::json!({"query": "rig"}),
    )]
}

#[test]
fn normalized_response_round_trips_through_serde() {
    let response = {
        let mut response = CompletionResponse::new(
            vec![AssistantContent::text("hello")],
            Usage {
                input_tokens: Some(3),
                output_tokens: Some(2),
                total_tokens: Some(5),
                cached_input_tokens: Some(1),
                cache_creation_input_tokens: Some(0),
                tool_use_prompt_tokens: Some(0),
                reasoning_tokens: Some(1),
            },
            crate::message::Origin::new("test.api", "example", ""),
            serde_json::json!({}),
        )
        .with_finish_reason(FinishReason::Stop);
        response.origin.response_model = Some("provider-model-v2".into());
        response
    };

    let encoded = serde_json::to_value(&response).expect("serialize response");
    let decoded =
        serde_json::from_value::<CompletionResponse>(encoded.clone()).expect("deserialize");

    assert_eq!(
        serde_json::to_value(decoded).expect("re-serialize"),
        encoded
    );
}

/// Serde must not be a back door around `reconcile_with_output`: a
/// persisted `"stop"` next to a tool-call choice deserializes as
/// `ToolCalls`, exactly as if it had gone through the setter.
#[test]
fn deserializing_stop_with_a_tool_call_reconciles_to_tool_calls() {
    let mut encoded = serde_json::to_value(CompletionResponse::new(
        tool_call_choice(),
        Usage::default(),
        crate::message::Origin::new("test.api", "example", ""),
        serde_json::json!({}),
    ))
    .expect("serialize response");
    encoded["finish_reason"] = serde_json::json!("stop");

    let decoded =
        serde_json::from_value::<CompletionResponse>(encoded).expect("deserialize response");

    assert_eq!(decoded.finish_reason(), Some(FinishReason::ToolCalls));
}

/// Serde must not be a back door around the empty-string filtering either:
/// a persisted `""` identifier deserializes as `None`.
#[test]
fn deserializing_empty_identifiers_yields_none() {
    let mut encoded = serde_json::to_value(CompletionResponse::new(
        vec![AssistantContent::text("hello")],
        Usage::default(),
        crate::message::Origin::new("test.api", "example", ""),
        serde_json::json!({}),
    ))
    .expect("serialize response");
    encoded["origin"]["response_id"] = serde_json::json!("");
    encoded["origin"]["response_model"] = serde_json::json!("");

    let decoded =
        serde_json::from_value::<CompletionResponse>(encoded).expect("deserialize response");

    assert_eq!(decoded.response_id(), None);
    assert_eq!(decoded.model(), None);
}

#[test]
fn unknown_finish_reason_survives_a_serde_round_trip_verbatim() {
    let reason = FinishReason::Other("provider_specific_stop".to_owned());
    let encoded = serde_json::to_string(&reason).expect("serialize");
    let decoded = serde_json::from_str::<FinishReason>(&encoded).expect("deserialize");

    assert_eq!(decoded, reason);
}

use super::*;

#[test]
fn completion_request_content_telemetry_is_opt_in_and_not_serialized() {
    let default_request = CompletionRequest::new("completion prompt");
    assert!(!default_request.record_telemetry_content);

    let default_json = serde_json::to_value(&default_request).expect("serialize request");
    assert!(
        default_json.get("record_telemetry_content").is_none(),
        "safe default should not serialize the telemetry opt-in field"
    );
    let default_roundtrip: CompletionRequest =
        serde_json::from_value(default_json).expect("deserialize default request");
    assert!(!default_roundtrip.record_telemetry_content);

    let opt_in_request = CompletionRequest::new("completion prompt").record_content_telemetry(true);
    assert!(opt_in_request.record_telemetry_content);

    let opt_in_json = serde_json::to_value(&opt_in_request).expect("serialize opt-in request");
    assert!(
        opt_in_json.get("record_telemetry_content").is_none(),
        "local telemetry policy must not be serialized into provider requests"
    );
    let without_field: CompletionRequest =
        serde_json::from_value(opt_in_json).expect("deserialize a request without the field");
    assert!(
        !without_field.record_telemetry_content,
        "missing field should deserialize to the safe default"
    );
}

/// The deserialization mirror carries `raw`: a response with a captured
/// payload survives serialize → deserialize with the payload intact, and a
/// response written without the field is refused rather than loaded with
/// `raw` invented.
#[test]
fn normalized_response_raw_round_trips_through_serde_mirror() {
    let payload = serde_json::json!({
        "id": "chatcmpl-1",
        "system_fingerprint": "fp_abc",
        "choices": [{"finish_reason": "stop"}]
    });
    let response = {
        let mut response = CompletionResponse::new(
            vec![AssistantContent::text("hello")],
            Usage::default(),
            crate::message::Origin::new("test.api", "example", ""),
            payload.clone(),
        );
        response.origin.response_id = Some("chatcmpl-1".into());
        response
    };

    let encoded = serde_json::to_value(&response).expect("serialize response");
    assert_eq!(encoded["raw"], payload);
    let decoded: CompletionResponse =
        serde_json::from_value(encoded.clone()).expect("deserialize response");
    assert_eq!(decoded.raw, payload);
    assert_eq!(decoded.response_id(), Some("chatcmpl-1"));
    assert_eq!(
        serde_json::to_value(&decoded).expect("re-serialize"),
        encoded
    );

    let without_raw = serde_json::json!({
        "choice": [{"type": "text", "text": "hello"}],
        "usage": serde_json::to_value(Usage::default()).unwrap(),
        "origin": {"api": "test.api", "provider": "example", "model": ""}
    });
    let error = serde_json::from_value::<CompletionResponse>(without_raw)
        .expect_err("a response without `raw` is refused");
    assert!(error.to_string().contains("raw"), "{error}");
}

fn test_document(id: &str, text: &str) -> Document {
    Document {
        id: id.to_string(),
        text: text.to_string(),
        additional_props: HashMap::new(),
    }
}

#[test]
fn test_document_display_with_metadata() {
    let mut additional_props = HashMap::new();
    additional_props.insert("author".to_string(), "John Doe".to_string());
    additional_props.insert("length".to_string(), "42".to_string());

    let doc = Document {
        id: "123".to_string(),
        text: "This is a test document.".to_string(),
        additional_props,
    };

    let expected = concat!(
        "<file id: 123>\n",
        "<metadata author: \"John Doe\" length: \"42\" />\n",
        "This is a test document.\n",
        "</file>\n"
    );
    assert_eq!(format!("{doc}"), expected);
}

#[test]
fn documents_join_the_first_user_message_so_roles_alternate() {
    let request = CompletionRequest::new(Message::user("Prompt"))
        .message(Message::system("System prompt"))
        .message(Message::user("Earlier user turn"))
        .document(test_document("doc1", "Document text."));
    let history = request.chat_history_with_documents();
    let roles: Vec<&str> = history
        .iter()
        .map(|message| match message {
            Message::System { .. } => "system",
            Message::User { .. } => "user",
            Message::Assistant(_) => "assistant",
        })
        .collect();
    assert_eq!(roles, ["system", "user", "user"]);
    let Message::User { content } = &history[1] else {
        panic!("the first user message: {history:?}");
    };
    assert!(matches!(content.first(), Some(UserContent::Document(_))));
    assert!(
        content.iter().any(
            |part| matches!(part, UserContent::Text(text) if text.text == "Earlier user turn")
        )
    );
}

#[test]
fn completion_error_provider_response_helpers_with_preserved_json_body() {
    let body = r#"{"error":{"code":"rate_limit","message":"slow down"}}"#;
    let error = ProviderError::ProviderResponse(
        provider_response::ProviderResponseError::without_status(body.to_string()),
    );

    assert_eq!(error.provider_response_body(), Some(body));
    assert_eq!(error.provider_response_status(), None);
    assert_eq!(
        error
            .provider_response_json()
            .expect("fixture body should parse as valid JSON"),
        Some(serde_json::json!({
            "error": {
                "code": "rate_limit",
                "message": "slow down"
            }
        }))
    );
}

#[test]
fn completion_error_provider_response_helpers_with_preserved_plain_text_body() {
    let error = ProviderError::ProviderResponse(
        provider_response::ProviderResponseError::without_status("provider exploded".to_string()),
    );

    assert_eq!(error.provider_response_body(), Some("provider exploded"));
    assert_eq!(error.provider_response_status(), None);
    assert!(error.provider_response_json().is_err());
}

#[test]
fn completion_error_provider_response_helpers_with_http_non_success_body_and_status() {
    let body = r#"{"error":{"type":"invalid_request","message":"bad request"}}"#;
    let error = ProviderError::from_transport_error(http_client::Error::non_success_with_details(
        http::StatusCode::BAD_REQUEST,
        http::HeaderMap::new(),
        body.to_string(),
    ));

    assert_eq!(error.provider_response_body(), Some(body));
    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::BAD_REQUEST)
    );
    assert_eq!(
        error.provider_response_json().expect("valid JSON body"),
        Some(serde_json::json!({
            "error": {
                "type": "invalid_request",
                "message": "bad request"
            }
        }))
    );
}

mod additional_params_precedence {}

/// An unknown finish reason fails the turn by default, and replay leaves the
/// turn out. A request that accepts unknown reasons gets a response that
/// stops normally, and the turn replays: the response's stop, the run rule
/// and replay read the one answer.
mod unknown_finish_reasons {
    use crate::completion::message::turn_failure;
    use crate::completion::{CompletionRequest, FinishReason};
    use crate::message::{AssistantContent, Message, StopReason};
    use crate::test_utils::{MockCompletionModel, MockTurn};

    fn weird(turn: MockTurn) -> MockTurn {
        turn.with_finish_reason(FinishReason::Other("weird".to_owned()))
    }

    async fn turn(accept: bool, turn: MockTurn) -> (Message, Option<String>, bool) {
        let model = MockCompletionModel::from_turns([turn, MockTurn::text("next")]);
        let response = model
            .call(CompletionRequest::new("hi").accept_unknown_finish_reasons(accept))
            .await
            .expect("the reply folds");
        assert_eq!(response.accepts_unknown_finish_reasons(), accept);
        let head = response.head();
        let failure = turn_failure(
            &response.choice,
            head.stop.as_ref(),
            response.finish_reason().as_ref(),
        );
        let message = response.message().expect("the turn has content");
        model
            .call(CompletionRequest::new("again").messages([Message::user("hi"), message.clone()]))
            .await
            .expect("the follow-up folds");
        let replayed = model.requests()[1]
            .chat_history
            .iter()
            .any(|message| matches!(message, Message::Assistant(_)));
        (message, failure, replayed)
    }

    #[tokio::test]
    async fn a_text_turn_fails_and_is_left_out_by_default() {
        let (message, failure, replayed) = turn(false, weird(MockTurn::text("answer"))).await;
        let Message::Assistant(turn) = message else {
            panic!("an assistant turn");
        };
        assert_eq!(
            turn.stop,
            Some(StopReason::Error(
                "Provider finish_reason: weird".to_owned()
            ))
        );
        assert!(failure.is_some_and(|failure| failure.contains("weird")));
        assert!(!replayed, "replay leaves the failed turn out");
    }

    #[tokio::test]
    async fn an_accepted_text_turn_succeeds_and_replays() {
        let (message, failure, replayed) = turn(true, weird(MockTurn::text("answer"))).await;
        let Message::Assistant(turn) = message else {
            panic!("an assistant turn");
        };
        assert_eq!(turn.stop, Some(StopReason::Stop));
        assert_eq!(failure, None);
        assert!(replayed, "replay keeps the accepted turn");
    }

    #[tokio::test]
    async fn a_turn_with_calls_runs_them_only_when_accepted() {
        let call = || {
            weird(MockTurn::tool_call(
                "call_1",
                "lookup",
                serde_json::json!({}),
            ))
        };
        let (_, failure, _) = turn(false, call()).await;
        assert!(failure.is_some_and(|failure| failure.contains("none of its tool calls ran")));
        let (message, failure, _) = turn(true, call()).await;
        let Message::Assistant(turn) = message else {
            panic!("an assistant turn");
        };
        assert_eq!(turn.stop, Some(StopReason::ToolUse));
        assert_eq!(failure, None);
    }

    /// Accepting unknown reasons never accepts filtered content or a failure
    /// the provider reported.
    #[test]
    fn filtered_content_and_reported_failures_still_fail() {
        let accepted = |reason: FinishReason| {
            super::CompletionResponse::new(
                vec![AssistantContent::text("partial")],
                super::Usage::default(),
                crate::message::Origin::new("mock", "mock", "mock"),
                serde_json::Value::Null,
            )
            .with_finish_reason(reason)
            .accept_unknown_finish_reasons(true)
        };
        assert!(accepted(FinishReason::ContentFilter).stop().is_failure());
        let mut reported = accepted(FinishReason::Other("weird".to_owned()));
        reported.error = Some("refused".to_owned());
        assert_eq!(reported.stop(), StopReason::Error("refused".to_owned()));
        assert_eq!(
            accepted(FinishReason::Other("weird".to_owned())).stop(),
            StopReason::Stop
        );
    }

    /// The choice travels with the response and the request, and neither
    /// writes it when it is off.
    #[test]
    fn the_choice_round_trips_and_is_omitted_when_off() {
        let request = CompletionRequest::new("hi").accept_unknown_finish_reasons(true);
        let json = serde_json::to_value(&request).expect("serializes");
        assert_eq!(json["accept_unknown_finish_reasons"], true);
        let back: CompletionRequest = serde_json::from_value(json).expect("parses");
        assert!(back.accept_unknown_finish_reasons);
        let off = serde_json::to_value(CompletionRequest::new("hi")).expect("serializes");
        assert!(off.get("accept_unknown_finish_reasons").is_none());

        let response = super::CompletionResponse::new(
            Vec::new(),
            super::Usage::default(),
            crate::message::Origin::new("mock", "mock", "mock"),
            serde_json::Value::Null,
        );
        let off = serde_json::to_value(&response).expect("serializes");
        assert!(off.get("accepts_unknown_finish_reasons").is_none());
        let on =
            serde_json::to_value(response.accept_unknown_finish_reasons(true)).expect("serializes");
        let back: super::CompletionResponse = serde_json::from_value(on).expect("parses");
        assert!(back.accepts_unknown_finish_reasons());
    }
}
