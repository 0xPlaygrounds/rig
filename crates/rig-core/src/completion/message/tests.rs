use serde::{Deserialize, Serialize};

use super::{Message, Text, ToolResultContent};

mod vec_content_serde {
    use super::super::{AssistantContent, Message, UserContent};

    #[test]
    fn message_content_still_serializes_as_a_plain_sequence() {
        // The removed container serialized as a bare sequence, which is why
        // this migration changes no persisted history and no recorded
        // provider fixture. Pin the wire shape so that stays true.
        let message = Message::User {
            content: vec![UserContent::text("hi")],
        };
        let json = serde_json::to_value(&message).expect("serialize");
        assert_eq!(
            json,
            serde_json::json!({
                "role": "user",
                "content": [{"type": "text", "text": "hi"}],
            })
        );
    }

    #[test]
    fn message_content_round_trips_byte_identically() {
        let message = Message::Assistant(crate::message::AssistantMessage::new(vec![
            AssistantContent::text("hello"),
        ]));
        let encoded = serde_json::to_string(&message).expect("serialize");
        let decoded: Message = serde_json::from_str(&encoded).expect("deserialize");
        assert_eq!(
            serde_json::to_string(&decoded).expect("re-serialize"),
            encoded
        );
    }

    #[test]
    fn an_empty_content_array_deserializes() {
        // The request boundary rejects it when it is sent, not serde.
        for message in [
            serde_json::json!({"role": "user", "content": []}),
            serde_json::json!({"role": "assistant", "content": []}),
        ] {
            let decoded = serde_json::from_value::<Message>(message.clone()).expect("parses");
            assert_eq!(serde_json::to_value(&decoded).expect("serializes"), message);
        }
    }
}

#[test]
fn system_message_constructor_and_serde_roundtrip() {
    let message = Message::system("You are concise.");

    match &message {
        Message::System { content } => assert_eq!(content, "You are concise."),
        _ => panic!("Expected system message"),
    }

    let json = serde_json::to_string(&message).expect("serialize");
    let roundtrip: Message = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(roundtrip, message);
}

#[test]
fn a_rig_issued_call_id_round_trips_as_local() {
    // A call the provider sent without an id carries a rig-issued one, and
    // the round trip never turns it into a provider id.
    let call = super::ToolCall::from_wire(
        "",
        super::ToolFunction::new(
            super::ToolName::new("add").expect("tool name"),
            serde_json::json!({}),
        ),
    );
    assert!(call.id.is_local());

    let json = serde_json::to_value(&call).expect("serialize");
    let roundtrip: super::ToolCall = serde_json::from_value(json).expect("deserialize");
    assert!(roundtrip.id.provider().is_none());
    assert_eq!(roundtrip, call);
}

#[test]
fn legacy_call_id_key_cannot_recover_an_untagged_identity() {
    let legacy = serde_json::json!({
        "id": "fc_123",
        "call_id": "call_abc",
        "function": {"name": "add", "arguments": {"x": 1}},
    });
    assert!(serde_json::from_value::<super::ToolCall>(legacy).is_err());
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct ExecutorLikeResponse {
    output: serde_json::Value,
    logs: Vec<String>,
    execution_time_ms: u64,
}

#[test]
fn tool_result_content_decodes_structured_and_legacy_json() {
    let response = ExecutorLikeResponse {
        output: serde_json::json!({"answer": 42}),
        logs: vec!["computed".to_string()],
        execution_time_ms: 7,
    };
    let value = serde_json::to_value(&response).expect("serialize response");

    let structured = ToolResultContent::json(value.clone());
    assert_eq!(structured.as_json(), Some(&value));
    assert_eq!(structured.as_text(), None);
    assert_eq!(
        structured
            .deserialize_json::<ExecutorLikeResponse>()
            .expect("decode structured response"),
        response
    );

    let legacy_json = value.to_string();
    let legacy_text = ToolResultContent::Text(Text::new(legacy_json.clone()));
    assert_eq!(legacy_text.as_text(), Some(legacy_json.as_str()));
    assert_eq!(legacy_text.as_json(), None);
    assert_eq!(
        legacy_text
            .deserialize_json::<ExecutorLikeResponse>()
            .expect("decode legacy response"),
        response
    );

    let image = ToolResultContent::image_url("https://example.com/result.png", None, None);
    let image_error = image.deserialize_json::<ExecutorLikeResponse>();
    assert!(image_error.is_err());
    if let Err(error) = image_error {
        assert_eq!(
            error.to_string(),
            "cannot decode image tool-result content as JSON"
        );
    }
}

/// The shapes persisted before calls had one id no longer parse.
#[test]
fn the_legacy_tool_call_id_shape_does_not_parse() {
    let legacy = serde_json::json!({
        "id": {"origin": "explicit", "id": "call_1"},
        "function": {"name": "add", "arguments": {}},
    });
    assert!(serde_json::from_value::<super::ToolCall>(legacy).is_err());
    assert!(serde_json::from_str::<super::CallId>(r#""call_1""#).is_err());
}

#[test]
fn a_stored_call_with_any_arguments_shape_loads() {
    let stored = serde_json::json!({"name": "search", "arguments": null});
    let function: super::ToolFunction =
        serde_json::from_value(stored).expect("a stored call loads");
    assert_eq!(function.arguments_value(), serde_json::json!({}));
    let stored = serde_json::json!({"name": "search", "arguments": "{\"q\":1}"});
    let function: super::ToolFunction =
        serde_json::from_value(stored).expect("a stored call loads");
    assert_eq!(function.arguments_value(), serde_json::json!({"q": 1}));
}

#[test]
fn an_unanswered_turn_fails_its_run_only_when_cut_or_failed() {
    use super::{AssistantContent, StopReason, turn_failure};
    use crate::completion::FinishReason;
    let empty: Vec<AssistantContent> = vec![AssistantContent::reasoning("thinking")];
    let answered = vec![AssistantContent::text("hi")];
    let failed = StopReason::Error("refused".into());
    assert_eq!(
        turn_failure(
            &empty,
            Some(&StopReason::Length),
            Some(&FinishReason::Length)
        ),
        Some(FinishReason::Length.no_answer_message())
    );
    assert!(turn_failure(&empty, Some(&failed), None).is_some_and(|m| m.contains("refused")));
    assert_eq!(
        turn_failure(&empty, Some(&StopReason::Stop), Some(&FinishReason::Stop)),
        None
    );
    assert_eq!(
        turn_failure(
            &answered,
            Some(&StopReason::Length),
            Some(&FinishReason::Length)
        ),
        None
    );
}

/// A turn the provider failed fails its run even when it holds an answer:
/// replay leaves it out, so the caller must not take it as a success.
#[test]
fn a_failed_turn_with_an_answer_fails_its_run() {
    use super::{AssistantContent, StopReason, turn_failure};
    use crate::completion::FinishReason;
    let answered = vec![AssistantContent::text("hi")];
    let failed = StopReason::Error("Provider finish_reason: weird".into());
    let message = turn_failure(
        &answered,
        Some(&failed),
        Some(&FinishReason::Other("weird".into())),
    )
    .expect("the run fails");
    assert_eq!(
        message,
        "the provider failed the turn: Provider finish_reason: weird"
    );
    let filtered = StopReason::Error("Provider finish_reason: content_filter".into());
    assert!(
        turn_failure(
            &answered,
            Some(&filtered),
            Some(&FinishReason::ContentFilter)
        )
        .is_some_and(|message| message.contains("content_filter"))
    );
    let aborted = StopReason::Aborted("the caller stopped".into());
    assert_eq!(turn_failure(&answered, Some(&aborted), None), None);
}

/// A turn that will not replay runs none of its tool calls, whatever else
/// it holds; a call in a turn the token limit ended is finished and runs.
#[test]
fn a_failed_turn_with_calls_fails_its_run() {
    use super::{AssistantContent, StopReason, ToolName, turn_failure};
    use crate::completion::FinishReason;
    let call = AssistantContent::tool_call(
        "call_1",
        ToolName::new("lookup").expect("a tool name"),
        serde_json::json!({}),
    );
    let turn = vec![AssistantContent::text("checking"), call];
    for stop in [
        StopReason::Error("Provider finish_reason: pause".into()),
        StopReason::Aborted("the stream ended".into()),
    ] {
        let message = turn_failure(&turn, Some(&stop), None).expect("the run fails");
        assert!(message.contains("none of its tool calls ran"), "{message}");
    }
    for (stop, finish) in [
        (StopReason::Length, FinishReason::Length),
        (StopReason::ToolUse, FinishReason::ToolCalls),
    ] {
        assert_eq!(turn_failure(&turn, Some(&stop), Some(&finish)), None);
    }
}

/// Redacted reasoning without a current provider item has nothing to send
/// (round-5 F7); with one, it replays.
#[test]
fn redacted_reasoning_without_its_item_is_blank() {
    use super::{AssistantContent, Reasoning};
    let redacted = AssistantContent::Reasoning(Reasoning {
        text: "kept".to_owned(),
        redacted: true,
        native: None,
    });
    assert!(redacted.is_blank());
    let with_item =
        redacted.with_native(serde_json::json!({"type": "redacted_thinking", "data": "x"}));
    assert!(!with_item.is_blank());
}

/// A document's text and its file keep the JSON they have always had: a
/// file is its source's spelling, text is `text`, and the old `string`
/// spelling loads as text for a document and as base64 data for media.
#[test]
fn document_data_keeps_its_json_and_reads_the_string_spelling() {
    use super::{DocumentData, DocumentSourceKind};
    use serde_json::json;

    let file = DocumentData::File(DocumentSourceKind::Base64("JVBERi0=".into()));
    let spelled = json!({"type": "base64", "value": "JVBERi0="});
    assert_eq!(serde_json::to_value(&file).ok(), Some(spelled.clone()));
    assert_eq!(
        serde_json::from_value::<DocumentData>(spelled).ok(),
        Some(file)
    );
    for source in [
        DocumentSourceKind::Url("https://example.com/a.pdf".into()),
        DocumentSourceKind::FileId("file_1".into()),
        DocumentSourceKind::Raw(vec![1, 2]),
        DocumentSourceKind::Unknown,
    ] {
        let as_source = serde_json::to_value(&source).ok();
        let data = DocumentData::from(source);
        let as_data = serde_json::to_value(&data).ok();
        assert_eq!(as_data, as_source);
        let back = as_data.and_then(|value| serde_json::from_value::<DocumentData>(value).ok());
        assert_eq!(back, Some(data));
    }

    let text = DocumentData::Text("notes".into());
    let spelled = json!({"type": "text", "value": "notes"});
    assert_eq!(serde_json::to_value(&text).ok(), Some(spelled.clone()));
    assert_eq!(
        serde_json::from_value::<DocumentData>(spelled).ok(),
        Some(text.clone())
    );

    let legacy = json!({"type": "string", "value": "notes"});
    assert_eq!(
        serde_json::from_value::<DocumentData>(legacy.clone()).ok(),
        Some(text)
    );
    assert_eq!(
        serde_json::from_value::<DocumentSourceKind>(legacy).ok(),
        Some(DocumentSourceKind::Base64("notes".into()))
    );
    assert_eq!(
        DocumentData::default(),
        DocumentData::File(DocumentSourceKind::Unknown)
    );
}

#[test]
fn a_call_answers_with_its_own_id_and_name_as_a_success_or_an_error() {
    let call = super::ToolCall::from_wire(
        "c1",
        super::ToolFunction::new(
            super::ToolName::new("add").expect("tool name"),
            serde_json::json!({}),
        ),
    );
    let ok = call.result(vec![ToolResultContent::text("3")]);
    let failed = call.error_result(vec![ToolResultContent::text("boom")]);
    assert!(!ok.is_error);
    assert!(failed.is_error);
    assert_eq!(
        (&failed.call, &failed.name),
        (&call.id, &call.function.name)
    );
    assert_eq!(failed.content, vec![ToolResultContent::text("boom")]);
}

mod image_media_type_sniff {
    use crate::message::ImageMediaType;

    #[test]
    fn known_signatures_are_recognised_and_name_their_extension() {
        let cases: [(&[u8], ImageMediaType, &str); 5] = [
            (b"\x89PNG\r\n\x1a\nrest", ImageMediaType::PNG, "png"),
            (b"\xff\xd8\xff\xe0", ImageMediaType::JPEG, "jpg"),
            (b"GIF87a..", ImageMediaType::GIF, "gif"),
            (b"GIF89a..", ImageMediaType::GIF, "gif"),
            (b"RIFF\0\0\0\0WEBPVP8 ", ImageMediaType::WEBP, "webp"),
        ];
        for (bytes, media_type, extension) in cases {
            assert_eq!(ImageMediaType::sniff(bytes), Some(media_type.clone()));
            assert_eq!(media_type.extension(), extension);
        }
    }

    #[test]
    fn unknown_or_truncated_bytes_are_not_guessed() {
        let inputs: [&[u8]; 5] = [b"", b"\x89PNG", b"GIF88a", b"RIFF\0\0\0\0WAVE", b"<svg"];
        for bytes in inputs {
            assert_eq!(ImageMediaType::sniff(bytes), None);
        }
        let others = [
            ImageMediaType::HEIC,
            ImageMediaType::HEIF,
            ImageMediaType::SVG,
        ];
        let extensions: Vec<&str> = others.iter().map(ImageMediaType::extension).collect();
        assert_eq!(extensions, ["heic", "heif", "svg"]);
    }
}
