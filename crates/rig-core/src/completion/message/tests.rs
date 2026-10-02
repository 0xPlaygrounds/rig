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
        super::ToolFunction {
            name: super::ToolName::new("add").expect("tool name"),
            arguments: serde_json::json!({}),
        },
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

/// Every call has exactly one id: the provider's, or a fresh rig-issued one.
#[test]
fn a_call_carries_the_providers_id_or_a_fresh_local_one() {
    use super::{CallId, ProviderCallId};
    let provider = CallId::from_wire("call_1");
    assert_eq!(
        provider.provider(),
        Some(&ProviderCallId::new("call_1").expect("non-empty"))
    );
    assert_eq!(provider.wire(), "call_1");

    let (first, second) = (CallId::from_wire(""), CallId::from_wire(""));
    assert!(first.is_local() && second.is_local());
    assert_ne!(first, second);
    assert_eq!(first.wire().len(), 36, "a hyphenated v4 UUID");
}

/// A result built from its call answers that call under its name.
#[test]
fn a_result_is_built_from_the_call_it_answers() {
    let call = super::ToolCall::from_wire(
        "call_1",
        super::ToolFunction::new(
            super::ToolName::new("add").expect("tool name"),
            serde_json::json!({}),
        ),
    );
    let result = call.result(vec![super::ToolResultContent::text("42")]);
    assert_eq!(result.call, call.id);
    assert_eq!(result.name, call.function.name);
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

/// An empty tool name is not a name.
#[test]
fn an_empty_tool_name_does_not_parse() {
    assert!(super::ToolName::new("").is_err());
    assert!(serde_json::from_str::<super::ToolName>(r#""""#).is_err());
}

#[test]
fn media_constructors_name_their_source_encoding() {
    use super::{
        AudioMediaType, DocumentMediaType, DocumentSourceKind, UserContent, VideoMediaType,
    };

    let source = |content: UserContent| match content {
        UserContent::Audio(audio) => audio.data,
        UserContent::Video(video) => video.data,
        UserContent::Document(document) => document.data,
        other => DocumentSourceKind::string(format!("not media: {other:?}")),
    };

    assert_eq!(
        source(UserContent::audio_base64(
            "UklGRg==",
            Some(AudioMediaType::WAV)
        )),
        DocumentSourceKind::base64("UklGRg==")
    );
    assert_eq!(
        source(UserContent::video_base64("AAAA", Some(VideoMediaType::MP4))),
        DocumentSourceKind::base64("AAAA")
    );
    assert_eq!(
        source(UserContent::document_base64(
            "JVBERi0=",
            Some(DocumentMediaType::PDF)
        )),
        DocumentSourceKind::base64("JVBERi0=")
    );
    assert_eq!(
        source(UserContent::document_text(
            "# Notes",
            Some(DocumentMediaType::MARKDOWN)
        )),
        DocumentSourceKind::string("# Notes")
    );
}
