use super::*;
use crate::{completion::Usage, message::ReasoningContent, streaming::MintKind};

#[test]
fn every_event_round_trips_through_serde() {
    let events = vec![
        StreamEvent::BlockStart {
            id: BlockId::wire("msg_1"),
            kind: BlockKind::Message,
        },
        StreamEvent::BlockStart {
            id: BlockId::minted(MintKind::Text, 0),
            kind: BlockKind::Text {
                additional_params: AdditionalParams::from_entries([("k", serde_json::json!(1))]),
            },
        },
        StreamEvent::text(BlockId::minted(MintKind::Text, 0), "hi"),
        StreamEvent::BlockDelta {
            id: BlockId::wire("rs_1"),
            delta: Delta::Reasoning {
                text: "think".into(),
            },
        },
        StreamEvent::BlockStart {
            id: BlockId::wire("call_1"),
            kind: BlockKind::ToolCall,
        },
        StreamEvent::BlockDelta {
            id: BlockId::wire("call_1"),
            delta: Delta::ToolName { name: "add".into() },
        },
        StreamEvent::BlockDelta {
            id: BlockId::wire("call_1"),
            delta: Delta::ToolArguments {
                arguments: "{\"x\":1}".into(),
            },
        },
        StreamEvent::BlockEnd {
            id: BlockId::wire("call_1"),
            end: BlockClose::ToolCall(
                ToolCallEnd::new(UnparseableToolInput::Drop).with_call_id("c1"),
            ),
            block: None,
        },
        StreamEvent::BlockEnd {
            id: BlockId::wire("rs_1"),
            end: BlockClose::Reasoning {
                reasoning: Some(Reasoning {
                    id: Some("rs_1".into()),
                    content: vec![ReasoningContent::Text {
                        text: "think".into(),
                        signature: Some("sig".into()),
                    }],
                }),
                signature: None,
                wire_sent: true,
            },
            block: None,
        },
        StreamEvent::BlockEnd {
            id: BlockId::minted(MintKind::Text, 0),
            end: BlockClose::Text,
            block: None,
        },
        StreamEvent::Final(StreamFinal::new("mock", Usage::new())),
        StreamEvent::Unknown(UnknownPayload::new(
            serde_json::json!({"type": "web_search_call"}),
        )),
    ];
    for event in events {
        let json = serde_json::to_string(&event).expect("serialize");
        let back: StreamEvent = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, event, "{json}");
    }
}

#[test]
fn empty_additional_params_are_a_decode_error() {
    let json =
        r#"{"event":"block_start","id":"wire:t","kind":{"kind":"text","additional_params":{}}}"#;
    assert!(serde_json::from_str::<StreamEvent>(json).is_err());
    let json = r#"{"event":"block_start","id":"wire:t","kind":{"kind":"text"}}"#;
    assert!(matches!(
        serde_json::from_str::<StreamEvent>(json).unwrap(),
        StreamEvent::BlockStart {
            kind: BlockKind::Text {
                additional_params: None
            },
            ..
        }
    ));
}
