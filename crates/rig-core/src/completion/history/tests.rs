use serde_json::json;

use super::*;
use crate::message::{
    AssistantContent, AssistantMessage, CallId, Image, Message, Opaque, Origin, Reasoning,
    StopReason, Text, ToolCall, ToolFunction, ToolName, ToolResult, ToolResultContent, UserContent,
};

#[derive(Debug)]
struct Target {
    accepts_images: bool,
}

impl ReplayTarget for Target {
    fn api(&self) -> Api {
        Api::from_static("test.api")
    }

    fn provider(&self) -> &str {
        "test"
    }

    fn model(&self) -> &str {
        "model-a"
    }

    fn accepts_images(&self, model: &str) -> bool {
        self.accepts_images && model != "text-only"
    }

    fn normalize_tool_call_id(&self, id: &str, _source: Option<&Origin>) -> String {
        id.replace('|', "_")
    }
}

const TARGET: Target = Target {
    accepts_images: true,
};

fn same() -> Origin {
    Origin::new("test.api", "test", "model-a")
}

fn other() -> Origin {
    Origin::new("test.api", "test", "model-b")
}

fn turn(origin: Option<Origin>, content: Vec<AssistantContent>) -> Message {
    Message::Assistant(AssistantMessage {
        content,
        origin,
        stop: Some(StopReason::Stop),
        native: None,
    })
}

fn call(id: &str) -> AssistantContent {
    AssistantContent::ToolCall(ToolCall::new(
        CallId::from_wire(id),
        ToolFunction::new(ToolName::new("lookup").expect("a name"), json!({})),
    ))
}

fn result(id: &str, text: &str) -> UserContent {
    UserContent::ToolResult(ToolResult {
        call: CallId::from_wire(id),
        name: ToolName::new("lookup").expect("a name"),
        content: vec![ToolResultContent::text(text)],
    })
}

fn signed_reasoning() -> AssistantContent {
    AssistantContent::Reasoning(Reasoning::new("thinking"))
        .with_native(json!({"type": "thinking", "thinking": "thinking", "signature": "sig"}))
}

fn opaque(replay: bool) -> AssistantContent {
    AssistantContent::Opaque(Opaque {
        item: json!({"type": "web_search_call", "id": "ws_1"}),
        replay,
    })
}

fn assistant(message: &Message) -> &AssistantMessage {
    match message {
        Message::Assistant(turn) => turn,
        other => panic!("expected an assistant message, got {other:?}"),
    }
}

#[test]
fn the_same_model_keeps_provider_items_and_replayable_opaque_items() {
    let text =
        AssistantContent::text("answer").with_native(json!({"type": "text", "text": "answer"}));
    let history = vec![
        Message::user("q"),
        turn(
            Some(same()),
            vec![signed_reasoning(), opaque(true), text.clone()],
        ),
    ];
    let adapted = adapt(&history, &TARGET);
    assert_eq!(adapted, history);
    assert!(assistant(&adapted[1]).content[0].native_item().is_some());
}

#[test]
fn opaque_items_that_do_not_replay_are_always_dropped() {
    let history = vec![turn(
        Some(same()),
        vec![opaque(false), AssistantContent::text("a")],
    )];
    let adapted = adapt(&history, &TARGET);
    assert_eq!(
        assistant(&adapted[0]).content,
        vec![AssistantContent::text("a")]
    );
}

#[test]
fn another_model_replays_canonical_fields_only() {
    let redacted = AssistantContent::Reasoning(Reasoning {
        text: String::new(),
        redacted: true,
        native: None,
    })
    .with_native(json!({"type": "redacted_thinking", "data": "x"}));
    let empty = AssistantContent::Reasoning(Reasoning::new("  "));
    let text = AssistantContent::text("answer").with_native(json!({"id": "msg_1"}));
    let history = vec![turn(
        Some(other()),
        vec![
            signed_reasoning(),
            redacted,
            empty,
            opaque(true),
            text,
            call("c|1"),
        ],
    )];
    let adapted = adapt(&history, &TARGET);
    let turn = assistant(&adapted[0]);
    assert_eq!(
        turn.content,
        vec![
            AssistantContent::Text(Text::new("thinking")),
            AssistantContent::text("answer"),
            call("c_1"),
        ]
    );
    assert!(turn.native.is_none());
}

#[test]
fn a_hand_built_turn_replays_canonically() {
    let text = AssistantContent::text("a").with_native(json!({"id": "msg_1"}));
    let adapted = adapt(&[turn(None, vec![text])], &TARGET);
    assert_eq!(
        assistant(&adapted[0]).content,
        vec![AssistantContent::text("a")]
    );
}

#[test]
fn sameness_needs_api_provider_and_model() {
    let text = AssistantContent::text("a").with_native(json!({"id": "msg_1"}));
    for origin in [
        Origin::new("other.api", "test", "model-a"),
        Origin::new("test.api", "other", "model-a"),
        other(),
    ] {
        let adapted = adapt(&[turn(Some(origin), vec![text.clone()])], &TARGET);
        assert!(assistant(&adapted[0]).content[0].native_item().is_none());
    }
}

#[test]
fn blank_text_and_empty_reasoning_without_items_are_dropped() {
    let history = vec![turn(
        Some(same()),
        vec![
            AssistantContent::text(" "),
            AssistantContent::Reasoning(Reasoning::new("")),
            AssistantContent::text("kept"),
        ],
    )];
    let adapted = adapt(&history, &TARGET);
    assert_eq!(
        assistant(&adapted[0]).content,
        vec![AssistantContent::text("kept")]
    );
}

#[test]
fn failed_turns_are_skipped_with_the_results_answering_them() {
    for stop in [
        StopReason::Error("refused".into()),
        StopReason::Aborted("cancelled".into()),
    ] {
        let failed = Message::Assistant(AssistantMessage {
            content: vec![call("c1")],
            origin: Some(same()),
            stop: Some(stop),
            native: None,
        });
        let history = vec![
            Message::user("q"),
            failed,
            Message::User {
                content: vec![result("c1", "late")],
            },
            Message::user("again"),
        ];
        assert_eq!(
            adapt(&history, &TARGET),
            vec![Message::user("q"), Message::user("again")]
        );
    }
}

#[test]
fn unanswered_calls_get_an_error_result_before_the_next_message() {
    let history = vec![
        Message::user("q"),
        turn(Some(same()), vec![call("c1"), call("c2")]),
        Message::User {
            content: vec![result("c2", "two"), UserContent::text("and then")],
        },
        turn(Some(same()), vec![call("c3")]),
        turn(Some(same()), vec![AssistantContent::text("done")]),
        turn(Some(same()), vec![call("c4")]),
    ];
    let adapted = adapt(&history, &TARGET);
    assert_eq!(
        adapted[2],
        Message::User {
            content: vec![
                result("c2", "two"),
                result("c1", NO_RESULT_PROVIDED),
                UserContent::text("and then"),
            ],
        }
    );
    assert_eq!(
        adapted[4],
        Message::User {
            content: vec![result("c3", NO_RESULT_PROVIDED)],
        }
    );
    assert_eq!(
        adapted.last(),
        Some(&Message::User {
            content: vec![result("c4", NO_RESULT_PROVIDED)],
        })
    );
    assert_eq!(adapted.len(), 8);
}

#[test]
fn system_messages_wait_for_pending_results() {
    let history = vec![
        turn(Some(same()), vec![call("c1")]),
        Message::system("later instructions"),
        Message::User {
            content: vec![result("c1", "one")],
        },
    ];
    assert_eq!(
        adapt(&history, &TARGET),
        vec![
            history[0].clone(),
            history[2].clone(),
            Message::system("later instructions"),
        ]
    );
}

#[test]
fn renamed_calls_rename_their_results() {
    let history = vec![
        turn(Some(other()), vec![call("call|fc_1")]),
        Message::User {
            content: vec![result("call|fc_1", "r")],
        },
    ];
    let adapted = adapt(&history, &TARGET);
    assert_eq!(assistant(&adapted[0]).content, vec![call("call_fc_1")]);
    assert_eq!(
        adapted[1],
        Message::User {
            content: vec![result("call_fc_1", "r")],
        }
    );
}

#[test]
fn images_become_placeholders_for_a_model_without_image_input() {
    let image = || Image {
        data: crate::message::DocumentSourceKind::url("https://example.invalid/a.png"),
        ..Image::default()
    };
    let history = vec![Message::User {
        content: vec![
            UserContent::Image(image()),
            UserContent::Image(image()),
            UserContent::text("look"),
            UserContent::ToolResult(ToolResult {
                call: CallId::from_wire("c1"),
                name: ToolName::new("lookup").expect("a name"),
                content: vec![
                    ToolResultContent::Image(image()),
                    ToolResultContent::Image(image()),
                ],
            }),
        ],
    }];
    let adapted = adapt(
        &history,
        &Target {
            accepts_images: false,
        },
    );
    assert_eq!(
        adapted,
        vec![Message::User {
            content: vec![
                UserContent::text(USER_IMAGE_OMITTED),
                UserContent::text("look"),
                UserContent::ToolResult(ToolResult {
                    call: CallId::from_wire("c1"),
                    name: ToolName::new("lookup").expect("a name"),
                    content: vec![ToolResultContent::text(TOOL_IMAGE_OMITTED)],
                }),
            ],
        }]
    );
}

#[test]
fn a_turn_left_empty_is_dropped() {
    let history = vec![
        Message::user("q"),
        turn(Some(other()), vec![opaque(true)]),
        Message::user("again"),
    ];
    assert_eq!(
        adapt(&history, &TARGET),
        vec![Message::user("q"), Message::user("again")]
    );
}

#[test]
fn a_request_model_override_is_the_model_compared() {
    let text = AssistantContent::text("a").with_native(json!({"id": "msg_1"}));
    let history = vec![turn(Some(other()), vec![text])];
    let adapted = adapt_for_model(&history, &TARGET, Some("model-b"));
    assert!(assistant(&adapted[0]).content[0].native_item().is_some());
}

#[test]
fn the_message_level_item_follows_sameness() {
    let mut message = AssistantMessage::new(vec![AssistantContent::text("a")])
        .with_native(json!({"role": "assistant", "content": "a"}));
    message.origin = Some(same());
    let adapted = adapt(&[Message::Assistant(message.clone())], &TARGET);
    assert!(assistant(&adapted[0]).native_item().is_some());

    message.origin = Some(other());
    let adapted = adapt(&[Message::Assistant(message)], &TARGET);
    assert!(assistant(&adapted[0]).native.is_none());
}

#[test]
fn a_request_model_override_decides_image_input() {
    let history = vec![Message::User {
        content: vec![UserContent::Image(Image {
            data: crate::message::DocumentSourceKind::url("https://example.invalid/a.png"),
            ..Image::default()
        })],
    }];
    assert_eq!(adapt(&history, &TARGET), history);
    assert_eq!(
        adapt_for_model(&history, &TARGET, Some("text-only")),
        vec![Message::User {
            content: vec![UserContent::text(USER_IMAGE_OMITTED)],
        }]
    );
}
