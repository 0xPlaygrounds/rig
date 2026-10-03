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

    fn accepts(&self, model: &str) -> Accepts {
        if self.accepts_images && model != "text-only" {
            Accepts::ALL
        } else {
            Accepts {
                tools: true,
                ..Accepts::TEXT
            }
        }
    }

    fn normalize_tool_call_id(&self, id: &str, _model: &str, _source: Option<&Origin>) -> String {
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
        is_error: false,
    })
}

/// The error result `adapt` gives a call nothing answered.
fn no_result(id: &str) -> UserContent {
    match result(id, NO_RESULT_PROVIDED) {
        UserContent::ToolResult(result) => UserContent::ToolResult(ToolResult {
            is_error: true,
            ..result
        }),
        other => other,
    }
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
            vec![Message::User {
                content: vec![UserContent::text("q"), UserContent::text("again")],
            }],
            "the user messages the skipped turn separated become one"
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
                no_result("c1"),
                UserContent::text("and then"),
            ],
        }
    );
    assert_eq!(
        adapted[4],
        Message::User {
            content: vec![no_result("c3")],
        }
    );
    assert_eq!(
        adapted.last(),
        Some(&Message::User {
            content: vec![no_result("c4")],
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
    let history = vec![
        turn(Some(same()), vec![call("c1")]),
        Message::User {
            content: vec![
                UserContent::Image(image()),
                UserContent::Image(image()),
                UserContent::text("look"),
                UserContent::ToolResult(ToolResult {
                    is_error: false,
                    call: CallId::from_wire("c1"),
                    name: ToolName::new("lookup").expect("a name"),
                    content: vec![
                        ToolResultContent::Image(image()),
                        ToolResultContent::Image(image()),
                    ],
                }),
            ],
        },
    ];
    let adapted = adapt(
        &history,
        &Target {
            accepts_images: false,
        },
    );
    assert_eq!(
        adapted[1..],
        vec![Message::User {
            content: vec![
                UserContent::ToolResult(ToolResult {
                    is_error: false,
                    call: CallId::from_wire("c1"),
                    name: ToolName::new("lookup").expect("a name"),
                    content: vec![ToolResultContent::text(TOOL_IMAGE_OMITTED)],
                }),
                UserContent::text(USER_IMAGE_OMITTED),
                UserContent::text("look"),
            ],
        }]
    );
}

/// A target whose encoder takes inline images only, and no audio, video or
/// documents.
#[derive(Debug)]
struct InlineOnly;

impl ReplayTarget for InlineOnly {
    fn api(&self) -> Api {
        Api::from_static("test.api")
    }

    fn provider(&self) -> &str {
        "test"
    }

    fn model(&self) -> &str {
        "model-a"
    }

    fn accepts(&self, _model: &str) -> Accepts {
        Accepts::ALL
    }

    fn encodes(&self, _model: &str, media: Media<'_>) -> bool {
        match media {
            Media::Image(image, _) => {
                matches!(image.data, DocumentSourceKind::Base64(_)) && image.media_type.is_some()
            }
            Media::Audio(_) | Media::Video(_) | Media::Document(_) => false,
        }
    }
}

#[test]
fn media_the_encoder_cannot_carry_becomes_a_placeholder_or_its_text() {
    use crate::message::{Audio, Document, DocumentMediaType, Video};
    let url = || Image {
        data: DocumentSourceKind::url("https://example.invalid/a.png"),
        ..Image::default()
    };
    let history = vec![
        turn(Some(same()), vec![call("c1")]),
        Message::User {
            content: vec![
                UserContent::Image(url()),
                UserContent::Image(Image {
                    data: DocumentSourceKind::Unknown,
                    ..Image::default()
                }),
                UserContent::Audio(Audio {
                    data: DocumentSourceKind::base64("SUQz"),
                    media_type: None,
                }),
                UserContent::Video(Video {
                    data: DocumentSourceKind::url("https://example.invalid/a.mp4"),
                    media_type: None,
                    additional_params: None,
                }),
                UserContent::Document(Document {
                    data: DocumentSourceKind::base64("YSxiCjEsMgo="),
                    media_type: Some(DocumentMediaType::CSV),
                    additional_params: None,
                }),
                UserContent::Document(Document {
                    data: DocumentSourceKind::base64("JVBERi0="),
                    media_type: Some(DocumentMediaType::PDF),
                    additional_params: None,
                }),
                UserContent::ToolResult(ToolResult {
                    is_error: false,
                    call: CallId::from_wire("c1"),
                    name: ToolName::new("lookup").expect("a name"),
                    content: vec![ToolResultContent::Image(url())],
                }),
            ],
        },
    ];
    assert_eq!(
        adapt(&history, &InlineOnly)[1..],
        vec![Message::User {
            content: vec![
                UserContent::ToolResult(ToolResult {
                    is_error: false,
                    call: CallId::from_wire("c1"),
                    name: ToolName::new("lookup").expect("a name"),
                    content: vec![ToolResultContent::text(TOOL_IMAGE_OMITTED)],
                }),
                UserContent::text(IMAGE_UNSENDABLE),
                UserContent::text(AUDIO_UNSENDABLE),
                UserContent::text(VIDEO_UNSENDABLE),
                UserContent::text("a,b\n1,2\n"),
                UserContent::text(DOCUMENT_UNSENDABLE),
            ],
        }]
    );
}

#[test]
fn inline_images_are_base64_with_the_type_their_bytes_name() {
    let raw_jpeg = Image {
        data: DocumentSourceKind::Raw(vec![0xFF, 0xD8, 0xFF, 0xE0]),
        ..Image::default()
    };
    let untyped_png = Image {
        data: DocumentSourceKind::base64("iVBORw0KGgoAAAANSUhEUg=="),
        ..Image::default()
    };
    let history = vec![Message::User {
        content: vec![
            UserContent::Image(raw_jpeg),
            UserContent::Image(untyped_png),
        ],
    }];
    assert_eq!(
        adapt(&history, &InlineOnly),
        vec![Message::User {
            content: vec![
                UserContent::Image(Image {
                    data: DocumentSourceKind::base64("/9j/4A=="),
                    media_type: Some(ImageMediaType::JPEG),
                    ..Image::default()
                }),
                UserContent::Image(Image {
                    data: DocumentSourceKind::base64("iVBORw0KGgoAAAANSUhEUg=="),
                    media_type: Some(ImageMediaType::PNG),
                    ..Image::default()
                }),
            ],
        }]
    );
}

#[test]
fn results_come_before_the_text_of_the_message_they_merge_into() {
    let history = vec![
        Message::user("q"),
        turn(Some(same()), vec![call("c1")]),
        Message::user("and also"),
        Message::User {
            content: vec![result("c1", "done")],
        },
    ];
    assert_eq!(
        adapt(&history, &TARGET)[2..],
        vec![Message::User {
            content: vec![result("c1", "done"), UserContent::text("and also")],
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
        vec![Message::User {
            content: vec![UserContent::text("q"), UserContent::text("again")],
        }]
    );
}

#[test]
fn a_request_model_override_is_the_model_compared() {
    let text = AssistantContent::text("a").with_native(json!({"id": "msg_1"}));
    let history = vec![turn(Some(other()), vec![text])];
    let adapted = adapt_for(
        &history,
        &TARGET,
        &Request {
            model: Some("model-b"),
            ..Request::default()
        },
    );
    assert!(assistant(&adapted[0]).content[0].native_item().is_some());
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
        adapt_for(
            &history,
            &TARGET,
            &Request {
                model: Some("text-only"),
                ..Request::default()
            }
        ),
        vec![Message::User {
            content: vec![UserContent::text(USER_IMAGE_OMITTED)],
        }]
    );
}

#[test]
fn a_skipped_turns_ids_do_not_drop_a_later_turns_results() {
    // Kimi numbers calls per conversation, so a later turn can reuse an id.
    let failed = Message::Assistant(AssistantMessage {
        content: vec![call("functions.f:0")],
        origin: Some(same()),
        stop: Some(StopReason::Error("refused".into())),
    });
    let history = vec![
        Message::user("q"),
        failed,
        Message::user("again"),
        turn(Some(same()), vec![call("functions.f:0")]),
        Message::User {
            content: vec![result("functions.f:0", "real")],
        },
    ];
    let adapted = adapt(&history, &TARGET);
    assert_eq!(
        adapted.last(),
        Some(&Message::User {
            content: vec![result("functions.f:0", "real")],
        })
    );
}

#[test]
fn a_same_model_block_edited_blank_is_dropped() {
    let mut text = AssistantContent::text("answer").with_native(json!({"text": "answer"}));
    if let AssistantContent::Text(text) = &mut text {
        text.text = String::new();
    }
    let history = vec![
        Message::user("q"),
        turn(Some(same()), vec![text, AssistantContent::text("kept")]),
    ];
    assert_eq!(
        assistant(&adapt(&history, &TARGET)[1]).content,
        vec![AssistantContent::text("kept")]
    );
}

/// A target that normalizes every id to its first three characters, so two
/// distinct ids collide, and reads what `accepts` says.
#[derive(Debug)]
struct Narrow {
    accepts: Accepts,
}

impl ReplayTarget for Narrow {
    fn api(&self) -> Api {
        Api::from_static("test.api")
    }

    fn provider(&self) -> &str {
        "test"
    }

    fn model(&self) -> &str {
        "model-a"
    }

    fn accepts(&self, _model: &str) -> Accepts {
        self.accepts
    }

    fn normalize_tool_call_id(&self, id: &str, _model: &str, _source: Option<&Origin>) -> String {
        id.chars().take(3).collect()
    }
}

#[test]
fn blank_system_messages_are_dropped() {
    let history = vec![
        Message::system(""),
        Message::system("  \n"),
        Message::user("hi"),
        Message::system("keep"),
    ];
    assert_eq!(
        adapt(&history, &TARGET),
        vec![Message::user("hi"), Message::system("keep")]
    );
}

#[test]
fn a_result_no_call_asked_for_is_dropped() {
    let history = vec![
        Message::User {
            content: vec![result("gone", "stale"), UserContent::text("next")],
        },
        turn(Some(same()), vec![call("c1")]),
        Message::User {
            content: vec![result("c1", "one"), result("c1", "again")],
        },
    ];
    assert_eq!(
        adapt(&history, &TARGET),
        vec![
            Message::user("next"),
            turn(Some(same()), vec![call("c1")]),
            Message::User {
                content: vec![result("c1", "one")],
            },
        ]
    );
}

/// A conversation the provider stores holds the calls its first results
/// answer, so they are kept; a result after a turn still answers that turn.
#[test]
fn results_before_the_first_turn_of_a_stored_conversation_are_kept() {
    let history = vec![
        Message::User {
            content: vec![result("held", "kept")],
        },
        turn(Some(same()), vec![call("c1")]),
        Message::User {
            content: vec![result("gone", "stale"), result("c1", "one")],
        },
    ];
    assert_eq!(
        adapt_for(
            &history,
            &TARGET,
            &Request {
                stored: true,
                ..Request::default()
            }
        ),
        vec![
            Message::User {
                content: vec![result("held", "kept")],
            },
            turn(Some(same()), vec![call("c1")]),
            Message::User {
                content: vec![result("c1", "one")],
            },
        ]
    );
}

#[test]
fn results_split_over_messages_answer_the_turn_before_them() {
    let history = vec![
        turn(Some(same()), vec![call("c1"), call("c2")]),
        Message::User {
            content: vec![result("c1", "one")],
        },
        Message::User {
            content: vec![result("c2", "two")],
        },
    ];
    assert_eq!(
        adapt(&history, &TARGET)[1],
        Message::User {
            content: vec![result("c1", "one"), result("c2", "two")],
        }
    );
}

#[test]
fn normalized_ids_that_collide_stay_distinct() {
    let history = vec![
        turn(Some(other()), vec![call("abc-1"), call("abc-2")]),
        Message::User {
            content: vec![result("abc-1", "one"), result("abc-2", "two")],
        },
    ];
    let target = Narrow {
        accepts: Accepts::ALL,
    };
    let adapted = adapt(&history, &target);
    let Message::Assistant(turn) = &adapted[0] else {
        panic!("the turn is first: {adapted:?}");
    };
    let ids: Vec<String> = turn.tool_calls().map(|call| call.id.to_string()).collect();
    assert_eq!(ids.len(), 2);
    assert_ne!(ids[0], ids[1], "{ids:?}");
    assert!(ids.iter().all(|id| id.len() == 3), "{ids:?}");
    let Message::User { content } = &adapted[1] else {
        panic!("the results follow: {adapted:?}");
    };
    let answered: Vec<String> = content
        .iter()
        .filter_map(|part| match part {
            UserContent::ToolResult(result) => Some(result.call.to_string()),
            _ => None,
        })
        .collect();
    assert_eq!(answered, ids, "each result follows its call's new id");
}

#[test]
fn tool_result_images_move_to_a_user_message_when_only_users_send_images() {
    let image = Image {
        data: crate::message::DocumentSourceKind::base64("aW1hZ2U="),
        ..Image::default()
    };
    let history = vec![
        turn(Some(same()), vec![call("c1")]),
        Message::User {
            content: vec![UserContent::ToolResult(ToolResult {
                call: CallId::from_wire("c1"),
                name: ToolName::new("lookup").expect("a name"),
                content: vec![
                    ToolResultContent::text("shot"),
                    ToolResultContent::Image(image.clone()),
                ],
                is_error: false,
            })],
        },
    ];
    let target = Narrow {
        accepts: Accepts {
            tool_result_images: false,
            ..Accepts::ALL
        },
    };
    assert_eq!(
        adapt(&history, &target)[1],
        Message::User {
            content: vec![
                UserContent::ToolResult(ToolResult {
                    call: CallId::from_wire("c1"),
                    name: ToolName::new("lookup").expect("a name"),
                    // A model that reads no multimodal results gets one text.
                    content: vec![ToolResultContent::text(format!(
                        "shot\n{TOOL_IMAGE_ATTACHED}"
                    ))],
                    is_error: false,
                }),
                UserContent::text(TOOL_IMAGES_HEADING),
                UserContent::Image(image),
            ],
        }
    );
}

#[test]
fn another_models_assistant_image_becomes_text_for_a_model_that_reads_none() {
    let image = AssistantContent::Image(Image {
        data: crate::message::DocumentSourceKind::base64("aW1hZ2U="),
        ..Image::default()
    });
    let history = vec![Message::user("draw"), turn(Some(other()), vec![image])];
    let target = Narrow {
        accepts: Accepts {
            assistant_images: false,
            ..Accepts::ALL
        },
    };
    assert_eq!(
        adapt(&history, &target)[1],
        turn(
            Some(other()),
            vec![AssistantContent::text(ASSISTANT_IMAGE_OMITTED)]
        )
    );
}

#[test]
fn a_model_without_tools_reads_calls_and_results_as_text() {
    let history = vec![
        turn(Some(other()), vec![call("c1")]),
        Message::User {
            content: vec![result("c1", "sunny")],
        },
    ];
    let target = Narrow {
        accepts: Accepts {
            tools: false,
            ..Accepts::ALL
        },
    };
    assert_eq!(
        adapt(&history, &target),
        vec![
            turn(
                Some(other()),
                vec![AssistantContent::text("[called tool lookup with {}]")]
            ),
            Message::user("[tool lookup result] sunny"),
        ]
    );
}

#[test]
fn a_store_that_reorders_keys_keeps_the_native() {
    let block = AssistantContent::ToolCall(ToolCall::new(
        CallId::from_wire("c1"),
        ToolFunction::new(
            ToolName::new("lookup").expect("a name"),
            json!({"b": 1, "a": {"y": 2, "x": 3}}),
        ),
    ))
    .with_native(json!({"type": "function_call", "id": "c1"}));
    fn sorted(value: serde_json::Value) -> serde_json::Value {
        match value {
            serde_json::Value::Object(fields) => {
                let mut fields: Vec<_> = fields.into_iter().collect();
                fields.sort_by(|left, right| left.0.cmp(&right.0));
                serde_json::Value::Object(
                    fields
                        .into_iter()
                        .map(|(key, value)| (key, sorted(value)))
                        .collect(),
                )
            }
            serde_json::Value::Array(values) => {
                serde_json::Value::Array(values.into_iter().map(sorted).collect())
            }
            value => value,
        }
    }
    let stored = sorted(serde_json::to_value(&block).expect("serializes"));
    let loaded: AssistantContent = serde_json::from_value(stored).expect("loads");
    assert!(loaded.native_item().is_some(), "{loaded:?}");
}

/// A target whose calls carry their id at `/id`, and whose items keep `id`
/// through an edit.
#[derive(Debug)]
struct Slotted;

impl ReplayTarget for Slotted {
    fn api(&self) -> Api {
        Api::from_static("test.api")
    }

    fn provider(&self) -> &str {
        "test"
    }

    fn model(&self) -> &str {
        "model-a"
    }

    fn accepts(&self, _model: &str) -> Accepts {
        Accepts::ALL
    }

    fn identity(&self, item: &serde_json::Value) -> serde_json::Map<String, serde_json::Value> {
        item.as_object()
            .and_then(|fields| fields.get("id"))
            .map(|id| serde_json::Map::from_iter([("id".to_owned(), id.clone())]))
            .unwrap_or_default()
    }

    fn call_id_slot(&self) -> Option<&'static str> {
        Some("/id")
    }
}

#[test]
fn a_current_call_item_carries_the_wire_id_its_result_gets() {
    let call = AssistantContent::ToolCall(ToolCall::new(
        CallId::Local(crate::message::LocalCallId::new()),
        ToolFunction::new(ToolName::new("lookup").expect("a name"), json!({})),
    ))
    .with_native(json!({"type": "call", "name": "lookup"}));
    let AssistantContent::ToolCall(tool_call) = &call else {
        panic!("a call");
    };
    let history = vec![
        turn(Some(same()), vec![call.clone()]),
        Message::User {
            content: vec![UserContent::ToolResult(
                tool_call.result(vec![ToolResultContent::text("ok")]),
            )],
        },
    ];
    let ids =
        crate::providers::internal::wire_ids::WireIds::for_target(&history, &Slotted, "model-a");
    assert_eq!(ids.of(&tool_call.id), Some("tool-0"));
    assert_eq!(
        call.replay(&Slotted, &ids),
        Replay::Item(std::borrow::Cow::Owned(
            json!({"type": "call", "name": "lookup", "id": "tool-0"})
        ))
    );
}

#[test]
fn an_edited_block_rebuilds_keeping_its_identity() {
    let ids = crate::providers::internal::wire_ids::WireIds::default();
    let mut text = AssistantContent::text("a").with_native(json!({"id": "msg_1", "text": "a"}));
    assert!(matches!(text.replay(&Slotted, &ids), Replay::Item(_)));
    if let AssistantContent::Text(inner) = &mut text {
        inner.text.push('!');
    }
    assert_eq!(
        text.replay(&Slotted, &ids),
        Replay::Identity(serde_json::Map::from_iter([(
            "id".to_owned(),
            json!("msg_1")
        )]))
    );
    assert_eq!(
        AssistantContent::text("plain").replay(&Slotted, &ids),
        Replay::Rebuild
    );
}

#[test]
fn calls_sharing_an_id_in_one_turn_become_distinct_and_keep_their_results() {
    let history = vec![
        Message::user("q"),
        turn(Some(same()), vec![call("dup"), call("dup")]),
        Message::User {
            content: vec![result("dup", "first"), result("dup", "second")],
        },
    ];
    let adapted = adapt(&history, &TARGET);
    let turn = assistant(&adapted[1]);
    let ids: Vec<String> = turn
        .tool_calls()
        .map(|call| call.id.wire().into_owned())
        .collect();
    assert_eq!(ids.len(), 2);
    assert_ne!(ids[0], ids[1], "{ids:?}");
    let Message::User { content } = &adapted[2] else {
        panic!("the results: {adapted:?}");
    };
    let answered: Vec<(String, Option<&str>)> = content
        .iter()
        .filter_map(|part| match part {
            UserContent::ToolResult(result) => Some((
                result.call.wire().into_owned(),
                result.content.first().and_then(ToolResultContent::as_text),
            )),
            _ => None,
        })
        .collect();
    assert_eq!(
        answered,
        [
            (ids[0].clone(), Some("first")),
            (ids[1].clone(), Some("second")),
        ]
    );
}

#[test]
fn an_id_reused_by_a_later_turn_is_made_distinct() {
    let history = vec![
        Message::user("q"),
        turn(Some(other()), vec![call("call_0")]),
        Message::User {
            content: vec![result("call_0", "one")],
        },
        turn(Some(other()), vec![call("call_0")]),
        Message::User {
            content: vec![result("call_0", "two")],
        },
    ];
    let adapted = adapt(&history, &TARGET);
    let first = assistant(&adapted[1])
        .tool_calls()
        .next()
        .map(|call| call.id.clone());
    let second = assistant(&adapted[3])
        .tool_calls()
        .next()
        .map(|call| call.id.clone());
    assert_ne!(first, second);
    let Message::User { content } = &adapted[4] else {
        panic!("the second results: {adapted:?}");
    };
    assert!(matches!(
        content.first(),
        Some(UserContent::ToolResult(result)) if Some(&result.call) == second.as_ref()
    ));
}

#[test]
fn an_empty_result_says_so() {
    let history = vec![
        turn(Some(same()), vec![call("c1")]),
        Message::User {
            content: vec![result("c1", "  ")],
        },
    ];
    let adapted = adapt(&history, &TARGET);
    let Message::User { content } = &adapted[1] else {
        panic!("the result: {adapted:?}");
    };
    assert!(matches!(
        content.first(),
        Some(UserContent::ToolResult(result))
            if result.content == [ToolResultContent::text(NO_TOOL_OUTPUT)]
    ));
}

#[test]
fn a_held_system_message_goes_between_the_results_and_the_users_text() {
    let history = vec![
        Message::user("q"),
        turn(Some(same()), vec![call("c1")]),
        Message::system("steer"),
        Message::User {
            content: vec![result("c1", "done"), UserContent::text("and then")],
        },
    ];
    let adapted = adapt(&history, &TARGET);
    assert_eq!(
        adapted[2..],
        [
            Message::User {
                content: vec![result("c1", "done")],
            },
            Message::system("steer"),
            Message::user("and then"),
        ]
    );
}

/// A target whose reasoning items need the item after them, and whose
/// hosted uses and results pair by id.
#[derive(Debug)]
struct Paired;

impl ReplayTarget for Paired {
    fn api(&self) -> Api {
        Api::from_static("test.api")
    }

    fn provider(&self) -> &str {
        "test"
    }

    fn model(&self) -> &str {
        "model-a"
    }

    fn accepts(&self, _model: &str) -> Accepts {
        Accepts::ALL
    }

    fn needs_next(&self, item: &serde_json::Value) -> bool {
        item.get("type").and_then(serde_json::Value::as_str) == Some("reasoning")
    }

    fn hosted_pair(&self, item: &serde_json::Value) -> Option<(Pairing, String)> {
        let id = item.get("id")?.as_str()?.to_owned();
        match item.get("type")?.as_str()? {
            "server_use" => Some((Pairing::Use, id)),
            "server_result" => Some((Pairing::Result, id)),
            _ => None,
        }
    }
}

#[test]
fn a_same_model_item_whose_partner_is_gone_goes_with_it() {
    let reasoning = AssistantContent::reasoning("plan").with_native(json!({"type": "reasoning"}));
    let blank = AssistantContent::text(" ");
    let hosted = |kind: &str, id: &str| {
        AssistantContent::Opaque(Opaque {
            item: json!({"type": kind, "id": id}),
            replay: true,
        })
    };
    let history = vec![
        Message::user("q"),
        turn(
            Some(same()),
            vec![
                reasoning.clone(),
                blank,
                hosted("server_use", "s1"),
                hosted("server_result", "s1"),
                hosted("server_use", "s2"),
                reasoning.clone(),
                AssistantContent::text("answer"),
            ],
        ),
    ];
    let adapted = adapt(&history, &Paired);
    assert_eq!(
        assistant(&adapted[1]).content,
        vec![
            hosted("server_use", "s1"),
            hosted("server_result", "s1"),
            reasoning,
            AssistantContent::text("answer"),
        ]
    );
}

#[test]
fn a_request_without_tools_gets_calls_and_results_as_text() {
    let history = vec![
        Message::user("q"),
        turn(Some(same()), vec![call("c1")]),
        Message::User {
            content: vec![result("c1", "done")],
        },
    ];
    let adapted = adapt_for(
        &history,
        &TARGET,
        &Request {
            tools: false,
            ..Request::default()
        },
    );
    assert!(
        adapted.iter().all(|message| match message {
            Message::Assistant(turn) => turn.tool_calls().next().is_none(),
            Message::User { content } => content
                .iter()
                .all(|part| !matches!(part, UserContent::ToolResult(_))),
            Message::System { .. } => true,
        }),
        "{adapted:?}"
    );
}

/// A target whose items bind to the request's tools and system prompt.
#[derive(Debug)]
struct Bound;

impl ReplayTarget for Bound {
    fn api(&self) -> Api {
        Api::from_static("test.api")
    }

    fn provider(&self) -> &str {
        "test"
    }

    fn model(&self) -> &str {
        "model-a"
    }

    fn accepts(&self, _model: &str) -> Accepts {
        Accepts::ALL
    }

    fn binds_context(&self, _model: &str) -> bool {
        true
    }
}

#[test]
fn a_turn_made_under_other_tools_replays_as_another_models() {
    let made = crate::message::Fingerprint::of(&json!("tools-a"));
    let now = crate::message::Fingerprint::of(&json!("tools-b"));
    let mut origin = same();
    origin.context = Some(made);
    let signed = AssistantContent::reasoning("plan").with_native(json!({"signature": "s"}));
    let history = vec![Message::user("q"), turn(Some(origin), vec![signed.clone()])];
    let under = |context| {
        adapt_for(
            &history,
            &Bound,
            &Request {
                context: Some(context),
                ..Request::default()
            },
        )
    };
    assert_eq!(assistant(&under(made)[1]).content, vec![signed]);
    assert_eq!(
        assistant(&under(now)[1]).content,
        vec![AssistantContent::text("plan")]
    );
}
