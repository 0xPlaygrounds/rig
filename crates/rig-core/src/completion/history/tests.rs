mod replay_stops;

use serde_json::json;

use super::*;
use crate::message::{
    AssistantContent, AssistantMessage, CallId, Image, Message, Opaque, Origin, Reasoning,
    StopReason, Text, ToolCall, ToolFunction, ToolName, ToolResult, ToolResultContent, UserContent,
};

/// [`super::adapt_for`] over the plain messages the tests write, after
/// every shorter history, so it adapts from where they left off, as a fresh
/// adaption does.
fn adapt_for(
    history: &[Message],
    target: &dyn ReplayTarget,
    request: &Request<'_>,
) -> Vec<Message> {
    let history: Vec<Arc<Message>> = history.iter().cloned().map(Arc::new).collect();
    let mut memo = Memo::default();
    let mut adapted = Vec::new();
    for end in 1..=history.len() {
        let shorter = history.get(..end).unwrap_or_default().to_vec();
        adapted = super::adapt_for(shorter.clone(), target, request, &mut memo);
        let fresh = super::adapt_for(shorter, target, request, &mut Memo::default());
        assert_eq!(adapted, fresh, "resumed after {end} messages");
    }
    adapted.into_iter().map(Arc::unwrap_or_clone).collect()
}

#[derive(Debug)]
struct Target {
    accepts_images: bool,
}

impl ReplayTarget for Target {
    fn map_options(
        &self,
        _request: &crate::completion::CompletionRequest,
        fields: crate::completion::options::OptionFields<'_>,
    ) -> crate::completion::options::OptionMap {
        crate::test_utils::refuse_options(fields)
    }

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

/// A target with a place for unsigned reasoning.
#[derive(Debug)]
struct ReadsReasoning;

impl ReplayTarget for ReadsReasoning {
    fn map_options(
        &self,
        _request: &crate::completion::CompletionRequest,
        fields: crate::completion::options::OptionFields<'_>,
    ) -> crate::completion::options::OptionMap {
        crate::test_utils::refuse_options(fields)
    }

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
        Accepts {
            foreign_reasoning: true,
            ..Accepts::ALL
        }
    }
}

/// Another model's reasoning stays reasoning, without its provider item,
/// only when the request asks for it and the target reads it; otherwise
/// it is text (pi's rule).
#[test]
fn foreign_reasoning_stays_reasoning_only_when_asked_and_read() {
    let history = vec![turn(Some(other()), vec![signed_reasoning()])];
    let asked = Request {
        foreign_reasoning: true,
        ..Request::default()
    };
    let first = |adapted: Vec<Message>| assistant(&adapted[0]).content.clone();
    assert_eq!(
        first(adapt_for(&history, &ReadsReasoning, &asked)),
        [AssistantContent::Reasoning(Reasoning::new("thinking"))]
    );
    let as_text = [AssistantContent::Text(Text::new("thinking"))];
    assert_eq!(first(adapt(&history, &ReadsReasoning)), as_text);
    assert_eq!(first(adapt_for(&history, &TARGET, &asked)), as_text);
}

/// A target whose encoder takes inline images only, and no audio, video or
/// documents.
#[derive(Debug)]
struct InlineOnly;

impl ReplayTarget for InlineOnly {
    fn map_options(
        &self,
        _request: &crate::completion::CompletionRequest,
        fields: crate::completion::options::OptionFields<'_>,
    ) -> crate::completion::options::OptionMap {
        crate::test_utils::refuse_options(fields)
    }

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
                    data: DocumentSourceKind::Unknown.into(),
                    media_type: Some(DocumentMediaType::TXT),
                    additional_params: None,
                }),
                UserContent::Document(Document {
                    data: DocumentSourceKind::base64("YSxiCjEsMgo=").into(),
                    media_type: Some(DocumentMediaType::CSV),
                    additional_params: None,
                }),
                UserContent::Document(Document {
                    data: DocumentSourceKind::base64("JVBERi0=").into(),
                    media_type: Some(DocumentMediaType::PDF),
                    additional_params: None,
                }),
                UserContent::Document(Document {
                    data: crate::message::DocumentData::Text("notes".to_owned()),
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
                UserContent::text(DOCUMENT_UNSENDABLE),
                UserContent::text("a,b\n1,2\n"),
                UserContent::text(DOCUMENT_UNSENDABLE),
                UserContent::text("notes"),
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
            content: vec![UserContent::text(IMAGE_OMITTED)],
        }]
    );
}

/// A target that normalizes every id to its first three characters, so two
/// distinct ids collide, and reads what `accepts` says.
#[derive(Debug)]
struct Narrow {
    accepts: Accepts,
}

impl ReplayTarget for Narrow {
    fn map_options(
        &self,
        _request: &crate::completion::CompletionRequest,
        fields: crate::completion::options::OptionFields<'_>,
    ) -> crate::completion::options::OptionMap {
        crate::test_utils::refuse_options(fields)
    }

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
    fn map_options(
        &self,
        _request: &crate::completion::CompletionRequest,
        fields: crate::completion::options::OptionFields<'_>,
    ) -> crate::completion::options::OptionMap {
        crate::test_utils::refuse_options(fields)
    }

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

#[test]
fn results_split_by_a_system_message_answer_their_calls() {
    let history = vec![
        Message::user("q"),
        turn(Some(same()), vec![call("c1"), call("c2")]),
        Message::User {
            content: vec![result("c1", "one")],
        },
        Message::system("steer"),
        Message::User {
            content: vec![result("c2", "two")],
        },
        Message::user("and then"),
    ];
    let adapted = adapt(&history, &TARGET);
    assert_eq!(
        adapted[2..],
        [
            Message::User {
                content: vec![result("c1", "one"), result("c2", "two")],
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
    fn map_options(
        &self,
        _request: &crate::completion::CompletionRequest,
        fields: crate::completion::options::OptionFields<'_>,
    ) -> crate::completion::options::OptionMap {
        crate::test_utils::refuse_options(fields)
    }

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
    let answer = AssistantContent::text("answer").with_native(json!({"type": "message"}));
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
                answer.clone(),
                reasoning.clone(),
                AssistantContent::text("rebuilt"),
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
            answer,
            AssistantContent::text("rebuilt"),
        ]
    );
}

fn hosted(kind: &str, id: &str) -> AssistantContent {
    AssistantContent::Opaque(Opaque {
        item: json!({"type": kind, "id": id}),
        replay: true,
    })
}

#[test]
fn a_hosted_use_pairs_with_its_result_in_a_later_turn() {
    let history = vec![
        Message::user("q"),
        turn(Some(same()), vec![hosted("server_use", "s1"), call("c1")]),
        Message::User {
            content: vec![result("c1", "done")],
        },
        turn(
            Some(same()),
            vec![hosted("server_result", "s1"), AssistantContent::text("a")],
        ),
        Message::user("next"),
    ];
    let adapted = adapt(&history, &Paired);
    assert_eq!(
        assistant(&adapted[1]).content,
        vec![hosted("server_use", "s1"), call("c1")]
    );
    assert_eq!(
        assistant(&adapted[3]).content,
        vec![hosted("server_result", "s1"), AssistantContent::text("a")]
    );
}

#[derive(Debug)]
struct UserFirst;

impl ReplayTarget for UserFirst {
    fn map_options(
        &self,
        _request: &crate::completion::CompletionRequest,
        fields: crate::completion::options::OptionFields<'_>,
    ) -> crate::completion::options::OptionMap {
        crate::test_utils::refuse_options(fields)
    }

    fn api(&self) -> Api {
        TARGET.api()
    }

    fn provider(&self) -> &str {
        TARGET.provider()
    }

    fn model(&self) -> &str {
        TARGET.model()
    }

    fn accepts(&self, model: &str) -> Accepts {
        TARGET.accepts(model)
    }

    fn starts_with_user(&self) -> bool {
        true
    }
}

#[test]
fn a_user_first_wire_drops_the_turns_before_the_first_user_message() {
    // A memory window that cut a conversation after a user message.
    let history = vec![
        Message::system("be brief"),
        turn(Some(same()), vec![call("c1")]),
        Message::User {
            content: vec![result("c1", "done"), UserContent::text("next")],
        },
        turn(Some(same()), vec![AssistantContent::text("ok")]),
    ];
    assert_eq!(
        adapt(&history, &UserFirst),
        vec![
            Message::system("be brief"),
            Message::user("next"),
            turn(Some(same()), vec![AssistantContent::text("ok")]),
        ]
    );
    assert_eq!(
        adapt(&history, &TARGET).len(),
        4,
        "other wires keep a leading turn"
    );
}

#[test]
fn the_context_is_the_same_before_and_after_adapt() {
    let history = vec![
        Message::system("be brief"),
        Message::system(" "),
        Message::user("q"),
        turn(Some(same()), vec![AssistantContent::text("a")]),
        Message::system("steer"),
        Message::user("next"),
    ];
    let request = |history: Vec<Message>| crate::completion::CompletionRequest {
        chat_history: history.into_iter().map(Into::into).collect(),
        ..crate::completion::CompletionRequest::new("q")
    };
    let targets: [&dyn ReplayTarget; 2] = [&TARGET, &LeadingSystemOnly];
    for target in targets {
        assert_eq!(
            context_of(&request(history.clone()), target, "model-a"),
            context_of(&request(adapt(&history, target)), target, "model-a"),
            "{target:?}"
        );
    }
}

#[derive(Debug)]
struct LeadingSystemOnly;

impl ReplayTarget for LeadingSystemOnly {
    fn map_options(
        &self,
        _request: &crate::completion::CompletionRequest,
        fields: crate::completion::options::OptionFields<'_>,
    ) -> crate::completion::options::OptionMap {
        crate::test_utils::refuse_options(fields)
    }

    fn api(&self) -> Api {
        TARGET.api()
    }

    fn provider(&self) -> &str {
        TARGET.provider()
    }

    fn model(&self) -> &str {
        TARGET.model()
    }

    fn accepts(&self, model: &str) -> Accepts {
        TARGET.accepts(model)
    }

    fn later_system(&self, _model: &str) -> LaterSystem {
        LaterSystem::Leading
    }
}

#[test]
fn a_call_renamed_for_a_reused_id_keeps_its_item_where_the_item_holds_its_id() {
    let call_with_item =
        || call("functions.f:0").with_native(json!({"id": "functions.f:0", "x": 1}));
    let history = vec![
        Message::user("q"),
        turn(Some(same()), vec![call_with_item()]),
        Message::User {
            content: vec![result("functions.f:0", "one")],
        },
        turn(Some(same()), vec![call_with_item()]),
        Message::User {
            content: vec![result("functions.f:0", "two")],
        },
    ];
    let renamed = |target: &dyn ReplayTarget| {
        let adapted = adapt(&history, target);
        let AssistantContent::ToolCall(call) = &assistant(&adapted[3]).content[0] else {
            panic!("a call");
        };
        assert_ne!(call.id.wire(), "functions.f:0");
        AssistantContent::ToolCall(call.clone())
            .native_item()
            .cloned()
    };
    assert_eq!(
        renamed(&Slotted),
        Some(json!({"id": "functions.f:0", "x": 1}))
    );
    assert_eq!(
        renamed(&TARGET),
        None,
        "without a slot the item would name the old id"
    );
}

#[test]
fn a_leading_turn_goes_with_its_results_and_its_user_message_stays() {
    let call = crate::message::ToolCall::from_wire(
        "c1",
        crate::message::ToolFunction::new(
            crate::message::ToolName::new("add").expect("tool name"),
            json!({}),
        ),
    );
    let result = UserContent::ToolResult(call.result(vec![ToolResultContent::text("3")]));
    let answered = vec![
        Message::system("s"),
        Message::Assistant(AssistantMessage::new(vec![AssistantContent::ToolCall(
            call.clone(),
        )])),
        Message::User {
            content: vec![result.clone()],
        },
        Message::user("q"),
    ];
    assert_eq!(
        from_first_user(answered),
        vec![Message::system("s"), Message::user("q")]
    );
    let mixed = vec![
        Message::Assistant(AssistantMessage::new(vec![AssistantContent::ToolCall(
            call.clone(),
        )])),
        Message::User {
            content: vec![result, UserContent::text("q")],
        },
    ];
    assert_eq!(from_first_user(mixed), vec![Message::user("q")]);
    let other = crate::message::ToolCall::from_wire("c2", call.function.clone());
    let unrelated = Message::User {
        content: vec![UserContent::ToolResult(
            other.result(vec![ToolResultContent::text("4")]),
        )],
    };
    let foreign = vec![
        Message::Assistant(AssistantMessage::new(vec![AssistantContent::ToolCall(
            call,
        )])),
        unrelated.clone(),
    ];
    assert_eq!(from_first_user(foreign), vec![unrelated]);
}

/// Cache points move with the history they name: a point inside a message
/// keeps its items, a point on a message the wire joins with the next one
/// ends inside the joined message, a point past the end goes, documents
/// count as the start of the first user message, and a point never covers
/// a result the wire moves in front of the text it ends after.
#[test]
fn cache_points_follow_the_history_through_adapting() {
    use crate::completion::CachePoint::{Messages, System, Tools, Within};
    let request = Request::default();
    let moved = |written: &[Message],
                 documents: Option<&Message>,
                 points: &[crate::completion::CachePoint]| {
        let written: Vec<Arc<Message>> = written.iter().cloned().map(Arc::new).collect();
        let mut sent = written.clone();
        if let Some(documents) = documents {
            crate::completion::request::insert_after_leading_system(&mut sent, documents.clone());
        }
        let adapted = super::adapt_for(sent, &TARGET, &request, &mut super::Memo::default());
        cache_points(points, &written, documents, &adapted, &TARGET, &request)
    };
    let written = vec![
        Message::user("a"),
        turn(
            Some(same()),
            vec![AssistantContent::text("b1"), AssistantContent::text("b2")],
        ),
        Message::user("c"),
        Message::user("d"),
    ];
    let points = [
        Tools,
        System,
        Messages(0),
        Within {
            message: 1,
            items: 1,
        },
        Within {
            message: 3,
            items: 1,
        },
        Messages(3),
        Messages(9),
    ];
    assert_eq!(
        moved(&written, None, &points),
        [
            Tools,
            System,
            Messages(0),
            Within {
                message: 1,
                items: 1
            },
            Messages(3),
            Within {
                message: 2,
                items: 1
            },
        ]
    );

    let documents = Message::User {
        content: vec![UserContent::text("doc")],
    };
    assert_eq!(
        moved(&written, Some(&documents), &[Messages(1), Messages(2)]),
        [Messages(1), Messages(2)]
    );

    let steered = vec![
        Message::user("q"),
        turn(Some(same()), vec![call("c1")]),
        Message::user("and also"),
        Message::User {
            content: vec![result("c1", "done")],
        },
    ];
    assert_eq!(moved(&steered, None, &[Messages(3)]), [Messages(2)]);
}

#[test]
fn a_growing_history_adapts_from_its_last_turn_as_a_fresh_one_does() {
    // Turns reuse a call id, so a later turn's call is renamed against the
    // ids the kept messages claimed.
    let session = [
        Message::system("be brief"),
        Message::user("q1"),
        turn(Some(same()), vec![call("c")]),
        Message::User {
            content: vec![result("c", "one")],
        },
        turn(Some(same()), vec![AssistantContent::text("a1")]),
        Message::user("q2"),
        turn(Some(same()), vec![call("c")]),
        Message::User {
            content: vec![result("c", "two")],
        },
        turn(Some(other()), vec![call("c|x")]),
        Message::User {
            content: vec![result("c|x", "three")],
        },
        turn(Some(same()), vec![AssistantContent::text("a2")]),
    ];
    let session: Vec<Arc<Message>> = session.into_iter().map(Arc::new).collect();
    let mut memo = Memo::default();
    for end in 1..=session.len() {
        let history = session.get(..end).unwrap_or_default().to_vec();
        let resumed = super::adapt_for(history.clone(), &TARGET, &Request::default(), &mut memo);
        let fresh = super::adapt_for(history, &TARGET, &Request::default(), &mut Memo::default());
        assert_eq!(resumed, fresh, "after {end} messages");
    }
    assert_eq!(memo.given.len(), session.len(), "the history was kept");
}

/// An assistant image the target cannot carry as it is gets the type its
/// bytes name, or becomes the placeholder when the encoder still cannot
/// carry it.
#[test]
fn an_assistant_image_is_typed_or_becomes_a_placeholder() {
    let url = Image {
        data: DocumentSourceKind::url("https://example.invalid/a.png"),
        ..Image::default()
    };
    let untyped_png = Image {
        data: DocumentSourceKind::base64("iVBORw0KGgoAAAANSUhEUg=="),
        ..Image::default()
    };
    let history = vec![
        Message::user("draw"),
        turn(
            Some(other()),
            vec![
                AssistantContent::Image(url),
                AssistantContent::Image(untyped_png),
            ],
        ),
    ];
    let adapted = adapt(&history, &InlineOnly);
    assert_eq!(
        adapted
            .get(1)
            .map(|message| assistant(message).content.clone()),
        Some(vec![
            AssistantContent::Text(Text::new(IMAGE_UNSENDABLE)),
            AssistantContent::Image(Image {
                data: DocumentSourceKind::base64("iVBORw0KGgoAAAANSUhEUg=="),
                media_type: Some(ImageMediaType::PNG),
                ..Image::default()
            }),
        ])
    );
}

/// A target whose wire requires user and assistant messages to alternate.
#[derive(Debug)]
struct Alternating;

impl ReplayTarget for Alternating {
    fn map_options(
        &self,
        _request: &crate::completion::CompletionRequest,
        fields: crate::completion::options::OptionFields<'_>,
    ) -> crate::completion::options::OptionMap {
        crate::test_utils::refuse_options(fields)
    }

    fn api(&self) -> Api {
        TARGET.api()
    }

    fn provider(&self) -> &str {
        TARGET.provider()
    }

    fn model(&self) -> &str {
        TARGET.model()
    }

    fn accepts(&self, model: &str) -> Accepts {
        TARGET.accepts(model)
    }

    fn alternates_roles(&self) -> bool {
        true
    }
}

#[test]
fn an_alternating_wire_joins_users_a_system_message_separates() {
    let history = vec![
        Message::user("q1"),
        Message::system("steer"),
        Message::user("q2"),
        turn(Some(same()), vec![AssistantContent::text("a")]),
    ];
    assert_eq!(
        adapt(&history, &Alternating),
        vec![
            Message::User {
                content: vec![UserContent::text("q1"), UserContent::text("q2")],
            },
            Message::system("steer"),
            turn(Some(same()), vec![AssistantContent::text("a")]),
        ]
    );
    assert_eq!(
        adapt(&history, &TARGET),
        history,
        "other wires keep them apart"
    );
}
