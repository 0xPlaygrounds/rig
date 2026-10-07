//! The `/api/chat` request built as JSON: where each `additional_params`
//! key lands, the format's deferral, and each message's shape. The recorded
//! requests are pinned by the `ollama` cassettes; these state every key's
//! placement, which no single recording shows.

use serde_json::{Value, json};

use super::Chat;
use crate::completion::{CompletionRequest, Message, ToolDefinition};
use crate::message::{
    AssistantContent, AssistantMessage, CallId, DocumentSourceKind, Image, ImageMediaType,
    Reasoning, ToolName, ToolResult, ToolResultContent, UserContent,
};
use crate::providers::ollama::OllamaConfig;
use crate::test_utils::json_body;
use crate::wire::{Mode, Operation, Wire};

const MODEL: &str = "qwen3:8b";

fn wire() -> Chat {
    OllamaConfig::new().native_completion(MODEL)
}

fn name(name: &str) -> ToolName {
    ToolName::new(name).expect("a tool name")
}

fn tool(tool: &str) -> ToolDefinition {
    ToolDefinition::new(name(tool), "a tool", json!({"type": "object"}))
}

/// The body `request` sends in `mode`, prepared as the driver prepares it.
fn sent(request: CompletionRequest, mode: Mode) -> Result<Value, crate::error::EncodeError> {
    let wire = wire();
    let request = crate::operation::Completion::prepare(request, &wire.describe())
        .expect("the request prepares");
    Ok(json_body(&wire.encode(request, mode)?.request))
}

fn body(request: CompletionRequest) -> Value {
    sent(request, Mode::Unary).expect("the request encodes")
}

fn with_params(params: Value) -> CompletionRequest {
    let mut request = CompletionRequest::new("What is 2 + 2?");
    request.additional_params = Some(params);
    request
}

/// `think` and `keep_alive` are top-level fields, `num_ctx` is a model
/// parameter in `options`, and so are `temperature` and `max_tokens` as
/// `num_predict`.
#[test]
fn think_keep_alive_and_num_ctx_land_where_the_daemon_reads_them() {
    let mut request = with_params(json!({"think": true, "keep_alive": "-1m", "num_ctx": 4096}));
    request
        .chat_history
        .insert(0, Message::system("You are a helpful assistant."));
    request.temperature = Some(0.7);
    request.max_tokens = Some(1024);
    assert_eq!(
        body(request),
        json!({
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2 + 2?"},
            ],
            "options": {"temperature": 0.7, "num_predict": 1024, "num_ctx": 4096},
            "stream": false,
            "think": true,
            "keep_alive": "-1m",
        })
    );
}

/// Every other key: a raw `think` is sent as written, an `options` object
/// merges over the typed options, `tools` join the request's, the other
/// top-level fields stay at the top, and any other key is an option. A
/// streamed request says so.
#[test]
fn every_additional_param_lands_in_its_place() {
    let mut request = with_params(json!({
        "think": "HIGH",
        "keep_alive": 300,
        "options": {"temperature": 0.2, "top_k": 40},
        "seed": 7,
        "logprobs": true,
        "top_logprobs": 2,
        "truncate": false,
        "shift": false,
        "format": "json",
        "tools": [{"type": "function", "function": {"name": "raw", "parameters": {}}}],
    }));
    request.temperature = Some(0.9);
    request.tools = vec![tool("typed")];
    assert_eq!(
        sent(request, Mode::Streaming).expect("the request encodes"),
        json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "What is 2 + 2?"}],
            "tools": [
                {"type": "function", "function": {
                    "name": "typed", "description": "a tool", "parameters": {"type": "object"}}},
                {"type": "function", "function": {"name": "raw", "parameters": {}}},
            ],
            "options": {"temperature": 0.2, "top_k": 40, "seed": 7},
            "stream": true,
            "think": "HIGH",
            "keep_alive": 300,
            "logprobs": true,
            "top_logprobs": 2,
            "truncate": false,
            "shift": false,
            "format": "json",
        })
    );
    // A request with nothing to tune sends no options.
    assert!(body(CompletionRequest::new("hi")).get("options").is_none());
}

/// `/api/chat` reads thinking from `think`, which the `reasoning` option
/// sends; a raw `reasoning_effort` key is a model option like any other.
#[test]
fn reasoning_is_think_and_reasoning_effort_is_an_option() {
    use crate::completion::{Effort, Reasoning};
    let sent = body(with_params(json!({"reasoning_effort": "high"})));
    assert_eq!(sent["options"], json!({"reasoning_effort": "high"}));
    assert!(sent.get("think").is_none());
    for (reasoning, think) in [
        (Reasoning::Off, json!(false)),
        (Effort::Medium.into(), json!("medium")),
        (Effort::Max.into(), json!("max")),
    ] {
        let sent = body(CompletionRequest::new("hi").reasoning(reasoning));
        assert_eq!(sent["think"], think, "{reasoning:?}");
    }
    let sent = body(
        CompletionRequest::new("hi")
            .top_p(0.4)
            .seed(3)
            .stop(["END"]),
    );
    assert_eq!(
        sent["options"],
        json!({"top_p": 0.4, "seed": 3, "stop": ["END"]})
    );
}

/// A value the daemon cannot read is a request error, never dropped.
#[test]
fn malformed_params_are_refused() {
    for (params, error) in [
        (json!({"keep_alive": true}), "`keep_alive`"),
        (json!({"options": [1]}), "`additional_params.options`"),
        (json!({"tools": {}}), "`additional_params.tools`"),
        (json!(["not", "an", "object"]), "`additional_params`"),
    ] {
        let refused =
            sent(with_params(params.clone()), Mode::Unary).expect_err("the request is refused");
        assert!(refused.to_string().contains(error), "{params}: {refused}");
    }
    let mut request = CompletionRequest::new("hi");
    request.tool_choice = Some(crate::message::ToolChoice::Required);
    request.tools = vec![tool("t")];
    assert!(body(request).get("tool_choice").is_none());
}

/// The output schema is `format`, deferred while the request's tools have
/// no result, as a constrained reply cannot call one.
#[test]
fn the_output_schema_is_format_once_tools_are_answered() {
    let schema: schemars::Schema =
        serde_json::from_value(json!({"type": "object"})).expect("a schema");
    let mut plain = CompletionRequest::new("hi");
    plain.output_schema = Some(schema.clone());
    assert_eq!(body(plain).get("format"), Some(&json!({"type": "object"})));

    let mut calling = CompletionRequest::new("hi");
    calling.output_schema = Some(schema.clone());
    calling.tools = vec![tool("lookup")];
    assert!(body(calling.clone()).get("format").is_none());

    calling.chat_history = vec![
        Message::user("hi"),
        Message::Assistant(AssistantMessage::new(vec![AssistantContent::tool_call(
            "call_1",
            name("lookup"),
            json!({}),
        )])),
        Message::User {
            content: vec![UserContent::ToolResult(ToolResult {
                call: CallId::from_wire("call_1"),
                name: name("lookup"),
                content: vec![ToolResultContent::text("found")],
                is_error: false,
            })],
        },
    ];
    assert_eq!(
        body(calling).get("format"),
        Some(&json!({"type": "object"}))
    );
}

/// Each message as `/api/chat` reads it: user text and base64 images, the
/// assistant's text, `thinking` and calls with their ids, and each result
/// as a `tool` message naming its tool and call.
#[test]
fn messages_carry_images_thinking_calls_and_results() {
    let reply = json!({
        "model": MODEL, "created_at": "2026-10-05T00:00:00Z", "done": true, "done_reason": "stop",
        "message": {"role": "assistant", "content": "Checking.", "thinking": "look it up",
            "tool_calls": [{"id": "call_1", "function": {"name": "lookup", "arguments": {"q": "png"}}}]},
    });
    let turn = crate::test_utils::history::decode(
        &wire(),
        Mode::Unary,
        [crate::wire::WireFrame::Text(reply.to_string())],
    )
    .expect("the reply decodes")
    .message()
    .expect("the turn is a message");
    let mut request = CompletionRequest::new("thanks");
    request.tools = vec![tool("lookup")];
    request.chat_history = vec![
        Message::system("Be brief."),
        Message::User {
            content: vec![
                UserContent::text("What is this?"),
                UserContent::Image(Image {
                    data: DocumentSourceKind::Base64("aW1hZ2U=".to_owned()),
                    media_type: Some(ImageMediaType::PNG),
                    detail: None,
                    native: None,
                }),
            ],
        },
        turn,
        Message::User {
            content: vec![
                UserContent::ToolResult(ToolResult {
                    call: CallId::from_wire("call_1"),
                    name: name("lookup"),
                    content: vec![
                        ToolResultContent::text("a"),
                        ToolResultContent::Json {
                            value: json!({"b": 1}),
                        },
                    ],
                    is_error: false,
                }),
                UserContent::text("thanks"),
            ],
        },
    ];
    assert_eq!(
        body(request)["messages"],
        json!([
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "What is this?", "images": ["aW1hZ2U="]},
            {"role": "assistant", "content": "Checking.", "thinking": "look it up",
                "tool_calls": [{"id": "call_1", "function": {"name": "lookup", "arguments": {"q": "png"}}}]},
            {"role": "tool", "content": "a\n{\"b\":1}", "tool_name": "lookup", "tool_call_id": "call_1"},
            {"role": "user", "content": "thanks"},
        ])
    );

    // Another model's reasoning is sent as text.
    let mut request = CompletionRequest::new("next");
    request.chat_history = vec![
        Message::user("first"),
        Message::Assistant(AssistantMessage::new(vec![
            AssistantContent::Reasoning(Reasoning::new("theirs")),
            AssistantContent::text("Answer."),
        ])),
    ];
    let messages = body(request)["messages"].clone();
    assert!(messages[1].get("thinking").is_none(), "{messages}");
}

/// A turn the same model produced replays its call items as the daemon sent
/// them, with the canonical name and arguments.
#[test]
fn a_same_model_turn_replays_its_call_items() {
    let reply = json!({
        "model": MODEL, "created_at": "2026-10-05T00:00:00Z", "done": true, "done_reason": "stop",
        "message": {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_9", "function": {"index": 3, "name": "lookup", "arguments": {"q": "x"}}}
        ]},
    });
    let turn = crate::test_utils::history::decode(
        &wire(),
        Mode::Unary,
        [crate::wire::WireFrame::Text(reply.to_string())],
    )
    .expect("the reply decodes");
    let mut request = CompletionRequest::new("next");
    request.tools = vec![tool("lookup")];
    request.chat_history = vec![
        Message::user("first"),
        turn.message().expect("the turn is a message"),
        Message::User {
            content: vec![UserContent::ToolResult(ToolResult {
                call: CallId::from_wire("call_9"),
                name: name("lookup"),
                content: vec![ToolResultContent::text("found")],
                is_error: false,
            })],
        },
    ];
    let messages = body(request)["messages"].clone();
    assert_eq!(
        messages[1]["tool_calls"],
        json!([{"id": "call_9", "function": {"index": 3, "name": "lookup", "arguments": {"q": "x"}}}])
    );
}

/// What the adapter replaces before encoding: images by URL, documents
/// that are not text, audio and video, and images in tool results and
/// assistant turns.
#[test]
fn the_wire_reads_only_base64_user_images() {
    use crate::completion::{Media, Place, ReplayTarget};
    let wire = wire();
    let image = |data| Image {
        data,
        media_type: Some(ImageMediaType::PNG),
        detail: None,
        native: None,
    };
    let base64 = image(DocumentSourceKind::Base64("aW1hZ2U=".to_owned()));
    let url = image(DocumentSourceKind::Url(
        "https://example.com/a.png".to_owned(),
    ));
    assert!(wire.encodes(MODEL, Media::Image(&base64, Place::User)));
    assert!(!wire.encodes(MODEL, Media::Image(&url, Place::User)));
    assert!(!wire.encodes(MODEL, Media::Image(&base64, Place::ToolResult)));
    let accepts = wire.accepts(MODEL);
    assert!(accepts.user_images && accepts.tools);
    assert!(!accepts.tool_result_images && !accepts.assistant_images);
    assert_eq!(wire.call_id_slot(), Some("/id"));
    assert!(!wire.sends_alone(&AssistantContent::text("")));
    assert!(wire.sends_alone(&AssistantContent::Reasoning(Reasoning::new("r"))));
}

/// The encoder refuses what the adapter would have replaced, rather than
/// sending something else, when handed an unprepared history.
#[test]
fn unprepared_content_is_refused() {
    use crate::message::{Audio, Document, Video};
    let unprepared = |content: UserContent| {
        let mut request = CompletionRequest::new("hi");
        request.chat_history = vec![Message::User {
            content: vec![content],
        }];
        wire()
            .encode(request, Mode::Unary)
            .expect_err("the content is refused")
            .to_string()
    };
    let url = DocumentSourceKind::Url("https://example.com/a".to_owned());
    let image = Image {
        data: url.clone(),
        media_type: Some(ImageMediaType::PNG),
        detail: None,
        native: None,
    };
    assert!(unprepared(UserContent::Image(image.clone())).contains("an image"));
    assert!(
        unprepared(UserContent::Document(Document {
            data: url.clone(),
            media_type: None,
            additional_params: None,
        }))
        .contains("a document")
    );
    assert!(
        unprepared(UserContent::Audio(Audio {
            data: url.clone(),
            media_type: None,
        }))
        .contains("audio")
    );
    assert!(
        unprepared(UserContent::Video(Video {
            data: url,
            media_type: None,
            additional_params: None,
        }))
        .contains("video")
    );
    assert!(
        unprepared(UserContent::ToolResult(ToolResult {
            call: CallId::from_wire("call_1"),
            name: name("lookup"),
            content: vec![ToolResultContent::Image(image)],
            is_error: false,
        }))
        .contains("tool result")
    );
    let mut empty = CompletionRequest::new("hi");
    empty.chat_history.clear();
    assert!(wire().encode(empty, Mode::Unary).is_err());
}
