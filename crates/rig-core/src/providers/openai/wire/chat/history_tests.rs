//! The replay findings of the Chat Completions audit, each as the test that
//! closes it: every part of a reply is a block, replay rebuilds the message
//! from its blocks, usage never fails a reply, every finish is mapped, and
//! the adapter downgrades what a dialect's models do not read.

use serde_json::{Value, json};

use super::{CallKind, Chat, Part};
use crate::completion::{CompletionRequest, Message};
use crate::message::{
    AssistantContent, AssistantMessage, CallId, DocumentSourceKind, Image, ImageMediaType, Origin,
    StopReason, ToolCall, ToolFunction, ToolName, ToolResultContent, UserContent,
};
use crate::providers::openai::wire::{
    DEEPSEEK, Dialect, LLAMACPP, MISTRAL, MOONSHOT, OPENAI, OPENROUTER, OpenAIConfig, PERPLEXITY,
    XIAOMIMIMO,
};
use crate::test_utils::history::{assert_every_variant, decode};
use crate::test_utils::json_body;
use crate::wire::{Mode, Operation, Wire, WireFrame};

fn wire(dialect: &'static Dialect, model: &str) -> Chat {
    OpenAIConfig::with_key(dialect, "key").chat(model)
}

fn chunk(delta: Value, finish: Option<&str>) -> WireFrame {
    WireFrame::Text(
        json!({"id": "c", "object": "chat.completion.chunk", "model": "m",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})
        .to_string(),
    )
}

fn whole(message: Value, finish: &str) -> WireFrame {
    WireFrame::Text(
        json!({"id": "c", "object": "chat.completion", "model": "m",
            "choices": [{"index": 0, "message": message, "finish_reason": finish}]})
        .to_string(),
    )
}

fn turn_of(response: &crate::completion::CompletionResponse) -> AssistantMessage {
    AssistantMessage {
        content: response.choice.clone(),
        ..response.head()
    }
}

/// The body `history` sends on `wire`, prepared as the driver prepares it,
/// declaring every tool the history calls as a tool loop does.
fn sent(wire: &Chat, history: Vec<Message>) -> Value {
    sent_with(wire, CompletionRequest::new("next"), history)
}

/// [`sent`] for `request`, whose other fields stand.
fn sent_with(wire: &Chat, mut request: CompletionRequest, history: Vec<Message>) -> Value {
    request.tools.extend(tools_of(&history));
    request.chat_history = history;
    request.chat_history.push(Message::user("next"));
    let request = crate::operation::Completion::prepare(request, &wire.describe())
        .expect("the request prepares");
    json_body(&wire.encode(request, Mode::Unary).expect("encodes").request)
}

/// A definition for every tool `history` calls or answers.
fn tools_of(history: &[Message]) -> Vec<crate::completion::ToolDefinition> {
    let mut names: Vec<ToolName> = Vec::new();
    for message in history {
        let found: Vec<ToolName> = match message {
            Message::Assistant(turn) => turn
                .tool_calls()
                .map(|call| call.function.name.clone())
                .collect(),
            Message::User { content } => content
                .iter()
                .filter_map(|part| match part {
                    UserContent::ToolResult(result) => Some(result.name.clone()),
                    _ => None,
                })
                .collect(),
            Message::System { .. } => Vec::new(),
        };
        for name in found {
            if !names.contains(&name) {
                names.push(name);
            }
        }
    }
    names
        .into_iter()
        .map(|name| {
            crate::completion::ToolDefinition::new(
                name,
                "a tool the history calls",
                json!({"type": "object", "properties": {}}),
            )
        })
        .collect()
}

fn name(name: &str) -> ToolName {
    ToolName::new(name).expect("a tool name")
}

fn png() -> Image {
    Image {
        data: DocumentSourceKind::base64("iVBORw0KGgo="),
        media_type: Some(ImageMediaType::PNG),
        ..Image::default()
    }
}

/// #807, #1835: every `images[]` entry is an image block in arrival order, an
/// image-only reply is not empty, and the same model never gets a generated
/// image back.
#[test]
fn openrouter_images_are_blocks_and_never_go_back() {
    let images = json!([
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
        {"type": "image_url", "image_url": {"url": "https://example.com/second.png"}},
    ]);
    let wire = wire(&OPENROUTER, "google/gemini-2.5-flash-image");
    let modes = [
        (
            Mode::Unary,
            vec![whole(
                json!({"role": "assistant", "content": "here", "images": images}),
                "stop",
            )],
        ),
        (
            Mode::Streaming,
            vec![
                chunk(json!({"role": "assistant", "content": "here"}), None),
                chunk(json!({"images": images}), Some("stop")),
                WireFrame::Text("[DONE]".to_owned()),
            ],
        ),
    ];
    for (mode, frames) in modes {
        let response = decode(&wire, mode, frames).expect("the reply decodes");
        let [
            AssistantContent::Text(text),
            AssistantContent::Image(first),
            AssistantContent::Image(second),
        ] = response.choice.as_slice()
        else {
            panic!("{mode:?}: text and two images: {:?}", response.choice);
        };
        assert_eq!(text.text, "here");
        assert_eq!(first.data, DocumentSourceKind::base64("iVBORw0KGgo="));
        assert_eq!(first.media_type, Some(ImageMediaType::PNG));
        assert_eq!(
            second.data,
            DocumentSourceKind::Url("https://example.com/second.png".to_owned())
        );
        let body = sent(
            &wire,
            vec![
                Message::user("draw"),
                Message::Assistant(turn_of(&response)),
            ],
        );
        assert_eq!(
            body["messages"][1],
            json!({"role": "assistant", "content": "here"}),
            "{mode:?}"
        );
    }
    let only = vec![whole(
        json!({"role": "assistant", "images": [images[0]]}),
        "stop",
    )];
    let response = decode(&wire, Mode::Unary, only).expect("an image-only reply decodes");
    assert!(matches!(
        response.choice.as_slice(),
        [AssistantContent::Image(_)]
    ));
}

/// chatB NEW (Mistral thinking): Magistral's `thinking` content parts are
/// reasoning in both modes, its message keeps them as parts, and the same
/// model gets them back as the part they came as.
#[test]
fn mistral_thinking_parts_are_reasoning() {
    let thinking =
        |text: &str| json!({"type": "thinking", "thinking": [{"type": "text", "text": text}]});
    let wire = wire(&MISTRAL, "magistral-medium-latest");
    let modes = [
        (
            Mode::Unary,
            vec![whole(
                json!({"role": "assistant",
                    "content": [thinking("plan it"), {"type": "text", "text": "Hi"}]}),
                "stop",
            )],
        ),
        (
            Mode::Streaming,
            vec![
                chunk(
                    json!({"role": "assistant", "content": [thinking("plan ")]}),
                    None,
                ),
                chunk(json!({"content": [thinking("it")]}), None),
                chunk(json!({"content": "Hi"}), Some("stop")),
                WireFrame::Text("[DONE]".to_owned()),
            ],
        ),
    ];
    for (mode, frames) in modes {
        let response = decode(&wire, mode, frames).expect("the reply decodes");
        let [
            AssistantContent::Reasoning(reasoning),
            AssistantContent::Text(text),
        ] = response.choice.as_slice()
        else {
            panic!("{mode:?}: reasoning and text: {:?}", response.choice);
        };
        assert_eq!(reasoning.text, "plan it", "{mode:?}");
        assert_eq!(text.text, "Hi", "{mode:?}");
        let body = sent(
            &wire,
            vec![Message::user("q"), Message::Assistant(turn_of(&response))],
        );
        assert_eq!(
            body["messages"][1],
            json!({"role": "assistant",
                "content": [thinking("plan it"), {"type": "text", "text": "Hi"}]}),
            "{mode:?}"
        );
    }
}

/// chatA NEW-3, chatB NEW (custom calls): a custom call is a call whose
/// arguments are its `{"input"}`, and an edited sibling leaves it a custom
/// call in the message's `tool_calls`.
#[test]
fn a_custom_call_stays_a_call_when_its_turn_is_edited() {
    let wire = wire(&OPENAI, "gpt-5.2");
    let frames = vec![whole(
        json!({"role": "assistant", "content": "grepping now", "tool_calls": [{"id": "call_c",
            "type": "custom", "custom": {"name": "grep", "input": "rig"}}]}),
        "tool_calls",
    )];
    let response = decode(&wire, Mode::Unary, frames).expect("the reply decodes");
    let mut turn = turn_of(&response);
    if let Some(AssistantContent::Text(text)) = turn.content.first_mut() {
        text.text = "edited".to_owned();
    }
    let body = sent(
        &wire,
        vec![
            Message::user("q"),
            Message::Assistant(turn),
            Message::tool_result(CallId::from_wire("call_c"), name("grep"), "found"),
        ],
    );
    assert_eq!(
        body["messages"][1],
        json!({"role": "assistant", "content": "edited", "tool_calls": [{"id": "call_c",
            "type": "custom", "custom": {"name": "grep", "input": "rig"}}]})
    );
}

/// chatB NEW (response-only fields): a reply's message never goes back as it
/// came: object-form arguments are sent as JSON text, and a call's stream
/// `index`, annotations and an answer's audio data stay home.
#[test]
fn response_only_fields_never_go_back() {
    let wire = wire(&OPENAI, "gpt-4o-audio-preview");
    let frames = vec![whole(
        json!({"role": "assistant", "content": "see rig.rs",
            "annotations": [{"type": "url_citation", "url_citation": {"url": "https://rig.rs"}}],
            "audio": {"id": "audio_1", "data": "UklG", "expires_at": 1, "transcript": "see"},
            "tool_calls": [{"id": "call_1", "type": "function",
                "function": {"name": "add", "arguments": {"x": 1}}}]}),
        "tool_calls",
    )];
    let response = decode(&wire, Mode::Unary, frames).expect("the reply decodes");
    let body = sent(
        &wire,
        vec![
            Message::user("q"),
            Message::Assistant(turn_of(&response)),
            Message::tool_result(CallId::from_wire("call_1"), name("add"), "2"),
        ],
    );
    assert_eq!(
        body["messages"][1],
        json!({"role": "assistant", "content": "see rig.rs", "audio": {"id": "audio_1"},
            "tool_calls": [{"id": "call_1", "type": "function",
                "function": {"name": "add", "arguments": "{\"x\":1}"}}]})
    );
}

/// #2591: usage is read leniently: a missing total, a `null` cached count and
/// a counter that is not a number never fail the reply.
#[test]
fn usage_never_fails_a_reply() {
    let wire = wire(&OPENAI, "gpt-4.1-nano");
    for usage in [
        json!({"prompt_tokens": 7, "completion_tokens": 3}),
        json!({"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10,
            "prompt_tokens_details": {"cached_tokens": null}}),
        json!({"prompt_tokens": "seven", "total_tokens": 10}),
    ] {
        let frame = WireFrame::Text(
            json!({"object": "chat.completion", "choices": [{"index": 0,
                "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
                "usage": usage})
            .to_string(),
        );
        let response = decode(&wire, Mode::Unary, vec![frame])
            .unwrap_or_else(|error| panic!("{usage}: {error}"));
        assert_eq!(response.choice.len(), 1, "{usage}");
    }
    let frame = WireFrame::Text(
        json!({"object": "chat.completion", "choices": [{"index": 0,
            "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3,
                "prompt_tokens_details": {"cached_tokens": null}, "prompt_cache_hit_tokens": 5}})
        .to_string(),
    );
    let usage = decode(&wire, Mode::Unary, vec![frame])
        .expect("the reply decodes")
        .usage;
    assert_eq!(usage.input_tokens, Some(7));
    assert_eq!(usage.output_tokens, Some(3));
    assert_eq!(usage.total_tokens, None);
    assert_eq!(usage.cached_input_tokens, Some(0));
}

/// chatB NEW (finish reasons): every finish Chat dialects document maps
/// explicitly; a failure and an unknown value fail the turn.
#[test]
fn every_chat_finish_maps_and_failures_fail() {
    let wire = wire(&OPENROUTER, "openai/gpt-5-mini");
    for (finish, stop) in [
        ("stop", Some(StopReason::Stop)),
        ("end", Some(StopReason::Stop)),
        ("length", Some(StopReason::Length)),
        ("max_tokens", Some(StopReason::Length)),
        ("model_length", Some(StopReason::Length)),
        ("tool_calls", Some(StopReason::ToolUse)),
        ("function_call", Some(StopReason::ToolUse)),
        ("content_filter", None),
        ("error", None),
        ("network_error", None),
        ("x_rig_invented", None),
    ] {
        let frame = whole(json!({"role": "assistant", "content": "done"}), finish);
        let response = decode(&wire, Mode::Unary, vec![frame]).expect("the reply decodes");
        match stop {
            Some(stop) => assert_eq!(response.stop(), stop, "{finish}"),
            None => assert!(
                response.stop().is_failure(),
                "{finish}: {:?}",
                response.stop()
            ),
        }
    }
}

/// #1085, #2447, #2554: double-stringified arguments are an object, malformed
/// ones never fail the reply, and `null` ones go back as an object.
#[test]
fn arguments_are_always_an_object() {
    let wire = wire(&OPENAI, "gpt-4.1-nano");
    let reply = |arguments: &str| {
        let frame = whole(
            json!({"role": "assistant", "tool_calls": [{"id": "call_1", "type": "function",
                "function": {"name": "add", "arguments": arguments}}]}),
            "tool_calls",
        );
        let response = decode(&wire, Mode::Unary, vec![frame]).expect("a call never fails");
        let calls: Vec<ToolCall> = response.tool_calls().cloned().collect();
        let [call] = calls.as_slice() else {
            panic!("the call is kept: {:?}", response.choice);
        };
        call.clone()
    };
    assert_eq!(
        reply(r#""{\"x\":1}""#).function.arguments_value(),
        json!({"x": 1})
    );
    let malformed = reply(r#"{"x": 1"#);
    assert_eq!(malformed.function.arguments_value(), json!({"x": 1}));
    assert!(malformed.function.invalid_arguments.is_some());
    let null = reply("null");
    assert_eq!(null.function.arguments_value(), json!({}));
    let turn = AssistantMessage {
        content: vec![AssistantContent::ToolCall(ToolCall::new(
            CallId::from_wire("call_1"),
            ToolFunction::new(name("add"), Value::Null),
        ))],
        origin: Some(Origin::new("openai.chat", "openai", "gpt-4.1-nano")),
        stop: Some(StopReason::ToolUse),
    };
    let body = sent(&wire, vec![Message::user("q"), Message::Assistant(turn)]);
    assert_eq!(
        body["messages"][1]["tool_calls"][0]["function"]["arguments"],
        "{}"
    );
}

/// chatB NEW (placeholders), chatA NEW-1, NEW-2, #2380: what a model does not
/// read is downgraded, never refused: DeepSeek gets a user image as text,
/// OpenAI gets a tool result's image in a user message after the results,
/// llama.cpp keeps it in the result, and another model's assistant image is
/// a placeholder everywhere.
#[test]
fn the_adapter_downgrades_what_a_dialect_does_not_read() {
    use crate::completion::history::{
        ASSISTANT_IMAGE_OMITTED, TOOL_IMAGE_ATTACHED, USER_IMAGE_OMITTED,
    };
    let call = ToolCall::new(
        CallId::from_wire("call_1"),
        ToolFunction::new(name("screenshot"), json!({})),
    );
    let history = vec![
        Message::User {
            content: vec![UserContent::text("look"), UserContent::Image(png())],
        },
        Message::Assistant(AssistantMessage {
            content: vec![
                AssistantContent::Image(png()),
                AssistantContent::ToolCall(call.clone()),
            ],
            origin: Some(Origin::new("gemini.generate_content", "gemini", "gemini-3")),
            stop: Some(StopReason::ToolUse),
        }),
        Message::User {
            content: vec![UserContent::ToolResult(
                call.result(vec![ToolResultContent::Image(png())]),
            )],
        },
    ];
    let text = |body: &Value| body.to_string();
    let deepseek = sent(&wire(&DEEPSEEK, "deepseek-v4-flash"), history.clone());
    assert!(!text(&deepseek).contains("image_url"), "{deepseek}");
    assert!(text(&deepseek).contains(USER_IMAGE_OMITTED), "{deepseek}");
    let openai = sent(&wire(&OPENAI, "gpt-4.1-mini"), history.clone());
    assert!(text(&openai).contains(ASSISTANT_IMAGE_OMITTED), "{openai}");
    assert_eq!(
        openai["messages"][2]["content"], TOOL_IMAGE_ATTACHED,
        "{openai}"
    );
    assert_eq!(
        openai["messages"][3]["content"][1]["type"], "image_url",
        "{openai}"
    );
    let llamacpp = sent(&wire(&LLAMACPP, "Qwen3-VL-2B-Instruct-Q8_0"), history);
    assert_eq!(
        llamacpp["messages"][2]["content"][0]["type"], "image_url",
        "{llamacpp}"
    );
}

/// chatA NEW-6: Perplexity takes no tools, so the adapter renders a call and
/// its result as text the model still reads, rather than dropping them.
#[test]
fn perplexity_reads_a_tool_exchange_as_text() {
    let call = ToolCall::new(
        CallId::from_wire("call_1"),
        ToolFunction::new(name("lookup"), json!({"q": "rig"})),
    );
    let history = vec![
        Message::user("look it up"),
        Message::Assistant(AssistantMessage {
            content: vec![AssistantContent::ToolCall(call.clone())],
            origin: Some(Origin::new("openai.chat", "openai", "gpt-4.1")),
            stop: Some(StopReason::ToolUse),
        }),
        Message::User {
            content: vec![UserContent::ToolResult(
                call.result(vec![ToolResultContent::text("crimson")]),
            )],
        },
    ];
    let body = sent(&wire(&PERPLEXITY, "sonar-pro"), history);
    assert_eq!(
        body["messages"],
        json!([
            {"role": "user", "content": "look it up"},
            {"role": "assistant", "content": "[called tool lookup with {\"q\":\"rig\"}]"},
            {"role": "user", "content": [
                {"type": "text", "text": "[tool lookup result] crimson"},
                {"type": "text", "text": "next"},
            ]},
        ])
    );
}

/// chatA NEW-4: two foreign ids that differ only in punctuation reach
/// Mistral as they are, distinct, each result following its own call.
/// Mistral takes any `[A-Za-z0-9_-]` id (checked live on eight models).
#[test]
fn mistral_call_ids_stay_distinct() {
    let calls: Vec<ToolCall> = ["abc-123456", "abc_123456"]
        .into_iter()
        .map(|id| {
            ToolCall::new(
                CallId::from_wire(id),
                ToolFunction::new(name("add"), json!({})),
            )
        })
        .collect();
    let history = vec![
        Message::user("q"),
        Message::Assistant(AssistantMessage {
            content: calls
                .iter()
                .cloned()
                .map(AssistantContent::ToolCall)
                .collect(),
            origin: Some(Origin::new("openai.chat", "openai", "gpt-4.1")),
            stop: Some(StopReason::ToolUse),
        }),
        Message::User {
            content: calls
                .iter()
                .map(|call| {
                    UserContent::ToolResult(
                        call.result(vec![ToolResultContent::text(call.id.wire())]),
                    )
                })
                .collect(),
        },
    ];
    let body = sent(&wire(&MISTRAL, "mistral-small-latest"), history);
    let ids: Vec<&str> = body["messages"][1]["tool_calls"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|call| call["id"].as_str())
        .collect();
    assert_eq!(ids, ["abc-123456", "abc_123456"], "{body}");
    assert_eq!(body["messages"][2]["tool_call_id"], ids[0]);
    assert_eq!(body["messages"][2]["content"], "abc-123456");
    assert_eq!(body["messages"][3]["tool_call_id"], ids[1]);
}

/// Every content part kind has a sample, an invented one among them.
#[test]
fn every_content_part_has_a_sample() {
    let index = |part: &Part| match part {
        Part::Text(_) => 0,
        Part::Thinking(_) => 1,
        Part::Image => 2,
        Part::Unknown => 3,
    };
    let samples = [
        Part::of(&json!({"type": "text", "text": "a"})),
        Part::of(&json!({"type": "refusal", "refusal": "no"})),
        Part::of(&json!({"type": "thinking", "thinking": [{"type": "text", "text": "t"}]})),
        Part::of(&json!({"type": "image_url", "image_url": {"url": "https://x"}})),
        Part::of(&json!({"type": "x_rig_invented"})),
    ];
    assert_eq!(samples[1], Part::Text("no".to_owned()));
    assert_eq!(samples[2], Part::Thinking("t".to_owned()));
    assert_eq!(samples[4], Part::Unknown);
    assert_every_variant(&samples, index, 4);
}

/// Every call kind has a sample, an invented one among them.
#[test]
fn every_call_kind_has_a_sample() {
    let index = |kind: &CallKind| match kind {
        CallKind::Function => 0,
        CallKind::Custom => 1,
        CallKind::Unknown => 2,
    };
    let samples = [
        CallKind::of(&json!({"type": "function"})),
        CallKind::of(&json!({})),
        CallKind::of(&json!({"type": "custom"})),
        CallKind::of(&json!({"type": "x_rig_invented"})),
    ];
    assert_eq!(samples[1], CallKind::Function);
    assert_eq!(samples[3], CallKind::Unknown);
    assert_every_variant(&samples, index, 3);
}

/// Change 7: an item is a block's native only when the provider stated it
/// complete. A call the budget cut keeps no native, and so replays from its
/// canonical fields; a finished call and the reasoning the reply ended keep
/// theirs.
#[test]
fn only_a_complete_item_becomes_a_native() {
    let wire = wire(&DEEPSEEK, "deepseek-v4-flash");
    let reply = |arguments: &str, finish: &str| {
        let frames = vec![
            chunk(
                json!({"role": "assistant", "reasoning_content": "plan"}),
                None,
            ),
            chunk(
                json!({"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                    "function": {"name": "add", "arguments": arguments}}]}),
                Some(finish),
            ),
            WireFrame::Text("[DONE]".to_owned()),
        ];
        decode(&wire, Mode::Streaming, frames).expect("the reply decodes")
    };
    let cut = reply(r#"{"x": 1"#, "length");
    let [
        AssistantContent::Reasoning(reasoning),
        AssistantContent::ToolCall(call),
    ] = cut.choice.as_slice()
    else {
        panic!("reasoning and the cut call: {:?}", cut.choice);
    };
    assert!(reasoning.native.is_some(), "the reply ended the reasoning");
    assert!(call.native.is_none(), "the budget cut the call: {call:?}");
    let whole = reply(r#"{"x": 1}"#, "tool_calls");
    assert!(
        whole.tool_calls().all(|call| call.native.is_some()),
        "{:?}",
        whole.choice
    );
}

/// chatB "accepts_images dead": the text-only models of Z.AI, Moonshot,
/// MiniMax and MiMo, on their own Chat dialects and through OpenRouter, get
/// a placeholder for an image, by the rule the Messages wire shares, and
/// their vision models still get the image.
#[test]
fn review_text_only_chat_dialect_models_get_placeholders() {
    use crate::providers::openai::wire::{MINIMAX, MOONSHOT, XIAOMIMIMO, ZAI};
    let history = vec![Message::User {
        content: vec![UserContent::text("look"), UserContent::Image(png())],
    }];
    let cases: [(&'static Dialect, &str, bool); 14] = [
        (&ZAI, "glm-4.6", false),
        (&ZAI, "glm-4.5v", true),
        (&MOONSHOT, "kimi-k2-0905-preview", false),
        (&MOONSHOT, "kimi-k2.6", true),
        (&MINIMAX, "MiniMax-M2", false),
        (&MINIMAX, "MiniMax-M3", true),
        (&XIAOMIMIMO, "mimo-v2-flash", false),
        (&XIAOMIMIMO, "mimo-v2.5", true),
        (&OPENROUTER, "z-ai/glm-4.6", false),
        (&OPENROUTER, "moonshotai/kimi-k2-0905", false),
        (&OPENROUTER, "minimax/minimax-m2", false),
        (&OPENROUTER, "xiaomi/mimo-v2-flash", false),
        (&OPENROUTER, "deepseek/deepseek-chat-v3.1", false),
        (&OPENROUTER, "z-ai/glm-4.5v", true),
    ];
    let mut wrong = Vec::new();
    for (dialect, model, images) in cases {
        let body = sent(&wire(dialect, model), history.clone()).to_string();
        if body.contains("image_url") != images {
            wrong.push(format!("{}/{model}", dialect.name));
        }
    }
    assert!(
        wrong.is_empty(),
        "image_url sent to text-only models, or withheld from vision models: {wrong:?}"
    );
}

/// chatB H5 class: a part of a type the decoder does not expect never fails
/// the reply. A streamed call with a `null` index and a numeric id is still
/// the call, and object content is a content part.
#[test]
fn a_mistyped_part_never_fails_a_reply() {
    let wire = wire(&OPENAI, "gpt-4.1-mini");
    let frames = vec![
        chunk(
            json!({"role": "assistant", "content": {"type": "text", "text": "looking"}}),
            None,
        ),
        chunk(
            json!({"tool_calls": [{"index": null, "id": 42, "type": "function",
                "function": {"name": "add", "arguments": "{\"x\":1}"}}]}),
            Some("tool_calls"),
        ),
        WireFrame::Text("[DONE]".to_owned()),
    ];
    let response = decode(&wire, Mode::Streaming, frames).expect("the reply decodes");
    let [
        AssistantContent::Text(text),
        AssistantContent::ToolCall(call),
    ] = response.choice.as_slice()
    else {
        panic!("the text and the call: {:?}", response.choice);
    };
    assert_eq!(text.text, "looking");
    assert_eq!(call.id.wire(), "42");
    assert_eq!(call.function.arguments_value(), json!({"x": 1}));
}

/// chatB (audio): an answer's audio transcript is a text block whose native
/// is the audio item, so the same model gets the audio back by its id with
/// the transcript, and another model gets the transcript.
#[test]
fn an_answers_audio_transcript_is_text() {
    let audio = json!({"id": "audio_1", "transcript": "hello there", "data": "UklG",
        "expires_at": 1});
    let modes = [
        (
            Mode::Unary,
            vec![whole(
                json!({"role": "assistant", "content": null, "audio": audio}),
                "stop",
            )],
        ),
        (
            Mode::Streaming,
            vec![
                chunk(
                    json!({"role": "assistant", "content": null,
                        "audio": {"id": "audio_1", "transcript": "hello "}}),
                    None,
                ),
                chunk(
                    json!({"audio": {"transcript": "there", "data": "UklG", "expires_at": 1}}),
                    Some("stop"),
                ),
                WireFrame::Text("[DONE]".to_owned()),
            ],
        ),
    ];
    let model = "gpt-4o-audio-preview";
    for (mode, frames) in modes {
        let response = decode(&wire(&OPENAI, model), mode, frames).expect("the reply decodes");
        let [AssistantContent::Text(text)] = response.choice.as_slice() else {
            panic!(
                "{mode:?}: the transcript is the one block: {:?}",
                response.choice
            );
        };
        assert_eq!(text.text, "hello there", "{mode:?}");
        assert_eq!(
            AssistantContent::Text(text.clone()).native_item(),
            Some(&json!({"audio": {"id": "audio_1"}})),
            "{mode:?}"
        );
        let history = vec![Message::user("hi"), Message::Assistant(turn_of(&response))];
        let same = sent(&wire(&OPENAI, model), history.clone());
        assert_eq!(
            same["messages"][1],
            json!({"role": "assistant", "content": "hello there", "audio": {"id": "audio_1"}}),
            "{mode:?}"
        );
        let other = sent(&wire(&OPENAI, "gpt-4.1-mini"), history);
        assert_eq!(
            other["messages"][1],
            json!({"role": "assistant", "content": "hello there"}),
            "{mode:?}"
        );
    }
}

/// chat NEW-3, chat_new NEW-4: a call rig issued an id for (the second of
/// two calls a reply named alike, or one an agent built) reaches Mistral
/// with one spelling on the call and its result, same model or not. One
/// `WireIds` spells every id through the target's normalizer.
#[test]
fn a_rig_issued_id_reaches_mistral_spelled_once() {
    let mistral = wire(&MISTRAL, "mistral-large-latest");
    let frames = vec![whole(
        json!({"role": "assistant", "content": "", "tool_calls": [
            {"id": "abcDEF123", "type": "function", "function": {"name": "f", "arguments": "{}"}},
            {"id": "abcDEF123", "type": "function", "function": {"name": "g", "arguments": "{}"}}]}),
        "tool_calls",
    )];
    let reply = decode(&mistral, Mode::Unary, frames).expect("the reply decodes");
    let calls: Vec<ToolCall> = reply.tool_calls().cloned().collect();
    let results = |calls: &[ToolCall]| Message::User {
        content: calls
            .iter()
            .map(|call| UserContent::ToolResult(call.result(vec![ToolResultContent::text("r")])))
            .collect(),
    };
    let hand_built = ToolCall::new(
        CallId::Local(crate::message::LocalCallId::new()),
        ToolFunction::new(name("h"), json!({})),
    );
    let hand_turn = AssistantMessage {
        content: vec![AssistantContent::ToolCall(hand_built.clone())],
        origin: Some(Origin::new(
            "openai.chat",
            "mistral",
            "mistral-large-latest",
        )),
        stop: Some(StopReason::ToolUse),
    };
    let body = sent(
        &mistral,
        vec![
            Message::user("q"),
            Message::Assistant(turn_of(&reply)),
            results(&calls),
            Message::Assistant(hand_turn),
            results(std::slice::from_ref(&hand_built)),
        ],
    );
    let messages = body["messages"].as_array().expect("messages");
    let ids: Vec<&str> = messages
        .iter()
        .flat_map(|message| message["tool_calls"].as_array().into_iter().flatten())
        .filter_map(|call| call["id"].as_str())
        .collect();
    let answers: Vec<&str> = messages
        .iter()
        .filter_map(|message| message["tool_call_id"].as_str())
        .collect();
    assert_eq!(ids.len(), 3, "{body}");
    let distinct: std::collections::HashSet<&&str> = ids.iter().collect();
    assert_eq!(distinct.len(), 3, "{body}");
    assert_eq!(ids, answers, "each result names its own call: {body}");
}

/// chat NEW-4: Together ends a successful turn with `eos`, which its API
/// reference lists; the turn is a stop and stays in history.
#[test]
fn together_eos_is_a_stop() {
    use crate::providers::openai::wire::TOGETHER;
    let together = wire(&TOGETHER, "meta-llama/Llama-3.3-70B-Instruct-Turbo");
    let frames = vec![whole(
        json!({"role": "assistant", "content": "Paris is the capital."}),
        "eos",
    )];
    let reply = decode(&together, Mode::Unary, frames).expect("the reply decodes");
    assert_eq!(reply.stop(), StopReason::Stop);
    let body = sent(
        &together,
        vec![
            Message::user("capital?"),
            Message::Assistant(turn_of(&reply)),
        ],
    );
    assert_eq!(
        body["messages"][1]["content"], "Paris is the capital.",
        "{body}"
    );
}

/// chat NEW-5, chat_new NEW-7, #1266, #1333: an edited same-model reasoning
/// block goes back under the field it arrived in, as pi keeps
/// `thinkingSignature`: DeepSeek and Kimi read their reasoning back on a
/// tool-call turn.
#[test]
fn an_edited_reasoning_block_keeps_its_field() {
    use crate::providers::openai::wire::MOONSHOT;
    for (dialect, model, id) in [
        (&DEEPSEEK, "deepseek-reasoner", "call_0"),
        (&MOONSHOT, "kimi-k2-thinking", "functions.f:0"),
    ] {
        let chat = wire(dialect, model);
        let frames = vec![whole(
            json!({"role": "assistant", "content": "", "reasoning_content": "plan",
                "tool_calls": [{"id": id, "type": "function",
                    "function": {"name": "f", "arguments": "{}"}}]}),
            "tool_calls",
        )];
        let reply = decode(&chat, Mode::Unary, frames).expect("the reply decodes");
        let mut turn = turn_of(&reply);
        for block in &mut turn.content {
            if let AssistantContent::Reasoning(reasoning) = block {
                reasoning.text = "plan (edited)".to_owned();
            }
        }
        let body = sent(
            &chat,
            vec![
                Message::user("q"),
                Message::Assistant(turn),
                Message::tool_result(CallId::from_wire(id), name("f"), "ok"),
            ],
        );
        assert_eq!(
            body["messages"][1]["reasoning_content"], "plan (edited)",
            "{}: {body}",
            dialect.name
        );
    }
}

/// #1317: a dialect that reads reasoning back on every assistant message
/// (DeepSeek) states its field, so a same-model reasoning block with no
/// item goes under it and a turn with none sends it empty.
#[test]
fn deepseek_states_its_reasoning_field() {
    let deepseek = wire(&DEEPSEEK, "deepseek-reasoner");
    let origin = Origin::new("openai.chat", "deepseek", "deepseek-reasoner");
    let turn = |content| {
        Message::Assistant(AssistantMessage {
            content,
            origin: Some(origin.clone()),
            stop: Some(StopReason::Stop),
        })
    };
    let body = sent(
        &deepseek,
        vec![
            Message::user("q"),
            turn(vec![
                AssistantContent::reasoning("weighed it"),
                AssistantContent::text("a"),
            ]),
            Message::user("again"),
            turn(vec![AssistantContent::text("b")]),
        ],
    );
    assert_eq!(
        body["messages"][1]["reasoning_content"], "weighed it",
        "{body}"
    );
    assert_eq!(body["messages"][3]["reasoning_content"], "", "{body}");
}

/// chat NEW-7: late `reasoning_details` (Gemini through OpenRouter) fold to
/// the same blocks from a whole message as from its stream: a detail that
/// signs one of the message's calls follows the answer, as a stream sends
/// it with the call.
#[test]
fn late_reasoning_details_fold_alike_whole_and_streamed() {
    let openrouter = wire(&OPENROUTER, "google/gemini-3-pro-preview");
    let detail = json!({"type": "reasoning.encrypted", "data": "ENC", "id": "tool_1",
        "format": "google-gemini-v1", "index": 0});
    let call = json!({"index": 0, "id": "tool_1", "type": "function",
        "function": {"name": "f", "arguments": "{}"}});
    let unary = decode(
        &openrouter,
        Mode::Unary,
        vec![whole(
            json!({"role": "assistant", "content": "Let me check.",
                "reasoning_details": [detail], "tool_calls": [call]}),
            "tool_calls",
        )],
    )
    .expect("the whole reply decodes");
    let streamed = decode(
        &openrouter,
        Mode::Streaming,
        vec![
            chunk(
                json!({"role": "assistant", "content": "Let me check."}),
                None,
            ),
            chunk(
                json!({"reasoning_details": [detail], "tool_calls": [call]}),
                Some("tool_calls"),
            ),
            WireFrame::Text("[DONE]".to_owned()),
        ],
    )
    .expect("the stream decodes");
    let canonical = |response: &crate::completion::CompletionResponse| {
        response
            .choice
            .iter()
            .map(AssistantContent::canonical)
            .collect::<Vec<_>>()
    };
    assert_eq!(canonical(&unary), canonical(&streamed));
    assert!(matches!(
        unary.choice.as_slice(),
        [
            AssistantContent::Text(_),
            AssistantContent::Reasoning(_),
            AssistantContent::ToolCall(_)
        ]
    ));
}

/// chat NEW-8, chat_new NEW-8: a custom call stays custom when its input is
/// edited (its kind survives as identity) and when another model replays it
/// to a request that declares the custom tool (pi decides by the declared
/// tools).
#[test]
fn a_custom_call_stays_custom_after_an_edit_and_across_models() {
    let frames = vec![whole(
        json!({"role": "assistant", "tool_calls": [{"id": "call_c",
            "type": "custom", "custom": {"name": "grep", "input": "rig"}}]}),
        "tool_calls",
    )];
    let reply = decode(&wire(&OPENAI, "gpt-5.2"), Mode::Unary, frames).expect("decodes");
    let mut edited = turn_of(&reply);
    for block in &mut edited.content {
        if let AssistantContent::ToolCall(call) = block {
            call.function
                .arguments
                .insert("input".into(), json!("rig2"));
        }
    }
    let result = Message::tool_result(CallId::from_wire("call_c"), name("grep"), "found");
    let same = sent(
        &wire(&OPENAI, "gpt-5.2"),
        vec![
            Message::user("q"),
            Message::Assistant(edited),
            result.clone(),
        ],
    );
    assert_eq!(
        same["messages"][1]["tool_calls"][0],
        json!({"type": "custom", "id": "call_c", "custom": {"name": "grep", "input": "rig2"}}),
        "{same}"
    );
    let declared = CompletionRequest::new("next").additional_params(json!({"tools": [
        {"type": "custom", "custom": {"name": "grep", "description": "search"}}]}));
    let other = sent_with(
        &wire(&OPENAI, "gpt-5.4"),
        declared,
        vec![
            Message::user("q"),
            Message::Assistant(turn_of(&reply)),
            result,
        ],
    );
    assert_eq!(
        other["messages"][1]["tool_calls"][0],
        json!({"type": "custom", "id": "call_c", "custom": {"name": "grep", "input": "rig"}}),
        "{other}"
    );
}

/// chat NEW-9: a result with no content, or only blank text, says so as pi
/// does, rather than failing the request or sending an empty string.
#[test]
fn an_empty_tool_result_says_it_has_no_output() {
    let openai = wire(&OPENAI, "gpt-4o");
    let call = ToolCall::from_wire("c1", ToolFunction::new(name("f"), json!({})));
    for content in [vec![], vec![ToolResultContent::text(" ")]] {
        let body = sent(
            &openai,
            vec![
                Message::user("q"),
                Message::Assistant(AssistantMessage::new(vec![AssistantContent::ToolCall(
                    call.clone(),
                )])),
                Message::User {
                    content: vec![UserContent::ToolResult(call.result(content))],
                },
            ],
        );
        assert_eq!(
            body["messages"][2],
            json!({"role": "tool", "tool_call_id": "c1",
                "content": crate::completion::history::NO_TOOL_OUTPUT}),
            "{body}"
        );
    }
}

/// chat_new NEW-2: Moonshot reports usage on the choice rather than the
/// chunk; pi reads it there.
#[test]
fn usage_on_the_choice_is_read() {
    use crate::providers::openai::wire::MOONSHOT;
    let frames = vec![
        chunk(json!({"role": "assistant", "content": "hi"}), None),
        WireFrame::Text(
            json!({"id": "c", "object": "chat.completion.chunk", "model": "m",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop",
                    "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9}}]})
            .to_string(),
        ),
        WireFrame::Text("[DONE]".to_owned()),
    ];
    let reply = decode(
        &wire(&MOONSHOT, "kimi-k2-0905-preview"),
        Mode::Streaming,
        frames,
    )
    .expect("the stream decodes");
    assert_eq!(reply.usage.input_tokens, Some(7));
    assert_eq!(reply.usage.total_tokens, Some(9));
}

/// chat NEW-2, chat_new NEW-6: a whole reply's calls are delimited by the
/// list, so calls that state one index, even without ids, stay apart.
#[test]
fn whole_reply_calls_are_delimited_by_the_list() {
    let frames = vec![whole(
        json!({"role": "assistant", "tool_calls": [
            {"index": 0, "type": "function",
                "function": {"name": "weather", "arguments": "{\"city\":\"Paris\"}"}},
            {"index": 0, "type": "function",
                "function": {"name": "weather", "arguments": "{\"city\":\"Rome\"}"}}]}),
        "tool_calls",
    )];
    let reply = decode(&wire(&OPENROUTER, "x/y"), Mode::Unary, frames).expect("decodes");
    let calls: Vec<ToolCall> = reply.tool_calls().cloned().collect();
    let [paris, rome] = calls.as_slice() else {
        panic!("two calls: {:?}", reply.choice);
    };
    assert_eq!(paris.function.arguments_value(), json!({"city": "Paris"}));
    assert_eq!(rome.function.arguments_value(), json!({"city": "Rome"}));
    assert_ne!(paris.id, rome.id);
}

fn call_turn(id: &str, tool: &str) -> Message {
    Message::Assistant(AssistantMessage::new(vec![AssistantContent::ToolCall(
        ToolCall::from_wire(id, ToolFunction::new(name(tool), json!({}))),
    )]))
}

fn result(id: &str, tool: &str) -> Message {
    Message::tool_result(CallId::from_wire(id), name(tool), "x")
}

/// A content part with no `type` and a string `text` is text, and nothing
/// of it lands on the replayed message's own fields.
#[test]
fn a_typeless_text_part_is_text() {
    let wire = wire(&OPENAI, "gpt-4o");
    let frames = vec![whole(
        json!({"role": "assistant", "content": [{"text": "hello"}]}),
        "stop",
    )];
    let response = decode(&wire, Mode::Unary, frames).expect("the reply decodes");
    assert_eq!(response.text(), "hello", "{:?}", response.choice);
    let body = sent(
        &wire,
        vec![Message::user("q"), Message::Assistant(turn_of(&response))],
    );
    let assistant = &body["messages"][1];
    assert_eq!(assistant["content"], "hello", "{body}");
    assert!(assistant.get("text").is_none(), "{body}");
}

/// pi's `requiresReasoningContentOnAssistantMessages` follows the model:
/// DeepSeek at its own base URL behind the OpenAI dialect, every MiMo
/// model, and Kimi K3 under any gateway path send an empty
/// `reasoning_content` on a call turn with none; other models send none.
#[test]
fn reasoning_content_follows_the_model() {
    let history = || {
        vec![
            Message::user("q"),
            call_turn("call_0", "f"),
            result("call_0", "f"),
        ]
    };
    let deepseek = OpenAIConfig::with_key(&OPENAI, "key")
        .with_base_url("https://api.deepseek.com")
        .chat("deepseek-reasoner");
    for wire in [
        deepseek,
        wire(&DEEPSEEK, "deepseek-reasoner"),
        wire(&XIAOMIMIMO, "mimo-v2.5"),
        wire(&MOONSHOT, "kimi-k3"),
        wire(&OPENAI, "accounts/fireworks/models/kimi-k3"),
        wire(&OPENROUTER, "moonshotai/kimi-k2.6"),
    ] {
        let body = sent(&wire, history());
        assert_eq!(
            body["messages"][1]["reasoning_content"],
            json!(""),
            "{}: {body}",
            wire.model
        );
    }
    for wire in [wire(&OPENAI, "gpt-4o"), wire(&MOONSHOT, "kimi-k2.6")] {
        let body = sent(&wire, history());
        assert!(
            body["messages"][1].get("reasoning_content").is_none(),
            "{}: {body}",
            wire.model
        );
    }
}

/// OpenRouter takes a Claude turn's signed `reasoning_details` back after
/// the tool list changes, on every route it serves Claude from, so they
/// replay verbatim whether or not the model binds its thinking.
#[test]
fn openrouter_claude_thinking_replays_whatever_the_tools() {
    let frames = || {
        vec![
            chunk(
                json!({"role": "assistant", "reasoning": "plan", "reasoning_details": [
                    {"type": "reasoning.text", "text": "plan", "signature": "SIG",
                     "format": "anthropic-claude-v1", "index": 0}]}),
                None,
            ),
            chunk(
                json!({"tool_calls": [{"index": 0, "id": "toolu_1", "type": "function",
                    "function": {"name": "f", "arguments": "{}"}}]}),
                Some("tool_calls"),
            ),
            WireFrame::Text("[DONE]".to_owned()),
        ]
    };
    for model in ["anthropic/claude-opus-5.5", "anthropic/claude-haiku-4.5"] {
        let wire = wire(&OPENROUTER, model);
        let response = decode(&wire, Mode::Streaming, frames()).expect("the reply decodes");
        let mut next = CompletionRequest::new("next");
        next.tools.push(crate::completion::ToolDefinition::new(
            name("g"),
            "a tool added since",
            json!({"type": "object", "properties": {}}),
        ));
        let body = sent_with(
            &wire,
            next,
            vec![
                Message::user("q"),
                Message::Assistant(turn_of(&response)),
                result("toolu_1", "f"),
            ],
        );
        assert!(
            body["messages"][1].get("reasoning_details").is_some(),
            "{model}: {body}"
        );
    }
}

/// Two whole calls streamed with neither an index nor an id are two calls:
/// a fragment continues the latest call only until its arguments are a
/// whole object.
#[test]
fn index_less_id_less_calls_split_once_whole() {
    let wire = wire(&OPENAI, "gpt-4o");
    let call = |name: &str, arguments: &str| {
        chunk(
            json!({"tool_calls": [{"type": "function",
                "function": {"name": name, "arguments": arguments}}]}),
            None,
        )
    };
    let frames = vec![
        chunk(json!({"role": "assistant"}), None),
        call("f", "{\"q\":"),
        chunk(
            json!({"tool_calls": [{"function": {"arguments": "\"a\"}"}}]}),
            None,
        ),
        call("f", "{\"q\":\"b\"}"),
        chunk(json!({}), Some("tool_calls")),
        WireFrame::Text("[DONE]".to_owned()),
    ];
    let response = decode(&wire, Mode::Streaming, frames).expect("the reply decodes");
    let arguments: Vec<Value> = response
        .choice
        .iter()
        .filter_map(|block| match block {
            AssistantContent::ToolCall(call) => {
                Some(Value::Object(call.function.arguments.clone()))
            }
            _ => None,
        })
        .collect();
    assert_eq!(
        arguments,
        [json!({"q": "a"}), json!({"q": "b"})],
        "{:?}",
        response.choice
    );
}

/// R6-NEW-7: a same-model turn of only reasoning makes no Chat message
/// (pi skips one with neither content nor calls), so `adapt` drops it and
/// the user messages around it become one; a Mistral thinking part is
/// content, so a thinking-only Mistral turn still goes back.
#[test]
fn a_turn_with_nothing_chat_can_send_leaves_no_adjacent_users() {
    let roles = |body: &Value| -> Vec<String> {
        body["messages"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|message| message["role"].as_str().map(str::to_owned))
            .collect()
    };
    let llama = wire(&LLAMACPP, "qwen3");
    let reasoning_only = decode(
        &llama,
        Mode::Unary,
        vec![whole(
            json!({"role": "assistant", "content": "", "reasoning_content": "Let me think"}),
            "length",
        )],
    )
    .expect("the reply decodes");
    assert_eq!(reasoning_only.stop(), StopReason::Length);
    let body = sent(
        &llama,
        vec![
            Message::user("q"),
            Message::Assistant(turn_of(&reasoning_only)),
        ],
    );
    assert_eq!(roles(&body), ["user"], "{body}");

    let mistral = wire(&MISTRAL, "magistral-medium-latest");
    let thinking_only = decode(
        &mistral,
        Mode::Unary,
        vec![whole(
            json!({"role": "assistant", "content": [
                {"type": "thinking", "thinking": [{"type": "text", "text": "plan"}]}
            ]}),
            "length",
        )],
    )
    .expect("the reply decodes");
    let body = sent(
        &mistral,
        vec![
            Message::user("q"),
            Message::Assistant(turn_of(&thinking_only)),
        ],
    );
    assert_eq!(roles(&body), ["user", "assistant", "user"], "{body}");
}

/// Perplexity, which declares no tools, gets what `adapt` makes of awkward
/// histories: a result no call asked for is dropped, as it is when tools are
/// declared (round-5 F11), blank user text goes (F8), and two assistant
/// turns stay two, with no text glued together (F5).
#[test]
fn perplexity_gets_orphans_dropped_blanks_gone_and_turns_apart() {
    let other = Some(Origin::new(
        "anthropic.messages",
        "anthropic",
        "claude-opus-5-5",
    ));
    let assistant = |text: &str| {
        Message::Assistant(AssistantMessage {
            content: vec![AssistantContent::text(text)],
            origin: other.clone(),
            stop: Some(StopReason::Stop),
        })
    };
    let orphan = Message::User {
        content: vec![UserContent::ToolResult(
            ToolCall::new(
                CallId::from_wire("gone"),
                ToolFunction::new(name("lookup"), json!({})),
            )
            .result(vec![ToolResultContent::text("r")]),
        )],
    };
    let wire = wire(&PERPLEXITY, "sonar");
    let body = sent(
        &wire,
        vec![
            orphan.clone(),
            Message::user("a"),
            assistant("First sentence."),
            orphan,
            Message::user("   "),
            assistant("Second sentence."),
            Message::user("q"),
        ],
    );
    let text = body["messages"].to_string();
    assert!(
        !text.contains("[tool lookup"),
        "the orphan result is dropped: {text}"
    );
    assert!(
        !text.contains("First sentence.Second"),
        "the turns stay apart: {text}"
    );
    for message in body["messages"].as_array().into_iter().flatten() {
        let content = message["content"].as_str().unwrap_or("x");
        assert!(!content.trim().is_empty(), "no blank message: {text}");
    }
}
