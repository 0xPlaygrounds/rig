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
    DEEPSEEK, Dialect, LLAMACPP, MISTRAL, OPENAI, OPENROUTER, OpenAIConfig, PERPLEXITY,
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

/// The body `history` sends on `wire`, prepared as the driver prepares it.
fn sent(wire: &Chat, history: Vec<Message>) -> Value {
    let mut request = CompletionRequest::new("next");
    request.chat_history = history;
    request.chat_history.push(Message::user("next"));
    let request = crate::operation::Completion::prepare(request, &wire.describe())
        .expect("the request prepares");
    json_body(&wire.encode(request, Mode::Unary).expect("encodes").request)
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
        let message = response.head().native.expect("the message").item;
        assert_eq!(
            message["content"],
            json!([thinking("plan it"), {"type": "text", "text": "Hi"}]),
            "{mode:?}"
        );
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
        native: None,
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
            native: None,
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
            native: None,
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
            {"role": "user", "content": "[tool lookup result] crimson\nnext"},
        ])
    );
}

/// chatA NEW-4: two foreign ids Mistral's rule maps to the same nine
/// alphanumerics stay distinct, each result following its own call.
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
            native: None,
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
    assert_eq!(ids.len(), 2);
    assert_ne!(ids[0], ids[1], "{body}");
    assert!(
        ids.iter()
            .all(|id| id.len() == 9 && id.chars().all(|c| c.is_ascii_alphanumeric()))
    );
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
