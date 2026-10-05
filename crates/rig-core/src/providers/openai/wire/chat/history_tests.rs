//! The replay findings of the Chat Completions audit, each as the test that
//! closes it: every part of a reply is a block, replay rebuilds the message
//! from its blocks, usage never fails a reply, every finish is mapped, and
//! the adapter downgrades what a dialect's models do not read.

use serde_json::{Value, json};

use super::Chat;
use crate::completion::{CompletionRequest, Message};
use crate::message::{
    AssistantContent, AssistantMessage, CallId, DocumentSourceKind, Image, ImageMediaType, Origin,
    StopReason, ToolCall, ToolFunction, ToolName, ToolResultContent, UserContent,
};
use crate::providers::openai::wire::{
    DEEPSEEK, Dialect, MISTRAL, MOONSHOT, OPENAI, OPENROUTER, OpenAIConfig, PERPLEXITY, XIAOMIMIMO,
};
use crate::test_utils::history::decode;
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
            {"role": "user", "content": "[tool lookup result] crimson\nnext"},
        ])
    );
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

/// Round 5 generated-history finding 2: `reasoning_details` join the
/// message's one reasoning block, the turn's first, wherever they arrive,
/// so a stream folds to the turn its whole message folds to. The first shape is the recorded
/// Gemini 3 stream through OpenRouter
/// (`openrouter/upstream_switch_matrix/switch_streamed.yaml`): a signature
/// alone after the content. The second interleaves reasoning, the answer
/// and a trailing ciphertext; the third has reasoning text after the
/// answer; the fourth is the recorded OpenAI ciphertext ahead of a call
/// (`openrouter/streaming_tools/stream_encrypted_reasoning_survives_into_the_next_turn.yaml`);
/// the fifth signs the first of two parallel calls, as Gemini 3 does.
/// Each replays the same bytes from either form.
#[test]
fn reasoning_details_fold_alike_whole_and_streamed_wherever_they_arrive() {
    let wire = wire(&OPENROUTER, "anthropic/claude-opus-5.5");
    let signature = json!({"type": "reasoning.text", "signature": "AY89a19Q",
        "format": "google-gemini-v1", "index": 0});
    let signed = json!({"type": "reasoning.text", "text": "plan", "signature": "EsYF",
        "format": "anthropic-claude-v1", "index": 0});
    let encrypted = json!({"type": "reasoning.encrypted", "data": "enc",
        "format": "openai-responses-v1", "index": 0, "id": "rs_1"});
    let call = json!({"id": "call_1", "type": "function",
        "function": {"name": "f", "arguments": "{}"}});
    let mut streamed_call = call.clone();
    streamed_call["index"] = json!(0);
    let second = json!({"id": "call_2", "type": "function",
        "function": {"name": "f", "arguments": "{}"}});
    let mut streamed_second = second.clone();
    streamed_second["index"] = json!(1);
    let signs_call = json!({"type": "reasoning.encrypted", "data": "gsig",
        "format": "google-gemini-v1", "index": 0, "id": "call_1"});
    let cases = [
        (
            json!({"role": "assistant", "content": "The code is amber.",
                "reasoning_details": [signature]}),
            vec![
                json!({"role": "assistant", "content": "The code is amber."}),
                json!({"role": "assistant", "content": "", "reasoning_details": [signature]}),
            ],
            vec!["reasoning", "text"],
        ),
        (
            json!({"role": "assistant", "content": "answer", "reasoning": "plan",
                "reasoning_details": [signed, encrypted]}),
            vec![
                json!({"role": "assistant", "reasoning": "plan", "reasoning_details": [signed]}),
                json!({"content": "answer"}),
                json!({"reasoning_details": [encrypted]}),
            ],
            vec!["reasoning", "text"],
        ),
        (
            json!({"role": "assistant", "content": "ab", "reasoning": "xy"}),
            vec![
                json!({"role": "assistant", "reasoning": "x"}),
                json!({"content": "a"}),
                json!({"reasoning": "y"}),
                json!({"content": "b"}),
            ],
            vec!["reasoning", "text"],
        ),
        (
            json!({"role": "assistant", "reasoning_details": [encrypted], "tool_calls": [call]}),
            vec![
                json!({"role": "assistant", "reasoning_details": [encrypted]}),
                json!({"tool_calls": [streamed_call]}),
            ],
            vec!["reasoning", "call"],
        ),
        (
            json!({"role": "assistant", "content": "Checking.", "tool_calls": [call, second],
                "reasoning_details": [signs_call]}),
            vec![
                json!({"role": "assistant", "content": "Checking."}),
                json!({"tool_calls": [streamed_call], "reasoning_details": [signs_call]}),
                json!({"tool_calls": [streamed_second]}),
            ],
            vec!["reasoning", "text", "call", "call"],
        ),
    ];
    for (message, deltas, kinds) in cases {
        let finish = if message.get("tool_calls").is_some() {
            "tool_calls"
        } else {
            "stop"
        };
        let unary = decode(&wire, Mode::Unary, vec![whole(message.clone(), finish)])
            .expect("the whole reply decodes");
        let mut frames: Vec<WireFrame> = deltas.into_iter().map(|d| chunk(d, None)).collect();
        frames.push(chunk(json!({}), Some(finish)));
        frames.push(WireFrame::Text("[DONE]".to_owned()));
        let streamed = decode(&wire, Mode::Streaming, frames).expect("the stream decodes");
        assert_eq!(unary.choice, streamed.choice, "{message}");
        let found: Vec<&str> = streamed
            .choice
            .iter()
            .map(|block| match block {
                AssistantContent::Reasoning(_) => "reasoning",
                AssistantContent::Text(_) => "text",
                AssistantContent::ToolCall(_) => "call",
                _ => "other",
            })
            .collect();
        assert_eq!(found, kinds, "{message}");
        let body = |response: &crate::completion::CompletionResponse| {
            sent(
                &wire,
                vec![Message::user("q"), Message::Assistant(turn_of(response))],
            )
        };
        let replayed = body(&streamed);
        assert_eq!(body(&unary), replayed, "{message}");
        for key in ["reasoning", "reasoning_details"] {
            assert_eq!(
                replayed["messages"][1].get(key),
                message.get(key),
                "{replayed}"
            );
        }
    }
}
