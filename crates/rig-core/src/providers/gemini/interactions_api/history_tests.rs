//! How Interactions replies become turns, and how turns go back: every
//! documented status, one block per model output item, provider items only
//! for steps the API stated complete, resumed turns, lenient reads, and
//! what each model accepts.

use super::{Interactions, create_request_body};
use crate::completion::{CompletionRequest, CompletionResponse, ReplayTarget, ToolDefinition};
use crate::message::{
    AssistantContent, CallId, Image, ImageMediaType, Message, StopReason, ToolName, ToolResult,
    ToolResultContent, UserContent,
};
use crate::test_utils::history::{assert_restated_agrees, decode};
use crate::wire::{Mode, Wire, WireFrame};
use serde_json::{Value, json};

const MODEL: &str = "gemini-3-flash-preview";

fn config() -> crate::providers::gemini::GeminiConfig {
    crate::providers::gemini::GeminiConfig::new("test-key")
}

fn wire(model: &str) -> Interactions {
    config().interactions(model)
}

fn frame(value: Value) -> WireFrame {
    WireFrame::Text(value.to_string())
}

/// The whole interaction resource holding `steps`, ended with `status`.
fn resource(steps: Vec<Value>, status: &str) -> WireFrame {
    frame(json!({"id": "int_1", "model": MODEL, "status": status, "steps": steps}))
}

fn whole(steps: Vec<Value>, status: &str) -> CompletionResponse {
    decode(&wire(MODEL), Mode::Unary, [resource(steps, status)]).expect("the resource decodes")
}

fn event(value: Value) -> WireFrame {
    frame(value)
}

fn completed(interaction: Value) -> WireFrame {
    event(json!({"event_type": "interaction.completed", "interaction": interaction}))
}

/// A tool named `name` that takes no arguments.
fn tool(name: &str) -> ToolDefinition {
    ToolDefinition {
        name: ToolName::new(name).expect("a tool name"),
        description: format!("The {name} tool."),
        parameters: json!({"type": "object", "properties": {}}),
    }
}

/// The tools the calls and results of `history` name, so that the request
/// boundary sends them as calls and results rather than as text.
fn tools_of(history: &[Message]) -> Vec<ToolDefinition> {
    let mut names: Vec<String> = Vec::new();
    for message in history {
        let named: Vec<String> = match message {
            Message::Assistant(turn) => turn
                .tool_calls()
                .map(|call| call.function.name.to_string())
                .collect(),
            Message::User { content } => content
                .iter()
                .filter_map(|part| match part {
                    UserContent::ToolResult(result) => Some(result.name.to_string()),
                    _ => None,
                })
                .collect(),
            Message::System { .. } => Vec::new(),
        };
        for name in named {
            if !names.contains(&name) {
                names.push(name);
            }
        }
    }
    names.iter().map(|name| tool(name)).collect()
}

/// The `input` steps `request` encodes to for `target`, adapted at the
/// request boundary as the driver adapts it.
fn input_of(target: &Interactions, request: CompletionRequest) -> Vec<Value> {
    let request = <crate::operation::Completion as crate::wire::Operation>::prepare(
        request,
        &target.describe(),
    )
    .expect("the request is valid");
    match create_request_body(target, request, None).expect("the request builds") {
        Value::Object(mut body) => match body.shift_remove("input") {
            Some(Value::Array(steps)) => steps,
            other => panic!("the body sends its steps as `input`: {other:?}"),
        },
        other => panic!("the body is an object: {other}"),
    }
}

/// The steps `history`, then a new user message, encode to for `target`,
/// with the tools its calls and results name declared.
fn sent(target: &Interactions, history: Vec<Message>) -> Vec<Value> {
    let mut history = history;
    history.push(Message::user("next"));
    let mut request = CompletionRequest::from(history);
    request.tools = tools_of(&request.chat_history);
    input_of(target, request)
}

/// What `response` replays to `target`, between the prompt and the next
/// user message.
fn replayed_to(target: &Interactions, response: &CompletionResponse) -> Vec<Value> {
    let turn = response.message().expect("a turn");
    let steps = sent(target, vec![Message::user("q"), turn]);
    let count = steps.len();
    steps
        .into_iter()
        .skip(1)
        .take(count.saturating_sub(2))
        .collect()
}

fn text_step(text: &str) -> Value {
    json!({"type": "model_output", "content": [{"type": "text", "text": text}]})
}

fn image_item() -> Value {
    json!({"type": "image", "data": "aW1n", "mime_type": "image/png"})
}

fn audio_item() -> Value {
    json!({"type": "audio", "data": "YXVk", "mime_type": "audio/wav"})
}

fn mixed_step() -> Value {
    json!({"type": "model_output", "content": [
        {"type": "text", "text": "Here it is."}, image_item(), audio_item(),
    ]})
}

/// Another model gets the text and the image rebuilt, and no audio.
#[test]
fn a_mixed_output_reaches_another_model_as_its_canonical_blocks() {
    let response = whole(vec![mixed_step()], "completed");
    assert_eq!(
        replayed_to(&wire("gemini-2.5-flash"), &response),
        [
            text_step("Here it is."),
            json!({"type": "model_output", "content": [image_item()]}),
        ]
    );
}

/// A mixed output streamed as text deltas and whole media items folds into
/// the turn its whole form does.
#[test]
fn a_streamed_mixed_output_agrees_with_the_whole() {
    let delta =
        |delta: Value| event(json!({"event_type": "step.delta", "index": 0, "delta": delta}));
    assert_restated_agrees(
        &wire(MODEL),
        [resource(vec![mixed_step()], "completed")],
        [
            event(
                json!({"event_type": "step.start", "index": 0, "step": {"type": "model_output"}}),
            ),
            delta(json!({"type": "text", "text": "Here "})),
            delta(json!({"type": "text", "text": "it is."})),
            delta(image_item()),
            delta(audio_item()),
            event(json!({"event_type": "step.stop", "index": 0})),
            completed(json!({"id": "int_1", "model": MODEL, "status": "completed"})),
        ],
    );
}

/// A resumed read names the interaction's model, whole or streamed, even
/// when the stream joins after the completion stopped naming it, so its
/// turn replays to that model as its own: the signed thought goes back.
#[test]
fn a_resumed_turn_replays_to_its_model_as_the_same_model() {
    let thought = json!({"type": "thought", "signature": "c2lnX3Jlc3VtZQ=="});
    let output = text_step("done");
    let resume = super::InteractionResume::new(config(), "int_1");
    let streamed = [
        event(json!({"event_type": "interaction.created",
            "interaction": {"id": "int_1", "model": MODEL, "status": "in_progress"}})),
        event(json!({"event_type": "step.start", "index": 0, "step": thought})),
        event(json!({"event_type": "step.stop", "index": 0})),
        event(json!({"event_type": "step.start", "index": 1, "step": output})),
        event(json!({"event_type": "step.stop", "index": 1})),
        completed(json!({"id": "int_1", "status": "completed"})),
    ];
    for (mode, frames) in [
        (
            Mode::Unary,
            vec![resource(vec![thought.clone(), output.clone()], "completed")],
        ),
        (Mode::Streaming, streamed.to_vec()),
    ] {
        let response = decode(&resume, mode, frames).expect("the resumed reply decodes");
        assert_eq!(response.origin.model, MODEL, "{mode:?}");
        assert_eq!(
            replayed_to(&wire(MODEL), &response),
            [thought.clone(), output.clone()],
            "{mode:?}"
        );
    }
}

/// A turn of another model that calls `name` with the id `c_1`.
fn calling(name: &str) -> Message {
    let call = crate::message::ToolCall::new(
        CallId::from_wire("c_1"),
        crate::message::ToolFunction::new(ToolName::new(name).expect("a tool name"), json!({})),
    );
    let mut turn = crate::message::AssistantMessage::new(vec![AssistantContent::ToolCall(call)]);
    turn.origin = Some(crate::message::Origin::new(
        "other.api",
        "other",
        "other-model",
    ));
    turn.stop = Some(StopReason::ToolUse);
    Message::Assistant(turn)
}

/// Fields of unexpected types, and types rig does not know, never fail a
/// reply: a count sent as a string of digits is read and one of another
/// type is unknown, an unknown tool or modality in the echoed request is
/// kept, a hosted step missing its fields is an opaque step, and a call
/// without a name is dropped.
#[test]
fn unexpected_fields_never_fail_the_reply() {
    let document = json!({
        "id": "int_1", "model": MODEL, "status": "completed",
        "usage": {"total_input_tokens": "10", "total_output_tokens": 5, "total_thought_tokens": [1]},
        "tools": [{"type": "x_rig_tool"}],
        "response_modalities": ["x_rig_modality"],
        "steps": [
            {"type": "file_search_result", "result": [{"title": 1}]},
            {"type": "function_call", "id": "c_1", "name": 5},
            text_step("done"),
        ],
    });
    for (mode, frame) in [
        (Mode::Unary, frame(document.clone())),
        (Mode::Streaming, completed(document.clone())),
    ] {
        let response = decode(&wire(MODEL), mode, [frame]).expect("the reply decodes");
        assert_eq!(response.usage.input_tokens, Some(10));
        assert_eq!(response.usage.output_tokens, Some(5));
        assert_eq!(response.usage.reasoning_tokens, None);
        assert_eq!(response.stop(), StopReason::Stop);
    }
    let response = whole(
        document["steps"].as_array().cloned().unwrap_or_default(),
        "completed",
    );
    assert!(
        matches!(
            response.choice.as_slice(),
            [AssistantContent::Opaque(opaque), AssistantContent::Text(text)]
                if opaque.replay && text.text == "done"
        ),
        "{:?}",
        response.choice
    );
}

/// A request that continues a stored interaction sends the results that
/// answer the calls the API holds, though no turn in its history made them.
#[test]
fn results_continuing_a_stored_interaction_are_sent() {
    let result = Message::User {
        content: vec![UserContent::ToolResult(ToolResult {
            call: CallId::from_wire("c_1"),
            name: ToolName::new("add").expect("a tool name"),
            content: vec![ToolResultContent::text("18")],
            is_error: false,
        })],
    };
    for (params, sent_result) in [
        (Some(json!({"previous_interaction_id": "int_0"})), true),
        (None, false),
    ] {
        let mut request = CompletionRequest::from(vec![result.clone(), Message::user("next")]);
        request.additional_params = params;
        request.tools = vec![tool("add")];
        let steps = input_of(&wire(MODEL), request);
        assert_eq!(
            steps.iter().any(|step| step["type"] == "function_result"),
            sent_result,
            "{steps:?}"
        );
    }
}

/// #2143, round 4 NEW-2: Gemini 2 reads no multimodal function results, so
/// a result of several text parts (an MCP tool's), or of text and an image
/// the adapter moves out, reaches it as one string, never as a list of
/// content blocks the API refuses ("Multimodal function responses are not
/// supported for this model").
#[test]
fn a_multi_part_result_reaches_gemini_2_as_one_string() {
    let result = |content: Vec<ToolResultContent>| Message::User {
        content: vec![UserContent::ToolResult(ToolResult {
            call: CallId::from_wire("c_1"),
            name: ToolName::new("read").expect("a tool name"),
            content,
            is_error: false,
        })],
    };
    let image = ToolResultContent::Image(Image {
        data: crate::message::DocumentSourceKind::base64("aW1n"),
        media_type: Some(ImageMediaType::PNG),
        ..Image::default()
    });
    for content in [
        vec![
            ToolResultContent::text("line one"),
            ToolResultContent::text("line two"),
        ],
        vec![ToolResultContent::text("line one"), image],
    ] {
        let steps = sent(
            &wire("gemini-2.5-flash"),
            vec![Message::user("q"), calling("read"), result(content)],
        );
        let step = steps
            .iter()
            .find(|step| step["type"] == "function_result")
            .expect("the result is sent");
        assert!(step["result"].is_string(), "{step}");
        assert!(
            step["result"]
                .as_str()
                .is_some_and(|text| text.starts_with("line one")),
            "{step}"
        );
    }
}

/// Every Gemini wire reads a model the same way: an alias that names no
/// version reads multimodal results on Interactions as on GenerateContent,
/// and a `models/` prefix does not hide the version.
#[test]
fn every_gemini_wire_classifies_a_model_alike() {
    let rest = crate::providers::gemini::GeminiConfig::new("k").completion(MODEL);
    for model in [
        "gemini-flash-latest",
        "models/gemini-2.5-flash",
        "gemini-3-flash-preview",
        "claude-sonnet-4",
    ] {
        assert_eq!(wire(model).accepts(model), rest.accepts(model), "{model}");
    }
    assert!(
        !wire("models/gemini-2.5-flash")
            .accepts("models/gemini-2.5-flash")
            .tool_result_images
    );
}

/// A stored interaction holds the tools its calls named, so results that
/// continue it go back as function results even when the request declares
/// no tools of its own, as the recorded `tool_result_roundtrip` sends them.
#[test]
fn a_stored_continuation_keeps_its_results_without_declared_tools() {
    let result = Message::User {
        content: vec![UserContent::ToolResult(ToolResult {
            call: CallId::from_wire("c_1"),
            name: ToolName::new("add").expect("a tool name"),
            content: vec![ToolResultContent::text("18")],
            is_error: false,
        })],
    };
    let mut request = CompletionRequest::from(vec![result]);
    request.additional_params = Some(json!({"previous_interaction_id": "int_0"}));
    let steps = input_of(&wire(MODEL), request);
    assert_eq!(
        steps,
        [json!({"type": "function_result", "name": "add", "call_id": "c_1", "result": "18"})]
    );
}
