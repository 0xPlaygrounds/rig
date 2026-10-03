//! Cohere's `/v2/chat`: a tool plan, thinking and text content items, and
//! calls, as a whole reply or the events of its stream. The rebuild sends
//! each content item in order and the plan as its field.

use rig_core::completion::CompletionRequest;
use rig_core::error::EncodeError;
use rig_core::message::{AssistantContent, AssistantMessage};
use rig_core::providers::cohere::{Chat, CohereConfig};
use rig_core::test_utils::history_conformance::{
    Ablation, Ending, HistoryFixture, Shape, http_body,
};
use rig_core::wire::{Mode, Wire, WireFrame};
use serde_json::{Map, Value, json};

pub struct CohereHistory;

fn usage() -> Value {
    json!({"billed_units": {"input_tokens": 8, "output_tokens": 5},
        "tokens": {"input_tokens": 10, "output_tokens": 5}, "cached_tokens": 2})
}

fn event(event: Value) -> WireFrame {
    WireFrame::Text(event.to_string())
}

fn call(arguments: &str) -> Value {
    json!({"id": "call_1", "type": "function",
        "function": {"name": "lookup", "arguments": arguments}})
}

/// A reply of a plan, content items and calls: the whole body, or the
/// events of its stream.
fn reply(
    plan: Option<&str>,
    content: &[Value],
    calls: &[Value],
    finish: &str,
    mode: Mode,
) -> Vec<WireFrame> {
    match mode {
        Mode::Unary => {
            let mut message = json!({"role": "assistant", "content": content,
                "tool_calls": calls, "citations": []});
            message["tool_plan"] = json!(plan.unwrap_or_default());
            vec![event(
                json!({"id": "msg_1", "finish_reason": finish, "message": message,
                "usage": usage()}),
            )]
        }
        Mode::Streaming => {
            let mut frames = vec![event(json!({"type": "message-start", "id": "msg_1",
                "delta": {"message": {"role": "assistant", "content": [], "tool_plan": "",
                    "tool_calls": [], "citations": []}}}))];
            if let Some(plan) = plan {
                frames.push(event(json!({"type": "tool-plan-delta",
                    "delta": {"message": {"tool_plan": plan}}})));
            }
            for (index, item) in content.iter().enumerate() {
                let key = if item["type"] == "thinking" {
                    "thinking"
                } else {
                    "text"
                };
                let mut start = item.clone();
                let text = start.get(key).cloned();
                if text.is_some() {
                    start[key] = json!("");
                }
                frames.push(event(json!({"type": "content-start", "index": index,
                    "delta": {"message": {"content": start}}})));
                if let Some(text) = text {
                    frames.push(event(json!({"type": "content-delta", "index": index,
                        "delta": {"message": {"content": {key: text}}}})));
                }
                frames.push(event(json!({"type": "content-end", "index": index})));
            }
            for (index, call) in calls.iter().enumerate() {
                let mut start = call.clone();
                start["function"]["arguments"] = json!("");
                frames.push(event(json!({"type": "tool-call-start", "index": index,
                    "delta": {"message": {"tool_calls": start}}})));
                frames.push(event(json!({"type": "tool-call-delta", "index": index,
                    "delta": {"message": {"tool_calls": {"function": {
                        "arguments": call["function"]["arguments"]}}}}})));
                frames.push(event(json!({"type": "tool-call-end", "index": index})));
            }
            frames.push(event(json!({"type": "message-end",
                "delta": {"finish_reason": finish, "usage": usage()}})));
            frames
        }
    }
}

fn thinking(text: &str) -> Value {
    json!({"type": "thinking", "thinking": text})
}

fn text(text: &str) -> Value {
    json!({"type": "text", "text": text})
}

impl HistoryFixture for CohereHistory {
    type Wire = Chat;

    fn wire(&self, model: &str) -> Chat {
        Chat {
            provider: CohereConfig::new("key"),
            model: model.to_owned(),
        }
    }

    fn model(&self) -> &'static str {
        "command-a-vision-07-2025"
    }

    fn text_only_model(&self) -> Option<&'static str> {
        Some("command-a-reasoning-08-2025")
    }

    fn other_model(&self) -> &'static str {
        "command-a-03-2025"
    }

    fn body(
        &self,
        wire: &Chat,
        request: CompletionRequest,
        mode: Mode,
    ) -> Result<Value, EncodeError> {
        http_body(&wire.encode(request, mode)?)
    }

    fn reply(&self, shape: Shape, mode: Mode) -> Option<Vec<WireFrame>> {
        let rig = call("{\"q\":\"rig\"}");
        Some(match shape {
            Shape::Rich => reply(
                Some("I will look it up."),
                &[thinking("plan the lookup"), text("looking it up")],
                &[rig],
                "TOOL_CALL",
                mode,
            ),
            Shape::Interleaved => reply(
                None,
                &[thinking("first"), text("between"), thinking("second")],
                &[rig],
                "TOOL_CALL",
                mode,
            ),
            Shape::Unknown => reply(
                None,
                &[
                    json!({"type": "text", "text": "noted", "x_rig_field": true}),
                    json!({"type": "x_rig_invented", "id": "x_1"}),
                ],
                &[],
                "COMPLETE",
                mode,
            ),
        })
    }

    fn call_reply(&self, arguments: &str, mode: Mode) -> Option<Vec<WireFrame>> {
        Some(reply(None, &[], &[call(arguments)], "TOOL_CALL", mode))
    }

    /// Every `finish_reason` Cohere documents, and an invented one.
    fn finishes(&self) -> Vec<(&'static str, Vec<WireFrame>, Ending)> {
        [
            ("COMPLETE", Ending::Success),
            ("STOP_SEQUENCE", Ending::Success),
            ("MAX_TOKENS", Ending::Success),
            ("TOOL_CALL", Ending::Success),
            ("ERROR", Ending::Failure),
            ("ERROR_TOXIC", Ending::Failure),
            ("ERROR_LIMIT", Ending::Failure),
            ("USER_CANCEL", Ending::Failure),
            ("TIMEOUT", Ending::Failure),
            ("X_RIG_INVENTED", Ending::Failure),
        ]
        .into_iter()
        .map(|(finish, ending)| {
            (
                finish,
                reply(None, &[text("done")], &[], finish, Mode::Unary),
                ending,
            )
        })
        .collect()
    }

    fn ablation(&self) -> Option<Ablation<WireFrame>> {
        let mut message = json!({"role": "assistant",
            "content": [thinking("plan the lookup"), text("looking it up")],
            "tool_plan": "I will look it up.", "tool_calls": [call("{\"q\":\"rig\"}")],
            "citations": [{"start": 0, "end": 7, "text": "looking", "type": "TEXT_CONTENT",
                "sources": [{"type": "tool", "id": "call_1:0", "tool_output": {"q": "rig"}}]}]});
        message["role"] = json!("assistant");
        Some(Ablation {
            document: json!({"id": "msg_1", "finish_reason": "TOOL_CALL", "message": message,
                "usage": usage()}),
            required: &[
                "/finish_reason",
                "/message",
                "/message/content/*/text",
                "/message/content/*/thinking",
            ],
            frames: |document| vec![event(document)],
        })
    }

    /// The message the rebuild sends: each content item in order (its own
    /// item while current), the plan as `tool_plan`, and each call's item
    /// with its canonical name and arguments as JSON text.
    fn replayed(&self, turn: &AssistantMessage) -> Vec<Value> {
        let (mut content, mut calls, mut plan) = (Vec::new(), Vec::new(), None);
        for piece in &turn.content {
            let native = piece.native_item();
            match piece {
                AssistantContent::Reasoning(block) => match native {
                    Some(item) if item.get("type").is_some() => content.push(item.clone()),
                    Some(item) => plan = item.get("tool_plan").cloned(),
                    None => content.push(thinking(&block.text)),
                },
                AssistantContent::Text(block) => {
                    content.push(native.cloned().unwrap_or_else(|| text(&block.text)))
                }
                AssistantContent::Opaque(opaque) if opaque.replay => {
                    content.push(opaque.item.clone())
                }
                AssistantContent::ToolCall(call) => {
                    let mut item = native
                        .cloned()
                        .unwrap_or_else(|| json!({"type": "function"}));
                    item["id"] = json!(call.id.wire());
                    item["function"]["name"] = json!(call.function.name.as_str());
                    item["function"]["arguments"] =
                        json!(call.function.arguments_value().to_string());
                    calls.push(item);
                }
                AssistantContent::Image(_) | AssistantContent::Opaque(_) => {}
            }
        }
        let mut message = Map::new();
        message.insert("role".to_owned(), "assistant".into());
        message.insert("content".to_owned(), content.into());
        message.insert("tool_calls".to_owned(), calls.into());
        if let Some(plan) =
            plan.filter(|plan| plan.as_str().is_some_and(|plan| !plan.trim().is_empty()))
        {
            message.insert("tool_plan".to_owned(), plan);
        }
        vec![Value::Object(message)]
    }
}

rig_core::history_conformance_suite! {
    wire: "cohere",
    fixture: CohereHistory,
}
