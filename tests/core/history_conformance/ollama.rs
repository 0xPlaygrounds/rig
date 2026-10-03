//! Ollama's `/api/chat`: thinking, text and whole calls with object
//! arguments, one NDJSON record per delta. The rebuild joins every thinking
//! block into `thinking` and every text block into `content`.

use rig_core::completion::CompletionRequest;
use rig_core::error::EncodeError;
use rig_core::message::{AssistantContent, AssistantMessage};
use rig_core::providers::ollama::{Chat, OllamaConfig};
use rig_core::wire::{Mode, Wire, WireFrame};
use rig_history_conformance::{Ablation, Ending, HistoryFixture, Shape, http_body};
use serde_json::{Map, Value, json};

pub struct OllamaHistory;

const MODEL: &str = "qwen3:4b";

fn record(message: Value, done: Option<&str>) -> Value {
    let mut record = json!({"model": MODEL, "created_at": "1970-01-01T00:00:00Z",
        "message": message, "done": done.is_some()});
    if let Some(reason) = done {
        record["done_reason"] = json!(reason);
        record["prompt_eval_count"] = json!(10);
        record["eval_count"] = json!(5);
        record["total_duration"] = json!(100);
    }
    record
}

fn line(record: Value) -> WireFrame {
    WireFrame::Text(record.to_string())
}

fn call(extra: Value) -> Value {
    let mut call = json!({"id": "call_1", "function": {"index": 0, "name": "lookup",
        "arguments": {"q": "rig"}}});
    if let (Some(call), Value::Object(extra)) = (call.as_object_mut(), extra) {
        call.extend(extra);
    }
    call
}

/// A whole reply of `message`, or the stream of `deltas` that restates it.
fn reply(message: Value, deltas: Vec<Value>, mode: Mode) -> Vec<WireFrame> {
    match mode {
        Mode::Unary => vec![line(record(message, Some("stop")))],
        Mode::Streaming => {
            let mut frames: Vec<WireFrame> = deltas
                .into_iter()
                .map(|delta| line(record(delta, None)))
                .collect();
            frames.push(line(record(
                json!({"role": "assistant", "content": ""}),
                Some("stop"),
            )));
            frames
        }
    }
}

impl HistoryFixture for OllamaHistory {
    type Wire = Chat;

    fn wire(&self, model: &str) -> Chat {
        Chat {
            provider: OllamaConfig::new(),
            model: model.to_owned(),
        }
    }

    fn model(&self) -> &'static str {
        MODEL
    }

    fn other_model(&self) -> &'static str {
        "llama3.2"
    }

    fn body(
        &self,
        wire: &Chat,
        request: CompletionRequest,
        mode: Mode,
    ) -> Result<Value, EncodeError> {
        http_body(&wire.encode(request, mode)?)
    }

    /// Ollama's message has one `thinking` and one `content`, so
    /// `[reasoning, text, reasoning, call]` has only a streamed form.
    fn reply(&self, shape: Shape, mode: Mode) -> Option<Vec<WireFrame>> {
        match shape {
            Shape::Rich => Some(reply(
                json!({"role": "assistant", "thinking": "plan the lookup",
                    "content": "looking it up", "tool_calls": [call(json!({}))]}),
                vec![
                    json!({"role": "assistant", "thinking": "plan ", "content": ""}),
                    json!({"role": "assistant", "thinking": "the lookup", "content": ""}),
                    json!({"role": "assistant", "content": "looking it up"}),
                    json!({"role": "assistant", "content": "", "tool_calls": [call(json!({}))]}),
                ],
                mode,
            )),
            Shape::Interleaved if mode == Mode::Streaming => Some(reply(
                Value::Null,
                vec![
                    json!({"role": "assistant", "thinking": "first", "content": ""}),
                    json!({"role": "assistant", "content": "between"}),
                    json!({"role": "assistant", "thinking": "second", "content": ""}),
                    json!({"role": "assistant", "content": "", "tool_calls": [call(json!({}))]}),
                ],
                mode,
            )),
            Shape::Interleaved => None,
            Shape::Unknown => {
                let invented = call(json!({"type": "x_rig_invented", "x_rig_field": true}));
                Some(reply(
                    json!({"role": "assistant", "content": "noted", "tool_calls": [invented]}),
                    vec![
                        json!({"role": "assistant", "content": "noted"}),
                        json!({"role": "assistant", "content": "", "tool_calls": [invented]}),
                    ],
                    mode,
                ))
            }
        }
    }

    /// Ollama sends arguments as JSON, so the text is the arguments value
    /// when it parses, and a string otherwise.
    fn call_reply(&self, arguments: &str, mode: Mode) -> Option<Vec<WireFrame>> {
        let arguments: Value =
            serde_json::from_str(arguments).unwrap_or_else(|_| Value::String(arguments.to_owned()));
        let call = json!({"function": {"name": "lookup", "arguments": arguments}});
        Some(reply(
            json!({"role": "assistant", "content": "", "tool_calls": [call]}),
            vec![json!({"role": "assistant", "content": "", "tool_calls": [call]})],
            mode,
        ))
    }

    /// Ollama's documented `done_reason`s; `load` and `unload` end a request
    /// that only loaded or unloaded the model.
    fn finishes(&self) -> Vec<(&'static str, Vec<WireFrame>, Ending)> {
        [
            ("stop", Ending::Success),
            ("length", Ending::Success),
            ("load", Ending::Success),
            ("unload", Ending::Success),
            ("x_rig_invented", Ending::Failure),
        ]
        .into_iter()
        .map(|(reason, ending)| {
            let message = json!({"role": "assistant", "content": "done"});
            (reason, vec![line(record(message, Some(reason)))], ending)
        })
        .collect()
    }

    fn ablation(&self) -> Option<Ablation<WireFrame>> {
        Some(Ablation {
            document: record(
                json!({"role": "assistant", "thinking": "plan the lookup",
                    "content": "looking it up", "tool_calls": [call(json!({}))]}),
                Some("stop"),
            ),
            required: &["/message", "/done", "/message/content"],
            frames: |document| vec![line(document)],
        })
    }

    /// The message the rebuild sends: text joined into `content`, reasoning
    /// joined with `"\n"` into `thinking`, and each call's item with its
    /// canonical name and object arguments.
    fn replayed(&self, turn: &AssistantMessage) -> Vec<Value> {
        let mut message = Map::new();
        message.insert("role".to_owned(), "assistant".into());
        let (mut text, mut thinking, mut calls) = (String::new(), Vec::new(), Vec::new());
        for content in &turn.content {
            let native = content.native_item();
            match content {
                AssistantContent::Text(block) if !block.text.trim().is_empty() => {
                    text.push_str(&block.text)
                }
                AssistantContent::Reasoning(block) if !block.text.trim().is_empty() => {
                    thinking.push(block.text.clone())
                }
                AssistantContent::ToolCall(call) => {
                    let mut item = native
                        .cloned()
                        .unwrap_or_else(|| json!({"type": "function"}));
                    if let Some(id) = call.id.provider() {
                        item["id"] = json!(id.as_str());
                    }
                    item["function"]["name"] = json!(call.function.name.as_str());
                    item["function"]["arguments"] = call.function.arguments_value();
                    calls.push(item);
                }
                _ => {}
            }
        }
        message.insert("content".to_owned(), text.into());
        if !thinking.is_empty() {
            message.insert("thinking".to_owned(), thinking.join("\n").into());
        }
        if !calls.is_empty() {
            message.insert("tool_calls".to_owned(), calls.into());
        }
        vec![Value::Object(message)]
    }
}

rig_history_conformance::history_conformance_suite! {
    wire: "ollama",
    fixture: OllamaHistory,
}
