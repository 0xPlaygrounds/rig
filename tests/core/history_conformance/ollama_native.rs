//! Ollama's native `/api/chat` wire: a whole reply or NDJSON records, each
//! with `thinking`, `content` and whole `tool_calls`, and the message its
//! rebuild sends for a turn.

use rig_core::completion::CompletionRequest;
use rig_core::error::EncodeError;
use rig_core::message::{AssistantContent, AssistantMessage};
use rig_core::providers::ollama::{Chat, OllamaConfig};
use rig_core::wire::{Mode, Wire, WireFrame};
use rig_history_conformance::{
    Ablation, CallShape, Ending, HistoryFixture, Shape, decode, http_body,
};
use serde_json::{Map, Value, json};

const MODEL: &str = "qwen3:4b";

pub struct OllamaNativeHistory;

fn call() -> Value {
    json!({"id": "call_1", "function": {"index": 0, "name": "lookup", "arguments": {"q": "rig"}}})
}

/// One record carrying `message`, ending the reply on `done` when set.
fn record(message: Value, done: Option<&str>) -> Value {
    let mut record = json!({
        "model": MODEL,
        "created_at": "2026-10-05T00:00:00Z",
        "message": message,
        "done": done.is_some(),
    });
    if let (Some(reason), Some(fields)) = (done, record.as_object_mut()) {
        fields.insert("done_reason".to_owned(), json!(reason));
        fields.insert("prompt_eval_count".to_owned(), json!(10));
        fields.insert("eval_count".to_owned(), json!(5));
        fields.insert("total_duration".to_owned(), json!(100));
    }
    record
}

fn assistant(fields: Value) -> Value {
    let mut message = json!({"role": "assistant", "content": ""});
    if let (Some(message), Value::Object(fields)) = (message.as_object_mut(), fields) {
        message.extend(fields);
    }
    message
}

fn frames(records: impl IntoIterator<Item = Value>) -> Vec<WireFrame> {
    records
        .into_iter()
        .map(|record| WireFrame::Text(record.to_string()))
        .collect()
}

/// A whole reply of `message`, or the same message as one record per part
/// and a `done` record.
fn reply_of(whole: Value, parts: Vec<Value>, mode: Mode) -> Vec<WireFrame> {
    match mode {
        Mode::Unary => frames([record(assistant(whole), Some("stop"))]),
        Mode::Streaming => frames(
            parts
                .into_iter()
                .map(|part| record(assistant(part), None))
                .chain([record(assistant(json!({})), Some("stop"))]),
        ),
    }
}

impl HistoryFixture for OllamaNativeHistory {
    type Wire = Chat;

    fn wire(&self, model: &str) -> Chat {
        Chat::new(OllamaConfig::new(), model)
    }

    fn model(&self) -> &'static str {
        MODEL
    }

    fn other_model(&self) -> &'static str {
        "llama3.2"
    }

    fn raw_tool(&self, tool: &rig_core::completion::ToolDefinition) -> Option<Value> {
        Some(json!({"type": "function", "function": {
            "name": tool.name.as_str(), "description": tool.description, "parameters": tool.parameters,
        }}))
    }

    fn body(
        &self,
        wire: &Chat,
        request: CompletionRequest,
        mode: Mode,
    ) -> Result<Value, EncodeError> {
        http_body(&wire.encode(request, mode)?)
    }

    /// A whole message holds one `thinking` and one `content`, so the
    /// interleaved shape exists only as a stream; a message has no item an
    /// invented type could sit on.
    fn reply(&self, shape: Shape, mode: Mode) -> Option<Vec<WireFrame>> {
        match shape {
            Shape::Rich => Some(reply_of(
                json!({"thinking": "plan the lookup", "content": "looking it up", "tool_calls": [call()]}),
                vec![
                    json!({"thinking": "plan "}),
                    json!({"thinking": "the lookup"}),
                    json!({"content": "looking "}),
                    json!({"content": "it up"}),
                    json!({"tool_calls": [call()]}),
                ],
                mode,
            )),
            Shape::Interleaved if mode == Mode::Streaming => Some(reply_of(
                Value::Null,
                vec![
                    json!({"thinking": "first"}),
                    json!({"content": "between"}),
                    json!({"thinking": "second"}),
                    json!({"tool_calls": [call()]}),
                ],
                mode,
            )),
            Shape::Interleaved | Shape::Unknown => None,
        }
    }

    /// Ollama sends arguments as an object, so only text that is a JSON
    /// object can arrive.
    fn call_reply(&self, arguments: &str, mode: Mode) -> Option<Vec<WireFrame>> {
        let arguments: Value = serde_json::from_str(arguments).ok()?;
        let calls = json!({"tool_calls": [
            {"id": "call_1", "function": {"name": "lookup", "arguments": arguments}}
        ]});
        Some(reply_of(calls.clone(), vec![calls], mode))
    }

    /// `stop` and `length`, the operational `load` and `unload`, and an
    /// invented reason.
    fn finishes(&self) -> Vec<(&'static str, Vec<WireFrame>, Ending)> {
        [
            ("stop", Ending::Success),
            ("length", Ending::Success),
            ("load", Ending::Failure),
            ("unload", Ending::Failure),
            ("x_rig_invented", Ending::Failure),
        ]
        .into_iter()
        .map(|(reason, ending)| {
            let reply = frames([record(assistant(json!({"content": "done"})), Some(reason))]);
            (reason, reply, ending)
        })
        .collect()
    }

    fn ablation(&self) -> Option<Ablation<WireFrame>> {
        Some(Ablation {
            document: record(
                assistant(json!({"thinking": "plan", "content": "text", "tool_calls": [call()]})),
                Some("stop"),
            ),
            required: &["/done", "/done_reason"],
            frames: |document| frames([document]),
        })
    }

    fn finish_reason_pointer(&self) -> Option<&'static str> {
        Some("/done_reason")
    }

    fn replayed(&self, turn: &AssistantMessage) -> Vec<Value> {
        vec![projection(turn)]
    }

    /// A reasoning item is its `thinking`, and a call item a `tool_calls`
    /// entry; a text block keeps no item.
    fn decode_item(&self, block: &AssistantContent) -> Option<AssistantContent> {
        let item = block.native_item()?.clone();
        let message = match block {
            AssistantContent::Reasoning(_) => assistant(item),
            AssistantContent::ToolCall(_) => assistant(json!({"tool_calls": [item]})),
            _ => return None,
        };
        let wire = self.wire(MODEL);
        let response = decode(
            &wire,
            &CompletionRequest::new("restate"),
            Mode::Unary,
            frames([record(message, Some("stop"))]),
        )
        .ok()?;
        response.choice.into_iter().next()
    }

    /// Each call arrives whole, so every shape is one record per call in a
    /// stream, or one list in a whole reply; `index` never tells them apart.
    fn calls_reply(&self, shape: CallShape, mode: Mode) -> Option<Vec<WireFrame>> {
        let call = |id: &str, city: &str| {
            let mut function = json!({"name": "weather", "arguments": {"city": city}});
            match shape {
                CallShape::Indexless => {}
                CallShape::NullIndex => function["index"] = Value::Null,
                CallShape::ReusedIndex | CallShape::WholeList => function["index"] = json!(0),
            }
            json!({"id": id, "function": function})
        };
        let (paris, rome) = (call("a1", "Paris"), call("b2", "Rome"));
        match (shape, mode) {
            (CallShape::WholeList, Mode::Streaming) => None,
            _ => Some(reply_of(
                json!({"tool_calls": [paris.clone(), rome.clone()]}),
                vec![
                    json!({"tool_calls": [paris]}),
                    json!({"tool_calls": [rome]}),
                ],
                mode,
            )),
        }
    }

    fn empty_reply(&self, mode: Mode) -> Option<Vec<WireFrame>> {
        Some(reply_of(json!({}), Vec::new(), mode))
    }

    fn error_frame(&self) -> Option<WireFrame> {
        Some(WireFrame::Text(json!({"error": "overloaded"}).to_string()))
    }
}

/// The message the encoder sends for `turn`: its text joined as `content`,
/// its reasoning joined as `thinking`, and each call's item with its
/// canonical name and arguments.
pub fn projection(turn: &AssistantMessage) -> Value {
    let (mut text, mut thinking, mut calls) = (String::new(), Vec::new(), Vec::new());
    for block in &turn.content {
        match block {
            AssistantContent::Text(block) => text.push_str(&block.text),
            AssistantContent::Reasoning(block) if !block.text.is_empty() => {
                thinking.push(block.text.clone());
            }
            AssistantContent::ToolCall(call) => {
                let mut item = match block.native_item() {
                    Some(Value::Object(item)) => item.clone(),
                    _ => Map::new(),
                };
                item.insert("id".to_owned(), json!(call.id.wire()));
                let function = item.entry("function").or_insert_with(|| json!({}));
                function["name"] = json!(call.function.name.as_str());
                function["arguments"] = Value::Object(call.function.arguments.clone());
                calls.push(Value::Object(item));
            }
            _ => {}
        }
    }
    let mut message = json!({"role": "assistant", "content": text});
    if !thinking.is_empty() {
        message["thinking"] = json!(thinking.join("\n"));
    }
    if !calls.is_empty() {
        message["tool_calls"] = Value::Array(calls);
    }
    message
}

rig_history_conformance::history_conformance_suite! {
    wire: "ollama_native",
    fixture: OllamaNativeHistory,
}
