//! The Chat Completions fixture every Chat dialect's suite shares: a
//! dialect's rich reply as one message and as a stream of its deltas, the
//! invented shapes, every documented finish, and the projection pi's
//! `openai-completions` rebuild sends for a turn.

use rig_core::completion::CompletionRequest;
use rig_core::error::EncodeError;
use rig_core::message::{AssistantContent, AssistantMessage};
use rig_core::providers::openai::wire::{Chat, DEEPSEEK, Dialect, MISTRAL, OpenAIConfig};
use rig_core::test_utils::history_conformance::{
    Ablation, Ending, HistoryFixture, Shape, http_body,
};
use rig_core::wire::{Mode, Wire, WireFrame};
use serde_json::{Map, Value, json};

/// One Chat dialect's side of the suite.
pub struct ChatHistory {
    pub dialect: &'static Dialect,
    pub model: &'static str,
    pub other_model: &'static str,
    pub text_only_model: Option<&'static str>,
    /// The rich reply's message, every block kind the dialect has.
    pub rich: fn() -> Value,
    /// The same message as the deltas a stream sends.
    pub rich_deltas: fn() -> Vec<Value>,
    /// The deltas of a stream whose blocks are `[reasoning, text,
    /// reasoning, call]`, on a dialect with reasoning. One message holds one
    /// reasoning and one text, so only a stream can interleave them.
    pub interleaved: Option<fn() -> Vec<Value>>,
    /// Whether the dialect's messages carry items an invented type and field
    /// can sit on: content parts or tool calls.
    pub has_items: bool,
}

/// The call every rich and invented reply makes; a Mistral id (nine
/// alphanumerics) is legal on every dialect.
pub fn call() -> Value {
    json!({
        "id": "abcDEF123",
        "type": "function",
        "function": {"name": "lookup", "arguments": "{\"q\":\"rig\"}"},
    })
}

/// A stream that reasons, answers, reasons again under `key`, and calls.
pub fn interleaved_under(key: &str) -> Vec<Value> {
    let mut whole = call();
    whole["index"] = json!(0);
    vec![
        json!({"role": "assistant", key: "plan the lookup"}),
        json!({"content": "looking it up"}),
        json!({key: "check the arguments"}),
        json!({"tool_calls": [whole]}),
    ]
}

/// The call's two stream fragments: its opening and its arguments.
pub fn call_deltas() -> Vec<Value> {
    vec![
        json!({"tool_calls": [{"index": 0, "id": "abcDEF123", "type": "function",
            "function": {"name": "lookup", "arguments": ""}}]}),
        json!({"tool_calls": [{"index": 0, "function": {"arguments": "{\"q\":\"rig\"}"}}]}),
    ]
}

fn usage() -> Value {
    json!({"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
        "prompt_tokens_details": {"cached_tokens": 2}})
}

impl ChatHistory {
    fn document(&self, message: Value, finish: &str) -> Value {
        json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": self.model,
            "choices": [{"index": 0, "finish_reason": finish, "message": message}],
            "usage": usage(),
        })
    }

    fn chunk(&self, delta: Value, finish: Option<&str>) -> WireFrame {
        WireFrame::Text(
            json!({
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": self.model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            })
            .to_string(),
        )
    }

    /// `deltas` as a stream that ends with `finish`, its usage and `[DONE]`.
    fn stream(&self, deltas: Vec<Value>, finish: &str) -> Vec<WireFrame> {
        let mut frames: Vec<WireFrame> = deltas
            .into_iter()
            .map(|delta| self.chunk(delta, None))
            .collect();
        frames.push(self.chunk(json!({}), Some(finish)));
        frames.push(WireFrame::Text(
            json!({"id": "chatcmpl-1", "object": "chat.completion.chunk", "model": self.model,
                "choices": [], "usage": usage()})
            .to_string(),
        ));
        frames.push(WireFrame::Text("[DONE]".to_owned()));
        frames
    }

    fn reply_of(&self, message: Value, deltas: Vec<Value>, mode: Mode) -> Vec<WireFrame> {
        let finish = if message.get("tool_calls").is_some() {
            "tool_calls"
        } else {
            "stop"
        };
        match mode {
            Mode::Unary => frames(self.document(message, finish)),
            Mode::Streaming => self.stream(deltas, finish),
        }
    }
}

fn frames(document: Value) -> Vec<WireFrame> {
    vec![WireFrame::Text(document.to_string())]
}

impl HistoryFixture for ChatHistory {
    type Wire = Chat;

    fn wire(&self, model: &str) -> Chat {
        OpenAIConfig::with_key(self.dialect, "key").chat(model)
    }

    fn model(&self) -> &'static str {
        self.model
    }

    fn other_model(&self) -> &'static str {
        self.other_model
    }

    fn text_only_model(&self) -> Option<&'static str> {
        self.text_only_model
    }

    fn body(
        &self,
        wire: &Chat,
        request: CompletionRequest,
        mode: Mode,
    ) -> Result<Value, EncodeError> {
        http_body(&wire.encode(request, mode)?)
    }

    /// A whole message carries its reasoning in one field (or Mistral's
    /// thinking parts) and its text in one content, so `[reasoning, text,
    /// reasoning, call]` has a Chat form only as a stream, where a block
    /// closes when the next one starts.
    fn reply(&self, shape: Shape, mode: Mode) -> Option<Vec<WireFrame>> {
        match shape {
            Shape::Rich => Some(self.reply_of((self.rich)(), (self.rich_deltas)(), mode)),
            Shape::Interleaved => match (self.interleaved, mode) {
                (Some(deltas), Mode::Streaming) => Some(self.stream(deltas(), "tool_calls")),
                _ => None,
            },
            Shape::Unknown if self.has_items => {
                let call = json!({"id": "abcDEF999", "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"}, "x_rig_field": true});
                let message = json!({
                    "role": "assistant",
                    "content": [{"type": "text", "text": "noted"}, {"type": "x_rig_invented", "id": "x_1"}],
                    "tool_calls": [call],
                });
                let mut streamed = call.clone();
                streamed["index"] = json!(0);
                let deltas = vec![
                    json!({"role": "assistant", "content": [{"type": "text", "text": "noted"}]}),
                    json!({"content": [{"type": "x_rig_invented", "id": "x_1"}]}),
                    json!({"tool_calls": [streamed]}),
                ];
                Some(self.reply_of(message, deltas, mode))
            }
            // A dialect whose message is one string content has no item an
            // invented type or field could sit on.
            Shape::Unknown => None,
        }
    }

    fn call_reply(&self, arguments: &str, mode: Mode) -> Option<Vec<WireFrame>> {
        let message = json!({"role": "assistant", "tool_calls": [{"id": "abcDEF123",
            "type": "function", "function": {"name": "lookup", "arguments": arguments}}]});
        let deltas = vec![
            json!({"role": "assistant", "tool_calls": [{"index": 0, "id": "abcDEF123",
                "type": "function", "function": {"name": "lookup", "arguments": ""}}]}),
            json!({"tool_calls": [{"index": 0, "function": {"arguments": arguments}}]}),
        ];
        Some(self.reply_of(message, deltas, mode))
    }

    /// Chat's documented finishes, OpenRouter's `error` and `network_error`
    /// and Mistral's `model_length` among them, and an invented one.
    fn finishes(&self) -> Vec<(&'static str, Vec<WireFrame>, Ending)> {
        [
            ("stop", Ending::Success),
            ("end", Ending::Success),
            ("length", Ending::Success),
            ("model_length", Ending::Success),
            ("tool_calls", Ending::Success),
            ("function_call", Ending::Success),
            ("content_filter", Ending::Failure),
            ("error", Ending::Failure),
            ("network_error", Ending::Failure),
            ("x_rig_invented", Ending::Failure),
        ]
        .into_iter()
        .map(|(finish, ending)| {
            let message = json!({"role": "assistant", "content": "done"});
            (finish, frames(self.document(message, finish)), ending)
        })
        .collect()
    }

    fn ablation(&self) -> Option<Ablation<WireFrame>> {
        let message = (self.rich)();
        let finish = if message.get("tool_calls").is_some() {
            "tool_calls"
        } else {
            "stop"
        };
        // A message whose only block is its text needs its content.
        let required: &'static [&'static str] = if self.has_items {
            &["/choices", "/choices/*/message"]
        } else {
            &[
                "/choices",
                "/choices/*/message",
                "/choices/*/message/content",
            ]
        };
        Some(Ablation {
            document: self.document(message, finish),
            required,
            frames,
        })
    }

    fn replayed(&self, turn: &AssistantMessage) -> Vec<Value> {
        vec![projection(turn, self.dialect)]
    }
}

/// The assistant message pi's rebuild sends for `turn` on `dialect`: the
/// text joined into `content` (a part array when a block is a content
/// part), reasoning under the field its item names, the fields beside it,
/// an opaque item's fields, and each call's item with its canonical name
/// and arguments as JSON text. A text block whose item is message fields
/// (an answer's audio) sends them beside the text. Images are never sent
/// back.
pub fn projection(turn: &AssistantMessage, dialect: &Dialect) -> Value {
    let mut message = Map::new();
    message.insert("role".to_owned(), "assistant".into());
    let (mut text, mut parts, mut has_parts) = (String::new(), Vec::new(), false);
    let mut reasoning: Vec<(String, String)> = Vec::new();
    let (mut fields, mut calls) = (Map::new(), Vec::new());
    for content in &turn.content {
        let native = content.native_item();
        match content {
            AssistantContent::Text(block) => {
                if !block.text.trim().is_empty() {
                    text.push_str(&block.text);
                    parts.push(json!({"type": "text", "text": block.text}));
                }
                if let Some(Value::Object(item)) = native.filter(|item| item.get("type").is_none())
                {
                    fields.extend(item.clone());
                }
            }
            AssistantContent::Reasoning(block) => match native {
                Some(item) if item.get("type").is_some() => {
                    has_parts = true;
                    parts.push(item.clone());
                }
                Some(Value::Object(item)) => {
                    let mut field = None;
                    for (key, value) in item {
                        if value.is_string() {
                            field.get_or_insert(key.clone());
                        } else {
                            fields.insert(key.clone(), value.clone());
                        }
                    }
                    if let Some(field) = field.filter(|_| !block.text.trim().is_empty()) {
                        match reasoning.iter_mut().find(|(name, _)| *name == field) {
                            Some((_, joined)) => {
                                joined.push('\n');
                                joined.push_str(&block.text);
                            }
                            None => reasoning.push((field, block.text.clone())),
                        }
                    }
                }
                _ => {}
            },
            AssistantContent::Opaque(opaque) if opaque.replay => {
                if opaque.item.get("type").is_some() {
                    has_parts = true;
                    parts.push(opaque.item.clone());
                } else if let Value::Object(item) = &opaque.item {
                    fields.extend(item.clone());
                }
            }
            AssistantContent::ToolCall(call) => {
                let mut item = native
                    .cloned()
                    .unwrap_or_else(|| json!({"type": "function"}));
                item["id"] = json!(call.id.wire());
                item["function"]["name"] = json!(call.function.name.as_str());
                item["function"]["arguments"] = json!(call.function.arguments_value().to_string());
                calls.push(item);
            }
            AssistantContent::Image(_) | AssistantContent::Opaque(_) => {}
        }
    }
    if has_parts {
        message.insert("content".to_owned(), parts.into());
    } else if !text.is_empty() {
        message.insert("content".to_owned(), text.into());
    }
    for (field, text) in reasoning {
        message.insert(field, text.into());
    }
    message.extend(fields);
    if !calls.is_empty() {
        message.insert("tool_calls".to_owned(), calls.into());
    }
    // DeepSeek takes `reasoning_content` and `content` on every assistant
    // turn, Mistral `content`.
    if dialect.name == DEEPSEEK.name {
        message
            .entry("reasoning_content")
            .or_insert_with(|| json!(""));
    }
    if dialect.name == DEEPSEEK.name || dialect.name == MISTRAL.name {
        message.entry("content").or_insert_with(|| json!(""));
    }
    Value::Object(message)
}
