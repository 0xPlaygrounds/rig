//! Cohere's native chat wire's history suite: a whole reply is one
//! `/v2/chat` document, and a stream is its events. Content parts, the tool
//! plan and each call are blocks whose items hold the citations that point
//! at them.

use rig_core::completion::CompletionRequest;
use rig_core::error::EncodeError;
use rig_core::message::{AssistantContent, AssistantMessage};
use rig_core::providers::cohere::{
    COMMAND_A_03_2025, COMMAND_A_REASONING_08_2025, COMMAND_A_VISION_07_2025, CohereConfig,
    NativeChat,
};
use rig_core::wire::{Mode, Wire, WireFrame};
use rig_history_conformance::{
    Ablation, CallShape, Ending, HistoryFixture, Shape, decode, http_body,
};
use serde_json::{Map, Value, json};

pub struct CohereNativeHistory;

fn call(id: &str, name: &str, arguments: &str) -> Value {
    json!({"id": id, "type": "function", "function": {"name": name, "arguments": arguments}})
}

fn citation(content_index: Option<usize>, kind: &str) -> Value {
    let mut citation = json!({"start": 0, "end": 4, "text": "rig!", "type": kind,
        "sources": [{"type": "document", "id": "doc_0", "document": {"id": "doc_0", "text": "rig"}}]});
    if let Some(index) = content_index {
        citation["content_index"] = json!(index);
    }
    citation
}

/// The whole reply holding `message`, ending on `finish`.
fn document(message: Value, finish: &str) -> Value {
    json!({"id": "resp_1", "message": message, "finish_reason": finish,
        "usage": {"tokens": {"input_tokens": 3, "output_tokens": 5}}})
}

fn frames(values: impl IntoIterator<Item = Value>) -> Vec<WireFrame> {
    values
        .into_iter()
        .map(|value| WireFrame::Text(value.to_string()))
        .collect()
}

/// `message` as Cohere streams it: the tool plan first, then each content
/// part with its citations, then each call, its arguments in two halves.
fn stream(message: &Value, finish: &str) -> Vec<WireFrame> {
    let mut events = vec![json!({"id": "resp_1", "type": "message-start",
        "delta": {"message": {"role": "assistant"}}})];
    let citations = message["citations"].as_array().cloned().unwrap_or_default();
    // Each citation streams inside the block it cites.
    let cite = |events: &mut Vec<Value>, cites: &dyn Fn(&Value) -> bool| {
        for (at, citation) in citations.iter().enumerate().filter(|(_, c)| cites(c)) {
            events.push(json!({"type": "citation-start", "index": at,
                "delta": {"message": {"citations": citation}}}));
            events.push(json!({"type": "citation-end", "index": at}));
        }
    };
    if let Some(plan) = message["tool_plan"].as_str() {
        events.push(json!({"type": "tool-plan-delta", "delta": {"message": {"tool_plan": plan}}}));
        cite(&mut events, &|citation| citation["type"] == "PLAN");
    }
    let parts = message["content"].as_array().cloned().unwrap_or_default();
    for (index, part) in parts.iter().enumerate() {
        let key = if part["type"] == "thinking" {
            "thinking"
        } else {
            "text"
        };
        let mut start = part.clone();
        let text = part[key].as_str().map(str::to_owned);
        if text.is_some() {
            start[key] = json!("");
        }
        events.push(json!({"type": "content-start", "index": index,
            "delta": {"message": {"content": start}}}));
        if let Some(text) = text {
            events.push(json!({"type": "content-delta", "index": index,
                "delta": {"message": {"content": {key: text}}}}));
        }
        cite(&mut events, &|citation| {
            citation["type"] != "PLAN"
                && citation["content_index"].as_u64().unwrap_or(0) == index as u64
        });
        events.push(json!({"type": "content-end", "index": index}));
    }
    let calls = message["tool_calls"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    for (index, call) in calls.iter().enumerate() {
        let mut start = call.clone();
        let arguments = call["function"]["arguments"].as_str().unwrap_or_default();
        start["function"]["arguments"] = json!("");
        events.push(json!({"type": "tool-call-start", "index": index,
            "delta": {"message": {"tool_calls": start}}}));
        let middle = arguments
            .char_indices()
            .nth(arguments.chars().count() / 2)
            .map_or(arguments.len(), |(at, _)| at);
        for half in [&arguments[..middle], &arguments[middle..]] {
            events.push(json!({"type": "tool-call-delta", "index": index,
                "delta": {"message": {"tool_calls": {"function": {"arguments": half}}}}}));
        }
        events.push(json!({"type": "tool-call-end", "index": index}));
    }
    events.push(
        json!({"type": "message-end", "delta": {"finish_reason": finish,
        "usage": {"tokens": {"input_tokens": 3, "output_tokens": 5}}}}),
    );
    frames(events)
}

fn reply_of(message: Value, finish: &str, mode: Mode) -> Vec<WireFrame> {
    match mode {
        Mode::Unary => frames([document(message, finish)]),
        Mode::Streaming => stream(&message, finish),
    }
}

fn rich() -> Value {
    json!({
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "plan the lookup"},
            {"type": "text", "text": "looking it up"},
        ],
        "tool_plan": "I will look it up.",
        "tool_calls": [call("lookup_1", "lookup", "{\"q\":\"rig\"}")],
        "citations": [citation(Some(1), "TEXT_CONTENT")],
    })
}

impl HistoryFixture for CohereNativeHistory {
    type Wire = NativeChat;

    fn wire(&self, model: &str) -> NativeChat {
        NativeChat::new(CohereConfig::new("key"), model)
    }

    fn raw_tool(&self, tool: &rig_core::completion::ToolDefinition) -> Option<Value> {
        Some(json!({"type": "function", "function": {
            "name": tool.name.as_str(), "description": tool.description,
            "parameters": tool.parameters,
        }}))
    }

    fn model(&self) -> &'static str {
        COMMAND_A_VISION_07_2025
    }

    fn other_model(&self) -> &'static str {
        COMMAND_A_REASONING_08_2025
    }

    fn text_only_model(&self) -> Option<&'static str> {
        Some(COMMAND_A_03_2025)
    }

    fn body(
        &self,
        wire: &NativeChat,
        request: CompletionRequest,
        mode: Mode,
    ) -> Result<Value, EncodeError> {
        http_body(&wire.encode(request, mode)?)
    }

    fn reply(&self, shape: Shape, mode: Mode) -> Option<Vec<WireFrame>> {
        let message = match shape {
            Shape::Rich => rich(),
            Shape::Interleaved => json!({
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "first"},
                    {"type": "text", "text": "between"},
                    {"type": "thinking", "thinking": "second"},
                ],
                "tool_calls": [call("lookup_1", "lookup", "{\"q\":\"rig\"}")],
            }),
            Shape::Unknown => json!({
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "noted", "x_rig_field": true},
                    {"type": "x_rig_invented", "id": "x_1"},
                ],
            }),
        };
        let finish = if message.get("tool_calls").is_some() {
            "TOOL_CALL"
        } else {
            "COMPLETE"
        };
        Some(reply_of(message, finish, mode))
    }

    fn call_reply(&self, arguments: &str, mode: Mode) -> Option<Vec<WireFrame>> {
        let message = json!({"role": "assistant",
            "tool_calls": [call("lookup_1", "lookup", arguments)]});
        Some(reply_of(message, "TOOL_CALL", mode))
    }

    /// Cohere's documented finishes, and an invented one.
    fn finishes(&self) -> Vec<(&'static str, Vec<WireFrame>, Ending)> {
        [
            ("COMPLETE", Ending::Success),
            ("STOP_SEQUENCE", Ending::Success),
            ("MAX_TOKENS", Ending::Success),
            ("TOOL_CALL", Ending::Success),
            ("ERROR", Ending::Failure),
            ("TIMEOUT", Ending::Failure),
            ("X_RIG_INVENTED", Ending::Failure),
        ]
        .into_iter()
        .map(|(finish, ending)| {
            let message = json!({"role": "assistant",
                "content": [{"type": "text", "text": "done"}]});
            (finish, frames([document(message, finish)]), ending)
        })
        .collect()
    }

    fn ablation(&self) -> Option<Ablation<WireFrame>> {
        Some(Ablation {
            document: document(rich(), "TOOL_CALL"),
            required: &["/message", "/finish_reason"],
            frames: |document| frames([document]),
        })
    }

    fn replayed(&self, turn: &AssistantMessage) -> Vec<Value> {
        vec![projection(turn)]
    }

    /// The block's item as the one item of a whole reply: a content part,
    /// the tool plan or a call.
    fn decode_item(&self, block: &AssistantContent) -> Option<AssistantContent> {
        let item = block.native_item()?.clone();
        let message = match (block, item["type"].as_str()) {
            (AssistantContent::Reasoning(_), Some("tool_plan")) => {
                json!({"role": "assistant", "tool_plan": item["tool_plan"],
                    "citations": item.get("citations").cloned().unwrap_or_else(|| json!([]))})
            }
            (AssistantContent::ToolCall(_), _) => {
                json!({"role": "assistant", "tool_calls": [item]})
            }
            (AssistantContent::Text(_) | AssistantContent::Reasoning(_), _) => {
                let mut part = item.clone();
                let citations = part
                    .as_object_mut()
                    .and_then(|part| part.shift_remove("citations"))
                    .unwrap_or_else(|| json!([]));
                json!({"role": "assistant", "content": [part], "citations": citations})
            }
            (AssistantContent::Image(_) | AssistantContent::Opaque(_), _) => return None,
        };
        let wire = self.wire(self.model());
        let response = decode(
            &wire,
            &CompletionRequest::new("restate"),
            Mode::Unary,
            frames([document(message, "COMPLETE")]),
        )
        .ok()?;
        response.choice.into_iter().next()
    }

    /// Cohere names each streamed call by its index, so only a whole reply
    /// lists two calls.
    fn calls_reply(&self, shape: CallShape, mode: Mode) -> Option<Vec<WireFrame>> {
        match (shape, mode) {
            (CallShape::WholeList, _) => Some(reply_of(
                json!({"role": "assistant", "tool_calls": [
                    call("a1", "weather", "{\"city\":\"Paris\"}"),
                    call("b2", "weather", "{\"city\":\"Rome\"}"),
                ]}),
                "TOOL_CALL",
                mode,
            )),
            _ => None,
        }
    }

    fn empty_reply(&self, mode: Mode) -> Option<Vec<WireFrame>> {
        Some(reply_of(json!({"role": "assistant"}), "COMPLETE", mode))
    }

    fn finish_reason_pointer(&self) -> Option<&'static str> {
        Some("/finish_reason")
    }

    fn error_frame(&self) -> Option<WireFrame> {
        Some(WireFrame::Text(
            json!({"type": "message-end", "delta": {"finish_reason": "ERROR", "error": "overloaded"}})
                .to_string(),
        ))
    }
}

/// What the native encoder sends for `turn`: each current text, thinking
/// or unknown part as its item less its citations (or rebuilt from its
/// text), the tool plan, each call with its canonical name and arguments
/// as JSON text, and every citation pointed at the part it cites.
pub fn projection(turn: &AssistantMessage) -> Value {
    let (mut content, mut plan, mut calls, mut citations) =
        (Vec::new(), String::new(), Vec::new(), Vec::new());
    for block in &turn.content {
        let mut item = match block.native_item() {
            Some(Value::Object(item)) => item.clone(),
            _ => Map::new(),
        };
        let cited = match item.shift_remove("citations") {
            Some(Value::Array(cited)) => cited,
            _ => Vec::new(),
        };
        let mut point = |at: Option<usize>| {
            citations.extend(cited.iter().cloned().map(|mut citation| {
                if let Some(at) = at {
                    citation["content_index"] = json!(at);
                }
                citation
            }));
        };
        match block {
            AssistantContent::Reasoning(reasoning)
                if item.get("type") == Some(&json!("tool_plan")) =>
            {
                point(None);
                plan.push_str(&reasoning.text);
            }
            AssistantContent::Text(text) if !text.text.is_empty() => {
                point(Some(content.len()));
                content.push(if item.is_empty() {
                    json!({"type": "text", "text": text.text})
                } else {
                    Value::Object(item)
                });
            }
            AssistantContent::Reasoning(reasoning) if !reasoning.text.is_empty() => {
                point(Some(content.len()));
                content.push(if item.is_empty() {
                    json!({"type": "thinking", "thinking": reasoning.text})
                } else {
                    Value::Object(item)
                });
            }
            AssistantContent::Opaque(opaque)
                if opaque.replay && opaque.item.get("type").is_some() =>
            {
                content.push(opaque.item.clone());
            }
            AssistantContent::ToolCall(call) => {
                let mut item = Value::Object(item);
                item["type"] = item
                    .get("type")
                    .cloned()
                    .unwrap_or_else(|| json!("function"));
                item["id"] = json!(call.id.wire());
                item["function"]["name"] = json!(call.function.name.as_str());
                item["function"]["arguments"] = json!(call.function.arguments_value().to_string());
                calls.push(item);
            }
            _ => {}
        }
    }
    let mut message = Map::new();
    message.insert("role".to_owned(), json!("assistant"));
    if !content.is_empty() {
        message.insert("content".to_owned(), content.into());
    }
    if !plan.is_empty() {
        message.insert("tool_plan".to_owned(), plan.into());
    }
    if !calls.is_empty() {
        message.insert("tool_calls".to_owned(), calls.into());
    }
    if !citations.is_empty() {
        message.insert("citations".to_owned(), citations.into());
    }
    Value::Object(message)
}

rig_history_conformance::history_conformance_suite! {
    wire: "cohere_native",
    fixture: CohereNativeHistory,
}
