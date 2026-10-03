//! The Anthropic Messages wire's history suite, and the fixture its
//! dialects share: replies are built as the whole message a unary call
//! answers and restated as the event stream a streamed call reads.

use rig_core::completion::CompletionRequest;
use rig_core::error::EncodeError;
use rig_core::message::{AssistantContent, Message};
use rig_core::providers::anthropic::completion::{CLAUDE_HAIKU_4_5, CLAUDE_SONNET_4_6};
use rig_core::providers::anthropic::{ANTHROPIC, AnthropicConfig, Dialect, Messages};
use rig_core::wire::{Mode, Wire, WireFrame};
use rig_history_conformance::{
    Ablation, CallShape, Ending, HistoryFixture, Shape, decode, http_body,
};
use serde_json::{Value, json};

/// A Messages-format dialect's side of the suite.
pub struct MessagesHistory {
    pub dialect: &'static Dialect,
    pub model: &'static str,
    pub other_model: &'static str,
    pub text_only_model: Option<&'static str>,
    /// The signature the dialect sends on thinking: Anthropic signs it,
    /// Kimi sends it empty (#1315).
    pub signature: &'static str,
    /// Whether the dialect has redacted thinking and server tools.
    pub hosted: bool,
}

pub const ANTHROPIC_HISTORY: MessagesHistory = MessagesHistory {
    dialect: &ANTHROPIC,
    model: CLAUDE_SONNET_4_6,
    other_model: CLAUDE_HAIKU_4_5,
    text_only_model: None,
    signature: "sig_1",
    hosted: true,
};

impl MessagesHistory {
    fn thinking(&self, text: &str) -> Value {
        json!({"type": "thinking", "thinking": text, "signature": self.signature})
    }

    fn call(&self) -> Value {
        json!({"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {"q": "rig"}})
    }

    /// The content blocks of `shape`, and the reason it stops.
    fn content(&self, shape: Shape) -> (Vec<Value>, &'static str) {
        match shape {
            Shape::Rich => {
                let mut content = vec![self.thinking("plan the lookup")];
                if self.hosted {
                    content.extend([
                        json!({"type": "redacted_thinking", "data": "opaque-payload"}),
                        json!({"type": "server_tool_use", "id": "srvtoolu_1",
                            "name": "web_search", "input": {"query": "rig"}}),
                        json!({"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1",
                            "content": [{"type": "web_search_result", "url": "https://rig.rs",
                                "title": "Rig", "encrypted_content": "enc"}]}),
                    ]);
                }
                content.extend([
                    json!({"type": "text", "text": "looking it up"}),
                    self.call(),
                ]);
                (content, "tool_use")
            }
            Shape::Interleaved => (
                vec![
                    self.thinking("first"),
                    json!({"type": "text", "text": "between"}),
                    self.thinking("second"),
                    self.call(),
                ],
                "tool_use",
            ),
            Shape::Unknown => (
                vec![
                    json!({"type": "x_rig_invented", "id": "x_1", "payload": {"n": 1}}),
                    json!({"type": "text", "text": "noted", "x_rig_field": true}),
                ],
                "end_turn",
            ),
        }
    }

    /// The whole message a unary call answers.
    fn message(&self, content: Vec<Value>, stop_reason: &str) -> Value {
        json!({
            "type": "message", "id": "msg_1", "role": "assistant", "model": self.model,
            "content": content, "stop_reason": stop_reason, "stop_sequence": null,
            "usage": {"input_tokens": 3, "output_tokens": 5}
        })
    }

    fn whole(&self, content: Vec<Value>, stop_reason: &str) -> Vec<WireFrame> {
        vec![WireFrame::Text(
            self.message(content, stop_reason).to_string(),
        )]
    }

    /// `content` as Anthropic streams it: each block opens with its text
    /// and input empty, then streams them as deltas.
    fn stream(&self, content: Vec<Value>, stop_reason: &str) -> Vec<WireFrame> {
        let mut events =
            vec![json!({"type": "message_start", "message": self.message(vec![], "")})];
        if let Some(message) = events[0]["message"].as_object_mut() {
            message.insert("stop_reason".to_owned(), Value::Null);
        }
        for (index, block) in content.into_iter().enumerate() {
            let mut start = block.clone();
            let mut deltas = Vec::new();
            for (key, delta) in [
                ("text", "text_delta"),
                ("thinking", "thinking_delta"),
                ("signature", "signature_delta"),
            ] {
                if let Some(text) = block.get(key).and_then(Value::as_str) {
                    start[key] = json!("");
                    if !text.is_empty() {
                        let mut fields = serde_json::Map::new();
                        fields.insert("type".to_owned(), json!(delta));
                        fields.insert(key.to_owned(), json!(text));
                        deltas.push(Value::Object(fields));
                    }
                }
            }
            if let Some(input) = block.get("input") {
                start["input"] = json!({});
                deltas.push(json!({"type": "input_json_delta", "partial_json": input.to_string()}));
            }
            events.push(
                json!({"type": "content_block_start", "index": index, "content_block": start}),
            );
            events.extend(deltas.into_iter().map(
                |delta| json!({"type": "content_block_delta", "index": index, "delta": delta}),
            ));
            events.push(json!({"type": "content_block_stop", "index": index}));
        }
        events.push(json!({"type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": null},
            "usage": {"output_tokens": 5}}));
        events
            .into_iter()
            .map(|event| WireFrame::Text(event.to_string()))
            .collect()
    }
}

impl HistoryFixture for MessagesHistory {
    type Wire = Messages;

    fn wire(&self, model: &str) -> Messages {
        Messages {
            provider: AnthropicConfig::with_key(self.dialect, "sk-test"),
            model: model.to_owned(),
            default_max_tokens: self.dialect.default_max_tokens(model),
            prompt_caching: false,
            automatic_caching: false,
            automatic_caching_ttl: None,
            static_prefix_cache_ttl: None,
            strict_tools: false,
        }
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
        wire: &Messages,
        request: CompletionRequest,
        mode: Mode,
    ) -> Result<Value, EncodeError> {
        http_body(&wire.encode(request, mode)?)
    }

    fn reply(&self, shape: Shape, mode: Mode) -> Option<Vec<WireFrame>> {
        let (content, stop_reason) = self.content(shape);
        Some(match mode {
            Mode::Unary => self.whole(content, stop_reason),
            Mode::Streaming => self.stream(content, stop_reason),
        })
    }

    /// A whole reply states the input as JSON, so it cannot carry text
    /// that is not JSON; a stream can.
    fn call_reply(&self, arguments: &str, mode: Mode) -> Option<Vec<WireFrame>> {
        match mode {
            Mode::Unary => {
                let input: Value = serde_json::from_str(arguments).ok()?;
                Some(self.whole(
                    vec![
                        json!({"type": "tool_use", "id": "toolu_1", "name": "lookup",
                        "input": input}),
                    ],
                    "tool_use",
                ))
            }
            Mode::Streaming => {
                let events = [
                    json!({"type": "message_start", "message": {"type": "message",
                        "id": "msg_1", "role": "assistant", "model": self.model,
                        "content": [], "stop_reason": null,
                        "usage": {"input_tokens": 3, "output_tokens": 1}}}),
                    json!({"type": "content_block_start", "index": 0, "content_block":
                        {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {}}}),
                    json!({"type": "content_block_delta", "index": 0, "delta":
                        {"type": "input_json_delta", "partial_json": arguments}}),
                    json!({"type": "content_block_stop", "index": 0}),
                    json!({"type": "message_delta", "delta": {"stop_reason": "tool_use"},
                        "usage": {"output_tokens": 5}}),
                ];
                Some(
                    events
                        .into_iter()
                        .map(|event| WireFrame::Text(event.to_string()))
                        .collect(),
                )
            }
        }
    }

    /// Every `stop_reason` Anthropic documents, the gateway `sensitive`
    /// stop pi handles, and one nobody documents.
    fn finishes(&self) -> Vec<(&'static str, Vec<WireFrame>, Ending)> {
        [
            ("end_turn", Ending::Success),
            ("stop_sequence", Ending::Success),
            ("pause_turn", Ending::Success),
            ("max_tokens", Ending::Success),
            ("model_context_window_exceeded", Ending::Success),
            ("tool_use", Ending::Success),
            ("refusal", Ending::Failure),
            ("sensitive", Ending::Failure),
            ("x_rig_reason", Ending::Failure),
        ]
        .into_iter()
        .map(|(reason, ending)| {
            let content = vec![json!({"type": "text", "text": "done"})];
            (reason, self.whole(content, reason), ending)
        })
        .collect()
    }

    /// A reply needs its tag, its content and each block's type; every
    /// other field is read leniently.
    fn ablation(&self) -> Option<Ablation<WireFrame>> {
        let (content, stop_reason) = self.content(Shape::Rich);
        Some(Ablation {
            document: self.message(content, stop_reason),
            required: &["/type", "/content", "/content/*/type"],
            frames: |document| vec![WireFrame::Text(document.to_string())],
        })
    }

    /// The item as the content of a whole message.
    fn decode_item(&self, block: &AssistantContent) -> Option<AssistantContent> {
        let item = block.native_item()?.clone();
        let wire = self.wire(self.model);
        let response = decode(
            &wire,
            &CompletionRequest::new("restate"),
            Mode::Unary,
            self.whole(vec![item], "end_turn"),
        )
        .ok()?;
        match response.message() {
            Some(Message::Assistant(turn)) => turn.content.into_iter().next(),
            _ => None,
        }
    }

    /// Messages keys every block by its index, so calls never come without
    /// one: a stream may reuse an index once its block stopped, and a whole
    /// message lists its calls.
    fn calls_reply(&self, shape: CallShape, mode: Mode) -> Option<Vec<WireFrame>> {
        let calls = vec![
            json!({"type": "tool_use", "id": "a1", "name": "weather", "input": {"city": "Paris"}}),
            json!({"type": "tool_use", "id": "b2", "name": "weather", "input": {"city": "Rome"}}),
        ];
        match (shape, mode) {
            (CallShape::WholeList, Mode::Unary) => Some(self.whole(calls, "tool_use")),
            (CallShape::ReusedIndex, Mode::Streaming) => {
                let mut frames = self.stream(calls, "tool_use");
                // Both calls at index 0: the second opens after the first stopped.
                for frame in &mut frames {
                    let WireFrame::Text(text) = frame else {
                        continue;
                    };
                    *text = text.replace("\"index\":1", "\"index\":0");
                }
                Some(frames)
            }
            _ => None,
        }
    }

    fn empty_reply(&self, mode: Mode) -> Option<Vec<WireFrame>> {
        Some(match mode {
            Mode::Unary => self.whole(Vec::new(), "end_turn"),
            Mode::Streaming => self.stream(Vec::new(), "end_turn"),
        })
    }

    fn finish_reason_pointer(&self) -> Option<&'static str> {
        Some("/stop_reason")
    }

    fn error_frame(&self) -> Option<WireFrame> {
        Some(WireFrame::Text(
            json!({"type": "error", "error": {"type": "overloaded_error", "message": "busy"}})
                .to_string(),
        ))
    }

    fn strict_roles(&self) -> bool {
        true
    }
}

rig_history_conformance::history_conformance_suite! {
    wire: "anthropic",
    fixture: ANTHROPIC_HISTORY,
}
