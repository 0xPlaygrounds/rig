//! Generated histories for row H17: a seeded generator of the shapes that
//! broke replay before (calls sharing ids, orphan and repeated results,
//! system messages mid-loop, failed turns, other models' turns), a greedy
//! shrinker, and a walker that finds calls and results in any wire's JSON.

use std::collections::HashMap;

use serde_json::Value;

use rig_core::message::{
    AssistantContent, AssistantMessage, CallId, Image, Message, Origin, StopReason, ToolCall,
    ToolFunction, ToolName, ToolResultContent, UserContent,
};

/// A seeded xorshift64* generator.
pub(crate) struct Rng(u64);

impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn below(&mut self, bound: usize) -> usize {
        usize::try_from(self.next() % u64::try_from(bound.max(1)).unwrap_or(1)).unwrap_or(0)
    }

    fn chance(&mut self, percent: usize) -> bool {
        self.below(100) < percent
    }
}

const IDS: [&str; 4] = ["call_a", "call_b", "dup", "call_c"];
const PNG: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";

fn tool(name: &str) -> ToolName {
    ToolName::new(name).unwrap_or_else(|_| panic!("`{name}` is a tool name"))
}

fn user(rng: &mut Rng) -> Message {
    let mut content = vec![UserContent::text(format!("question {}", rng.below(100)))];
    if rng.chance(20) {
        content.push(UserContent::Image(Image {
            data: rig_core::message::DocumentSourceKind::base64(PNG),
            media_type: Some(rig_core::message::ImageMediaType::PNG),
            ..Image::default()
        }));
    }
    Message::User { content }
}

fn other_turn(rng: &mut Rng) -> AssistantMessage {
    let mut content = Vec::new();
    if rng.chance(40) {
        content.push(AssistantContent::reasoning("thinking it over"));
    }
    if rng.chance(70) {
        content.push(AssistantContent::text("an answer"));
    }
    for _ in 0..rng.below(3) {
        let id = IDS[rng.below(IDS.len())];
        content.push(AssistantContent::ToolCall(ToolCall::new(
            CallId::from_wire(id),
            ToolFunction::new(tool("lookup"), serde_json::json!({"q": rng.below(9)})),
        )));
    }
    if content.is_empty() {
        content.push(AssistantContent::text("ok"));
    }
    let failed = rng.chance(10);
    AssistantMessage {
        content,
        origin: Some(Origin::new("other.api", "other", "other-model")),
        stop: Some(if failed {
            StopReason::Error("refused".to_owned())
        } else {
            StopReason::ToolUse
        }),
    }
}

fn results(rng: &mut Rng, turn: &AssistantMessage) -> Vec<UserContent> {
    let mut parts = Vec::new();
    for call in turn.tool_calls() {
        if rng.chance(80) {
            parts.push(UserContent::ToolResult(
                call.result(vec![ToolResultContent::text("done")]),
            ));
        }
        if rng.chance(10) {
            parts.push(UserContent::ToolResult(
                call.result(vec![ToolResultContent::text("again")]),
            ));
        }
    }
    if rng.chance(10) {
        parts.push(UserContent::ToolResult(rig_core::message::ToolResult {
            call: CallId::from_wire("call_gone"),
            name: tool("lookup"),
            content: vec![ToolResultContent::text("stale")],
            is_error: false,
        }));
    }
    parts
}

/// A history of up to a few rounds: user turns, assistant turns from the
/// fixture's own replies (`own`) or another model, results answering random
/// subsets of their calls, and system messages in random places. It ends
/// with a user message.
pub(crate) fn history(rng: &mut Rng, own: &[AssistantMessage]) -> Vec<Message> {
    let mut history = vec![user(rng)];
    for _ in 0..1 + rng.below(3) {
        if rng.chance(15) {
            history.push(Message::system("steer"));
        }
        let turn = match own.get(rng.below(own.len() + 1)) {
            // The model's own turn with only its reasoning left, as a reply
            // cut while thinking is.
            Some(turn)
                if rng.chance(10)
                    && turn
                        .content
                        .iter()
                        .any(|block| matches!(block, AssistantContent::Reasoning(_))) =>
            {
                AssistantMessage {
                    content: turn
                        .content
                        .iter()
                        .filter(|block| matches!(block, AssistantContent::Reasoning(_)))
                        .cloned()
                        .collect(),
                    ..turn.clone()
                }
            }
            Some(turn) if rng.chance(50) => turn.clone(),
            _ => other_turn(rng),
        };
        let mut answer = results(rng, &turn);
        history.push(Message::Assistant(turn));
        if rng.chance(15) {
            history.push(Message::system("steer"));
        }
        if rng.chance(30) || answer.is_empty() {
            answer.push(UserContent::text("and then?"));
        }
        history.push(Message::User { content: answer });
    }
    history
}

/// The smallest prefix-preserving subset of `history` that still fails
/// `fails`, removing one message at a time.
pub(crate) fn shrink(
    mut history: Vec<Message>,
    fails: impl Fn(&[Message]) -> bool,
) -> Vec<Message> {
    let mut at = 0;
    while at < history.len() {
        let mut candidate = history.clone();
        candidate.remove(at);
        let ends_with_user = matches!(candidate.last(), Some(Message::User { .. }));
        if ends_with_user && !candidate.is_empty() && fails(&candidate) {
            history = candidate;
        } else {
            at += 1;
        }
    }
    history
}

enum Event {
    Call(String),
    Result(String),
}

fn text_of(value: &Value, key: &str) -> Option<String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
}

/// Every call and result in a request body, in order, across the wire
/// shapes Rig sends: Messages `tool_use`/`tool_result`, Responses
/// `function_call`/`function_call_output`, Chat `tool_calls` and `tool`
/// messages, Gemini `functionCall`/`functionResponse` (by name when they
/// carry no id), and Converse `toolUse`/`toolResult`.
fn events(body: &Value) -> Vec<Event> {
    fn walk(value: &Value, out: &mut Vec<Event>) {
        match value {
            Value::Object(fields) => {
                match fields.get("type").and_then(Value::as_str) {
                    Some("tool_use") => {
                        out.push(Event::Call(text_of(value, "id").unwrap_or_default()))
                    }
                    Some("function_call" | "custom_tool_call") => out.push(Event::Call(
                        text_of(value, "call_id")
                            .or_else(|| text_of(value, "id"))
                            .unwrap_or_default(),
                    )),
                    Some("tool_result") => {
                        out.push(Event::Result(
                            text_of(value, "tool_use_id").unwrap_or_default(),
                        ));
                    }
                    Some(
                        "function_call_output" | "custom_tool_call_output" | "function_result",
                    ) => {
                        out.push(Event::Result(text_of(value, "call_id").unwrap_or_default()));
                    }
                    _ => {}
                }
                if let Some(call) = fields.get("toolUse") {
                    out.push(Event::Call(text_of(call, "toolUseId").unwrap_or_default()));
                }
                if let Some(result) = fields.get("toolResult") {
                    out.push(Event::Result(
                        text_of(result, "toolUseId").unwrap_or_default(),
                    ));
                }
                if let Some(call) = fields.get("functionCall") {
                    out.push(Event::Call(text_of(call, "id").unwrap_or_else(|| {
                        format!("name:{}", text_of(call, "name").unwrap_or_default())
                    })));
                }
                if let Some(result) = fields.get("functionResponse") {
                    out.push(Event::Result(text_of(result, "id").unwrap_or_else(|| {
                        format!("name:{}", text_of(result, "name").unwrap_or_default())
                    })));
                }
                if fields.get("role").and_then(Value::as_str) == Some("tool") {
                    out.push(Event::Result(
                        text_of(value, "tool_call_id").unwrap_or_else(|| {
                            format!("name:{}", text_of(value, "tool_name").unwrap_or_default())
                        }),
                    ));
                }
                if let Some(Value::Array(calls)) = fields.get("tool_calls") {
                    for call in calls {
                        out.push(Event::Call(text_of(call, "id").unwrap_or_else(|| {
                            let name = call
                                .pointer("/function/name")
                                .and_then(Value::as_str)
                                .unwrap_or_default();
                            format!("name:{name}")
                        })));
                    }
                }
                for (key, value) in fields {
                    if matches!(key.as_str(), "tools" | "tool_calls" | "toolConfig") {
                        continue;
                    }
                    walk(value, out);
                }
            }
            Value::Array(values) => values.iter().for_each(|value| walk(value, out)),
            _ => {}
        }
    }
    let mut out = Vec::new();
    walk(body, &mut out);
    out
}

/// What is wrong with `body`'s pairing: a call sent again while its first
/// is unanswered, a result no open call asked for, or a call with no result.
pub(crate) fn pairing(body: &Value) -> Vec<String> {
    let mut open: HashMap<String, usize> = HashMap::new();
    let mut problems = Vec::new();
    for event in events(body) {
        match event {
            Event::Call(id) => {
                let count = open.entry(id.clone()).or_default();
                if *count > 0 && !id.starts_with("name:") {
                    problems.push(format!(
                        "the call `{id}` is sent again before it is answered"
                    ));
                }
                *count += 1;
            }
            Event::Result(id) => {
                let count = open.entry(id.clone()).or_default();
                if *count == 0 {
                    problems.push(format!("a result answers `{id}`, which no open call has"));
                } else {
                    *count -= 1;
                }
            }
        }
    }
    problems.extend(
        open.into_iter()
            .filter(|(_, count)| *count > 0)
            .map(|(id, _)| format!("the call `{id}` has no result")),
    );
    problems.sort();
    problems
}

/// What breaks role alternation in `body`: two user or two assistant
/// messages in a row, system messages aside, in its `messages` or
/// `contents`.
pub(crate) fn alternation(body: &Value) -> Vec<String> {
    let messages = body
        .get("messages")
        .or_else(|| body.get("contents"))
        .and_then(Value::as_array)
        .map_or(&[][..], Vec::as_slice);
    let roles: Vec<&str> = messages
        .iter()
        .filter_map(|message| message.get("role").and_then(Value::as_str))
        .filter(|role| *role != "system")
        .map(|role| if role == "model" { "assistant" } else { role })
        .collect();
    roles
        .windows(2)
        .filter(|pair| pair[0] == pair[1])
        .map(|pair| format!("two `{}` messages in a row", pair[0]))
        .collect()
}

/// Two user messages in a row in `body`'s `messages`, which no
/// message-shaped wire needs: the adapter joins them. Gemini's `contents`
/// are left out, since it splits a user turn's function responses from its
/// text by design.
pub(crate) fn adjacent_users(body: &Value) -> Vec<String> {
    let messages = body
        .get("messages")
        .and_then(Value::as_array)
        .map_or(&[][..], Vec::as_slice);
    messages
        .windows(2)
        .filter(|pair| {
            pair.iter()
                .all(|message| message.get("role").and_then(Value::as_str) == Some("user"))
        })
        .map(|_| "two user messages in a row".to_owned())
        .collect()
}

/// A short rendering of `history`, for a failure report.
pub(crate) fn render(history: &[Message]) -> String {
    history
        .iter()
        .map(|message| match message {
            Message::System { content } => format!("system({content})"),
            Message::User { content } => format!(
                "user[{}]",
                content
                    .iter()
                    .map(|part| match part {
                        UserContent::ToolResult(result) =>
                            format!("result({})", result.call.wire()),
                        UserContent::Image(_) => "image".to_owned(),
                        _ => "text".to_owned(),
                    })
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Message::Assistant(turn) => format!(
                "assistant{}[{}]",
                if turn.stop.as_ref().is_some_and(StopReason::is_failure) {
                    "(failed)"
                } else {
                    ""
                },
                turn.content
                    .iter()
                    .map(|block| match block {
                        AssistantContent::ToolCall(call) => format!("call({})", call.id.wire()),
                        AssistantContent::Text(_) => "text".to_owned(),
                        AssistantContent::Reasoning(_) => "reasoning".to_owned(),
                        AssistantContent::Image(_) => "image".to_owned(),
                        AssistantContent::Opaque(_) => "opaque".to_owned(),
                    })
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        })
        .collect::<Vec<_>>()
        .join(" / ")
}
