//! Generated histories for row H17: a seeded generator of the shapes that
//! broke replay before (calls sharing ids, orphan, repeated, late and split
//! results, system messages mid-loop, failed turns, other models' turns
//! with hosted pairs, windows a memory policy cuts), a greedy shrinker, and
//! a walker that finds calls and results in any wire's JSON.

use std::collections::HashMap;

use serde_json::Value;

use rig_core::message::{
    AssistantContent, AssistantMessage, CallId, Image, Message, Origin, StopReason, ToolCall,
    ToolFunction, ToolName, ToolResultContent, UserContent,
};

/// A seeded xorshift64* generator, shared by the generated histories (H17)
/// and the generated replies (H18, H19).
#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    /// A generator from `seed`.
    pub fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    /// The next draw.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// A draw in `0..bound` (0 when `bound` is 0).
    pub fn below(&mut self, bound: usize) -> usize {
        usize::try_from(self.next() % u64::try_from(bound.max(1)).unwrap_or(1)).unwrap_or(0)
    }

    /// True `percent` times in a hundred.
    pub fn chance(&mut self, percent: usize) -> bool {
        self.below(100) < percent
    }

    /// One of `items`, which must not be empty.
    pub fn pick<T: Clone>(&mut self, items: &[T]) -> T {
        items[self.below(items.len())].clone()
    }

    /// A draw in `low..=high`.
    pub fn range(&mut self, low: usize, high: usize) -> usize {
        low + self.below(high.saturating_sub(low) + 1)
    }

    /// `text` split into one to three pieces at char boundaries.
    pub fn split(&mut self, text: &str) -> Vec<String> {
        let chars: Vec<char> = text.chars().collect();
        if chars.len() < 2 || self.chance(30) {
            return vec![text.to_owned()];
        }
        let mut cuts = vec![self.range(1, chars.len() - 1)];
        if chars.len() > 2 && self.chance(40) {
            cuts.push(self.range(1, chars.len() - 1));
        }
        cuts.sort_unstable();
        cuts.dedup();
        let mut pieces = Vec::new();
        let mut at = 0;
        for cut in cuts {
            pieces.push(chars[at..cut].iter().collect());
            at = cut;
        }
        pieces.push(chars[at..].iter().collect());
        pieces
    }
}

/// Call ids, including ones the wires must normalize: odd characters, an
/// id longer than any wire takes, Kimi's per-conversation form, a repeat,
/// and an empty one rig issues its own id for.
const IDS: [&str; 8] = [
    "call_a",
    "call_b",
    "dup",
    "call_c",
    "call a/b:c|d",
    "functions.f:0",
    "id_0123456789012345678901234567890123456789012345678901234567890123456789",
    "",
];
const TOOLS: [&str; 3] = ["lookup", "weather", "search_v2"];
const PNG: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";
const PDF: &str = "JVBERi0xLjQKJcOkw7zDtsOfCjIgMCBvYmoKPDwvTGVuZ3RoIDMgMCBSPj4Kc3RyZWFtCmVuZHN0cmVhbQplbmRvYmoKdHJhaWxlcgo8PC9Sb290IDEgMCBSPj4KJSVFT0YK";

fn tool(name: &str) -> ToolName {
    ToolName::new(name).unwrap_or_else(|_| panic!("`{name}` is a tool name"))
}

/// How a generated request declares its tools.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Tools {
    /// In `request.tools`, one definition per tool the history names.
    Declared,
    /// Only in `additional_params.tools`, as a provider's own JSON.
    Raw,
    /// Declared, with `ToolChoice::None`.
    ChoiceNone,
}

/// The tools of a generated request.
pub(crate) fn tools(rng: &mut Rng) -> Tools {
    match rng.below(20) {
        0 | 1 => Tools::Raw,
        2 => Tools::ChoiceNone,
        _ => Tools::Declared,
    }
}

fn user(rng: &mut Rng) -> Message {
    use rig_core::message::{
        Audio, AudioMediaType, DocumentMediaType, DocumentSourceKind, ImageMediaType, Video,
        VideoMediaType,
    };
    let mut content = vec![UserContent::text(format!("question {}", rng.below(100)))];
    if rng.chance(20) {
        content.push(UserContent::Image(Image {
            data: DocumentSourceKind::base64(PNG),
            media_type: Some(ImageMediaType::PNG),
            ..Image::default()
        }));
    }
    if rng.chance(10) {
        content.push(UserContent::document_text(
            "the hours are 9 to 5",
            Some(DocumentMediaType::TXT),
        ));
    }
    if rng.chance(8) {
        content.push(UserContent::document_base64(
            PDF,
            Some(DocumentMediaType::PDF),
        ));
    }
    if rng.chance(5) {
        content.push(UserContent::Audio(Audio {
            data: DocumentSourceKind::base64("UklGRgAAAABXQVZF"),
            media_type: Some(AudioMediaType::WAV),
        }));
    }
    if rng.chance(5) {
        content.push(UserContent::Video(Video {
            data: DocumentSourceKind::url("https://example.invalid/clip.mp4"),
            media_type: Some(VideoMediaType::MP4),
            additional_params: None,
        }));
    }
    Message::User { content }
}

/// A blank text, or blank reasoning, sometimes holding an item.
fn blank(rng: &mut Rng) -> AssistantContent {
    let block = if rng.chance(50) {
        AssistantContent::text(rng.pick(&["", " ", "\n"]))
    } else {
        AssistantContent::reasoning("")
    };
    if rng.chance(30) {
        block.with_native(serde_json::json!({"type": "x_rig_item", "id": "item_blank"}))
    } else {
        block
    }
}

fn other_turn(rng: &mut Rng) -> AssistantMessage {
    let mut content = Vec::new();
    if rng.chance(40) {
        content.push(AssistantContent::reasoning("thinking it over"));
    }
    if rng.chance(10) {
        content.push(blank(rng));
    }
    if rng.chance(70) {
        content.push(AssistantContent::text("an answer"));
    }
    for _ in 0..rng.below(3) {
        let id = IDS[rng.below(IDS.len())];
        let name = TOOLS[rng.below(TOOLS.len())];
        content.push(AssistantContent::ToolCall(ToolCall::new(
            CallId::from_wire(id),
            ToolFunction::new(tool(name), serde_json::json!({"q": rng.below(9)})),
        )));
    }
    if rng.chance(5) {
        content.push(AssistantContent::Opaque(rig_core::message::Opaque {
            item: serde_json::json!({"type": "computer_call", "id": "cu_1"}),
            replay: false,
        }));
    }
    // A hosted use and its result, which only their own model replays.
    if rng.chance(8) {
        for item in [
            serde_json::json!({"type": "server_tool_use", "id": "srvtoolu_other",
                "name": "web_search", "input": {"query": "rig"}}),
            serde_json::json!({"type": "web_search_tool_result", "tool_use_id": "srvtoolu_other",
                "content": []}),
        ] {
            content.push(AssistantContent::Opaque(rig_core::message::Opaque {
                item,
                replay: true,
            }));
        }
    }
    if content.is_empty() {
        content.push(AssistantContent::text("ok"));
    }
    let stop = match rng.below(20) {
        0 | 1 => StopReason::Error("refused".to_owned()),
        2 => StopReason::Aborted("the caller stopped reading".to_owned()),
        3 => StopReason::Length,
        4 => StopReason::Stop,
        _ => StopReason::ToolUse,
    };
    AssistantMessage::new(content)
        .with_origin(Origin::new("other.api", "other", "other-model"))
        .with_stop(stop)
}

/// `turn`, one of the model's own, edited as a hook or repair edits one:
/// a text rewritten, a call's arguments changed, or reasoning reworded, so
/// that block's item is stale.
fn edited(rng: &mut Rng, turn: &AssistantMessage) -> AssistantMessage {
    let mut turn = turn.clone();
    let at = rng.below(turn.content.len());
    if let Some(block) = turn.content.get_mut(at) {
        match block {
            AssistantContent::Text(text) => text.text.push_str(" (edited)"),
            AssistantContent::Reasoning(reasoning) => reasoning.text.push_str(" (edited)"),
            AssistantContent::ToolCall(call) => {
                call.function = ToolFunction::new(
                    call.function.name.clone(),
                    serde_json::json!({"q": "edited"}),
                );
            }
            _ => {}
        }
    }
    turn
}

/// `turn` cut short: its first blocks, stopped by the output budget or
/// aborted, as a reply mid-history that never finished.
fn cut(rng: &mut Rng, turn: &AssistantMessage) -> AssistantMessage {
    let keep = 1 + rng.below(turn.content.len().max(1));
    turn.clone()
        .with_content(turn.content.iter().take(keep).cloned().collect())
        .with_stop(if rng.chance(50) {
            StopReason::Length
        } else {
            StopReason::Aborted("the run stopped".to_owned())
        })
}

/// Results for some of `turn`'s calls, some repeated and one orphan; a
/// result given a turn late goes to `late` instead.
fn results(
    rng: &mut Rng,
    turn: &AssistantMessage,
    late: &mut Vec<UserContent>,
) -> Vec<UserContent> {
    let mut parts = Vec::new();
    for call in turn.tool_calls() {
        if rng.chance(8) {
            late.push(UserContent::ToolResult(
                call.result(vec![ToolResultContent::text("late")]),
            ));
        } else if rng.chance(80) {
            let content = match rng.below(10) {
                0 => vec![
                    ToolResultContent::text("see the chart"),
                    ToolResultContent::image_base64(
                        PNG,
                        Some(rig_core::message::ImageMediaType::PNG),
                        None,
                    ),
                ],
                1 => vec![ToolResultContent::json(
                    serde_json::json!({"ok": true, "n": 2}),
                )],
                2 => vec![
                    ToolResultContent::text("first"),
                    ToolResultContent::text("second"),
                ],
                3 => Vec::new(),
                _ => vec![ToolResultContent::text("done")],
            };
            parts.push(UserContent::ToolResult(rig_core::message::ToolResult {
                is_error: rng.chance(10),
                ..call.result(content)
            }));
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

/// A history of up to a few rounds: user turns with media and documents,
/// assistant turns from the fixture's own replies (`own`, sometimes edited,
/// cut short or reduced to their reasoning) or another model, results
/// answering random subsets of their calls (some a turn late, some split
/// across two user messages), and system messages at the start and in
/// random places. It may start with an assistant turn, and it ends with a
/// user message. Some histories are a window of a longer one, as a memory
/// policy cuts it.
pub(crate) fn history(rng: &mut Rng, own: &[AssistantMessage]) -> Vec<Message> {
    let mut history = Vec::new();
    let mut late = Vec::new();
    if rng.chance(20) {
        history.push(Message::system("be brief"));
    }
    if rng.chance(10) {
        let turn = other_turn(rng);
        let answer = results(rng, &turn, &mut late);
        history.push(Message::Assistant(turn));
        if !answer.is_empty() {
            history.push(Message::User { content: answer });
        }
    }
    history.push(user(rng));
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
                turn.clone().with_content(
                    turn.content
                        .iter()
                        .filter(|block| matches!(block, AssistantContent::Reasoning(_)))
                        .cloned()
                        .collect(),
                )
            }
            Some(turn) if rng.chance(15) => edited(rng, turn),
            Some(turn) if rng.chance(10) => cut(rng, turn),
            Some(turn) if rng.chance(50) => turn.clone(),
            _ => other_turn(rng),
        };
        let mut answer = std::mem::take(&mut late);
        answer.extend(results(rng, &turn, &mut late));
        history.push(Message::Assistant(turn));
        if rng.chance(15) {
            history.push(Message::system("steer"));
        }
        if rng.chance(30) || answer.is_empty() {
            answer.push(UserContent::text("and then?"));
        }
        if answer.len() > 1 && rng.chance(15) {
            let rest = answer.split_off(answer.len() / 2);
            history.push(Message::User { content: answer });
            history.push(Message::User { content: rest });
        } else {
            history.push(Message::User { content: answer });
        }
    }
    if !late.is_empty() {
        history.push(Message::User { content: late });
    }
    if rng.chance(20) {
        history = window(rng, history);
    }
    history
}

/// The last messages of `history`: as rig-memory's sliding window keeps
/// them, which drops results that lost their calls, or as a policy that
/// cuts anywhere does, between a call and its result too.
fn window(rng: &mut Rng, history: Vec<Message>) -> Vec<Message> {
    use rig_memory::MemoryPolicy;
    let keep = rng.range(1, history.len());
    if rng.chance(50) {
        rig_memory::SlidingWindowMemory::last_messages(keep)
            .apply(history.clone())
            .ok()
            .filter(|window| !window.is_empty())
            .unwrap_or(history)
    } else {
        let from = history.len() - keep;
        history.into_iter().skip(from).collect()
    }
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

pub(crate) enum Event {
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
pub(crate) fn events(body: &Value) -> Vec<Event> {
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
/// What breaks a user-first rule in `body`: its first message after any
/// system message is not a user message, in its `messages` or `contents`.
pub(crate) fn first_is_user(body: &Value) -> Vec<String> {
    let first = body
        .get("messages")
        .or_else(|| body.get("contents"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|message| message.get("role").and_then(Value::as_str))
        .find(|role| *role != "system");
    match first {
        Some("user") | None => Vec::new(),
        Some(role) => vec![format!("the first message is a `{role}` message")],
    }
}

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

/// Where a call's results do not come right after it, in the shape each
/// message wire requires: Chat `tool` messages right after the assistant
/// message, Messages `tool_result` blocks leading the next user message,
/// Converse results in the next message, and as many Gemini
/// `functionResponse` parts in the next content as its model turn has
/// calls.
pub(crate) fn adjacency(body: &Value) -> Vec<String> {
    let mut problems = Vec::new();
    let strs = |value: Option<&Value>, key: &str| -> Vec<String> {
        value
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|item| text_of(item, key))
            .collect()
    };
    let messages = body
        .get("messages")
        .and_then(Value::as_array)
        .map_or(&[][..], Vec::as_slice);
    for (at, message) in messages.iter().enumerate() {
        let next = messages.get(at + 1);
        if message.get("role").and_then(Value::as_str) != Some("assistant") {
            continue;
        }
        let answered: Vec<String> = messages[at + 1..]
            .iter()
            .take_while(|next| next.get("role").and_then(Value::as_str) == Some("tool"))
            .filter_map(|next| text_of(next, "tool_call_id"))
            .collect();
        for id in strs(message.get("tool_calls"), "id") {
            if !answered.contains(&id) {
                problems.push(format!(
                    "chat: the call `{id}` at message {at} has no tool message right after it"
                ));
            }
        }
        let blocks = message
            .get("content")
            .and_then(Value::as_array)
            .map_or(&[][..], Vec::as_slice);
        let next_blocks = next
            .and_then(|next| next.get("content"))
            .and_then(Value::as_array)
            .map_or(&[][..], Vec::as_slice);
        let leading: Vec<String> = next_blocks
            .iter()
            .take_while(|block| block.get("type").and_then(Value::as_str) == Some("tool_result"))
            .filter_map(|block| text_of(block, "tool_use_id"))
            .collect();
        for block in blocks {
            if block.get("type").and_then(Value::as_str) == Some("tool_use")
                && let Some(id) = text_of(block, "id")
                && !leading.contains(&id)
            {
                problems.push(format!(
                    "messages: the tool_use `{id}` at message {at} has no tool_result leading the next message"
                ));
            }
        }
        let results: Vec<String> = next_blocks
            .iter()
            .filter_map(|block| block.get("toolResult"))
            .filter_map(|result| text_of(result, "toolUseId"))
            .collect();
        let calls = blocks
            .iter()
            .filter_map(|block| block.get("toolUse"))
            .filter(|call| call.get("type").and_then(Value::as_str) != Some("server_tool_use"));
        for call in calls {
            if let Some(id) = text_of(call, "toolUseId")
                && !results.contains(&id)
            {
                problems.push(format!(
                    "converse: the toolUse `{id}` at message {at} has no toolResult in the next message"
                ));
            }
        }
    }
    let contents = body
        .get("contents")
        .and_then(Value::as_array)
        .map_or(&[][..], Vec::as_slice);
    let count = |content: Option<&Value>, key: &str| {
        content
            .and_then(|content| content.get("parts"))
            .and_then(Value::as_array)
            .map_or(0, |parts| {
                parts.iter().filter(|part| part.get(key).is_some()).count()
            })
    };
    for (at, content) in contents.iter().enumerate() {
        let calls = count(Some(content), "functionCall");
        let responses = count(contents.get(at + 1), "functionResponse");
        if calls > 0 && responses != calls {
            problems.push(format!(
                "gemini: content {at} has {calls} calls and the next has {responses} responses"
            ));
        }
    }
    problems
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
                        UserContent::Document(_) => "document".to_owned(),
                        UserContent::Audio(_) => "audio".to_owned(),
                        UserContent::Video(_) => "video".to_owned(),
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
                        _ => "other".to_owned(),
                    })
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        })
        .collect::<Vec<_>>()
        .join(" / ")
}
