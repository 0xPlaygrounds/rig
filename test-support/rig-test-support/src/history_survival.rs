//! The history round-trip rule: opaque fields a provider delivered must reach
//! the next request, and every tool call a request carries must be paired
//! with its result.
//!
//! Two callers apply the same rule: `tests/cassette_history_survival.rs`
//! sweeps every committed cassette at zero provider cost, and the recorded
//! round-trip cells apply it to the exchanges they just recorded. The rule
//! reads wire bytes, not Rig's normalized history, so a field that Rig's
//! decoder never modeled is still caught when the next request lacks it.
//! Survival is presence: a delivered value must appear as some string leaf
//! of the next request. The rule does not check which slot carries it, so a
//! value kept on one leg of a pair and dropped from the other passes.
//!
//! ```
//! use rig_test_support::history_survival::{lost_tokens, Dialect};
//! let request = serde_json::json!({ "messages": [] });
//! assert!(lost_tokens(Dialect::from_path("/v1/messages"), "{}", &request).is_empty());
//! ```
#![allow(dead_code)]

pub mod driver;
pub mod portability;

use std::collections::BTreeSet;

use serde_json::Value;

/// The request/response grammar a recorded path speaks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dialect {
    /// Anthropic Messages: `content` blocks with `thinking` signatures,
    /// `redacted_thinking` data and `tool_use` ids.
    AnthropicMessages,
    /// Gemini `generateContent`: parts carrying `thoughtSignature`.
    GeminiGenerateContent,
    /// Gemini Interactions: parts carrying `signature`.
    GeminiInteractions,
    /// OpenAI Responses and its dialects: `reasoning` items with
    /// `encrypted_content`, `function_call` items with `call_id`.
    OpenAiResponses,
    /// Chat Completions and its dialects: `tool_calls[].id`.
    ChatCompletions,
    /// Ollama `/api/chat`: `tool_calls[].id` when the daemon issues one.
    OllamaChat,
    /// Bedrock Converse: `reasoningText.signature`, `toolUse.toolUseId`.
    BedrockConverse,
    /// Cohere v2 chat: `tool_calls[].id`.
    CohereChat,
    /// An endpoint this rule does not model.
    Unmodeled,
}

impl Dialect {
    /// Classify a recorded request path.
    pub fn from_path(path: &str) -> Self {
        if path.ends_with("/v1/messages") {
            Self::AnthropicMessages
        } else if path.contains(":generateContent") || path.contains(":streamGenerateContent") {
            Self::GeminiGenerateContent
        } else if path.contains("/interactions") {
            Self::GeminiInteractions
        } else if path.ends_with("/responses") {
            Self::OpenAiResponses
        } else if path.ends_with("/chat/completions") {
            Self::ChatCompletions
        } else if path.ends_with("/api/chat") {
            Self::OllamaChat
        } else if path.contains("/converse") {
            Self::BedrockConverse
        } else if path.ends_with("/v2/chat") {
            Self::CohereChat
        } else {
            Self::Unmodeled
        }
    }

    /// Whether the provider always issues tool-call ids. Gemini pairs
    /// function calls with responses by name; the 2.5 models issue no id
    /// and the 3 models do, so its ids are optional rather than absent.
    pub fn issues_tool_call_ids(self) -> bool {
        !matches!(self, Self::GeminiGenerateContent | Self::GeminiInteractions)
    }

    /// The request field holding the conversation.
    pub fn conversation_field(self) -> Option<&'static str> {
        match self {
            Self::AnthropicMessages
            | Self::ChatCompletions
            | Self::OllamaChat
            | Self::BedrockConverse
            | Self::CohereChat => Some("messages"),
            Self::GeminiGenerateContent => Some("contents"),
            Self::GeminiInteractions | Self::OpenAiResponses => Some("input"),
            Self::Unmodeled => None,
        }
    }
}

/// One opaque value a response delivered.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Token {
    /// The content kind: `signature`, `thought_signature`, `encrypted_content`,
    /// `redacted_reasoning`, `reasoning_id` or `tool_call_id`.
    pub kind: &'static str,
    /// The delivered value, as scrubbed in the cassette.
    pub value: String,
}

/// Every content kind the rule can observe, for coverage censuses.
pub const TOKEN_KINDS: &[&str] = &[
    "signature",
    "thought_signature",
    "encrypted_content",
    "redacted_reasoning",
    "reasoning_id",
    "tool_call_id",
];

/// Split a recorded response body into its JSON documents: one for a whole
/// reply, one per `data:` line for SSE, one per line for NDJSON.
pub fn response_documents(body: &str) -> Vec<Value> {
    let trimmed = body.trim_start();
    if trimmed.starts_with('{') || trimmed.starts_with('[') {
        if let Ok(value) = serde_json::from_str::<Value>(trimmed) {
            return vec![value];
        }
        // NDJSON: a whole reply parses above, so a failed parse is a
        // multi-line record stream.
        return body
            .lines()
            .filter_map(|line| serde_json::from_str::<Value>(line.trim()).ok())
            .collect();
    }
    body.lines()
        .filter_map(|line| line.strip_prefix("data:"))
        .filter_map(|data| serde_json::from_str::<Value>(data.trim()).ok())
        .collect()
}

/// The opaque values a response delivered that the next request must carry.
///
/// A Responses stream restates the same item at `output_item.added`,
/// `output_item.done` and `response.completed`, and the provider re-encrypts
/// reasoning at each restatement, so only the `done` item is the value Rig
/// replays. Every other dialect delivers each value once.
pub fn response_tokens(dialect: Dialect, body: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    for document in response_documents(body) {
        let document = match dialect {
            Dialect::OpenAiResponses => match document.get("type").and_then(Value::as_str) {
                Some("response.output_item.done") => {
                    document.get("item").cloned().unwrap_or(Value::Null)
                }
                Some(_) => continue,
                None => document,
            },
            _ => document,
        };
        collect_tokens(&document, &mut tokens);
    }
    tokens.sort();
    tokens.dedup();
    tokens
}

fn collect_tokens(value: &Value, tokens: &mut Vec<Token>) {
    match value {
        Value::Object(object) => {
            let kind_of = object.get("type").and_then(Value::as_str);
            let mut push = |kind: &'static str, key: &str| {
                if let Some(text) = object.get(key).and_then(Value::as_str)
                    && !text.is_empty()
                {
                    tokens.push(Token {
                        kind,
                        value: text.to_owned(),
                    });
                }
            };
            push("signature", "signature");
            push("thought_signature", "thoughtSignature");
            push("encrypted_content", "encrypted_content");
            match kind_of {
                Some("redacted_thinking") => push("redacted_reasoning", "data"),
                Some("reasoning") => push("reasoning_id", "id"),
                Some("tool_use") => push("tool_call_id", "id"),
                Some("function_call") => push("tool_call_id", "call_id"),
                _ => {}
            }
            // Chat Completions, Ollama and Cohere calls: an object holding a
            // `function` beside its `id`.
            if object.contains_key("function") {
                push("tool_call_id", "id");
            }
            if object.contains_key("toolUseId") {
                push("tool_call_id", "toolUseId");
            }
            // Gemini function calls carry an optional `id` beside `name`/`args`.
            if object.contains_key("args") && object.contains_key("name") {
                push("tool_call_id", "id");
            }
            for child in object.values() {
                collect_tokens(child, tokens);
            }
        }
        Value::Array(items) => {
            for item in items {
                collect_tokens(item, tokens);
            }
        }
        _ => {}
    }
}

/// Every string leaf in a request body.
pub fn string_values(value: &Value) -> BTreeSet<String> {
    let mut values = BTreeSet::new();
    collect_strings(value, &mut values);
    values
}

fn collect_strings(value: &Value, values: &mut BTreeSet<String>) {
    match value {
        Value::String(text) => {
            values.insert(text.clone());
        }
        Value::Array(items) => items.iter().for_each(|item| collect_strings(item, values)),
        Value::Object(object) => object
            .values()
            .for_each(|item| collect_strings(item, values)),
        _ => {}
    }
}

/// The tokens a response delivered that the continuation request lacks.
pub fn lost_tokens(dialect: Dialect, response_body: &str, next_request: &Value) -> Vec<Token> {
    let present = string_values(next_request);
    response_tokens(dialect, response_body)
        .into_iter()
        .filter(|token| !present.contains(&token.value))
        .collect()
}

/// Whether `later` continues the conversation `earlier` sent: the same
/// opening turn, and more turns.
pub fn continues(dialect: Dialect, earlier: &Value, later: &Value) -> bool {
    let Some(field) = dialect.conversation_field() else {
        return false;
    };
    let (Some(earlier), Some(later)) = (
        earlier.get(field).and_then(Value::as_array),
        later.get(field).and_then(Value::as_array),
    ) else {
        return false;
    };
    let opening = |turns: &[Value]| {
        turns
            .iter()
            .find(|turn| {
                !matches!(
                    turn.get("role").and_then(Value::as_str),
                    Some("system" | "developer")
                )
            })
            .cloned()
    };
    match (opening(earlier), opening(later)) {
        (Some(first), Some(second)) => first == second && later.len() > earlier.len(),
        _ => false,
    }
}

/// A tool call without its result, or a result without its call.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Unpaired {
    /// `call` or `result`.
    pub side: &'static str,
    /// The tool name or id the dangling side names.
    pub label: String,
}

impl std::fmt::Display for Unpaired {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {} has no partner", self.side, self.label)
    }
}

/// Every tool call in a request must be answered before the conversation
/// moves on, and every result must answer a call. Ollama and Gemini pair by
/// name and order; the other dialects pair by id.
pub fn unpaired_tool_calls(dialect: Dialect, request: &Value) -> Vec<Unpaired> {
    let Some(field) = dialect.conversation_field() else {
        return Vec::new();
    };
    let Some(turns) = request.get(field).and_then(Value::as_array) else {
        return Vec::new();
    };
    let mut open: Vec<String> = Vec::new();
    let mut unpaired = Vec::new();
    let mut previous_assistant = false;
    for turn in turns {
        let (calls, results) = turn_calls_and_results(dialect, turn);
        let assistant = is_assistant_turn(dialect, turn);
        // A Responses turn is several items in a row; a new assistant run,
        // not a new item, closes the previous one, and anything still open
        // was never answered.
        if assistant && !previous_assistant {
            unpaired.extend(open.drain(..).map(|label| Unpaired {
                side: "call",
                label,
            }));
        }
        if assistant {
            open.extend(calls);
        }
        previous_assistant = assistant;
        for result in results {
            match open.iter().position(|call| *call == result) {
                Some(index) => {
                    open.remove(index);
                }
                None => unpaired.push(Unpaired {
                    side: "result",
                    label: result,
                }),
            }
        }
    }
    unpaired.extend(open.into_iter().map(|label| Unpaired {
        side: "call",
        label,
    }));
    unpaired
}

/// The same pairing rule over Rig's normalized history as serialized in the
/// effect goldens: every `toolcall` id is answered by a `toolresult` whose
/// `call` is that id before the next assistant turn, and no result is stray.
pub fn unpaired_normalized_tool_calls(history: &[Value]) -> Vec<Unpaired> {
    let mut open: Vec<Value> = Vec::new();
    let mut unpaired = Vec::new();
    let label = |id: &Value| {
        id.get("id")
            .and_then(Value::as_str)
            .map_or_else(|| id.to_string(), str::to_owned)
    };
    for message in history {
        let content = message.get("content").and_then(Value::as_array);
        match message.get("role").and_then(Value::as_str) {
            Some("assistant") => {
                unpaired.extend(open.drain(..).map(|id| Unpaired {
                    side: "call",
                    label: label(&id),
                }));
                for part in content.into_iter().flatten() {
                    if part.get("type").and_then(Value::as_str) == Some("toolcall")
                        && let Some(id) = part.get("id")
                    {
                        open.push(id.clone());
                    }
                }
            }
            Some("user") => {
                for part in content.into_iter().flatten() {
                    if part.get("type").and_then(Value::as_str) != Some("toolresult") {
                        continue;
                    }
                    let Some(call) = part.get("call") else {
                        continue;
                    };
                    match open.iter().position(|id| id == call) {
                        Some(index) => {
                            open.remove(index);
                        }
                        None => unpaired.push(Unpaired {
                            side: "result",
                            label: label(call),
                        }),
                    }
                }
            }
            _ => {}
        }
    }
    unpaired.extend(open.into_iter().map(|id| Unpaired {
        side: "call",
        label: label(&id),
    }));
    unpaired
}

fn is_assistant_turn(dialect: Dialect, turn: &Value) -> bool {
    let role = turn.get("role").and_then(Value::as_str);
    match dialect {
        Dialect::GeminiGenerateContent => role == Some("model"),
        Dialect::OpenAiResponses => {
            matches!(
                turn.get("type").and_then(Value::as_str),
                Some("message" | "function_call" | "reasoning")
            ) && role != Some("user")
        }
        Dialect::GeminiInteractions => role == Some("model") || role == Some("assistant"),
        _ => role == Some("assistant"),
    }
}

/// The call labels a turn opens and the result labels it answers.
fn turn_calls_and_results(dialect: Dialect, turn: &Value) -> (Vec<String>, Vec<String>) {
    let mut calls = Vec::new();
    let mut results = Vec::new();
    let text = |value: Option<&Value>| value.and_then(Value::as_str).map(str::to_owned);
    match dialect {
        Dialect::ChatCompletions | Dialect::CohereChat => {
            for call in turn
                .get("tool_calls")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                calls.extend(text(call.get("id")));
            }
            if turn.get("role").and_then(Value::as_str) == Some("tool") {
                results.extend(text(turn.get("tool_call_id")));
            }
        }
        Dialect::OllamaChat => {
            for call in turn
                .get("tool_calls")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                calls.extend(text(call.pointer("/function/name")));
            }
            if turn.get("role").and_then(Value::as_str) == Some("tool") {
                results.extend(text(turn.get("tool_name")));
            }
        }
        Dialect::AnthropicMessages => {
            for block in turn
                .get("content")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                match block.get("type").and_then(Value::as_str) {
                    Some("tool_use") => calls.extend(text(block.get("id"))),
                    Some("tool_result") => results.extend(text(block.get("tool_use_id"))),
                    _ => {}
                }
            }
        }
        Dialect::OpenAiResponses => match turn.get("type").and_then(Value::as_str) {
            Some("function_call") => calls.extend(text(turn.get("call_id"))),
            Some("function_call_output") => results.extend(text(turn.get("call_id"))),
            _ => {}
        },
        Dialect::GeminiGenerateContent | Dialect::GeminiInteractions => {
            for part in turn
                .get("parts")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                if let Some(call) = part
                    .get("functionCall")
                    .or_else(|| part.get("function_call"))
                {
                    calls.extend(text(call.get("name")));
                }
                if let Some(result) = part
                    .get("functionResponse")
                    .or_else(|| part.get("function_response"))
                {
                    results.extend(text(result.get("name")));
                }
            }
        }
        Dialect::BedrockConverse => {
            for block in turn
                .get("content")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                if let Some(call) = block.get("toolUse") {
                    calls.extend(text(call.get("toolUseId")));
                }
                if let Some(result) = block.get("toolResult") {
                    results.extend(text(result.get("toolUseId")));
                }
            }
        }
        Dialect::Unmodeled => {}
    }
    (calls, results)
}

#[cfg(test)]
#[path = "history_survival/tests.rs"]
mod tests;
