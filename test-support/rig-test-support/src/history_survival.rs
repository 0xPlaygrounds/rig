//! The history round-trip rule: opaque fields a provider delivered must reach
//! the next request, and every tool call a request carries must be paired
//! with its result.
//!
//! Two callers apply the same rule: `tests/cassette_history_survival.rs`
//! sweeps every committed cassette at zero provider cost, and the recorded
//! round-trip cells apply it to the exchanges they just recorded. The rule
//! reads wire bytes, not Rig's normalized history, so a field that Rig's
//! decoder never modeled is still caught when the next request lacks it.
//! Survival is by slot: a delivered value must appear where the next request
//! carries that kind of value, and a tool-call id on both the call and its
//! result. Values recorded as legacy placeholders cannot be proven and are
//! counted apart.
//!
//! ```
//! use rig_test_support::history_survival::{lost_tokens, Dialect};
//! let request = serde_json::json!({ "messages": [] });
//! assert!(lost_tokens(Dialect::from_path("/v1/messages"), "{}", &request).is_empty());
//! ```
#![allow(dead_code)]

pub mod adversarial;
pub mod driver;
pub mod portability;
pub mod sessions;

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
    /// What the value belongs to, when the response names it: a call's tool
    /// name and arguments, a signed thinking block's text, a Gemini part's
    /// kind, an encrypted item's id. The request must carry the value on an
    /// owner with the same anchor. Streamed deltas that deliver a value apart
    /// from its owner leave it `None`.
    pub anchor: Option<String>,
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
    let documents = response_documents(body);
    // A whole reply states every owner complete; a stream's deltas may not.
    let whole = documents.len() == 1;
    for document in documents {
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
        // Responses `done` items are complete even inside a stream.
        let complete = whole || dialect == Dialect::OpenAiResponses;
        collect_tokens(&document, complete, &mut tokens);
    }
    tokens.sort();
    tokens.dedup();
    tokens
}

fn collect_tokens(value: &Value, complete: bool, tokens: &mut Vec<Token>) {
    match value {
        Value::Object(object) => {
            let kind_of = object.get("type").and_then(Value::as_str);
            let mut push = |kind: &'static str, key: &str, anchor: Option<String>| {
                if let Some(text) = object.get(key).and_then(Value::as_str)
                    && !text.is_empty()
                {
                    tokens.push(Token {
                        kind,
                        value: text.to_owned(),
                        anchor,
                    });
                }
            };
            let thinking = complete
                .then(|| object.get("thinking").and_then(Value::as_str))
                .flatten()
                .map(str::to_owned);
            // A Responses reasoning item (OpenRouter relaying Claude) carries
            // its signature beside its text, owned by the item's id.
            let signature_anchor = if kind_of == Some("reasoning") {
                object.get("id").and_then(Value::as_str).map(str::to_owned)
            } else {
                thinking
            };
            push("signature", "signature", signature_anchor);
            push("thought_signature", "thoughtSignature", part_anchor(object));
            push(
                "encrypted_content",
                "encrypted_content",
                object.get("id").and_then(Value::as_str).map(str::to_owned),
            );
            let call = call_anchor(object, complete);
            match kind_of {
                Some("redacted_thinking") => push("redacted_reasoning", "data", None),
                // OpenRouter encrypted reasoning details.
                Some("reasoning.encrypted") => push(
                    "encrypted_content",
                    "data",
                    object.get("id").and_then(Value::as_str).map(str::to_owned),
                ),
                Some("reasoning") => push("reasoning_id", "id", None),
                Some("tool_use") => push("tool_call_id", "id", call.clone()),
                Some("function_call") => push("tool_call_id", "call_id", call.clone()),
                _ => {}
            }
            // Chat Completions, Ollama and Cohere calls: an object holding a
            // `function` beside its `id`.
            if object.contains_key("function") {
                push("tool_call_id", "id", call.clone());
            }
            if object.contains_key("toolUseId") {
                push("tool_call_id", "toolUseId", call.clone());
            }
            // Gemini function calls carry an optional `id` beside `name`/`args`.
            if object.contains_key("args") && object.contains_key("name") {
                push("tool_call_id", "id", call);
            }
            for child in object.values() {
                collect_tokens(child, complete, tokens);
            }
        }
        Value::Array(items) => {
            for item in items {
                collect_tokens(item, complete, tokens);
            }
        }
        _ => {}
    }
}

/// A call's owner: its tool name, with its arguments when the call is
/// complete. Arguments compare as JSON, whatever their key order or
/// string encoding.
fn call_anchor(object: &serde_json::Map<String, Value>, complete: bool) -> Option<String> {
    let function = object
        .get("function")
        .and_then(Value::as_object)
        .unwrap_or(object);
    let name = function.get("name").and_then(Value::as_str)?;
    if !complete {
        return Some(name.to_owned());
    }
    let arguments = ["arguments", "input", "args"]
        .iter()
        .find_map(|key| function.get(*key))
        .map(|arguments| match arguments {
            Value::String(text) => serde_json::from_str(text).unwrap_or(arguments.clone()),
            other => other.clone(),
        })
        .map(|arguments| canonical(&arguments).to_string())
        .unwrap_or_default();
    Some(format!("{name}{arguments}"))
}

fn canonical(value: &Value) -> Value {
    match value {
        Value::Object(object) => {
            let sorted: std::collections::BTreeMap<_, _> = object
                .iter()
                .map(|(key, value)| (key.clone(), canonical(value)))
                .collect();
            Value::Object(sorted.into_iter().collect())
        }
        Value::Array(items) => Value::Array(items.iter().map(canonical).collect()),
        other => other.clone(),
    }
}

/// A Gemini part's owner: a function call, a thought, or answer text. The
/// call's name is not part of it, since a repair hook may rename the call
/// it signs.
fn part_anchor(object: &serde_json::Map<String, Value>) -> Option<String> {
    if object.contains_key("functionCall") || object.contains_key("function_call") {
        return Some("functionCall".to_owned());
    }
    object.contains_key("text").then(|| {
        if object.get("thought").and_then(Value::as_bool) == Some(true) {
            "thought".to_owned()
        } else {
            "text".to_owned()
        }
    })
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

/// The tokens a response delivered that the continuation request does not
/// carry in the slot that must hold them. A tool-call id must be on both
/// legs: the call and its result. Values recorded as legacy placeholders
/// (`call_REDACTED_1`) are excluded; see [`legacy_tokens`].
pub fn lost_tokens(dialect: Dialect, response_body: &str, next_request: &Value) -> Vec<Token> {
    let slots = request_slots(next_request);
    response_tokens(dialect, response_body)
        .into_iter()
        .filter(|token| !is_legacy_placeholder(&token.value))
        .filter(|token| {
            let legs: &[Leg] = if token.kind == "tool_call_id" {
                &[Leg::Call, Leg::Result]
            } else {
                &[Leg::Call]
            };
            !legs.iter().all(|leg| {
                slots
                    .get(&(token.kind, *leg))
                    .and_then(|values| values.get(&token.value))
                    .is_some_and(|anchors| match (&token.anchor, leg) {
                        (Some(anchor), Leg::Call) => anchors.contains(&Some(anchor.clone())),
                        _ => true,
                    })
            })
        })
        .collect()
}

/// Whether `later` addresses a different reasoning family than the model
/// that answered `earlier`: both name a gateway model (`vendor/name`) and the
/// vendors differ. The earlier family is the model `earlier_response` names,
/// since reasoning belongs to the upstream that produced it, falling back to
/// the earlier request's model. Reasoning is not expected to carry across
/// such a switch; tool-call ids still are. A later router or preset
/// (`openrouter/auto`, `@preset/name`) names no family and replays every one,
/// so it is no switch.
pub fn switches_reasoning_family(earlier: &Value, earlier_response: &str, later: &Value) -> bool {
    fn vendor(model: &str) -> Option<String> {
        if model.starts_with('@') {
            return None;
        }
        let (vendor, _) = model.split_once('/')?;
        let vendor = vendor.trim_start_matches('~');
        (vendor != "openrouter").then(|| vendor.to_owned())
    }
    let served = response_documents(earlier_response)
        .into_iter()
        .find_map(|document| {
            [&document["model"], &document["response"]["model"]]
                .into_iter()
                .find_map(Value::as_str)
                .map(str::to_owned)
        });
    let earlier = served
        .or_else(|| {
            earlier
                .get("model")
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
        .and_then(|model| vendor(&model));
    let later = later.get("model").and_then(Value::as_str).and_then(vendor);
    matches!((earlier, later), (Some(a), Some(b)) if a != b)
}

/// The tokens a response delivered as legacy placeholders: recorded before
/// cassettes kept provider values verbatim, so their slot cannot be proven.
pub fn legacy_tokens(dialect: Dialect, response_body: &str) -> Vec<Token> {
    response_tokens(dialect, response_body)
        .into_iter()
        .filter(|token| is_legacy_placeholder(&token.value))
        .collect()
}

fn is_legacy_placeholder(value: &str) -> bool {
    value.split_once("REDACTED_").is_some_and(|(_, counter)| {
        !counter.is_empty() && counter.bytes().all(|b| b.is_ascii_digit())
    })
}

/// Which side of a pairing a value occupies. Only tool-call ids have two.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Leg {
    /// The call, or the single slot of a one-legged kind.
    Call,
    /// The result that answers the call.
    Result,
}

/// Every value a request carries, indexed by the slot it occupies (the kind
/// of value that slot holds and the leg of the pairing), with the anchors of
/// the owners that carry it.
pub type Slots = std::collections::BTreeMap<
    (&'static str, Leg),
    std::collections::BTreeMap<String, BTreeSet<Option<String>>>,
>;

/// Every value a request carries, indexed by the slot it occupies: the kind
/// of value that slot holds and the leg of the pairing.
pub fn request_slots(request: &Value) -> Slots {
    let mut slots = std::collections::BTreeMap::new();
    collect_slots(None, request, &mut slots);
    slots
}

fn collect_slots(parent_key: Option<&str>, value: &Value, slots: &mut Slots) {
    match value {
        Value::Object(object) => {
            let kind_of = object.get("type").and_then(Value::as_str);
            let role = object.get("role").and_then(Value::as_str);
            let mut put =
                |kind: &'static str, leg: Leg, key: &str, anchors: Vec<Option<String>>| {
                    if let Some(text) = object.get(key).and_then(Value::as_str)
                        && !text.is_empty()
                    {
                        slots
                            .entry((kind, leg))
                            .or_default()
                            .entry(text.to_owned())
                            .or_default()
                            .extend(anchors);
                    }
                };
            // A streamed call names its owner by tool name alone; a whole one
            // by name and arguments. The request states both.
            let call = vec![call_anchor(object, false), call_anchor(object, true)];
            let id = object.get("id").and_then(Value::as_str).map(str::to_owned);
            match kind_of {
                // Anthropic thinking blocks, OpenRouter `reasoning.text`
                // details, Gemini Interactions thought steps.
                Some("thinking" | "reasoning.text" | "thought") => {
                    let thinking = object
                        .get("thinking")
                        .and_then(Value::as_str)
                        .map(str::to_owned);
                    put("signature", Leg::Call, "signature", vec![thinking]);
                }
                Some("redacted_thinking") => {
                    put("redacted_reasoning", Leg::Call, "data", vec![None])
                }
                // OpenAI Responses reasoning input items.
                Some("reasoning") => {
                    put("reasoning_id", Leg::Call, "id", vec![None]);
                    put("signature", Leg::Call, "signature", vec![id.clone()]);
                    put(
                        "encrypted_content",
                        Leg::Call,
                        "encrypted_content",
                        vec![id],
                    );
                }
                // OpenRouter encrypted reasoning details.
                Some("reasoning.encrypted") => {
                    put("encrypted_content", Leg::Call, "data", vec![id.clone()]);
                    put("reasoning_id", Leg::Call, "id", vec![None]);
                }
                Some("tool_use") => put("tool_call_id", Leg::Call, "id", call.clone()),
                Some("tool_result") => put("tool_call_id", Leg::Result, "tool_use_id", vec![None]),
                Some("function_call") => put("tool_call_id", Leg::Call, "call_id", call.clone()),
                Some("function_call_output" | "function_result") => {
                    put("tool_call_id", Leg::Result, "call_id", vec![None]);
                }
                _ => {}
            }
            // Chat Completions, Ollama and Cohere: calls under `tool_calls`,
            // results as `role: tool` messages.
            if parent_key == Some("tool_calls") {
                put("tool_call_id", Leg::Call, "id", call.clone());
            }
            if role == Some("tool") {
                put("tool_call_id", Leg::Result, "tool_call_id", vec![None]);
            }
            // Gemini: part-level thought signatures and optional call ids.
            put(
                "thought_signature",
                Leg::Call,
                "thoughtSignature",
                vec![part_anchor(object)],
            );
            match parent_key {
                Some("functionCall" | "function_call") => {
                    put("tool_call_id", Leg::Call, "id", call);
                }
                Some("functionResponse" | "function_response") => {
                    put("tool_call_id", Leg::Result, "id", vec![None]);
                }
                // Bedrock Converse, and its reasoning signature.
                Some("toolUse") => put("tool_call_id", Leg::Call, "toolUseId", call),
                Some("toolResult") => put("tool_call_id", Leg::Result, "toolUseId", vec![None]),
                Some("reasoningText") => put("signature", Leg::Call, "signature", vec![None]),
                _ => {}
            }
            for (key, child) in object {
                collect_slots(Some(key), child, slots);
            }
        }
        Value::Array(items) => {
            for item in items {
                collect_slots(parent_key, item, slots);
            }
        }
        _ => {}
    }
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
    // A stored Responses chain sends only the results; the calls they answer
    // live in the response `previous_response_id` names.
    let mut server_held =
        dialect == Dialect::OpenAiResponses && request.get("previous_response_id").is_some();
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
            server_held = false;
        }
        previous_assistant = assistant;
        for result in results {
            match open.iter().position(|call| *call == result) {
                Some(index) => {
                    open.remove(index);
                }
                None if server_held => {}
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
