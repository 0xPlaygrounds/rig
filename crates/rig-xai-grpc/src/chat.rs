//! The bridge between Chat Completions JSON and xAI's gRPC messages. A
//! request body built by rig-core's xAI Chat encoder becomes the proto3 JSON
//! of a `GetCompletionsRequest`, read through the proto descriptors, so a
//! field the proto does not declare is an error rather than dropped. A reply
//! becomes the Chat Completions JSON the Chat decoder reads.

use std::collections::HashSet;

use rig_core::error::EncodeError;
use serde::Deserialize;
use serde_json::{Map, Value, json};

use crate::json::from_json;
use crate::proto::{
    self, CompletionMessage, Delta, GetChatCompletionChunk, GetChatCompletionResponse,
    GetCompletionsRequest, MessageRole, SamplingUsage, ToolCall, ToolCallType,
};

fn unsendable(what: impl std::fmt::Display) -> EncodeError {
    EncodeError::request(format!("the xAI gRPC API cannot carry {what}"))
}

/// A Chat Completions request body: the fields whose spelling differs from
/// the proto's, and every other field as sent.
#[derive(Debug, Deserialize)]
pub(crate) struct ChatBody {
    messages: Option<Value>,
    tools: Option<Value>,
    tool_choice: Option<Value>,
    response_format: Option<Value>,
    reasoning_effort: Option<Value>,
    service_tier: Option<Value>,
    stop: Option<Value>,
    max_completion_tokens: Option<Value>,
    prompt_cache_key: Option<Value>,
    #[serde(flatten)]
    rest: Map<String, Value>,
}

/// The `GetCompletionsRequest` the Chat Completions body `body` describes.
///
/// # Errors
///
/// When the body holds a field, content part, tool choice or response
/// format the proto has no place for.
pub(crate) fn request(body: ChatBody) -> Result<GetCompletionsRequest, EncodeError> {
    let ChatBody {
        messages: chat_messages,
        tools: chat_tools,
        tool_choice: chat_tool_choice,
        response_format: chat_response_format,
        reasoning_effort,
        service_tier,
        stop,
        max_completion_tokens,
        prompt_cache_key,
        mut rest,
    } = body;
    if prompt_cache_key.is_some() {
        return Err(unsendable("`prompt_cache_key`"));
    }
    if let Some(tokens) = max_completion_tokens {
        rest.entry("max_tokens").or_insert(tokens);
    }
    let stop = stop.map(|stop| match stop {
        Value::String(stop) => json!([stop]),
        stop => stop,
    });
    let fields = [
        ("messages", chat_messages.map(messages).transpose()?),
        ("tools", chat_tools.map(tools).transpose()?),
        (
            "tool_choice",
            chat_tool_choice.map(tool_choice).transpose()?,
        ),
        (
            "response_format",
            chat_response_format.map(response_format).transpose()?,
        ),
        (
            "reasoning_effort",
            reasoning_effort
                .map(|effort| enumerated("EFFORT_", effort))
                .transpose()?,
        ),
        (
            "service_tier",
            service_tier
                .map(|tier| enumerated("SERVICE_TIER_", tier))
                .transpose()?,
        ),
        ("stop", stop),
    ];
    for (key, value) in fields {
        if let Some(value) = value {
            rest.insert(key.to_owned(), value);
        }
    }
    from_json(Value::Object(rest)).map_err(|error| EncodeError::request(error.to_string()))
}

/// A Chat Completions enum string as the proto enum value `prefix` names.
fn enumerated(prefix: &str, value: Value) -> Result<Value, EncodeError> {
    match value {
        Value::String(name) => Ok(Value::String(format!(
            "{prefix}{}",
            name.to_ascii_uppercase()
        ))),
        Value::Null => Ok(Value::Null),
        other => Err(unsendable(format_args!("`{other}` as an enum value"))),
    }
}

fn array(
    value: Value,
    map: impl Fn(Value) -> Result<Value, EncodeError>,
) -> Result<Value, EncodeError> {
    match value {
        Value::Array(values) => values
            .into_iter()
            .map(map)
            .collect::<Result<Vec<_>, _>>()
            .map(Value::Array),
        Value::Null => Ok(Value::Null),
        other => Err(unsendable(format_args!("`{other}` as a list"))),
    }
}

fn messages(messages: Value) -> Result<Value, EncodeError> {
    array(messages, message)
}

/// One Chat message as a proto `Message`: its role as the enum, its content
/// as content parts, and its client-side tool calls.
fn message(message: Value) -> Result<Value, EncodeError> {
    let Value::Object(mut message) = message else {
        return Err(unsendable("a message that is not an object"));
    };
    let role = match message
        .shift_remove("role")
        .as_ref()
        .and_then(Value::as_str)
    {
        Some("system") => "ROLE_SYSTEM",
        Some("developer") => "ROLE_DEVELOPER",
        Some("user") => "ROLE_USER",
        Some("assistant") => "ROLE_ASSISTANT",
        Some("tool") => "ROLE_TOOL",
        role => return Err(unsendable(format_args!("the message role {role:?}"))),
    };
    message.insert("role".to_owned(), role.into());
    let content = match message.shift_remove("content") {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::String(text)) if text.is_empty() => Vec::new(),
        Some(Value::String(text)) => vec![json!({ "text": text })],
        Some(Value::Array(parts)) => parts.into_iter().map(part).collect::<Result<_, _>>()?,
        Some(other) => return Err(unsendable(format_args!("`{other}` as message content"))),
    };
    message.insert("content".to_owned(), Value::Array(content));
    if let Some(calls) = message.shift_remove("tool_calls") {
        message.insert("tool_calls".to_owned(), array(calls, tool_call)?);
    }
    Ok(Value::Object(message))
}

/// A Chat content part as a proto `Content`.
fn part(part: Value) -> Result<Value, EncodeError> {
    let field = |key: &str| part.get(key).cloned().unwrap_or(Value::Null);
    Ok(match part.get("type").and_then(Value::as_str) {
        Some("text") => json!({ "text": field("text") }),
        Some("image_url") => {
            let image = field("image_url");
            let mut content = json!({ "image_url": image.get("url") });
            if let (Some(content), Some(detail)) = (content.as_object_mut(), image.get("detail")) {
                content.insert("detail".to_owned(), enumerated("DETAIL_", detail.clone())?);
            }
            json!({ "image_url": content })
        }
        Some("file") => json!({ "file": file(&field("file")) }),
        kind => return Err(unsendable(format_args!("a {kind:?} content part"))),
    })
}

/// A Chat `file` part as a proto `FileContent`: a file id, a data URI's
/// bytes and type, or a URL the API fetches.
fn file(file: &Value) -> Value {
    let filename = file.get("filename").cloned();
    let mut content = match (
        file.get("file_id"),
        file.get("file_data").and_then(Value::as_str),
    ) {
        (Some(id), _) => json!({ "file_id": id }),
        (None, Some(data)) => match data
            .strip_prefix("data:")
            .and_then(|uri| uri.split_once(";base64,"))
        {
            Some((mime, data)) => json!({ "data": data, "mime_type": mime }),
            None => json!({ "url": data }),
        },
        (None, None) => json!({}),
    };
    if let (Some(content), Some(filename)) = (content.as_object_mut(), filename) {
        content.insert("filename".to_owned(), filename);
    }
    content
}

/// A Chat tool call in an assistant message as a proto client-side call.
fn tool_call(call: Value) -> Result<Value, EncodeError> {
    if call.get("type").and_then(Value::as_str) != Some("function") {
        return Err(unsendable("a tool call that is not a function call"));
    }
    Ok(json!({
        "id": call.get("id"),
        "type": "TOOL_CALL_TYPE_CLIENT_SIDE_TOOL",
        "status": "TOOL_CALL_STATUS_COMPLETED",
        "function": call.get("function"),
    }))
}

/// Function tools as proto `Function`s, their schema as JSON text. Any
/// other tool is read as a proto `Tool` already, such as
/// `{"web_search": {}}`.
fn tools(tools: Value) -> Result<Value, EncodeError> {
    array(tools, |tool| {
        if tool.get("type").and_then(Value::as_str) != Some("function") {
            return Ok(tool);
        }
        let mut function = tool.get("function").cloned().unwrap_or_default();
        if let Some(function) = function.as_object_mut()
            && let Some(parameters) = function.get_mut("parameters")
            && !parameters.is_string()
        {
            *parameters = Value::String(parameters.to_string());
        }
        Ok(json!({ "function": function }))
    })
}

fn tool_choice(choice: Value) -> Result<Value, EncodeError> {
    Ok(match &choice {
        Value::String(mode) => match mode.as_str() {
            "auto" | "none" | "required" => {
                json!({ "mode": format!("TOOL_MODE_{}", mode.to_ascii_uppercase()) })
            }
            mode => return Err(unsendable(format_args!("the tool choice {mode:?}"))),
        },
        Value::Object(_) => match choice.pointer("/function/name") {
            Some(name) => json!({ "function_name": name }),
            None => return Err(unsendable(format_args!("the tool choice `{choice}`"))),
        },
        Value::Null => Value::Null,
        _ => return Err(unsendable(format_args!("the tool choice `{choice}`"))),
    })
}

fn response_format(format: Value) -> Result<Value, EncodeError> {
    Ok(match format.get("type").and_then(Value::as_str) {
        Some("text") => json!({ "format_type": "FORMAT_TYPE_TEXT" }),
        Some("json_object") => json!({ "format_type": "FORMAT_TYPE_JSON_OBJECT" }),
        Some("json_schema") => {
            let schema = format
                .pointer("/json_schema/schema")
                .map(Value::to_string)
                .ok_or_else(|| unsendable("a JSON schema response format without a schema"))?;
            json!({ "format_type": "FORMAT_TYPE_JSON_SCHEMA", "schema": schema })
        }
        _ if format.is_null() => Value::Null,
        _ => return Err(unsendable(format_args!("the response format `{format}`"))),
    })
}

/// The calls of one reply streamed in fragments, so the completed copy
/// that follows them is not read twice.
#[derive(Debug, Default)]
pub(crate) struct Calls {
    fragmented: HashSet<String>,
}

impl Calls {
    /// `call` as a Chat tool-call delta, keyed by its id: a fragment of a
    /// call streamed in pieces, or a whole call once. Server-side calls are
    /// the API's own and are not the caller's to run.
    fn delta(&mut self, call: &ToolCall) -> Option<Value> {
        if call.r#type() != ToolCallType::ClientSideTool && call.r#type() != ToolCallType::Invalid {
            return None;
        }
        if call.index.is_some() {
            self.fragmented.insert(call.id.clone());
        } else if self.fragmented.contains(&call.id) {
            return None;
        }
        let mut function = Map::new();
        if let Some(proto::tool_call::Tool::Function(called)) = &call.tool {
            if !called.name.is_empty() {
                function.insert("name".to_owned(), called.name.clone().into());
            }
            function.insert("arguments".to_owned(), called.arguments.clone().into());
        }
        Some(json!({ "id": call.id, "type": "function", "function": function }))
    }
}

/// Whether an output is the model's own message: server-side tool outputs
/// carry the tool role.
fn assistant(role: i32) -> bool {
    role == MessageRole::RoleAssistant as i32 || role == MessageRole::InvalidRole as i32
}

/// The Chat Completions message fields of an output's message or delta.
fn message_json(
    content: &str,
    reasoning: &str,
    calls: impl Iterator<Item = Value>,
) -> Map<String, Value> {
    let mut message = Map::from_iter([("role".to_owned(), Value::from("assistant"))]);
    if !content.is_empty() {
        message.insert("content".to_owned(), content.into());
    }
    if !reasoning.is_empty() {
        message.insert("reasoning_content".to_owned(), reasoning.into());
    }
    let calls: Vec<Value> = calls.collect();
    if !calls.is_empty() {
        message.insert("tool_calls".to_owned(), Value::Array(calls));
    }
    message
}

/// A proto finish reason as Chat Completions spells it; `None` before the
/// output finished.
fn finish_reason(reason: i32) -> Option<&'static str> {
    use proto::FinishReason as Reason;
    Some(match Reason::try_from(reason).ok()? {
        Reason::ReasonInvalid => return None,
        Reason::ReasonStop => "stop",
        Reason::ReasonMaxLen | Reason::ReasonMaxContext => "length",
        Reason::ReasonToolCalls => "tool_calls",
        Reason::ReasonTimeLimit => "time_limit",
    })
}

/// Usage under the REST API's field names.
fn usage(usage: &SamplingUsage) -> Value {
    let mut json = json!({
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "prompt_tokens_details": {
            "text_tokens": usage.prompt_text_tokens,
            "image_tokens": usage.prompt_image_tokens,
            "cached_tokens": usage.cached_prompt_text_tokens,
        },
        "completion_tokens_details": { "reasoning_tokens": usage.reasoning_tokens },
        "num_sources_used": usage.num_sources_used,
    });
    if let (Some(json), Some(ticks)) = (json.as_object_mut(), usage.cost_in_usd_ticks) {
        json.insert("cost_in_usd_ticks".to_owned(), ticks.into());
    }
    json
}

/// The fields a reply's envelope shares between unary and streamed.
fn envelope(
    object: &str,
    id: &str,
    model: &str,
    created: Option<&prost_types::Timestamp>,
    usage_: Option<&SamplingUsage>,
    citations: &[String],
    choices: Vec<Value>,
) -> Value {
    let mut reply = Map::from_iter([
        ("id".to_owned(), Value::from(id)),
        ("object".to_owned(), object.into()),
        ("model".to_owned(), model.into()),
        ("choices".to_owned(), Value::Array(choices)),
    ]);
    if let Some(created) = created {
        reply.insert("created".to_owned(), created.seconds.into());
    }
    if let Some(usage_) = usage_ {
        reply.insert("usage".to_owned(), usage(usage_));
    }
    if !citations.is_empty() {
        reply.insert("citations".to_owned(), citations.into());
    }
    Value::Object(reply)
}

fn choice(index: i32, key: &str, message: Map<String, Value>, finish: i32) -> Value {
    json!({
        "index": index,
        key: message,
        "finish_reason": finish_reason(finish),
    })
}

/// A unary reply as a `chat.completion`.
pub(crate) fn whole(response: &GetChatCompletionResponse) -> Value {
    let mut calls = Calls::default();
    let choices = response
        .outputs
        .iter()
        .filter_map(|output| {
            let default = CompletionMessage::default();
            let message = output.message.as_ref().unwrap_or(&default);
            let CompletionMessage {
                content,
                reasoning_content,
                role,
                tool_calls,
                ..
            } = message;
            assistant(*role).then(|| {
                let calls = tool_calls.iter().filter_map(|call| calls.delta(call));
                let message = message_json(content, reasoning_content, calls);
                choice(output.index, "message", message, output.finish_reason)
            })
        })
        .collect();
    envelope(
        "chat.completion",
        &response.id,
        &response.model,
        response.created.as_ref(),
        response.usage.as_ref(),
        &response.citations,
        choices,
    )
}

/// A streamed chunk as a `chat.completion.chunk`; `calls` holds the
/// reply's calls streamed in fragments.
pub(crate) fn chunk(chunk: &GetChatCompletionChunk, calls: &mut Calls) -> Value {
    let choices = chunk
        .outputs
        .iter()
        .filter_map(|output| {
            let default = Delta::default();
            let delta = output.delta.as_ref().unwrap_or(&default);
            let Delta {
                content,
                reasoning_content,
                role,
                tool_calls,
                ..
            } = delta;
            assistant(*role).then(|| {
                let deltas: Vec<Value> = tool_calls
                    .iter()
                    .filter_map(|call| calls.delta(call))
                    .collect();
                let message = message_json(content, reasoning_content, deltas.into_iter());
                choice(output.index, "delta", message, output.finish_reason)
            })
        })
        .collect();
    envelope(
        "chat.completion.chunk",
        &chunk.id,
        &chunk.model,
        chunk.created.as_ref(),
        chunk.usage.as_ref(),
        &chunk.citations,
        choices,
    )
}
