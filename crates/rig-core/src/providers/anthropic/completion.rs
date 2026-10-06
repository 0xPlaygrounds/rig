//! Anthropic Messages requests: model identifiers, the request body built
//! as JSON from a prepared request, prompt-cache breakpoints, and the strict
//! tool schemas Anthropic's constrained decoding takes.
//!
//! ```
//! use rig_core::providers::anthropic::completion::{CLAUDE_SONNET_4_6, CacheTtl};
//!
//! let ttl = CacheTtl::OneHour;
//! # let _ = (ttl, CLAUDE_SONNET_4_6);
//! ```

use base64::{Engine as _, prelude::BASE64_STANDARD};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use super::wire::Messages;
use crate::completion::{self, CompletionRequest, Replay};
use crate::error::EncodeError;
use crate::json_utils::Lenient;
use crate::message::{
    self, AssistantContent, DocumentMediaType, DocumentSourceKind, ImageMediaType, Message,
    ToolResultContent, UserContent,
};
use crate::providers::internal::wire_ids::WireIds;
use crate::wire::Mode;

/// Claude Fable 5.1, API ID `claude-fable-5-1`: 128K default `max_tokens`,
/// mid-conversation system messages kept in `messages`. It rejects a forced
/// tool choice, so the extractor uses native structured output instead of
/// forcing its `submit` tool.
pub const CLAUDE_FABLE_5_1: &str = "claude-fable-5-1";
/// Claude Opus 5.5, API ID `claude-opus-5-5`: 128K default `max_tokens`,
/// mid-conversation system messages kept in `messages`. It rejects a forced
/// tool choice, so the extractor uses native structured output instead of
/// forcing its `submit` tool.
pub const CLAUDE_OPUS_5_5: &str = "claude-opus-5-5";
/// Claude Sonnet 5.5, API ID `claude-sonnet-5-5`: 128K default `max_tokens`,
/// mid-conversation system messages kept in `messages` (unlike Claude
/// Sonnet 5). It rejects a forced tool choice, so the extractor uses native
/// structured output instead of forcing its `submit` tool.
pub const CLAUDE_SONNET_5_5: &str = "claude-sonnet-5-5";
/// `claude-fable-5` completion model
pub const CLAUDE_FABLE_5: &str = "claude-fable-5";
/// `claude-opus-5` completion model
pub const CLAUDE_OPUS_5: &str = "claude-opus-5";
/// `claude-sonnet-5` completion model
pub const CLAUDE_SONNET_5: &str = "claude-sonnet-5";
/// `claude-opus-4-6` completion model
pub const CLAUDE_OPUS_4_6: &str = "claude-opus-4-6";
/// `claude-opus-4-7` completion model
pub const CLAUDE_OPUS_4_7: &str = "claude-opus-4-7";
/// `claude-opus-4-8` completion model
pub const CLAUDE_OPUS_4_8: &str = "claude-opus-4-8";
/// `claude-sonnet-4-6` completion model
pub const CLAUDE_SONNET_4_6: &str = "claude-sonnet-4-6";
/// `claude-haiku-4-5` completion model
pub const CLAUDE_HAIKU_4_5: &str = "claude-haiku-4-5";

pub const ANTHROPIC_VERSION_2023_01_01: &str = "2023-01-01";
pub const ANTHROPIC_VERSION_2023_06_01: &str = "2023-06-01";
pub const ANTHROPIC_VERSION_LATEST: &str = ANTHROPIC_VERSION_2023_06_01;

/// Cache-breakpoint lifetime: five minutes by default, or one hour.
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Default)]
pub enum CacheTtl {
    /// 5-minute TTL (default).
    #[default]
    #[serde(rename = "5m")]
    FiveMinutes,
    /// 1-hour TTL.
    #[serde(rename = "1h")]
    OneHour,
}

/// What Anthropic publishes of each model with a 128K synchronous output
/// limit: whether it takes `role: "system"` inside `messages` (the
/// mid-conversation system messages page; Claude Sonnet 5 does not), whether
/// it answers a forced `tool_choice` with a 400 (the what's-new pages), and
/// whether its thinking binds to the request's tools and system prompt (the
/// recorded 400 on Claude Opus 5.5, and the models pi sends `drop_block` for).
const MODELS: [(&str, [bool; 3]); 10] = [
    (CLAUDE_FABLE_5_1, [true, true, true]),
    (CLAUDE_FABLE_5, [true, false, false]),
    (CLAUDE_OPUS_5_5, [true, true, true]),
    (CLAUDE_OPUS_5, [true, false, true]),
    (CLAUDE_SONNET_5_5, [true, true, true]),
    (CLAUDE_SONNET_5, [false; 3]),
    (CLAUDE_OPUS_4_8, [true, false, false]),
    (CLAUDE_OPUS_4_7, [false; 3]),
    (CLAUDE_OPUS_4_6, [false; 3]),
    (CLAUDE_SONNET_4_6, [false; 3]),
];

/// `model`'s row of [`MODELS`]: the model, or one of its dated snapshots
/// (`<id>-YYYYMMDD`), never a later model whose id merely starts with one.
fn listed(model: &str) -> Option<[bool; 3]> {
    MODELS.iter().find_map(|(id, flags)| {
        let rest = model.strip_prefix(id)?;
        (rest.is_empty() || rest.starts_with("-20")).then_some(*flags)
    })
}

/// The published synchronous output limit of a recognized model. Unknown
/// models require an explicit `max_tokens` value.
pub(super) fn default_max_tokens_for_model(model: &str) -> Option<u64> {
    let older = ["claude-opus-4", "claude-sonnet-4", "claude-haiku-4-5"];
    match listed(model) {
        Some(_) => Some(128_000),
        None => older
            .iter()
            .any(|prefix| model.starts_with(prefix))
            .then_some(64_000),
    }
}

/// Whether `model` rejects a forced tool choice.
pub(super) fn rejects_forced_tool_choice(model: &str) -> bool {
    listed(model).is_some_and(|[_, rejects, _]| rejects)
}

/// Whether `model` takes `role: "system"` inside `messages` (the
/// mid-conversation system messages page). A model the table does not list
/// takes system text only in the request's `system`.
pub(super) fn takes_mid_conversation_system(model: &str) -> bool {
    listed(model).is_some_and(|[mid, _, _]| mid)
}

/// Whether the Claude `model` binds its thinking blocks to the request's
/// tools and system prompt, however the serving API spells it: Anthropic's
/// `claude-opus-5-5`, OpenRouter's `anthropic/claude-opus-5.5`, or Bedrock's
/// `us.anthropic.claude-opus-5-5-v1:0`. Every wire that serves Claude reads
/// this one table.
pub fn binds_context(model: &str) -> bool {
    let model = model
        .rsplit_once("anthropic.")
        .map_or(model, |(_, rest)| rest);
    let model = model.strip_prefix("anthropic/").unwrap_or(model);
    let model = model.split_once("-v1:").map_or(model, |(id, _)| id);
    listed(&model.replace('.', "-")).is_some_and(|[_, _, binds]| binds)
}

/// The beta that lets a request ask Anthropic to drop a thinking block bound
/// to another context instead of rejecting the request.
pub(super) const THINKING_BINDING_BETA: &str = "thinking-binding-controls-2026-08-01";

/// The beta flag that streams tool input as it is written, for a provider
/// that rejects the per-tool `eager_input_streaming` field.
pub(super) const FINE_GRAINED_TOOL_STREAMING_BETA: &str = "fine-grained-tool-streaming-2025-05-14";

/// Whether a request to `model` on `wire`, with `thinking` as the caller
/// sets it, asks Anthropic to drop thinking bound to another context (pi's
/// `drop_block`): Anthropic's own API, for a model whose thinking binds, in
/// adaptive thinking (the default when the caller names none), the only
/// mode that takes the binding. Its turns then replay verbatim whatever the
/// context. In another mode (`disabled`, `between_tools`) a turn made under
/// another context replays as another model's.
pub(super) fn drops_unbound_thinking(
    wire: &Messages,
    model: &str,
    thinking: Option<&Value>,
) -> bool {
    let adaptive = thinking.is_none_or(|thinking| thinking.str("type") == Some("adaptive"));
    wire.provider.dialect.name == super::ANTHROPIC.name && binds_context(model) && adaptive
}

/// `body`'s `thinking` with `drop_block` set: the caller's adaptive
/// settings, or the model's default adaptive thinking when it names none.
pub(crate) fn drop_unbound_thinking(body: &mut Map<String, Value>) {
    let binding = json!({ "prefix_mismatch_behavior": "drop_block" });
    match body.get_mut("thinking") {
        Some(Value::Object(thinking)) => {
            thinking.entry("block_binding").or_insert(binding);
        }
        Some(_) => {}
        None => {
            body.insert(
                "thinking".into(),
                json!({ "type": "adaptive", "block_binding": binding }),
            );
        }
    }
}

/// The Messages request body for `request`, already prepared, on `wire`.
///
/// # Errors
///
/// When no `max_tokens` applies, `additional_params` is not an object or
/// carries malformed `tools` or `cache_control`, the caching settings
/// conflict, or a part has a form the wire's `encodes` refuses.
pub(super) fn body(
    wire: &Messages,
    request: CompletionRequest,
    mode: Mode,
) -> Result<Value, EncodeError> {
    let model = request.model.clone().unwrap_or_else(|| wire.model.clone());
    // A request that addresses another model gets that model's default; the
    // wire's own model keeps the one it was built with.
    let max_tokens = match request.max_tokens {
        Some(tokens) => Some(tokens),
        None if model == wire.model => wire.default_max_tokens,
        None => wire
            .provider
            .dialect
            .default_max_tokens(&model)
            .or(wire.default_max_tokens),
    }
    .ok_or_else(|| EncodeError::request("`max_tokens` must be set for Anthropic"))?;
    let mut params = match request.additional_params {
        None | Some(Value::Null) => Map::new(),
        Some(Value::Object(params)) => params,
        Some(_) => {
            return Err(EncodeError::request(
                "Anthropic `additional_params` must be a JSON object",
            ));
        }
    };
    let top = top_level_cache_control(wire, &mut params)?;
    let strict = wire.strict_tools && wire.provider.dialect.quirks.strict_tool_schemas;
    let eager = wire.tool_input_streaming == super::wire::ToolInputStreaming::Eager;
    let mut tools = tools(request.tools, &mut params, strict, eager)?;
    let (mut system, history) =
        split_system(&request.chat_history, takes_mid_conversation_system(&model));
    let ids = WireIds::for_target(&history, wire, &model);
    let mut messages: Vec<Value> = Vec::new();
    for message in &history {
        messages.extend(message_json(message, wire, &ids)?);
    }
    apply_cache_control(wire, top.as_ref(), &mut system, &mut messages, &mut tools)?;
    let has_tools = !tools.is_empty();
    let output_config = request.output_schema.map(|schema| {
        let mut schema = schema.to_value();
        sanitize_schema(&mut schema);
        json!({ "format": { "type": "json_schema", "schema": schema } })
    });
    let container = (!params.contains_key("container"))
        .then(|| container(&history))
        .flatten();
    let mut body = object([
        ("model", Some(json!(model))),
        ("messages", Some(Value::Array(messages))),
        ("max_tokens", Some(json!(max_tokens))),
        ("system", (!system.is_empty()).then(|| Value::Array(system))),
        (
            "temperature",
            request.temperature.map(|temperature| json!(temperature)),
        ),
        (
            "tool_choice",
            request.tool_choice.map(tool_choice).transpose()?,
        ),
        ("tools", has_tools.then(|| Value::Array(tools))),
        ("output_config", output_config),
        ("container", container.map(Value::String)),
    ]);
    body.extend(params);
    if drops_unbound_thinking(wire, &model, body.get("thinking")) {
        drop_unbound_thinking(&mut body);
    }
    if let Some(top) = top {
        body.insert("cache_control".into(), top);
    }
    if mode == Mode::Streaming {
        body.insert("stream".into(), Value::Bool(true));
        // Anthropic rejects tool_choice without tools.
        if has_tools {
            body.entry("tool_choice")
                .or_insert_with(|| json!({ "type": "auto" }));
        } else {
            body.shift_remove("tool_choice");
        }
    }
    Ok(Value::Object(body))
}

/// A text content block.
fn text(text: &str) -> Value {
    json!({ "type": "text", "text": text })
}

/// An object of the `fields` that are present, in order.
pub(super) fn object<const N: usize>(fields: [(&str, Option<Value>); N]) -> Map<String, Value> {
    fields
        .into_iter()
        .filter_map(|(key, value)| Some((key.to_owned(), value?)))
        .collect()
}

/// `message` on the wire, its tool ids spelled by `ids`. `None` when no
/// block is left to send.
fn message_json(
    message: &Message,
    target: &Messages,
    ids: &WireIds,
) -> Result<Option<Value>, EncodeError> {
    let (role, content): (&str, Vec<Value>) = match message {
        Message::System { content } => ("system", vec![text(content)]),
        Message::User { content } => {
            let mut parts = Vec::with_capacity(content.len());
            for part in content {
                parts.extend(user_part(part, ids)?);
            }
            ("user", parts)
        }
        Message::Assistant(turn) => (
            "assistant",
            turn.content
                .iter()
                .filter_map(|block| assistant_part(block, target, ids))
                .collect(),
        ),
    };
    Ok((!content.is_empty()).then(|| json!({ "role": role, "content": content })))
}

/// One user block on the wire. `None` for blank text, which Anthropic
/// rejects (pi's rule).
fn user_part(part: &UserContent, ids: &WireIds) -> Result<Option<Value>, EncodeError> {
    let image = |image: &message::Image| {
        let source = image_source(image).ok_or_else(unsendable)?;
        Ok::<_, EncodeError>(json!({ "type": "image", "source": source }))
    };
    Ok(Some(match part {
        UserContent::Text(part) if part.text.trim().is_empty() => return Ok(None),
        UserContent::Text(part) => text(&part.text),
        UserContent::ToolResult(result) => {
            let mut content = Vec::with_capacity(result.content.len());
            for part in &result.content {
                content.push(match part {
                    ToolResultContent::Text(part) => text(&part.text),
                    ToolResultContent::Json { value } => text(&value.to_string()),
                    ToolResultContent::Image(part) => image(part)?,
                });
            }
            Value::Object(object([
                ("type", Some(json!("tool_result"))),
                ("tool_use_id", Some(json!(ids.spell(&result.call)))),
                ("content", Some(Value::Array(content))),
                ("is_error", result.is_error.then_some(Value::Bool(true))),
            ]))
        }
        UserContent::Image(part) => image(part)?,
        UserContent::Document(document) => document_part(document)?,
        UserContent::Audio(_) | UserContent::Video(_) => return Err(unsendable()),
    }))
}

/// A document block: its source, and the `title`, `context` and
/// `citations` its `additional_params` name.
fn document_part(document: &message::Document) -> Result<Value, EncodeError> {
    let params = document.additional_params.as_ref();
    let param = |key: &str| params.and_then(|params| params.get(key));
    let text = |key: &str| param(key).filter(|value| value.is_string()).cloned();
    let citations = match param("citations") {
        None => None,
        Some(citations) => Some(json!({ "enabled": citations.bool("enabled").ok_or_else(|| {
            EncodeError::request("Document `additional_params.citations` must be `{\"enabled\": bool}`")
        })? })),
    };
    Ok(Value::Object(object([
        ("type", Some(json!("document"))),
        (
            "source",
            Some(document_source(document).ok_or_else(unsendable)?),
        ),
        ("title", text("title")),
        ("context", text("context")),
        ("citations", citations),
    ])))
}

/// One assistant block on the wire: the provider's item while it is
/// current, else the block rebuilt from its canonical fields as pi rebuilds
/// it, with the identity keys of an edited item. `None` for a block with
/// nothing Anthropic takes.
pub(super) fn assistant_part(
    block: &AssistantContent,
    target: &Messages,
    ids: &WireIds,
) -> Option<Value> {
    if let AssistantContent::Opaque(opaque) = block {
        // The reply's container is conversation state, sent as the
        // request's `container`, never as content.
        return (opaque.kind() != Some("container")).then(|| opaque.item.clone());
    }
    let identity = match block.replay(target, ids) {
        Replay::Item(item) => return Some(item.into_owned()),
        Replay::Identity(identity) => identity,
        Replay::Rebuild => Map::new(),
    };
    Some(match block {
        AssistantContent::Text(part) => text(&part.text),
        // Thinking without its signature is text, as pi sends it. A
        // redacted block's payload lives only in its item.
        AssistantContent::Reasoning(reasoning) => {
            if reasoning.redacted || reasoning.text.trim().is_empty() {
                return None;
            }
            text(&reasoning.text)
        }
        AssistantContent::ToolCall(call) => {
            let mut part = object([
                ("type", Some(json!("tool_use"))),
                ("id", Some(json!(ids.spell(&call.id)))),
                ("name", Some(json!(call.function.name.as_str()))),
                (
                    "input",
                    Some(Value::Object(call.function.arguments.clone())),
                ),
            ]);
            part.extend(identity);
            Value::Object(part)
        }
        // Assistant turns take no images; `adapt` downgrades every one, so
        // only a turn sent without it reaches here.
        AssistantContent::Image(_) => text(completion::history::ASSISTANT_IMAGE_OMITTED),
        AssistantContent::Opaque(_) => return None,
    })
}

/// The error for media [`Messages`]'s `encodes` refuses, which `adapt`
/// replaces before a prepared request reaches the encoder.
fn unsendable() -> EncodeError {
    EncodeError::request("Anthropic cannot receive this media in its form")
}

/// An image's source on the wire, in a user turn or a tool result: base64
/// data of a type Anthropic reads, a URL, or a Files API id. `None` for any
/// other form.
pub(super) fn image_source(image: &message::Image) -> Option<Value> {
    Some(match &image.data {
        DocumentSourceKind::Base64(data) => {
            let media_type = image.media_type.as_ref().filter(|media_type| {
                use ImageMediaType::{GIF, JPEG, PNG, WEBP};
                matches!(media_type, JPEG | PNG | GIF | WEBP)
            })?;
            let media_type = message::MimeType::to_mime_type(media_type);
            json!({ "type": "base64", "media_type": media_type, "data": data })
        }
        DocumentSourceKind::Url(url) => json!({ "type": "url", "url": url }),
        DocumentSourceKind::FileId(file_id) => json!({ "type": "file", "file_id": file_id }),
        DocumentSourceKind::Raw(_)
        | DocumentSourceKind::String(_)
        | DocumentSourceKind::Unknown => {
            return None;
        }
    })
}

/// A document's source on the wire: a Files API id, a PDF as data or by
/// URL, or any other document as the text it holds. `None` for any other
/// form.
pub(super) fn document_source(document: &message::Document) -> Option<Value> {
    let text = |data: &str| json!({ "type": "text", "media_type": "text/plain", "data": data });
    Some(match (&document.data, &document.media_type) {
        (DocumentSourceKind::FileId(file_id), _) => json!({ "type": "file", "file_id": file_id }),
        // Anthropic's URL source is defined for PDFs and has no media-type
        // field, so an untyped URL is one.
        (DocumentSourceKind::Url(url), None | Some(DocumentMediaType::PDF)) => {
            json!({ "type": "url", "url": url })
        }
        (
            DocumentSourceKind::Base64(data) | DocumentSourceKind::String(data),
            Some(DocumentMediaType::PDF),
        ) => json!({ "type": "base64", "media_type": "application/pdf", "data": data }),
        (DocumentSourceKind::String(data), _) => text(data),
        (DocumentSourceKind::Base64(data), Some(_)) => {
            let bytes = BASE64_STANDARD.decode(data).ok()?;
            text(&String::from_utf8(bytes).ok()?)
        }
        _ => return None,
    })
}

/// A `tool_choice` on the wire.
fn tool_choice(choice: message::ToolChoice) -> Result<Value, EncodeError> {
    Ok(match choice {
        message::ToolChoice::Auto => json!({ "type": "auto" }),
        message::ToolChoice::None => json!({ "type": "none" }),
        message::ToolChoice::Required => json!({ "type": "any" }),
        message::ToolChoice::Specific { function_names } => match function_names.as_slice() {
            [name] => json!({ "type": "tool", "name": name.as_str() }),
            _ => {
                return Err(EncodeError::request(
                    "Only one tool may be specified to be used by Claude",
                ));
            }
        },
    })
}

/// The request's tools: Rig's, strict when `strict` and asking for their
/// input as it is written when `eager`, then those
/// `additional_params.tools` names, verbatim.
fn tools(
    tools: Vec<completion::ToolDefinition>,
    params: &mut Map<String, Value>,
    strict: bool,
    eager: bool,
) -> Result<Vec<Value>, EncodeError> {
    let extra = match params.shift_remove("tools") {
        None => Vec::new(),
        Some(Value::Array(tools)) => tools,
        Some(_) => {
            return Err(EncodeError::request(
                "Invalid Anthropic `additional_params.tools` payload: expected an array",
            ));
        }
    };
    let rig = tools.into_iter().map(|tool| {
        let mut schema = tool.parameters;
        if strict {
            sanitize_strict_tool_schema(&mut schema);
        }
        Value::Object(object([
            ("name", Some(json!(tool.name.as_str()))),
            ("description", Some(json!(tool.description))),
            ("input_schema", Some(schema)),
            ("strict", strict.then_some(Value::Bool(true))),
            ("eager_input_streaming", eager.then_some(Value::Bool(true))),
        ]))
    });
    Ok(rig.chain(extra).collect())
}

/// Split `history` into the top-level `system` blocks and the messages.
///
/// On a model that takes mid-conversation system messages, one in a position
/// Anthropic rejects (after an assistant turn, or before a user turn) moves to
/// the next valid slot: right after the next user turn that ends the array or
/// precedes an assistant turn. Hoisting it into `system` instead would change
/// the prompt prefix, which misses the cache from the first token and, on
/// models that bind thinking blocks to their conversation, turns every earlier
/// thinking block into a 400. It is hoisted only when no such slot exists.
pub(super) fn split_system(history: &[Message], mid: bool) -> (Vec<Value>, Vec<Message>) {
    let mut system = Vec::new();
    let mut remaining = Vec::new();
    let mut deferred: Vec<&str> = Vec::new();
    let lead = history
        .iter()
        .take_while(|message| matches!(message, Message::System { .. }))
        .count();
    for (index, message) in history.iter().enumerate() {
        match message {
            Message::System { content } if content.is_empty() => {}
            // The system messages that lead the history are the prompt.
            Message::System { content } if index < lead => system.push(text(content)),
            Message::System { .. } if mid && valid_system_message(history, index) => {
                remaining.push(message.clone());
            }
            Message::System { content }
                if mid
                    && index > 0
                    && (index + 1..history.len()).any(|slot| system_slot(history, slot)) =>
            {
                deferred.push(content);
            }
            Message::System { content } => system.push(text(content)),
            other => {
                remaining.push(other.clone());
                if !deferred.is_empty() && system_slot(history, index) {
                    remaining.push(Message::System {
                        content: deferred.join("\n\n"),
                    });
                    deferred.clear();
                }
            }
        }
    }
    (system, remaining)
}

/// Whether a system message may sit right after `history[index]`: it is a
/// user turn, and the next turn that is not a system message is an
/// assistant turn or there is none, since Anthropic takes several system
/// messages in a row there.
fn system_slot(history: &[Message], index: usize) -> bool {
    matches!(history.get(index), Some(Message::User { .. }))
        && history
            .get(index + 1..)
            .into_iter()
            .flatten()
            .find(|message| !matches!(message, Message::System { .. }))
            .is_none_or(|message| matches!(message, Message::Assistant(_)))
}

/// Whether the system message at `index` sits where Anthropic takes one:
/// after a user turn, or an assistant turn ending in a server tool's result,
/// and before an assistant turn or the end.
fn valid_system_message(history: &[Message], index: usize) -> bool {
    // A run of system messages shares one slot.
    let not_system = |message: &&Message| !matches!(message, Message::System { .. });
    let after = history
        .get(..index)
        .into_iter()
        .flatten()
        .rev()
        .find(not_system);
    let follows = match after {
        Some(Message::User { .. }) => true,
        // The `container` block goes to the request's top level, not the turn.
        Some(Message::Assistant(turn)) => matches!(
            turn.content.iter().rev().find(|block| !matches!(
                block,
                AssistantContent::Opaque(opaque) if opaque.kind() == Some("container")
            )),
            Some(AssistantContent::Opaque(opaque))
                if opaque.kind().is_some_and(|kind| kind.ends_with("_tool_result"))
        ),
        Some(Message::System { .. }) | None => false,
    };
    follows
        && history
            .get(index + 1..)
            .into_iter()
            .flatten()
            .find(not_system)
            .is_none_or(|message| matches!(message, Message::Assistant(_)))
}

/// The id of the container the last turn holding one ran in. Anthropic
/// requires it on a request that answers a programmatic tool call, and it
/// keeps a code-execution session's state. The decoder keeps it as an
/// opaque `container` block, which `adapt` leaves only on turns of the
/// same model.
fn container(history: &[Message]) -> Option<String> {
    history
        .iter()
        .rev()
        .filter_map(|message| match message {
            Message::Assistant(turn) => Some(turn),
            Message::User { .. } | Message::System { .. } => None,
        })
        .flat_map(|turn| turn.content.iter().rev())
        .find_map(|block| match block {
            AssistantContent::Opaque(opaque) if opaque.kind() == Some("container") => {
                opaque.item.at("/container/id")?.as_str().map(str::to_owned)
            }
            _ => None,
        })
}

/// A `cache_control` marker.
fn ephemeral(ttl: Option<&CacheTtl>) -> Value {
    match ttl {
        Some(ttl) => json!({ "type": "ephemeral", "ttl": ttl }),
        None => json!({ "type": "ephemeral" }),
    }
}

/// Whether a `cache_control` marker has the one-hour TTL.
fn is_1h(marker: &Value) -> bool {
    marker.str("ttl") == Some("1h")
}

/// The most `cache_control` markers Anthropic takes in one request.
const MAX_CACHE_CONTROL_MARKERS: usize = 4;

/// The request's top-level `cache_control`: the one `additional_params`
/// names, taken out of them, or the automatic caching setting's.
fn top_level_cache_control(
    wire: &Messages,
    params: &mut Map<String, Value>,
) -> Result<Option<Value>, EncodeError> {
    let raw = match params.shift_remove("cache_control") {
        None | Some(Value::Null) => None,
        Some(raw) => {
            match Option::<CacheTtl>::deserialize(raw.get("ttl").unwrap_or(&Value::Null)) {
                Ok(ttl) if raw.str("type") == Some("ephemeral") => Some(ephemeral(ttl.as_ref())),
                _ => {
                    return Err(EncodeError::request(format!(
                        "Invalid Anthropic `additional_params.cache_control` payload: {raw}"
                    )));
                }
            }
        }
    };
    let typed = wire
        .automatic_caching
        .then(|| ephemeral(wire.automatic_caching_ttl.as_ref()));
    match (typed, raw) {
        (Some(typed), Some(raw))
            if wire.automatic_caching_ttl.is_some() && is_1h(&typed) != is_1h(&raw) =>
        {
            Err(EncodeError::request(
                "Anthropic `additional_params.cache_control` conflicts with the typed \
                 automatic caching TTL",
            ))
        }
        (typed, raw) => Ok(raw.or(typed)),
    }
}

/// Place the request's cache breakpoints within Anthropic's budget of four,
/// one fewer with a top-level marker. Manual prompt caching marks the final
/// non-deferred tool, the system prompt and the last block of the last user
/// or system message; a static-prefix TTL alone marks only the first two.
/// Markers a tool already carries are kept and count toward the budget, and
/// every one-hour marker must precede the five-minute ones.
fn apply_cache_control(
    wire: &Messages,
    top: Option<&Value>,
    system: &mut [Value],
    messages: &mut [Value],
    tools: &mut [Value],
) -> Result<(), EncodeError> {
    for tool in tools.iter_mut().filter_map(Value::as_object_mut) {
        if tool.get("cache_control").is_some_and(Value::is_null) {
            tool.shift_remove("cache_control");
        }
    }
    let budget = MAX_CACHE_CONTROL_MARKERS - usize::from(top.is_some());
    let marked = tools
        .iter()
        .filter(|tool| tool.get("cache_control").is_some())
        .count();
    let Some(mut remaining) = budget.checked_sub(marked) else {
        return Err(EncodeError::request(format!(
            "Too many Anthropic tool `cache_control` markers: {marked} exceeds the available \
             prompt caching budget of {budget}"
        )));
    };
    let top_ttl = top.and_then(|top| CacheTtl::deserialize(top.get("ttl")?).ok());
    if wire.static_prefix_cache_ttl == Some(CacheTtl::FiveMinutes)
        && top_ttl == Some(CacheTtl::OneHour)
    {
        return Err(EncodeError::request(
            "`with_static_prefix_cache_ttl(CacheTtl::FiveMinutes)` conflicts with the 1-hour \
             top-level cache TTL (`with_automatic_caching_1h` or a raw top-level \
             `cache_control`): Anthropic requires 1h markers to precede 5-minute ones, and the \
             static prefix precedes the conversation tail",
        ));
    }
    if wire.prompt_caching || wire.static_prefix_cache_ttl.is_some() {
        let marker = ephemeral(wire.static_prefix_cache_ttl.as_ref().or(top_ttl.as_ref()));
        let last_tool = tools
            .iter_mut()
            .filter_map(Value::as_object_mut)
            .rfind(|tool| tool.get("defer_loading") != Some(&Value::Bool(true)));
        if let Some(tool) = last_tool
            && !tool.contains_key("cache_control")
        {
            if remaining == 0 {
                return Err(EncodeError::request(
                    "Anthropic manual prompt caching requires a cache_control marker on the \
                     final non-deferred tool, but explicit tool markers exhaust the available \
                     cache point budget",
                ));
            }
            tool.insert("cache_control".into(), marker.clone());
            remaining -= 1;
        }
        if remaining > 0
            && let Some(Value::Object(block)) = system.last_mut()
            && !block.contains_key("cache_control")
        {
            block.insert("cache_control".into(), marker);
            remaining -= 1;
        }
    }
    if wire.prompt_caching && top.is_none() && remaining > 0 {
        let block = messages
            .last_mut()
            .filter(|message| message.str("role") != Some("assistant"))
            .and_then(|message| message.get_mut("content")?.as_array_mut()?.last_mut())
            .and_then(Value::as_object_mut);
        if let Some(block) = block {
            block.insert("cache_control".into(), ephemeral(None));
        }
    }
    let mut short_seen = false;
    let markers = tools
        .iter()
        .chain(system.iter())
        .chain(messages.iter().flat_map(|message| message.arr("content")))
        .filter_map(|block| block.get("cache_control"))
        .chain(top);
    for marker in markers {
        if !is_1h(marker) {
            short_seen = true;
        } else if short_seen {
            return Err(EncodeError::request(
                "Anthropic cache_control markers with ttl `1h` must appear before markers with \
                 the default 5-minute TTL",
            ));
        }
    }
    Ok(())
}

/// Require all object properties, disallow additional properties, and remove
/// numeric constraints for Anthropic structured output.
fn sanitize_schema(schema: &mut Value) {
    crate::providers::internal::schema::sanitize_schema(
        schema,
        crate::providers::internal::schema::SanitizeOptions {
            strip_ref_siblings: false,
            inject_empty_properties: false,
            strip_numeric_constraints: true,
        },
    );
}

/// Adapt a strict tool schema using Anthropic's SDK transformation policy.
///
/// Strict tools support optional parameters, so declared `required` lists are
/// preserved. Unsupported validation keywords are moved into descriptions as
/// model guidance instead of reaching the constrained-decoding compiler. A
/// local root `$ref` is inlined and a root `allOf` flattened, since Anthropic
/// rejects both at the top of a tool input, which must be `type: object`.
pub(super) fn sanitize_strict_tool_schema(schema: &mut Value) {
    let mut original = std::mem::take(schema);
    inline_local_root_reference(&mut original);
    if let Value::Object(root) = &mut original {
        if let Some(all_of) = root.shift_remove("allOf") {
            let mut conflicts = Map::new();
            merge_all_of(root, all_of, &mut conflicts);
            if !conflicts.is_empty() {
                root.insert("rootAllOfConstraints".into(), Value::Object(conflicts));
            }
        }
        if !root.contains_key("type")
            && (root.contains_key("properties") || root.contains_key("$ref"))
        {
            root.insert("type".into(), json!("object"));
        }
    }
    *schema = strict_schema(original);
}

/// The keywords that hold a schema's definitions.
const DEFINITIONS: [&str; 2] = ["$defs", "definitions"];

/// The string formats Anthropic's strict schemas take.
const STRICT_FORMATS: [&str; 10] = [
    "date-time",
    "time",
    "date",
    "duration",
    "email",
    "hostname",
    "uri",
    "ipv4",
    "ipv6",
    "uuid",
];

/// Resolve a local root `$ref` into the root, keeping the definitions nested
/// references need and the root's sibling keywords, which JSON Schema
/// applies conjunctively.
fn inline_local_root_reference(schema: &mut Value) {
    let mut seen = std::collections::BTreeSet::new();
    while let Some(reference) = schema
        .get("$ref")
        .and_then(Value::as_str)
        .map(str::to_owned)
    {
        let target = reference
            .strip_prefix('#')
            .and_then(|pointer| schema.pointer(pointer));
        let (Some(Value::Object(mut referenced)), Some(mut root)) =
            (target.cloned(), schema.as_object().cloned())
        else {
            return;
        };
        if !seen.insert(reference) {
            return;
        }
        root.shift_remove("$ref");
        for keyword in DEFINITIONS {
            if let Some(definitions) = root.shift_remove(keyword) {
                let merged = merge_definitions(definitions, referenced.shift_remove(keyword));
                referenced.insert(keyword.into(), merged);
            }
        }
        merge_siblings(&mut referenced, root);
        *schema = Value::Object(referenced);
    }
}

/// The root's definitions over the inlined schema's: absolute pointers
/// still resolve from the document root.
fn merge_definitions(root: Value, local: Option<Value>) -> Value {
    match (root, local) {
        (Value::Object(root), Some(Value::Object(mut local))) => {
            local.extend(root);
            Value::Object(local)
        }
        (root, _) => root,
    }
}

/// Merge keywords beside a root `$ref` into its resolved object. Unions and
/// conflicting constraints, which a root cannot hold, stay as guidance.
fn merge_siblings(referenced: &mut Map<String, Value>, siblings: Map<String, Value>) {
    let mut conflicts = Map::new();
    for (keyword, sibling) in siblings {
        match keyword.as_str() {
            "properties" => merge_properties(referenced, sibling),
            "required" => merge_required(referenced, sibling),
            "allOf" => merge_all_of(referenced, sibling, &mut conflicts),
            "anyOf" | "oneOf" => {
                conflicts.insert(keyword, sibling);
            }
            // Annotations of the root document: the root's wins.
            "description" | "title" | "$schema" | "$id" | "$comment" | "default" | "examples"
            | "deprecated" | "readOnly" | "writeOnly" => {
                referenced.insert(keyword, sibling);
            }
            _ => match referenced.get(&keyword) {
                None => {
                    referenced.insert(keyword, sibling);
                }
                Some(existing) if *existing == sibling => {}
                Some(_) => {
                    conflicts.insert(keyword, sibling);
                }
            },
        }
    }
    if !conflicts.is_empty() {
        referenced.insert("rootRefSiblingConstraints".into(), Value::Object(conflicts));
    }
}

/// Merge each object branch of a root `allOf` into `schema`; branches that
/// are not objects stay as guidance.
fn merge_all_of(
    schema: &mut Map<String, Value>,
    sibling: Value,
    conflicts: &mut Map<String, Value>,
) {
    let Value::Array(branches) = sibling else {
        conflicts.insert("allOf".into(), sibling);
        return;
    };
    let mut unsupported = Vec::new();
    for branch in branches {
        let Value::Object(mut branch) = branch else {
            unsupported.push(branch);
            continue;
        };
        if branch.contains_key("$ref") {
            for keyword in DEFINITIONS {
                if let Some(definitions) = schema.get(keyword).cloned() {
                    let merged = merge_definitions(definitions, branch.shift_remove(keyword));
                    branch.insert(keyword.into(), merged);
                }
            }
            let mut inlined = Value::Object(branch);
            inline_local_root_reference(&mut inlined);
            if let Value::Object(inlined) = inlined {
                merge_siblings(schema, inlined);
            }
        } else {
            merge_siblings(schema, branch);
        }
    }
    if !unsupported.is_empty() {
        conflicts.insert("allOf".into(), Value::Array(unsupported));
    }
}

/// Merge sibling properties; a property both define must meet both.
fn merge_properties(schema: &mut Map<String, Value>, sibling: Value) {
    let Value::Object(siblings) = sibling else {
        schema.entry("properties").or_insert(sibling);
        return;
    };
    let Value::Object(properties) = schema.entry("properties").or_insert_with(|| json!({})) else {
        return;
    };
    for (name, sibling) in siblings {
        let merged = match properties.shift_remove(&name) {
            None => sibling,
            Some(existing) if existing == sibling => existing,
            Some(existing) => json!({ "allOf": [existing, sibling] }),
        };
        properties.insert(name, merged);
    }
}

/// Merge sibling `required` names, each once.
fn merge_required(schema: &mut Map<String, Value>, sibling: Value) {
    let Value::Array(names) = sibling else {
        schema.entry("required").or_insert(sibling);
        return;
    };
    let Value::Array(required) = schema.entry("required").or_insert_with(|| json!([])) else {
        return;
    };
    for name in names {
        if !required.contains(&name) {
            required.push(name);
        }
    }
}

/// `schema` in the subset strict tool use compiles: objects closed, the
/// supported keywords kept, and every other one appended to the
/// description as `{keyword: value, ...}`.
fn strict_schema(schema: Value) -> Value {
    let Value::Object(mut source) = schema else {
        return schema;
    };
    let each = |schemas: Map<String, Value>| {
        Value::Object(
            schemas
                .into_iter()
                .map(|(name, schema)| (name, strict_schema(schema)))
                .collect(),
        )
    };
    let mut strict = Map::new();
    for keyword in DEFINITIONS {
        match source.shift_remove(keyword) {
            Some(Value::Object(definitions)) => {
                strict.insert(keyword.into(), each(definitions));
            }
            Some(definitions) => {
                source.insert(keyword.into(), definitions);
            }
            None => {}
        }
    }
    if let Some(reference) = source.shift_remove("$ref") {
        strict.insert("$ref".into(), reference);
        return Value::Object(strict);
    }
    let kind = source.shift_remove("type");
    let is = |expected: &str| match &kind {
        Some(Value::String(kind)) => kind == expected,
        Some(Value::Array(kinds)) => kinds.iter().any(|kind| kind.as_str() == Some(expected)),
        _ => false,
    };
    let alternatives = match (
        source.shift_remove("anyOf"),
        source.shift_remove("oneOf"),
        source.shift_remove("allOf"),
    ) {
        (Some(Value::Array(variants)), _, _) | (_, Some(Value::Array(variants)), _) => {
            Some(("anyOf", variants))
        }
        (_, _, Some(Value::Array(variants))) => Some(("allOf", variants)),
        _ => None,
    };
    match (alternatives, &kind) {
        (Some((keyword, variants)), _) => {
            let variants = variants.into_iter().map(strict_schema).collect();
            strict.insert(keyword.into(), Value::Array(variants));
        }
        (None, Some(kind)) => {
            strict.insert("type".into(), kind.clone());
        }
        (None, None) => {}
    }
    if let Some(Value::Array(values)) = source.shift_remove("enum") {
        strict.insert("enum".into(), Value::Array(values));
    }
    if let Some(constant) = source.shift_remove("const") {
        strict.insert("const".into(), constant);
    }
    for keyword in ["description", "title"] {
        if let Some(Value::String(value)) = source.shift_remove(keyword) {
            strict.insert(keyword.into(), Value::String(value));
        }
    }
    let has_properties = source.contains_key("properties");
    if kind.is_none() && has_properties {
        strict.insert("type".into(), json!("object"));
    }
    if is("object") || has_properties {
        let properties = match source.shift_remove("properties") {
            Some(Value::Object(properties)) => each(properties),
            _ => json!({}),
        };
        strict.insert("properties".into(), properties);
        source.shift_remove("additionalProperties");
        strict.insert("additionalProperties".into(), Value::Bool(false));
        if let Some(Value::Array(required)) = source.shift_remove("required") {
            strict.insert("required".into(), Value::Array(required));
        }
    }
    if is("string")
        && let Some(format) = source.shift_remove("format")
    {
        if format
            .as_str()
            .is_some_and(|format| STRICT_FORMATS.contains(&format))
        {
            strict.insert("format".into(), format);
        } else {
            source.insert("format".into(), format);
        }
    }
    if is("array") {
        if let Some(items) = source.shift_remove("items") {
            strict.insert("items".into(), strict_schema(items));
        }
        if let Some(min_items) = source.shift_remove("minItems") {
            if matches!(min_items.as_u64(), Some(0 | 1)) {
                strict.insert("minItems".into(), min_items);
            } else {
                source.insert("minItems".into(), min_items);
            }
        }
    }
    if !source.is_empty() {
        let hints: Vec<String> = source
            .into_iter()
            .map(|(keyword, value)| match value {
                Value::String(value) => format!("{keyword}: {value}"),
                value => format!("{keyword}: {value}"),
            })
            .collect();
        let suffix = format!("{{{}}}", hints.join(", "));
        match strict.get_mut("description") {
            Some(Value::String(description)) => {
                description.push_str("\n\n");
                description.push_str(&suffix);
            }
            _ => {
                strict.insert("description".into(), Value::String(suffix));
            }
        }
    }
    Value::Object(strict)
}

#[cfg(test)]
mod tests;
