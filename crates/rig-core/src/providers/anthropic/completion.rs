//! Anthropic Messages requests: model identifiers, the request body built
//! as JSON from a prepared request, prompt-cache breakpoints, and the strict
//! tool schemas Anthropic's constrained decoding takes.
//!
//! Cache markers are validated after raw request overrides: at most four,
//! with one-hour markers before five-minute markers. Native thinking replay
//! omits response-only token estimates without changing stored signed text,
//! signatures, or raw responses.
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
use std::borrow::Cow;
use std::sync::Arc;

#[doc(hidden)]
use super::wire::Messages;
use crate::catalog::{ModelFacts, ModelSpec};
use crate::completion::options::{BaseInput, FinalBody, RawAt, Rewrite, request_params};
use crate::completion::{self, CompletionRequest, Replay};
use crate::error::EncodeError;
use crate::json_utils::Lenient;
use crate::json_utils::remove_preserving_order;
use crate::message::{
    self, AssistantContent, DocumentData, DocumentMediaType, DocumentSourceKind, ImageMediaType,
    Message, ToolResultContent, UserContent,
};
use crate::providers::internal::set_fields;
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

/// An `anthropic-version` header value.
pub const ANTHROPIC_VERSION_2023_01_01: &str = "2023-01-01";
/// An `anthropic-version` header value.
pub const ANTHROPIC_VERSION_2023_06_01: &str = "2023-06-01";
/// The `anthropic-version` header value rig sends.
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

/// The facts of Anthropic's `model`: the spec the wire was connected to,
/// else its catalog's entry by the one lookup rule (an id as listed, or a
/// dated snapshot of one). `None` for a model the catalog does not list,
/// which gets the newest models' behaviour.
pub(super) fn spec<'f>(facts: &'f ModelFacts, model: &str) -> Option<&'f ModelSpec> {
    facts.for_model(super::ANTHROPIC.name, model)
}

/// The published synchronous output limit of a listed model. A model the
/// catalog does not list under the `claude-opus-4`, `claude-sonnet-4` or
/// `claude-haiku-4-5` prefixes takes 64000; any other requires an explicit
/// `max_tokens` value.
pub(super) fn default_max_tokens_for_model(facts: &ModelFacts, model: &str) -> Option<u64> {
    match spec(facts, model) {
        Some(spec) => spec.max_output_tokens.map(u64::from),
        None => ["claude-opus-4", "claude-sonnet-4", "claude-haiku-4-5"]
            .iter()
            .any(|prefix| model.starts_with(prefix))
            .then_some(64_000),
    }
}

/// Whether `model` rejects a forced tool choice.
pub(super) fn rejects_forced_tool_choice(facts: &ModelFacts, model: &str) -> bool {
    spec(facts, model).is_some_and(|spec| spec.compat.rejects_forced_tool_choice)
}

/// Whether `model` takes `role: "system"` inside `messages` (the
/// mid-conversation system messages page). A model the catalog does not
/// list takes system text only in the request's `system`.
pub(super) fn takes_mid_conversation_system(facts: &ModelFacts, model: &str) -> bool {
    spec(facts, model).is_some_and(|spec| spec.compat.mid_conversation_system)
}

/// Whether the Claude `model` binds its thinking blocks to the request's
/// tools and system prompt.
pub(super) fn binds_context(facts: &ModelFacts, model: &str) -> bool {
    spec(facts, model).is_some_and(|spec| spec.compat.binds_context)
}

/// The beta that lets a request ask Anthropic to drop a thinking block bound
/// to another context instead of rejecting the request.
pub(super) const THINKING_BINDING_BETA: &str = "thinking-binding-controls-2026-08-01";

/// The beta flag that streams tool input as it is written, for a provider
/// that rejects the per-tool `eager_input_streaming` field.
pub(super) const FINE_GRAINED_TOOL_STREAMING_BETA: &str = "fine-grained-tool-streaming-2025-05-14";

/// Whether a request to `model` on `wire`, with `thinking` as the caller
/// sets it, asks Anthropic to drop thinking bound to another context:
/// Anthropic's own API, for a model whose thinking binds, in
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
    wire.provider.dialect.name == super::ANTHROPIC.name
        && binds_context(&wire.facts, model)
        && adaptive
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

/// The Messages request body for `request`, already prepared, on `wire`:
/// the wire's encoding of the request, then the mapped options, then
/// `additional_params`, merged by [`request_params`].
///
/// # Errors
///
/// When no `max_tokens` applies, an option is refused, `additional_params`
/// is not an object or carries malformed `tools` or `cache_control`, the
/// caching settings conflict, or a part has a form the wire's `encodes`
/// refuses.
pub(super) fn body(
    wire: &Messages,
    request: &CompletionRequest,
    mode: Mode,
) -> Result<FinalBody, EncodeError> {
    let model = request.model.clone().unwrap_or_else(|| wire.model.clone());
    // A request that addresses another model gets that model's default; the
    // wire's own model keeps the one it was built with.
    let max_tokens = match request.max_tokens {
        Some(tokens) => Some(tokens),
        None if model == wire.model => wire.default_max_tokens,
        None => wire
            .provider
            .dialect
            .default_max_tokens(&wire.facts, &model)
            .or(wire.default_max_tokens),
    }
    .ok_or_else(|| EncodeError::request("`max_tokens` must be set for Anthropic"))?;
    let mut rewrites = Vec::new();
    if wire.provider.dialect.name == super::ANTHROPIC.name && binds_context(&wire.facts, &model) {
        rewrites.push(Rewrite::DropUnboundThinking);
    }
    if wire.provider.addressing == super::wire::Addressing::Url {
        rewrites.push(Rewrite::AnthropicVersionInBody(
            wire.provider.version.clone(),
        ));
    }
    if mode == Mode::Streaming {
        rewrites.push(Rewrite::Stream(true));
        // Anthropic rejects tool_choice without tools.
        rewrites.push(Rewrite::ToolChoiceNeedsTools);
    }
    let body = request_params(
        wire,
        request,
        |input| base(wire, request, &model, max_tokens, input),
        RawAt::Top,
        &rewrites,
    )?;
    validate_cache_control(&body)?;
    Ok(body)
}

/// The wire's own encoding of `request`: messages, tools, the typed fields
/// and the cache markers placed inside them.
fn base(
    wire: &Messages,
    request: &CompletionRequest,
    model: &str,
    max_tokens: u64,
    input: &mut BaseInput<'_>,
) -> Result<Map<String, Value>, EncodeError> {
    let top = top_level_cache_control(input.param("cache_control"))?;
    let strict = wire.strict_tools && wire.provider.dialect.quirks.strict_tool_schemas;
    let eager = wire.tool_input_streaming == super::wire::ToolInputStreaming::Eager;
    let mut tools = tools(&request.tools, input.raw_tools()?, strict, eager);
    let (mut system, history) = split_system(
        &request.chat_history,
        takes_mid_conversation_system(&wire.facts, model),
    );
    let ids = WireIds::for_target(&history, wire, model);
    let mut messages: Vec<Value> = Vec::new();
    // How many wire messages the first `n` turns became, at `n`. System
    // messages are placed apart, so the turns are counted alone.
    let mut after = vec![0];
    // Each turn with the index its wire message takes, if it sends one.
    let mut turns = Vec::new();
    for message in &history {
        let at = messages.len();
        messages.extend(message_json(message, wire, &ids)?);
        if !matches!(&**message, Message::System { .. }) {
            after.push(messages.len());
            turns.push((at, message));
        }
    }
    let ending = |turns: usize| {
        after
            .get(turns)
            .map(|wired| wired.checked_sub(1).map_or(Spot::System, Spot::last))
    };
    let mut points = Vec::with_capacity(request.cache_points.len());
    for point in &request.cache_points {
        let spot = match *point {
            completion::CachePoint::Tools => Some(Spot::Tools),
            completion::CachePoint::System => Some(Spot::System),
            completion::CachePoint::Messages(turns) => ending(turns),
            completion::CachePoint::Within { message, items } => {
                let Some(&(at, turn)) = turns.get(message) else {
                    continue;
                };
                let blocks = match &**turn {
                    Message::User { content } => {
                        let mut blocks = 0;
                        for part in content.iter().take(items) {
                            blocks += usize::from(user_part(part, &ids)?.is_some());
                        }
                        blocks
                    }
                    Message::Assistant(turn) => turn
                        .content
                        .iter()
                        .take(items)
                        .filter_map(|block| assistant_part(block, wire, &ids))
                        .count(),
                    Message::System { .. } => 0,
                };
                if blocks == 0 {
                    ending(message)
                } else {
                    Some(Spot::Message(at, blocks))
                }
            }
        };
        points.extend(spot);
    }
    // MiniMax honours a cache through block markers the manual placement
    // writes.
    let placed = input.cache().is_some() && wire.provider.dialect.name == super::wire::MINIMAX.name;
    apply_cache_control(
        wire,
        Markers {
            placed,
            top: top.as_ref(),
            points,
        },
        &mut system,
        &mut messages,
        &mut tools,
    )?;
    let has_tools = !tools.is_empty();
    let output_format = wire.provider.dialect.quirks.supports_output_format;
    if request.output_schema.is_some() && !output_format {
        tracing::warn!(
            "Structured outputs are not supported by this provider; ignoring output_schema"
        );
    }
    let output_config = request
        .output_schema
        .clone()
        .filter(|_| output_format)
        .map(|schema| {
            let mut schema = schema.to_value();
            sanitize_schema(&mut schema);
            json!({ "format": { "type": "json_schema", "schema": schema } })
        });
    let container = input
        .param("container")
        .is_none()
        .then(|| container(&history))
        .flatten();
    Ok(set_fields([
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
            request.tool_choice.clone().map(tool_choice).transpose()?,
        ),
        ("tools", has_tools.then(|| Value::Array(tools))),
        ("output_config", output_config),
        ("container", container.map(Value::String)),
    ]))
}

/// A text content block.
fn text(text: &str) -> Value {
    json!({ "type": "text", "text": text })
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
            let mut images = Vec::new();
            for part in content {
                parts.extend(user_part(part, ids)?);
                if let UserContent::ToolResult(result) = part
                    && result.is_error
                {
                    for part in &result.content {
                        if let ToolResultContent::Image(image) = part {
                            images.push(image_part(image)?);
                        }
                    }
                }
            }
            // Error results accept text only; keep their images after the result run.
            parts.extend(images);
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
    Ok((!content.is_empty()).then(|| {
        Value::Object(set_fields([
            ("role", Some(Value::from(role))),
            ("content", Some(Value::Array(content))),
        ]))
    }))
}

/// One user block on the wire. `None` for blank text, which Anthropic
/// rejects.
fn user_part(part: &UserContent, ids: &WireIds) -> Result<Option<Value>, EncodeError> {
    Ok(Some(match part {
        UserContent::Text(part) if part.text.trim().is_empty() => return Ok(None),
        UserContent::Text(part) => text(&part.text),
        UserContent::ToolResult(result) => {
            let mut content = Vec::with_capacity(result.content.len());
            for part in &result.content {
                content.push(match part {
                    ToolResultContent::Text(part) if part.text.trim().is_empty() => continue,
                    ToolResultContent::Text(part) => text(&part.text),
                    ToolResultContent::Json { value } => text(&value.to_string()),
                    ToolResultContent::Image(_) if result.is_error => continue,
                    ToolResultContent::Image(part) => image_part(part)?,
                });
            }
            let content = if content.is_empty() {
                Value::from(if result.is_error {
                    "Tool failed with no output."
                } else {
                    ""
                })
            } else {
                Value::Array(content)
            };
            Value::Object(set_fields([
                ("type", Some(json!("tool_result"))),
                ("tool_use_id", Some(Value::from(ids.spell(&result.call)))),
                ("content", Some(content)),
                ("is_error", result.is_error.then_some(Value::Bool(true))),
            ]))
        }
        UserContent::Image(part) => image_part(part)?,
        UserContent::Document(document) => document_part(document)?,
        UserContent::Audio(_) | UserContent::Video(_) => return Err(unsendable()),
    }))
}

fn image_part(image: &message::Image) -> Result<Value, EncodeError> {
    let source = image_source(image).ok_or_else(unsendable)?;
    Ok(Value::Object(set_fields([
        ("type", Some(Value::from("image"))),
        ("source", Some(source)),
    ])))
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
    Ok(Value::Object(set_fields([
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

/// One assistant block on the wire: the current provider item without
/// response-only thinking token estimates, else its canonical fields with the
/// identity keys of an edited item. `None` for a block with
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
        Replay::Item(item) => {
            let mut item = item.into_owned();
            if item.get("type").and_then(Value::as_str) == Some("thinking")
                && let Some(fields) = item.as_object_mut()
            {
                // Token estimates are response metadata, not accepted thinking inputs.
                // Keep the signed text and signature unchanged, including in saved history.
                remove_preserving_order(fields, "estimated_tokens");
            }
            return Some(item);
        }
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
            let mut part = set_fields([
                ("type", Some(json!("tool_use"))),
                ("id", Some(Value::from(ids.spell(&call.id)))),
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
        AssistantContent::Image(_) => text(completion::history::IMAGE_OMITTED),
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
        DocumentSourceKind::Raw(_) | DocumentSourceKind::Unknown => return None,
    })
}

/// A document's source on the wire: a Files API id, a PDF as data or by
/// URL, or any other document as the text it holds. `None` for any other
/// form.
pub(super) fn document_source(document: &message::Document) -> Option<Value> {
    let text = |data: &str| json!({ "type": "text", "media_type": "text/plain", "data": data });
    let source = match &document.data {
        DocumentData::Text(data) => return Some(text(data)),
        DocumentData::File(source) => source,
    };
    Some(match (source, &document.media_type) {
        (DocumentSourceKind::FileId(file_id), _) => json!({ "type": "file", "file_id": file_id }),
        // Anthropic's URL source is defined for PDFs and has no media-type
        // field, so an untyped URL is one.
        (DocumentSourceKind::Url(url), None | Some(DocumentMediaType::PDF)) => {
            json!({ "type": "url", "url": url })
        }
        (DocumentSourceKind::Base64(data), Some(DocumentMediaType::PDF)) => {
            json!({ "type": "base64", "media_type": "application/pdf", "data": data })
        }
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
/// input as it is written when `eager`, then `extra`, the tools
/// `additional_params` names, verbatim.
fn tools(
    tools: &[Arc<completion::ToolDefinition>],
    extra: Vec<Value>,
    strict: bool,
    eager: bool,
) -> Vec<Value> {
    let rig = tools.iter().map(|tool| {
        let mut schema = tool.parameters.clone();
        if strict {
            sanitize_strict_tool_schema(&mut schema);
        }
        Value::Object(set_fields([
            ("name", Some(Value::from(tool.name.as_str()))),
            ("description", Some(Value::from(tool.description.as_str()))),
            ("input_schema", Some(schema)),
            ("strict", strict.then_some(Value::Bool(true))),
            ("eager_input_streaming", eager.then_some(Value::Bool(true))),
        ]))
    });
    rig.chain(extra).collect()
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
pub(super) fn split_system(
    history: &[Arc<Message>],
    mid: bool,
) -> (Vec<Value>, Vec<Cow<'_, Message>>) {
    let mut system = Vec::new();
    let mut remaining = Vec::new();
    let mut deferred: Vec<&str> = Vec::new();
    let lead = history
        .iter()
        .take_while(|message| matches!(&***message, Message::System { .. }))
        .count();
    for (index, message) in history.iter().enumerate() {
        match &**message {
            Message::System { content } if content.is_empty() => {}
            // The system messages that lead the history are the prompt.
            Message::System { content } if index < lead => system.push(text(content)),
            Message::System { .. } if mid && valid_system_message(history, index) => {
                remaining.push(Cow::Borrowed(&**message));
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
                remaining.push(Cow::Borrowed(other));
                if !deferred.is_empty() && system_slot(history, index) {
                    remaining.push(Cow::Owned(Message::System {
                        content: deferred.join("\n\n"),
                    }));
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
fn system_slot(history: &[Arc<Message>], index: usize) -> bool {
    matches!(
        history.get(index).map(Arc::as_ref),
        Some(Message::User { .. })
    ) && history
        .get(index + 1..)
        .into_iter()
        .flatten()
        .map(Arc::as_ref)
        .find(|message| !matches!(message, Message::System { .. }))
        .is_none_or(|message| matches!(message, Message::Assistant(_)))
}

/// Whether the system message at `index` sits where Anthropic takes one:
/// after a user turn, or an assistant turn ending in a server tool's result,
/// and before an assistant turn or the end.
fn valid_system_message(history: &[Arc<Message>], index: usize) -> bool {
    // A run of system messages shares one slot.
    let not_system = |message: &&Message| !matches!(message, Message::System { .. });
    let after = history
        .get(..index)
        .into_iter()
        .flatten()
        .map(Arc::as_ref)
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
            .map(Arc::as_ref)
            .find(not_system)
            .is_none_or(|message| matches!(message, Message::Assistant(_)))
}

/// The id of the container the last turn holding one ran in. Anthropic
/// requires it on a request that answers a programmatic tool call, and it
/// keeps a code-execution session's state. The decoder keeps it as an
/// opaque `container` block, which `adapt` leaves only on turns of the
/// same model.
fn container(history: &[Cow<'_, Message>]) -> Option<String> {
    history
        .iter()
        .rev()
        .filter_map(|message| match &**message {
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

/// The request's top-level `cache_control`, as the layers above the base
/// set it: the mapped cache retention, or a raw one over it. `null` is no
/// marker. The value is checked, never rebuilt, so it is sent as written.
fn top_level_cache_control(marker: Option<&Value>) -> Result<Option<Value>, EncodeError> {
    match marker {
        None | Some(Value::Null) => Ok(None),
        Some(marker) => {
            match Option::<CacheTtl>::deserialize(marker.get("ttl").unwrap_or(&Value::Null)) {
                Ok(_) if marker.str("type") == Some("ephemeral") => Ok(Some(marker.clone())),
                _ => Err(EncodeError::request(format!(
                    "Invalid Anthropic `additional_params.cache_control` payload: {marker}"
                ))),
            }
        }
    }
}

/// Validate the sent markers after raw overrides and request rewrites.
fn validate_cache_control(body: &FinalBody) -> Result<(), EncodeError> {
    let array = |key| {
        body.get(key)
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
    };
    let markers = array("tools")
        .chain(array("system"))
        .chain(array("messages").flat_map(|message| message.arr("content")))
        .filter_map(|block| block.get("cache_control"))
        .chain(body.get("cache_control"))
        .filter(|marker| !marker.is_null());
    let mut count = 0;
    let mut short_seen = false;
    for marker in markers {
        top_level_cache_control(Some(marker))?;
        count += 1;
        if count > MAX_CACHE_CONTROL_MARKERS {
            return Err(EncodeError::request(
                "Too many Anthropic `cache_control` markers: the final request exceeds the budget of four",
            ));
        }
        if !is_1h(marker) {
            short_seen = true;
        } else if short_seen {
            return Err(EncodeError::request(
                "Anthropic cache_control markers with ttl `1h` must appear before markers with the default 5-minute TTL",
            ));
        }
    }
    Ok(())
}

/// What asks for cache markers besides the wire's own settings.
struct Markers<'a> {
    /// Whether the option mapping asked for block markers.
    placed: bool,
    /// The request's top-level marker, if any.
    top: Option<&'a Value>,
    /// Where the request's cache points land.
    points: Vec<Spot>,
}

/// Where a cache point lands, in the order the request is sent.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Spot {
    /// The final non-deferred tool.
    Tools,
    /// The last system block.
    System,
    /// The last markable block among the first blocks, as many as the
    /// second number, of the wire message at the first.
    Message(usize, usize),
}

impl Spot {
    /// The last markable block of the wire message at `at`.
    fn last(at: usize) -> Self {
        Self::Message(at, usize::MAX)
    }
}

/// The block a cache point at `spot` marks, if the request has one there.
fn spot_block<'a>(
    spot: Spot,
    system: &'a mut [Value],
    messages: &'a mut [Value],
    tools: &'a mut [Value],
) -> Option<&'a mut Map<String, Value>> {
    let block = match spot {
        Spot::Tools => tools
            .iter_mut()
            .rfind(|tool| tool.get("defer_loading") != Some(&Value::Bool(true))),
        Spot::System => system.last_mut(),
        Spot::Message(at, blocks) => messages
            .get_mut(at)?
            .get_mut("content")?
            .as_array_mut()?
            .iter_mut()
            .take(blocks)
            .rev()
            .find(|block| takes_marker(block)),
    };
    block?.as_object_mut()
}

/// Place the request's cache markers within Anthropic's budget of four,
/// one fewer with a top-level marker. Manual prompt caching, or a cache the
/// mapping `placed`, marks the final non-deferred tool, the system prompt
/// and the last block of the last user or system message; a static-prefix
/// TTL alone marks only the first two. The request's cache points then
/// take the markers left, ahead of manual caching's last-message marker:
/// when too few are left, the points furthest into the request win, and a
/// block already marked takes no second one. Markers a tool already
/// carries are kept and count toward the budget, and every one-hour marker
/// must precede the five-minute ones.
fn apply_cache_control(
    wire: &Messages,
    Markers {
        placed,
        top,
        mut points,
    }: Markers<'_>,
    system: &mut [Value],
    messages: &mut [Value],
    tools: &mut [Value],
) -> Result<(), EncodeError> {
    for tool in tools.iter_mut().filter_map(Value::as_object_mut) {
        if tool.get("cache_control").is_some_and(Value::is_null) {
            remove_preserving_order(tool, "cache_control");
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
             top-level cache TTL (`CacheRetention::Long` or a raw top-level `cache_control`): \
             Anthropic requires 1h markers to precede 5-minute ones, and the static prefix \
             precedes the conversation tail",
        ));
    }
    let manual = wire.prompt_caching || placed;
    if manual || wire.static_prefix_cache_ttl.is_some() {
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
    // Furthest into the request first. Distinct spots are distinct blocks,
    // so a repeated spot finds its block already marked.
    points.sort_unstable();
    for spot in points.into_iter().rev() {
        if remaining == 0 {
            break;
        }
        if let Some(block) = spot_block(spot, system, messages, tools)
            && !block.contains_key("cache_control")
        {
            block.insert("cache_control".into(), ephemeral(top_ttl.as_ref()));
            remaining -= 1;
        }
    }
    if manual && top.is_none() && remaining > 0 {
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

/// Whether a content block can carry a cache marker: any but a thinking
/// block.
fn takes_marker(block: &Value) -> bool {
    !matches!(block.str("type"), Some("thinking" | "redacted_thinking"))
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
        if let Some(all_of) = remove_preserving_order(root, "allOf") {
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
        remove_preserving_order(&mut root, "$ref");
        for keyword in DEFINITIONS {
            if let Some(definitions) = remove_preserving_order(&mut root, keyword) {
                let merged = merge_definitions(
                    definitions,
                    remove_preserving_order(&mut referenced, keyword),
                );
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
                    let merged = merge_definitions(
                        definitions,
                        remove_preserving_order(&mut branch, keyword),
                    );
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
        let merged = match remove_preserving_order(properties, &name) {
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
        match remove_preserving_order(&mut source, keyword) {
            Some(Value::Object(definitions)) => {
                strict.insert(keyword.into(), each(definitions));
            }
            Some(definitions) => {
                source.insert(keyword.into(), definitions);
            }
            None => {}
        }
    }
    if let Some(reference) = remove_preserving_order(&mut source, "$ref") {
        strict.insert("$ref".into(), reference);
        return Value::Object(strict);
    }
    let kind = remove_preserving_order(&mut source, "type");
    let is = |expected: &str| match &kind {
        Some(Value::String(kind)) => kind == expected,
        Some(Value::Array(kinds)) => kinds.iter().any(|kind| kind.as_str() == Some(expected)),
        _ => false,
    };
    let alternatives = match (
        remove_preserving_order(&mut source, "anyOf"),
        remove_preserving_order(&mut source, "oneOf"),
        remove_preserving_order(&mut source, "allOf"),
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
    if let Some(Value::Array(values)) = remove_preserving_order(&mut source, "enum") {
        strict.insert("enum".into(), Value::Array(values));
    }
    if let Some(constant) = remove_preserving_order(&mut source, "const") {
        strict.insert("const".into(), constant);
    }
    for keyword in ["description", "title"] {
        if let Some(Value::String(value)) = remove_preserving_order(&mut source, keyword) {
            strict.insert(keyword.into(), Value::String(value));
        }
    }
    let has_properties = source.contains_key("properties");
    if kind.is_none() && has_properties {
        strict.insert("type".into(), json!("object"));
    }
    if is("object") || has_properties {
        let properties = match remove_preserving_order(&mut source, "properties") {
            Some(Value::Object(properties)) => each(properties),
            _ => json!({}),
        };
        strict.insert("properties".into(), properties);
        remove_preserving_order(&mut source, "additionalProperties");
        strict.insert("additionalProperties".into(), Value::Bool(false));
        if let Some(Value::Array(required)) = remove_preserving_order(&mut source, "required") {
            strict.insert("required".into(), Value::Array(required));
        }
    }
    if is("string")
        && let Some(format) = remove_preserving_order(&mut source, "format")
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
        if let Some(items) = remove_preserving_order(&mut source, "items") {
            strict.insert("items".into(), strict_schema(items));
        }
        if let Some(min_items) = remove_preserving_order(&mut source, "minItems") {
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
