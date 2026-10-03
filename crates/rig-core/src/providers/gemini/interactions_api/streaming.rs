//! The decoder of Interactions replies: a whole interaction resource, or a
//! stream of `event_type`-tagged step events, read as JSON.
//!
//! ```
//! use rig_core::providers::gemini::interactions_api::streaming::InteractionsDecoder;
//!
//! let decoder = InteractionsDecoder::default();
//! # let _ = decoder;
//! ```

use std::collections::BTreeMap;

use serde_json::{Map, Value, json};

use crate::completion::{FinishReason, Usage};
use crate::error::ProviderError;
use crate::json_utils::Lenient;
use crate::message::{DocumentSourceKind, Image, ImageMediaType, MimeType};
use crate::operation::completion::merge;
use crate::operation::{Block, CallFragment, Completion, Finish};
use crate::providers::internal::wire;
use crate::wire::{Decoder, Flow, Out, WireEvent, WireFrame};

/// Recognized Interactions SSE tags; unlisted tags classify as unknown.
const KNOWN_EVENT_TYPES: &[&str] = &[
    "interaction.created",
    "interaction.completed",
    "interaction.status_update",
    "step.start",
    "step.delta",
    "step.stop",
    "error",
];

/// Top-level keys that mark an untagged frame as a whole interaction
/// resource.
const INTERACTION_MARKER_KEYS: &[&str] = &["steps", "status", "usage", "object", "id"];

/// A decoded frame, verbatim: the decoder reads the fields it needs.
pub enum InteractionsEvent {
    /// One `event_type`-tagged streaming event.
    Sse(SseEvent),
    /// The whole interaction resource, as the unary reply delivers it.
    Whole(Map<String, Value>),
}

/// A streaming event: its tag, and its other fields verbatim.
#[derive(Debug, serde::Deserialize)]
pub struct SseEvent {
    /// The event's `event_type`.
    pub event_type: String,
    /// Every other field.
    #[serde(flatten)]
    pub fields: Map<String, Value>,
}

/// The Gemini Interactions wire's decoder. Steps decode in step order: a
/// thought, a call and any other step is one block, and a model output is
/// one block per content item, whose provider item is the model output step
/// holding that item alone. A step opens on `step.start`, grows with its
/// `step.delta`s, and becomes its blocks' provider item on `step.stop`, where
/// the API states it complete. A whole interaction states each step
/// complete. A step still open when the reply ends keeps no provider item.
#[derive(Default)]
pub struct InteractionsDecoder {
    /// The open steps, by wire index.
    steps: BTreeMap<usize, Open>,
    /// The model `interaction.created` named, for a completion that does
    /// not.
    model: Option<String>,
    /// The last status a stream reported.
    status: Option<String>,
}

/// What an open step decodes to.
enum Open {
    Thought,
    /// A model output: the step's other fields, and each content item with
    /// its block's writer index.
    Output(Value, Vec<(usize, Value)>),
    /// A function call, with the argument JSON streamed so far.
    Call(String),
    Opaque,
}

/// Rig's usage for an interaction's `usage`, read leniently: input is
/// `total_input_tokens` plus the tool-use tokens, output
/// `total_output_tokens` plus the thought tokens, and the total their sum;
/// without a base count, that side and the total stay absent.
pub fn usage_of(usage: &Value) -> Usage {
    let tool_use = usage.u64("total_tool_use_tokens");
    let thoughts = usage.u64("total_thought_tokens");
    let input_tokens = usage
        .u64("total_input_tokens")
        .map(|input| input + tool_use.unwrap_or(0));
    let output_tokens = usage
        .u64("total_output_tokens")
        .map(|output| output + thoughts.unwrap_or(0));
    Usage {
        input_tokens,
        output_tokens,
        cached_input_tokens: usage.u64("total_cached_tokens"),
        reasoning_tokens: thoughts,
        tool_use_prompt_tokens: tool_use,
        total_tokens: input_tokens
            .zip(output_tokens)
            .map(|(input, output)| input + output),
        cache_creation_input_tokens: None,
    }
}

/// The image a content item states, when it has data or a URI.
fn image_of(content: &Value) -> Option<Image> {
    let data = match (content.str("data"), content.str("uri")) {
        (Some(data), _) => DocumentSourceKind::Base64(data.to_owned()),
        (None, Some(uri)) => DocumentSourceKind::Url(uri.to_owned()),
        (None, None) => return None,
    };
    Some(Image {
        data,
        media_type: content
            .str("mime_type")
            .and_then(ImageMediaType::from_mime_type),
        detail: None,
        native: None,
    })
}

/// Open the block for `content`, a model output's newest content item.
fn open_content(
    content: Value,
    out: &mut Out<'_, Completion>,
) -> Result<(usize, Value), ProviderError> {
    let index = out.fresh_index();
    let block = match content.str("type") {
        Some("text") => Block::Text,
        Some("image") => image_of(&content).map_or(Block::Opaque { replay: true }, Block::Image),
        _ => Block::Opaque { replay: true },
    };
    let text = matches!(block, Block::Text);
    out.open(index, block, Value::Null)?;
    if let Some(fragment) = content.str("text").filter(|_| text) {
        out.push(index, fragment)?;
    }
    Ok((index, content))
}

impl InteractionsDecoder {
    /// Open the step at `index` as it starts, writing the content it
    /// already states.
    fn start(
        &mut self,
        index: usize,
        step: Value,
        out: &mut Out<'_, Completion>,
    ) -> Result<(), ProviderError> {
        let open = match step.str("type") {
            Some("thought") => {
                let summary: Vec<String> = step
                    .arr("summary")
                    .iter()
                    .filter(|summary| summary.str("type") == Some("text"))
                    .filter_map(|summary| summary.str("text").map(str::to_owned))
                    .collect();
                out.open(index, Block::Reasoning { redacted: false }, step)?;
                for text in summary {
                    out.push(index, &text)?;
                }
                Open::Thought
            }
            Some("model_output") => {
                let items = step.arr("content").iter();
                let items = items.map(|content| open_content(content.clone(), out));
                let items = items.collect::<Result<_, _>>()?;
                let mut step = step;
                if let Some(step) = step.as_object_mut() {
                    step.insert("content".to_owned(), json!([]));
                }
                Open::Output(step, items)
            }
            Some("function_call") => {
                call_fields(index, &step, out)?;
                out.edit(index, |slot| *slot = step)?;
                Open::Call(String::new())
            }
            // Input a reply restates is kept and never sent back; a
            // hosted-tool step, or one rig does not know, replays.
            kind => {
                let replay = !matches!(kind, Some("user_input" | "function_result"));
                out.open(index, Block::Opaque { replay }, step)?;
                Open::Opaque
            }
        };
        self.steps.insert(index, open);
        Ok(())
    }

    /// Apply one delta to the step at `index`. A delta for a step that
    /// never started opens the step it implies: a resumed stream can join
    /// a step after its start.
    fn delta(
        &mut self,
        index: usize,
        delta: Map<String, Value>,
        out: &mut Out<'_, Completion>,
    ) -> Result<(), ProviderError> {
        let kind = delta
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        if !self.steps.contains_key(&index) {
            let step = match kind.as_str() {
                "text" | "image" | "audio" | "document" | "video" => "model_output",
                "thought_summary" | "thought_signature" => "thought",
                "arguments_delta" => "function_call",
                other => other,
            };
            self.start(index, json!({ "type": step }), out)?;
        }
        let Some(open) = self.steps.get_mut(&index) else {
            return Ok(());
        };
        match (open, kind.as_str()) {
            // Text extends the last text item and its block; any other delta
            // is a content item of its own.
            (Open::Output(_, items), _) => match items.last_mut() {
                Some((block, item)) if kind == "text" && item.str("type") == Some("text") => {
                    if let Some(fragment) = delta.get("text").and_then(Value::as_str) {
                        out.push(*block, fragment)?;
                    }
                    let mut delta = delta;
                    delta.shift_remove("type");
                    merge(item, &delta);
                    Ok(())
                }
                _ => {
                    items.push(open_content(Value::Object(delta), out)?);
                    Ok(())
                }
            },
            (Open::Thought, "thought_summary") => {
                let content = delta.get("content").cloned().unwrap_or_default();
                if let Some(text) = content
                    .str("text")
                    .filter(|_| content.str("type") == Some("text"))
                {
                    out.push(index, text)?;
                }
                out.edit(index, |item| {
                    merge(
                        item,
                        &Map::from_iter([("summary".to_owned(), json!([content]))]),
                    )
                })
            }
            (Open::Call(arguments), "arguments_delta") => {
                let fragment = match delta.get("arguments") {
                    Some(Value::String(fragment)) => fragment.clone(),
                    Some(Value::Null) | None => String::new(),
                    Some(other) => other.to_string(),
                };
                arguments.push_str(&fragment);
                let fragment = CallFragment {
                    arguments: Some(&fragment),
                    ..CallFragment::default()
                };
                out.fragment(Some(index), fragment)
            }
            (open, _) => {
                if matches!(open, Open::Call(_)) && kind == "function_call" {
                    call_fields(index, &Value::Object(delta.clone()), out)?;
                }
                // A delta that restates its step (it carries the step's own
                // `type`) replaces the fields it names; any other merges.
                out.edit(index, |item| {
                    if item.get("type") != delta.get("type") {
                        merge(item, &delta);
                    } else if let Some(item) = item.as_object_mut() {
                        item.extend(delta);
                    }
                })
            }
        }
    }

    /// Finish the step at `index`: the API states it complete.
    fn stop(&mut self, index: usize, out: &mut Out<'_, Completion>) -> Result<(), ProviderError> {
        match self.steps.remove(&index) {
            None => Ok(()),
            Some(Open::Call(arguments)) => {
                // Streamed fragments supersede the arguments the start
                // announced.
                if !arguments.is_empty()
                    && let Ok(arguments) = crate::json_utils::parse_tool_arguments(&arguments)
                {
                    out.edit(index, |item| {
                        if let Some(item) = item.as_object_mut() {
                            item.insert("arguments".to_owned(), arguments);
                        }
                    })?;
                }
                out.finish(index)
            }
            // An output with no content is one empty text block holding it.
            Some(Open::Output(step, items)) if items.is_empty() => {
                let block = out.fresh_index();
                out.open(block, Block::Text, step)?;
                out.finish(block)
            }
            Some(Open::Output(step, items)) => {
                for (block, item) in items {
                    let mut step = step.clone();
                    if let Some(step) = step.as_object_mut() {
                        step.insert("content".to_owned(), json!([item]));
                    }
                    out.edit(block, |slot| *slot = step)?;
                    out.finish(block)?;
                }
                Ok(())
            }
            Some(Open::Thought | Open::Opaque) => out.finish(index),
        }
    }

    /// End the reply with the interaction resource it completed with.
    fn complete(&mut self, interaction: Map<String, Value>, mut out: Out<'_, Completion>) -> Flow {
        let interaction = Value::Object(interaction);
        let field = |key: &str| interaction.str(key).map(str::to_owned);
        // An agent interaction names its agent in place of a model.
        let model = field("model")
            .or_else(|| field("agent"))
            .or_else(|| self.model.take());
        let status = field("status").or_else(|| self.status.take());
        let (reason, error) = ending(status.as_deref(), interaction.arr("errors"));
        let usage = interaction.get("usage").cloned().unwrap_or(Value::Null);
        let finish = Finish {
            usage: usage_of(&usage),
            reason,
            response_id: field("id"),
            model: model.clone(),
            error,
        };
        let mut raw = Map::from_iter([
            ("usage".to_owned(), usage),
            ("interaction".to_owned(), interaction),
        ]);
        if let Some(model) = model {
            raw.insert("model_version".to_owned(), Value::String(model));
        }
        out.raw(Value::Object(raw));
        out.end(finish)
    }
}

/// The finish an interaction's `status` states, and the failure it reports.
/// `completed` and `requires_action` (calls wait for results) are
/// successes; `incomplete` is the token or execution budget running out,
/// and `budget_exceeded` its deprecated spelling. `failed` and `cancelled`
/// are failures carrying the interaction's `errors`; `in_progress` and
/// `queued` are an interaction read before it ended. An unknown status, or
/// none, is a failure.
fn ending(status: Option<&str>, errors: &[Value]) -> (Option<FinishReason>, Option<String>) {
    let messages: Vec<&str> = errors
        .iter()
        .filter_map(|error| error.str("message"))
        .collect();
    let detail = match messages.is_empty() {
        true => String::new(),
        false => format!(": {}", messages.join("; ")),
    };
    let other = |status: &str| Some(FinishReason::Other(status.to_owned()));
    match status {
        Some("completed") => (Some(FinishReason::Stop), None),
        Some("requires_action") => (Some(FinishReason::ToolCalls), None),
        Some("incomplete" | "budget_exceeded") => (Some(FinishReason::Length), None),
        Some(status @ ("failed" | "cancelled")) => (
            other(status),
            Some(format!("The interaction {status}{detail}")),
        ),
        Some(status @ ("in_progress" | "queued")) => (
            other(status),
            Some(format!("The interaction was read while {status}")),
        ),
        Some(status) => (other(status), None),
        None => (None, Some("The interaction states no status".to_owned())),
    }
}

/// Write the id, name and announced arguments a call step states.
fn call_fields(
    index: usize,
    step: &Value,
    out: &mut Out<'_, Completion>,
) -> Result<(), ProviderError> {
    let fragment = CallFragment {
        id: step.str("id"),
        name: step.str("name"),
        arguments: None,
    };
    out.fragment(Some(index), fragment)?;
    match step.get("arguments") {
        Some(arguments) => out.announce(index, arguments.clone()),
        None => Ok(()),
    }
}

/// EOF without `interaction.completed` is truncation, not successful
/// completion, so the decoder has nothing to add at the end of the reply.
impl<'id> Decoder<'id, Completion> for InteractionsDecoder {
    type Event = InteractionsEvent;

    /// Classify a frame by its `event_type` tag, or as a whole resource by
    /// its marker keys when it carries no tag. A tagged frame never passes
    /// for a whole resource.
    fn classify(&self, frame: WireFrame) -> WireEvent<InteractionsEvent> {
        wire::classify_or_untagged(
            &frame.as_str(),
            "event_type",
            |data| {
                wire::classify_tagged_frame(data, "event_type", |tag| {
                    KNOWN_EVENT_TYPES.contains(&tag)
                })
                .map(InteractionsEvent::Sse)
            },
            |data| {
                wire::classify_marker_keyed_frame(data, INTERACTION_MARKER_KEYS)
                    .map(InteractionsEvent::Whole)
            },
        )
    }

    fn decode(
        &mut self,
        event: InteractionsEvent,
        mut out: Out<'id, Completion>,
    ) -> Result<Flow, ProviderError> {
        let SseEvent {
            event_type,
            fields: event,
        } = match event {
            InteractionsEvent::Sse(event) => event,
            InteractionsEvent::Whole(interaction) => {
                let steps = Value::Object(interaction.clone());
                for (index, step) in steps.arr("steps").iter().enumerate() {
                    self.start(index, step.clone(), &mut out)?;
                    self.stop(index, &mut out)?;
                }
                return Ok(self.complete(interaction, out));
            }
        };
        let event = Value::Object(event);
        let index = || {
            event
                .u64("index")
                .and_then(|index| usize::try_from(index).ok())
                .ok_or_else(|| {
                    ProviderError::Response(format!(
                        "an Interactions `{event_type}` event names no step index"
                    ))
                })
        };
        let malformed = |field: &str| {
            ProviderError::Response(format!(
                "an Interactions `{event_type}` event carries no `{field}` object"
            ))
        };
        match event_type.as_str() {
            "step.start" => match event.get("step") {
                // A start that states no step leaves its deltas to open it.
                None => {}
                Some(step @ Value::Object(_)) => self.start(index()?, step.clone(), &mut out)?,
                Some(_) => return Err(malformed("step")),
            },
            "step.delta" => match event.get("delta") {
                Some(Value::Object(delta)) => self.delta(index()?, delta.clone(), &mut out)?,
                _ => return Err(malformed("delta")),
            },
            "step.stop" => self.stop(index()?, &mut out)?,
            "interaction.created" => {
                let field = |key: &str| {
                    event
                        .at(&format!("/interaction/{key}"))
                        .and_then(Value::as_str)
                        .map(str::to_owned)
                };
                self.model = field("model").or_else(|| field("agent"));
                self.status = field("status");
            }
            "interaction.status_update" => {
                if let Some(status) = event.str("status") {
                    self.status = Some(status.to_owned());
                }
            }
            "interaction.completed" => {
                let interaction = event.obj("interaction").cloned().unwrap_or_default();
                return Ok(self.complete(interaction, out));
            }
            "error" => {
                let Value::Object(mut body) = event else {
                    return Err(ProviderError::from_provider_body(event_type));
                };
                body.insert("event_type".to_owned(), Value::from(event_type));
                return Err(ProviderError::from_provider_body(
                    Value::Object(body).to_string(),
                ));
            }
            _ => {}
        }
        Ok(Flow::More)
    }
}

#[cfg(test)]
mod tests;
