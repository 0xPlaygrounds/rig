use serde::{Deserialize, Serialize};

use std::collections::BTreeMap;

use super::interactions_api_types::{Interaction, InteractionUsage};
use crate::completion::FinishReason;
use crate::error::ProviderError;
use crate::message::{DocumentSourceKind, Image, ImageMediaType, MimeType};
use crate::operation::{Block, CallFragment, Completion, Finish};
use crate::providers::internal::wire;
use crate::wire::{Decoder, Flow, Out, WireEvent, WireFrame};
use serde_json::{Map, Value, json};

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
#[derive(Debug, Deserialize)]
pub struct SseEvent {
    /// The event's `event_type`.
    pub event_type: String,
    /// Every other field.
    #[serde(flatten)]
    pub fields: Map<String, Value>,
}

/// Classify a frame by its `event_type` tag, or as a whole resource by its
/// marker keys when it carries no tag. A tagged frame never passes for a
/// whole resource.
fn classify_interactions_frame(data: &str) -> WireEvent<InteractionsEvent> {
    wire::classify_or_untagged(
        data,
        "event_type",
        |data| {
            wire::classify_tagged_frame(data, "event_type", |tag| KNOWN_EVENT_TYPES.contains(&tag))
                .map(InteractionsEvent::Sse)
        },
        |data| {
            wire::classify_marker_keyed_frame(data, INTERACTION_MARKER_KEYS)
                .map(InteractionsEvent::Whole)
        },
    )
}

/// The reply's `raw`: the interaction resource the provider ended with,
/// verbatim, beside its usage and model.
#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub struct StreamingCompletionResponse {
    pub usage: Option<InteractionUsage>,
    pub interaction: Option<Interaction>,
    /// The model the interaction names (e.g. `gemini-3-flash-preview`).
    /// The Interactions API has no finish reason; `interaction.status`
    /// states how it ended.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_version: Option<String>,
}

impl From<&StreamingCompletionResponse> for crate::completion::Usage {
    fn from(value: &StreamingCompletionResponse) -> crate::completion::Usage {
        value
            .usage
            .as_ref()
            .map(crate::completion::Usage::from)
            .unwrap_or_default()
    }
}

impl From<StreamingCompletionResponse> for crate::completion::Usage {
    fn from(value: StreamingCompletionResponse) -> crate::completion::Usage {
        (&value).into()
    }
}

/// The Gemini Interactions wire's decoder. Steps decode in step order: a
/// thought, a call and any other step is one block, and a model output is
/// one block per content item. A step opens on `step.start`, grows with its
/// `step.delta`s, and becomes its blocks' provider item on `step.stop`,
/// where the API states it complete. A whole interaction states each step
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
    Output(Output),
    /// A function call, with the argument JSON streamed so far.
    Call(String),
    Opaque,
}

/// How a step decodes, by its `type`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StepKind {
    /// `thought`: reasoning, its signature kept in the step.
    Thought,
    /// `model_output`: one block per content item.
    Output,
    /// `function_call`: a call the client answers.
    Call,
    /// Input a reply restates (`user_input`, `function_result`): kept,
    /// never sent back.
    Input,
    /// A hosted-tool step, or a step type rig does not know: sent back to
    /// the model that produced it.
    Other,
}

fn step_kind(step: &Value) -> StepKind {
    match step.get("type").and_then(Value::as_str) {
        Some("thought") => StepKind::Thought,
        Some("model_output") => StepKind::Output,
        Some("function_call") => StepKind::Call,
        Some("user_input" | "function_result") => StepKind::Input,
        _ => StepKind::Other,
    }
}

/// How a model output content item decodes, by its `type`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ContentKind {
    Text,
    /// An image block, or an opaque one when it states no data or URI.
    Image,
    /// Audio, video, a document, or a type rig does not know.
    Other,
}

fn content_kind(content: &Value) -> ContentKind {
    match content.get("type").and_then(Value::as_str) {
        Some("text") => ContentKind::Text,
        Some("image") => ContentKind::Image,
        _ => ContentKind::Other,
    }
}

fn text_of(value: &Value) -> Option<&str> {
    value.get("text").and_then(Value::as_str)
}

/// An open model output step: the step as its events state it, and the
/// writer index of the block each content item became, in order.
struct Output {
    step: Value,
    blocks: Vec<usize>,
    /// The block the next text delta extends.
    text: Option<usize>,
}

impl Output {
    /// Open the block for `content`, the step's newest content item.
    fn open(
        &mut self,
        content: &Value,
        out: &mut Out<'_, Completion>,
    ) -> Result<(), ProviderError> {
        let index = out.fresh_index();
        let block = match content_kind(content) {
            ContentKind::Text => Block::Text,
            ContentKind::Image => {
                image_of(content).map_or(Block::Opaque { replay: true }, Block::Image)
            }
            ContentKind::Other => Block::Opaque { replay: true },
        };
        let text = matches!(block, Block::Text);
        out.open(index, block, Value::Null)?;
        if let Some(fragment) = text_of(content).filter(|_| text) {
            out.push(index, fragment)?;
        }
        self.blocks.push(index);
        self.text = text.then_some(index);
        Ok(())
    }

    /// Apply one delta: text extends the last text item and its block, any
    /// other delta is a whole content item of its own.
    fn delta(
        &mut self,
        delta: Map<String, Value>,
        out: &mut Out<'_, Completion>,
    ) -> Result<(), ProviderError> {
        let is_text = delta.get("type").and_then(Value::as_str) == Some("text");
        match self.text.filter(|_| is_text) {
            Some(index) => {
                if let Some(fragment) = delta.get("text").and_then(Value::as_str) {
                    out.push(index, fragment)?;
                }
                extend_text(&mut self.step, delta);
                Ok(())
            }
            None => {
                let content = Value::Object(delta);
                if let Some(items) = content_items(&mut self.step) {
                    items.push(content.clone());
                }
                self.open(&content, out)
            }
        }
    }

    /// Finish every block with the whole step as its provider item; an
    /// output with no content is one empty text block holding it.
    fn finish(self, out: &mut Out<'_, Completion>) -> Result<(), ProviderError> {
        if self.blocks.is_empty() {
            let index = out.fresh_index();
            out.open(index, Block::Text, Value::Null)?;
            return out.finish_with(index, self.step);
        }
        for index in self.blocks {
            out.finish_with(index, self.step.clone())?;
        }
        Ok(())
    }
}

impl InteractionsDecoder {
    /// Open the step at `index` as it starts, writing the content it
    /// already states.
    #[deny(clippy::wildcard_enum_match_arm)]
    fn start(
        &mut self,
        index: usize,
        step: Value,
        out: &mut Out<'_, Completion>,
    ) -> Result<(), ProviderError> {
        let open = match step_kind(&step) {
            StepKind::Thought => {
                let summary: Vec<String> = step
                    .get("summary")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .filter(|summary| summary.get("type").and_then(Value::as_str) == Some("text"))
                    .filter_map(|summary| text_of(summary).map(str::to_owned))
                    .collect();
                out.open(index, Block::Reasoning { redacted: false }, step)?;
                for text in summary {
                    out.push(index, &text)?;
                }
                Open::Thought
            }
            StepKind::Output => {
                let contents: Vec<Value> = step
                    .get("content")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default();
                let mut output = Output {
                    step,
                    blocks: Vec::new(),
                    text: None,
                };
                for content in &contents {
                    output.open(content, out)?;
                }
                Open::Output(output)
            }
            StepKind::Call => {
                call_fields(index, &step, out)?;
                out.edit(index, |slot| *slot = step)?;
                Open::Call(String::new())
            }
            StepKind::Input => {
                out.open(index, Block::Opaque { replay: false }, step)?;
                Open::Opaque
            }
            StepKind::Other => {
                out.open(index, Block::Opaque { replay: true }, step)?;
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
        let delta_type = delta
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        if !self.steps.contains_key(&index) {
            let step = match delta_type.as_str() {
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
        match (open, delta_type.as_str()) {
            (Open::Output(output), _) => output.delta(delta, out),
            (Open::Thought, "thought_summary") => {
                let content = delta.get("content").cloned().unwrap_or_default();
                if let Some(text) = content
                    .get("type")
                    .filter(|kind| *kind == "text")
                    .and(text_of(&content))
                {
                    out.push(index, text)?;
                }
                out.merge(
                    index,
                    &Map::from_iter([("summary".to_owned(), json!([content]))]),
                )
            }
            (Open::Call(arguments), "arguments_delta") => {
                let fragment = match delta.get("arguments") {
                    Some(Value::String(fragment)) => fragment.clone(),
                    Some(Value::Null) | None => String::new(),
                    Some(other) => other.to_string(),
                };
                arguments.push_str(&fragment);
                out.fragment(
                    Some(index),
                    CallFragment {
                        arguments: Some(&fragment),
                        ..CallFragment::default()
                    },
                )
            }
            (open, _) => {
                if let Open::Call(_) = open
                    && delta_type == "function_call"
                {
                    call_fields(index, &Value::Object(delta.clone()), out)?;
                }
                restate_or_merge(index, delta, out)
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
            Some(Open::Output(output)) => output.finish(out),
            Some(Open::Thought | Open::Opaque) => out.finish(index),
        }
    }

    /// End the reply with the interaction resource it completed with.
    fn complete(&mut self, interaction: Map<String, Value>, mut out: Out<'_, Completion>) -> Flow {
        let field = |key: &str| {
            interaction
                .get(key)
                .and_then(Value::as_str)
                .map(str::to_owned)
        };
        // An agent interaction names its agent in place of a model.
        let model = field("model")
            .or_else(|| field("agent"))
            .or_else(|| self.model.take());
        let status = field("status").or_else(|| self.status.take());
        let (reason, error) = ending(status.as_deref(), interaction.get("errors"));
        let usage = interaction.get("usage").cloned();
        let counts = usage
            .as_ref()
            .map(InteractionUsage::read)
            .unwrap_or_default();
        let response_id = field("id");
        let mut raw = Map::from_iter([
            ("usage".to_owned(), usage.unwrap_or(Value::Null)),
            ("interaction".to_owned(), Value::Object(interaction)),
        ]);
        if let Some(model) = &model {
            raw.insert("model_version".to_owned(), Value::from(model.as_str()));
        }
        out.raw(Value::Object(raw));
        out.end(Finish {
            usage: (&counts).into(),
            reason,
            response_id,
            model,
            error,
        })
    }
}

/// The finish an interaction's `status` states, and the failure it reports.
/// `completed` and `requires_action` (calls wait for results) are
/// successes; `incomplete` is the token or execution budget running out,
/// and `budget_exceeded` its deprecated spelling. `failed` and `cancelled`
/// are failures carrying the interaction's `errors`; `in_progress` and
/// `queued` are an interaction read before it ended. An unknown status, or
/// none, is a failure.
fn ending(status: Option<&str>, errors: Option<&Value>) -> (Option<FinishReason>, Option<String>) {
    let detail = || {
        let messages: Vec<&str> = errors
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|error| error.get("message").and_then(Value::as_str))
            .collect();
        if messages.is_empty() {
            String::new()
        } else {
            format!(": {}", messages.join("; "))
        }
    };
    match status {
        Some("completed") => (Some(FinishReason::Stop), None),
        Some("requires_action") => (Some(FinishReason::ToolCalls), None),
        Some("incomplete" | "budget_exceeded") => (Some(FinishReason::Length), None),
        Some(status @ ("failed" | "cancelled")) => (
            Some(FinishReason::Other(status.to_owned())),
            Some(format!("The interaction {status}{}", detail())),
        ),
        Some(status @ ("in_progress" | "queued")) => (
            Some(FinishReason::Other(status.to_owned())),
            Some(format!("The interaction was read while {status}")),
        ),
        Some(status) => (Some(FinishReason::Other(status.to_owned())), None),
        None => (None, Some("The interaction states no status".to_owned())),
    }
}

/// The error for an `event_type` event whose `field`, the content of its
/// block, is not an object.
fn malformed(event_type: &str, field: &str) -> ProviderError {
    ProviderError::Response(format!(
        "an Interactions `{event_type}` event carries no `{field}` object"
    ))
}

/// Write the id, name and announced arguments a call step states.
fn call_fields(
    index: usize,
    step: &Value,
    out: &mut Out<'_, Completion>,
) -> Result<(), ProviderError> {
    let field = |key: &str| step.get(key).and_then(Value::as_str);
    out.fragment(
        Some(index),
        CallFragment {
            id: field("id"),
            name: field("name"),
            arguments: None,
        },
    )?;
    match step.get("arguments") {
        Some(arguments) => out.announce(index, arguments.clone()),
        None => Ok(()),
    }
}

/// A delta that restates its step (it carries the step's own `type`)
/// replaces the fields it names; any other merges into the step.
fn restate_or_merge(
    index: usize,
    delta: Map<String, Value>,
    out: &mut Out<'_, Completion>,
) -> Result<(), ProviderError> {
    let mut merged = None;
    out.edit(index, |item| {
        if item.get("type") != delta.get("type") {
            merged = Some(delta);
        } else if let Some(item) = item.as_object_mut() {
            item.extend(delta);
        }
    })?;
    match merged {
        Some(delta) => out.merge(index, &delta),
        None => Ok(()),
    }
}

/// The content array of a model output, created when the step has none.
fn content_items(item: &mut Value) -> Option<&mut Vec<Value>> {
    item.as_object_mut()?
        .entry("content")
        .or_insert_with(|| json!([]))
        .as_array_mut()
}

/// Append a text delta to the step's last content item when that is text,
/// else add it as a new one.
fn extend_text(item: &mut Value, delta: Map<String, Value>) {
    let Some(content) = content_items(item) else {
        return;
    };
    match content.last_mut().and_then(Value::as_object_mut) {
        Some(last) if last.get("type") == delta.get("type") => {
            for (key, value) in delta.into_iter().filter(|(key, _)| key != "type") {
                match (last.get_mut(&key), value) {
                    (Some(Value::String(text)), Value::String(more)) => text.push_str(&more),
                    (Some(Value::Array(items)), Value::Array(more)) => items.extend(more),
                    (_, value) => {
                        last.insert(key, value);
                    }
                }
            }
        }
        _ => content.push(Value::Object(delta)),
    }
}

/// The image an image content item states, when it has data or a URI.
fn image_of(content: &Value) -> Option<Image> {
    let field = |key: &str| content.get(key).and_then(Value::as_str);
    let data = match (field("data"), field("uri")) {
        (Some(data), _) => DocumentSourceKind::Base64(data.to_owned()),
        (None, Some(uri)) => DocumentSourceKind::Url(uri.to_owned()),
        (None, None) => return None,
    };
    Some(Image {
        data,
        media_type: field("mime_type").and_then(ImageMediaType::from_mime_type),
        detail: None,
        native: None,
    })
}

/// EOF without `interaction.completed` is truncation, not successful
/// completion, so the decoder has nothing to add at the end of the reply.
impl<'id> Decoder<'id, Completion> for InteractionsDecoder {
    type Event = InteractionsEvent;

    fn classify(&self, frame: WireFrame) -> WireEvent<InteractionsEvent> {
        classify_interactions_frame(&frame.as_str())
    }

    fn decode(
        &mut self,
        event: InteractionsEvent,
        mut out: Out<'id, Completion>,
    ) -> Result<Flow, ProviderError> {
        let event = match event {
            InteractionsEvent::Sse(event) => event,
            InteractionsEvent::Whole(interaction) => {
                let steps = interaction
                    .get("steps")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default();
                for (index, step) in steps.into_iter().enumerate() {
                    self.start(index, step, &mut out)?;
                    self.stop(index, &mut out)?;
                }
                return Ok(self.complete(interaction, out));
            }
        };
        let SseEvent {
            event_type,
            fields: event,
        } = event;
        let index = || {
            event
                .get("index")
                .and_then(Value::as_u64)
                .and_then(|index| usize::try_from(index).ok())
                .ok_or_else(|| {
                    ProviderError::Response(format!(
                        "an Interactions `{event_type}` event names no step index"
                    ))
                })
        };
        match event_type.as_str() {
            "step.start" => match event.get("step") {
                // A start that states no step leaves its deltas to open it.
                None => {}
                Some(step @ Value::Object(_)) => self.start(index()?, step.clone(), &mut out)?,
                Some(_) => return Err(malformed("step.start", "step")),
            },
            "step.delta" => match event.get("delta") {
                Some(Value::Object(delta)) => self.delta(index()?, delta.clone(), &mut out)?,
                _ => return Err(malformed("step.delta", "delta")),
            },
            "step.stop" => self.stop(index()?, &mut out)?,
            "interaction.created" => {
                let interaction = event.get("interaction");
                let field = |key: &str| {
                    interaction
                        .and_then(|interaction| interaction.get(key))
                        .and_then(Value::as_str)
                        .map(str::to_owned)
                };
                self.model = field("model").or_else(|| field("agent"));
                self.status = field("status");
            }
            "interaction.status_update" => {
                if let Some(status) = event.get("status").and_then(Value::as_str) {
                    self.status = Some(status.to_owned());
                }
            }
            "interaction.completed" => {
                let interaction = match event.get("interaction") {
                    Some(Value::Object(interaction)) => interaction.clone(),
                    _ => Map::new(),
                };
                return Ok(self.complete(interaction, out));
            }
            "error" => {
                let mut body = event;
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
