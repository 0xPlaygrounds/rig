use serde::{Deserialize, Serialize};

use std::collections::BTreeMap;

use super::interactions_api_types::{
    Content, FunctionCallContent, ImageContent, Interaction, InteractionSseEvent, InteractionUsage,
    Step, ThoughtSummaryContent, map_interaction_status,
};
use crate::error::ProviderError;
use crate::message::{AssistantContent, DocumentSourceKind, Image, ImageMediaType, MimeType};
use crate::operation::{Block, CallFragment, Completion, Finish};
use crate::providers::internal::wire;
use crate::wire::{Decoder, Flow, Out, WireEvent, WireFrame};
use serde_json::{Map, Value, json};

/// Recognized Interactions SSE tags. Listed events must decode fully;
/// unlisted tags classify as unknown.
const KNOWN_EVENT_TYPES: &[&str] = &[
    "interaction.created",
    "interaction.completed",
    "interaction.status_update",
    "step.start",
    "step.delta",
    "step.stop",
    "error",
];

/// Classify an Interactions SSE frame by its `event_type` tag.
fn classify_interaction_frame(data: &str) -> WireEvent<InteractionSseEvent> {
    wire::classify_tagged_frame(data, "event_type", |event_type| {
        KNOWN_EVENT_TYPES.contains(&event_type)
    })
}

/// Whole-resource markers that prevent malformed SSE frames from decoding
/// as default interactions.
const INTERACTION_MARKER_KEYS: &[&str] = &["steps", "status", "usage", "object", "id"];

/// A decoded SSE event or whole unary interaction resource.
pub enum InteractionsEvent {
    /// One `event_type`-tagged streaming event.
    Sse(InteractionSseEvent),
    /// The whole interaction resource, as the unary reply delivers it.
    Whole(WholeInteraction),
}

/// A whole interaction resource: its steps verbatim, beside the rest of the
/// resource.
#[derive(Deserialize)]
pub struct WholeInteraction {
    #[serde(default)]
    steps: Vec<Map<String, Value>>,
    #[serde(flatten)]
    interaction: Interaction,
}

/// Classify a tagged SSE event, falling back to whole-resource classification
/// through [`wire::classify_or_untagged`]. Only an untagged frame falls back:
/// a frame carrying `event_type` that fails its typed decode stays corrupt,
/// rather than passing for a whole interaction and ending the reply.
fn classify_interactions_frame(data: &str) -> WireEvent<InteractionsEvent> {
    wire::classify_or_untagged(
        data,
        "event_type",
        |data| classify_interaction_frame(data).map(InteractionsEvent::Sse),
        |data| {
            wire::classify_marker_keyed_frame::<WholeInteraction>(data, INTERACTION_MARKER_KEYS)
                .map(InteractionsEvent::Whole)
        },
    )
}

/// Final metadata yielded by an Interactions streaming response.
#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub struct StreamingCompletionResponse {
    pub usage: Option<InteractionUsage>,
    pub interaction: Option<Interaction>,
    /// Resolved model identifier (e.g. `gemini-2.5-pro-preview-05-06`), extracted from
    /// `Interaction.model`. The Interactions API has no `FinishReason` field; use
    /// `interaction.status` for lifecycle state.
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

/// The Gemini Interactions wire's decoder. Each step of the interaction is
/// one block, in step order: it opens on `step.start`, grows with its
/// `step.delta`s and closes on `step.stop`, with the step, as its deltas
/// rebuilt it, as the block's provider item. A whole interaction is
/// restated step by step through the same calls.
#[derive(Default)]
pub struct InteractionsDecoder {
    /// The open steps, by wire index.
    steps: BTreeMap<usize, Kind>,
}

/// What an open step decodes to.
enum Kind {
    Thought,
    Output,
    /// A function call, with the argument JSON streamed so far.
    Call(String),
    Opaque,
}

impl InteractionsDecoder {
    /// Open the step at `index` as it starts, writing the content it
    /// already states.
    #[deny(clippy::wildcard_enum_match_arm)]
    fn start(
        &mut self,
        index: usize,
        step: Map<String, Value>,
        out: &mut Out<'_, Completion>,
    ) -> Result<(), ProviderError> {
        let item = Value::Object(step);
        let kind = match parse_step(&item)? {
            Some(Step::Thought(thought)) => {
                out.open(index, Block::Reasoning { redacted: false }, item)?;
                for summary in thought.summary.into_iter().flatten() {
                    if let ThoughtSummaryContent::Text(text) = summary {
                        out.push(index, &text.text)?;
                    }
                }
                Kind::Thought
            }
            Some(Step::ModelOutput { content }) => {
                out.open(index, Block::Text, item)?;
                for content in content {
                    if let Content::Text(text) = content {
                        out.push(index, &text.text)?;
                    }
                }
                Kind::Output
            }
            Some(Step::FunctionCall(call)) => {
                out.fragment(
                    index,
                    CallFragment {
                        id: call.id.as_deref(),
                        name: call.name.as_deref(),
                        arguments: None,
                    },
                )?;
                if let Some(arguments) = call.arguments {
                    out.announce(index, arguments)?;
                }
                out.edit(index, |slot| *slot = item)?;
                Kind::Call(String::new())
            }
            // Input a reply restates: kept, never sent back.
            Some(Step::UserInput { .. } | Step::FunctionResult(_)) => {
                out.open(index, Block::Opaque { replay: false }, item)?;
                Kind::Opaque
            }
            // Hosted-tool steps, and step types rig does not know, go back
            // to the model that produced them.
            Some(
                Step::CodeExecutionCall(_)
                | Step::CodeExecutionResult(_)
                | Step::UrlContextCall(_)
                | Step::UrlContextResult(_)
                | Step::GoogleSearchCall(_)
                | Step::GoogleSearchResult(_)
                | Step::McpServerToolCall(_)
                | Step::McpServerToolResult(_)
                | Step::FileSearchResult(_),
            )
            | None => {
                out.open(index, Block::Opaque { replay: true }, item)?;
                Kind::Opaque
            }
        };
        self.steps.insert(index, kind);
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
        let kind_of = |delta: &Map<String, Value>| {
            delta
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned()
        };
        let delta_type = kind_of(&delta);
        if !self.steps.contains_key(&index) {
            let step = match delta_type.as_str() {
                "text" | "image" | "audio" | "document" | "video" => "model_output",
                "thought_summary" | "thought_signature" => "thought",
                "arguments_delta" => "function_call",
                other => other,
            };
            let step = Map::from_iter([("type".to_owned(), Value::from(step))]);
            self.start(index, step, out)?;
        }
        let Some(kind) = self.steps.get_mut(&index) else {
            return Ok(());
        };
        match (kind, delta_type.as_str()) {
            (Kind::Output, "text") => {
                if let Some(text) = delta.get("text").and_then(Value::as_str) {
                    out.push(index, text)?;
                }
                out.edit(index, |item| extend_text(item, delta))
            }
            // Media arrive whole, one content item per delta.
            (Kind::Output, _) => out.edit(index, |item| {
                if let Some(content) = content(item) {
                    content.push(Value::Object(delta));
                }
            }),
            (Kind::Thought, "thought_summary") => {
                let content = delta.get("content").cloned().unwrap_or_default();
                if let Some(text) = content
                    .get("type")
                    .filter(|kind| *kind == "text")
                    .and(content.get("text"))
                    .and_then(Value::as_str)
                {
                    out.push(index, text)?;
                }
                out.merge(
                    index,
                    &Map::from_iter([("summary".to_owned(), json!([content]))]),
                )
            }
            (Kind::Call(arguments), "arguments_delta") => {
                let fragment = delta
                    .get("arguments")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                arguments.push_str(fragment);
                out.fragment(
                    index,
                    CallFragment {
                        arguments: Some(fragment),
                        ..CallFragment::default()
                    },
                )
            }
            (kind, _) => {
                if let Kind::Call(_) = kind
                    && delta_type == "function_call"
                {
                    let call: FunctionCallContent =
                        serde_json::from_value(Value::Object(delta.clone()))?;
                    out.fragment(
                        index,
                        CallFragment {
                            id: call.id.as_deref(),
                            name: call.name.as_deref(),
                            arguments: None,
                        },
                    )?;
                    if let Some(arguments) = call.arguments {
                        out.announce(index, arguments)?;
                    }
                }
                restate_or_merge(index, delta, out)
            }
        }
    }

    /// Close the step at `index`. A model output that is a single image
    /// becomes an image block.
    fn stop(&mut self, index: usize, out: &mut Out<'_, Completion>) -> Result<(), ProviderError> {
        match self.steps.remove(&index) {
            None => Ok(()),
            Some(Kind::Call(arguments)) => {
                // Streamed fragments supersede the arguments the start
                // announced.
                if !arguments.is_empty()
                    && let Ok(arguments) = crate::json_utils::parse_tool_arguments(&arguments)
                {
                    out.edit(index, |item| item["arguments"] = arguments)?;
                }
                out.finish(index)
            }
            Some(Kind::Output) => {
                let mut image = None;
                out.edit(index, |item| {
                    image = sole_image(item)
                        .map(|image| AssistantContent::Image(image).with_native(item.clone()));
                })?;
                match image {
                    Some(image) => {
                        out.discard(index);
                        out.content(image)
                    }
                    None => out.finish(index),
                }
            }
            Some(Kind::Thought | Kind::Opaque) => out.finish(index),
        }
    }
}

/// The step `item` is, or `None` for a step type rig does not know. A known
/// step that does not decode is an error.
fn parse_step(item: &Value) -> Result<Option<Step>, ProviderError> {
    match serde_json::from_value(item.clone()) {
        Ok(step) => Ok(Some(step)),
        Err(error) => {
            // Every step's fields are optional, so the tag alone decodes
            // exactly when the type is one rig knows.
            let tag = json!({ "type": item.get("type") });
            if serde_json::from_value::<Step>(tag).is_ok() {
                Err(ProviderError::Response(format!(
                    "malformed Interactions step: {error}"
                )))
            } else {
                Ok(None)
            }
        }
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
fn content(item: &mut Value) -> Option<&mut Vec<Value>> {
    item.as_object_mut()?
        .entry("content")
        .or_insert_with(|| json!([]))
        .as_array_mut()
}

/// Append a text delta to the step's last content item when that is text,
/// else add it as a new one.
fn extend_text(item: &mut Value, delta: Map<String, Value>) {
    let Some(content) = content(item) else {
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

/// The image a model output states alone.
fn sole_image(item: &Value) -> Option<Image> {
    let [image] = item.get("content")?.as_array()?.as_slice() else {
        return None;
    };
    if image.get("type")? != "image" {
        return None;
    }
    let image: ImageContent = serde_json::from_value(image.clone()).ok()?;
    let data = match (image.data, image.uri) {
        (Some(data), _) => DocumentSourceKind::Base64(data),
        (None, Some(uri)) => DocumentSourceKind::Url(uri),
        (None, None) => return None,
    };
    Some(Image {
        data,
        media_type: image
            .mime_type
            .as_deref()
            .and_then(ImageMediaType::from_mime_type),
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
            InteractionsEvent::Whole(WholeInteraction { steps, interaction }) => {
                for (index, step) in steps.into_iter().enumerate() {
                    self.start(index, step, &mut out)?;
                    self.stop(index, &mut out)?;
                }
                InteractionSseEvent::InteractionCompleted {
                    interaction,
                    event_id: None,
                }
            }
        };

        match event {
            InteractionSseEvent::StepStart { index, step, .. } => {
                self.start(index as usize, step, &mut out)?;
            }
            InteractionSseEvent::StepDelta { index, delta, .. } => {
                self.delta(index as usize, delta, &mut out)?;
            }
            InteractionSseEvent::StepStop { index, .. } => {
                self.stop(index as usize, &mut out)?;
            }
            InteractionSseEvent::InteractionCompleted { interaction, .. } => {
                // Provider completion finalizes steps even without step.stop.
                let open: Vec<usize> = self.steps.keys().copied().collect();
                for index in open {
                    self.stop(index, &mut out)?;
                }
                // Lifecycle status supplies the finish reason; absent status stays unknown.
                let model_version = interaction.model.clone();
                let native = StreamingCompletionResponse {
                    usage: interaction.usage,
                    interaction: Some(interaction),
                    model_version,
                };
                out.raw(serde_json::to_value(&native)?);
                let usage = (&native).into();
                let interaction = native.interaction.as_ref();
                let finish_reason = interaction
                    .and_then(|interaction| interaction.status.as_ref())
                    .map(map_interaction_status);
                let response_id = interaction.map(|interaction| interaction.id.clone());
                return Ok(out.end(Finish {
                    usage,
                    reason: finish_reason,
                    response_id,
                    model: native.model_version,
                    ..Finish::default()
                }));
            }
            event @ InteractionSseEvent::Error { .. } => {
                // Preserve modeled error fields without inventing an HTTP
                // status for an in-band failure.
                let body = serde_json::to_string(&event).unwrap_or_default();
                return Err(crate::error::ProviderError::from_provider_body(body));
            }
            InteractionSseEvent::InteractionCreated { .. }
            | InteractionSseEvent::InteractionStatusUpdate { .. } => {}
        }
        Ok(Flow::More)
    }
}

#[cfg(test)]
mod tests;
