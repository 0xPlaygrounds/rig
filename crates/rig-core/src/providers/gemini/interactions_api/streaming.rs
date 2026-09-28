//! The Interactions decoder: a whole interaction resource, or its stream of
//! step events, written through the [`edge`].

use std::collections::BTreeMap;

use serde::Deserialize;
use serde_json::value::RawValue;
use serde_json::{Map, Value};

use super::api::{self, InteractionStatus};
use super::{InteractionsDialect, annotations};
use crate::completion::FinishReason;
use crate::error::ProviderError;
use crate::message::NativePart;
use crate::operation::{CallFragment, Completion, Finish, IfMalformed};
use crate::providers::gemini::edge::{self, Dialect, Unit};
use crate::providers::internal::wire;
use crate::wire::{Decoder, Flow, Out, WireEvent, WireFrame};

/// The stream events this decoder reads; others are unknown.
const EVENT_TYPES: &[&str] = &[
    "interaction.created",
    "interaction.completed",
    "interaction.status_update",
    "step.start",
    "step.delta",
    "step.stop",
    "error",
];

/// What marks a whole interaction resource.
const RESOURCE_KEYS: &[&str] = &["steps", "status", "usage", "object", "id"];

/// A stream event, or the whole resource a unary reply carries.
pub enum InteractionsEvent {
    /// One `event_type`-tagged event.
    Event(Box<api::Event>),
    /// The whole interaction, and each step's JSON as it arrived.
    Whole(Box<Whole>),
}

/// A whole interaction and its raw steps.
pub struct Whole {
    interaction: api::Interaction,
    steps: Vec<Box<RawValue>>,
}

impl<'de> Deserialize<'de> for Whole {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Steps {
            #[serde(default)]
            steps: Vec<Box<RawValue>>,
        }
        use serde::de::Error;
        let text = Box::<RawValue>::deserialize(deserializer)?;
        let steps: Steps = serde_json::from_str(text.get()).map_err(D::Error::custom)?;
        let interaction = serde_json::from_str(text.get()).map_err(D::Error::custom)?;
        Ok(Self {
            interaction,
            steps: steps.steps,
        })
    }
}

fn classify(data: &str) -> WireEvent<InteractionsEvent> {
    wire::classify_or(
        data,
        |data| {
            wire::classify_tagged_frame::<api::Event>(data, "event_type", |event_type| {
                EVENT_TYPES.contains(&event_type)
            })
            .map(|event| InteractionsEvent::Event(Box::new(event)))
        },
        |data| {
            wire::classify_marker_keyed_frame::<Whole>(data, RESOURCE_KEYS)
                .map(|whole| InteractionsEvent::Whole(Box::new(whole)))
        },
    )
}

/// A streamed step, as its events have described it so far.
struct Open {
    kind: String,
    /// A hosted or unknown step: its start, with each delta laid over it.
    merged: Map<String, Value>,
}

/// Decodes one interaction, whole or streamed.
#[derive(Default)]
pub struct InteractionsDecoder<'id> {
    writer: edge::Writer<'id>,
    steps: BTreeMap<usize, Open>,
}

impl<'id> Decoder<'id, Completion> for InteractionsDecoder<'id> {
    type Event = InteractionsEvent;

    fn classify(&self, frame: WireFrame) -> WireEvent<InteractionsEvent> {
        classify(&frame.as_str())
    }

    fn decode(
        &mut self,
        event: InteractionsEvent,
        mut out: Out<'id, Completion>,
    ) -> Result<Flow, ProviderError> {
        match event {
            InteractionsEvent::Whole(whole) => {
                let Whole { interaction, steps } = *whole;
                for raw in &steps {
                    for unit in InteractionsDialect::units(raw)? {
                        let answers = !hosted(&unit);
                        self.writer.unit(unit, answers, &mut out)?;
                    }
                }
                self.complete(interaction, out)
            }
            InteractionsEvent::Event(event) => self.event(*event, out),
        }
    }
}

/// Whether `unit` is a hosted tool's step, which alone is no answer.
fn hosted(unit: &Unit) -> bool {
    match unit {
        Unit::Native(native) => {
            native.schema == api::STEP_SCHEMA
                && serde_json::from_str::<Map<String, Value>>(native.json())
                    .ok()
                    .and_then(|step| step.get("type").and_then(Value::as_str).map(api::is_hosted))
                    .unwrap_or(false)
        }
        _ => false,
    }
}

impl<'id> InteractionsDecoder<'id> {
    fn event(
        &mut self,
        event: api::Event,
        mut out: Out<'id, Completion>,
    ) -> Result<Flow, ProviderError> {
        let index = event.index.unwrap_or_default();
        match event.event_type.as_str() {
            "step.start" => {
                if let Some(step) = event.step {
                    self.start(index, &step, &mut out)?;
                }
            }
            "step.delta" => {
                if let Some(delta) = event.delta {
                    self.delta(index, &delta, &mut out)?;
                }
            }
            "step.stop" => self.stop(index, &mut out)?,
            "interaction.completed" => {
                return self.complete(event.interaction.unwrap_or_default(), out);
            }
            "error" => {
                let body = serde_json::json!({ "error": event.error }).to_string();
                return Err(ProviderError::from_provider_body(body));
            }
            _ => {}
        }
        Ok(Flow::More)
    }

    fn start(
        &mut self,
        index: usize,
        step: &RawValue,
        out: &mut Out<'id, Completion>,
    ) -> Result<(), ProviderError> {
        let merged: Map<String, Value> = serde_json::from_str(step.get())?;
        let kind = merged
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        match kind.as_str() {
            "function_call" => {
                let api::Step::FunctionCall(call) = serde_json::from_str(step.get())? else {
                    return Ok(());
                };
                self.writer.interrupt(out);
                out.call_fragment(
                    index,
                    CallFragment {
                        id: call.id.as_deref(),
                        name: call.name.as_deref(),
                        ..CallFragment::default()
                    },
                )?;
                // Announced arguments are a fallback, not a fragment to
                // append to.
                if let Some(arguments) = call.arguments.filter(|arguments| !arguments.is_empty()) {
                    out.announce_pending(index, Value::Object(arguments));
                }
                if call.signature.is_some() {
                    out.decorate_pending_at(index, call.signature, None);
                }
            }
            "model_output" | "thought" => {
                // A start that already states content writes it now.
                for unit in InteractionsDialect::units(step)? {
                    let empty = matches!(
                        &unit,
                        Unit::Thought { text, signature: None } if text.is_empty()
                    );
                    if !empty {
                        self.writer.unit(unit, true, out)?;
                    }
                }
            }
            _ => {}
        }
        self.steps.insert(index, Open { kind, merged });
        Ok(())
    }

    fn delta(
        &mut self,
        index: usize,
        delta: &RawValue,
        out: &mut Out<'id, Completion>,
    ) -> Result<(), ProviderError> {
        let fields: Map<String, Value> = serde_json::from_str(delta.get())?;
        let kind = fields
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        let step_kind = self.steps.get(&index).map(|open| open.kind.clone());
        match kind.as_str() {
            "text" => {
                let text = fields
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned();
                self.writer.unit(
                    Unit::Text {
                        text,
                        signature: None,
                    },
                    true,
                    out,
                )?;
                if let Some(Value::Array(items)) = fields.get("annotations") {
                    self.writer.unit(annotations(items.clone())?, false, out)?;
                }
            }
            "thought_summary" => {
                let text = fields
                    .get("content")
                    .and_then(|content| content.get("text"))
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned();
                self.writer.unit(
                    Unit::Thought {
                        text,
                        signature: None,
                    },
                    true,
                    out,
                )?;
            }
            "thought_signature" => {
                let signature = fields
                    .get("signature")
                    .and_then(Value::as_str)
                    .map(str::to_owned);
                self.writer.unit(
                    Unit::Thought {
                        text: String::new(),
                        signature,
                    },
                    true,
                    out,
                )?;
            }
            "arguments_delta" => {
                if let Some(fragment) = fields.get("arguments").and_then(Value::as_str) {
                    out.call_fragment(
                        index,
                        CallFragment {
                            arguments: Some(fragment),
                            ..CallFragment::default()
                        },
                    )?;
                }
            }
            "function_call" => {
                let api::Step::FunctionCall(call) = serde_json::from_str(delta.get())? else {
                    return Ok(());
                };
                // A delta that carries its arguments is the whole call.
                if let (Some(name), Some(args)) = (call.name.clone(), call.arguments.clone()) {
                    return self.writer.unit(
                        Unit::Call {
                            id: call.id,
                            name,
                            args,
                            signature: call.signature,
                        },
                        true,
                        out,
                    );
                }
                self.writer.interrupt(out);
                out.call_fragment(
                    index,
                    CallFragment {
                        id: call.id.as_deref(),
                        name: call.name.as_deref(),
                        ..CallFragment::default()
                    },
                )?;
                if let Some(arguments) = call.arguments.filter(|arguments| !arguments.is_empty()) {
                    out.announce_pending(index, Value::Object(arguments));
                }
                if call.signature.is_some() {
                    out.decorate_pending_at(index, call.signature, None);
                }
            }
            _ if step_kind.as_deref().is_some_and(|step| step == kind) => {
                // A hosted or unknown step restates itself in its delta.
                if let Some(open) = self.steps.get_mut(&index) {
                    for (key, value) in fields {
                        let blank = value.as_str().is_some_and(str::is_empty);
                        if !blank || !open.merged.contains_key(&key) {
                            open.merged.insert(key, value);
                        }
                    }
                }
            }
            _ => {
                // Content a model output carries that rig has no type for.
                self.writer
                    .unit(edge::native(delta, api::CONTENT_SCHEMA), true, out)?;
            }
        }
        Ok(())
    }

    fn stop(&mut self, index: usize, out: &mut Out<'id, Completion>) -> Result<(), ProviderError> {
        let Some(open) = self.steps.remove(&index) else {
            return Ok(());
        };
        match open.kind.as_str() {
            "function_call" => out.close_pending(index, IfMalformed::Fail)?,
            "model_output" | "thought" => {}
            kind => {
                let json = if api::is_hosted(kind) {
                    let mut step: api::HostedStep =
                        serde_json::from_value(Value::Object(open.merged))?;
                    step.restore();
                    serde_json::value::to_raw_value(&step)?
                } else {
                    serde_json::value::to_raw_value(&open.merged)?
                };
                let answers = !api::is_hosted(kind);
                self.writer.unit(
                    Unit::Native(NativePart::new(api::STEP_SCHEMA, json)),
                    answers,
                    out,
                )?;
            }
        }
        Ok(())
    }

    fn complete(
        &mut self,
        interaction: api::Interaction,
        mut out: Out<'id, Completion>,
    ) -> Result<Flow, ProviderError> {
        let span = tracing::Span::current();
        if let Some(id) = &interaction.id {
            span.record("gen_ai.response.id", id.as_str());
        }
        if let Some(model) = &interaction.model {
            span.record("gen_ai.response.model", model.as_str());
        }
        // Completion finalizes calls even without their step.stop.
        for index in out.pending_calls() {
            out.close_pending(index, IfMalformed::Fail)?;
        }
        self.writer.close(&mut out);
        let usage = interaction
            .usage
            .as_ref()
            .map(counts)
            .map(edge::usage)
            .unwrap_or_default();
        let reason = interaction.status.as_ref().map(finish_reason);
        let response_id = interaction.id.clone().filter(|id| !id.is_empty());
        let model = interaction.model.clone();
        if let Ok(raw) = serde_json::to_value(&interaction) {
            out.raw(raw);
        }
        Ok(out.end(
            Finish::new(usage)
                .with_optional_reason(reason)
                .with_optional_response_id(response_id)
                .with_optional_model(model),
        ))
    }
}

/// Interactions' counts under the edge's names.
pub(crate) fn counts(usage: &api::Usage) -> edge::Counts {
    edge::Counts {
        prompt: usage.total_input_tokens,
        tool_use_prompt: usage.total_tool_use_tokens,
        cached: usage.total_cached_tokens,
        candidates: usage.total_output_tokens,
        thoughts: usage.total_thought_tokens,
    }
}

/// Rig's finish reason for an interaction's status.
pub(crate) fn finish_reason(status: &InteractionStatus) -> FinishReason {
    match status {
        InteractionStatus::Completed => FinishReason::Stop,
        InteractionStatus::RequiresAction => FinishReason::ToolCalls,
        InteractionStatus::BudgetExceeded => FinishReason::Length,
        other => FinishReason::Other(other.as_str().to_owned()),
    }
}
