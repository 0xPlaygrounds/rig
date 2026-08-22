//! The protocol-facing configuration of a run, as plain data.
//!
//! [`RunSpec`] is the half of an agent definition that the protocol consumes:
//! prompt-shaping (preamble, static context, sampling parameters, additional
//! params), the turn budget, tool choice, structured-output policy. It carries
//! no model, no tools, no hooks, no memory — those are a driver's. Being plain
//! `Serialize + Deserialize` data it can be stored, diffed, loaded from a file,
//! or kept as an ECS component; [`AgentRun::from_spec`] turns it into a run.

use rig_core::completion::Document;
use rig_core::message::{Message, ToolChoice};
use serde::{Deserialize, Serialize};

use crate::output_mode::OutputMode;
use crate::run::{AgentRun, DEFAULT_OUTPUT_RETRIES};

/// Protocol-facing run configuration. See the [module docs](self).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RunSpec {
    /// System prompt.
    pub preamble: Option<String>,
    /// Documents always attached to the request.
    pub static_context: Vec<Document>,
    /// Provider-specific request parameters, merged into every request.
    pub additional_params: Option<serde_json::Value>,
    /// Maximum tokens the model may generate per call.
    pub max_tokens: Option<u64>,
    /// Sampling temperature.
    pub temperature: Option<f64>,
    /// Tool choice applied to every model call.
    pub tool_choice: Option<ToolChoice>,
    /// Total model-call budget for the run. `None` means the protocol default
    /// of one call.
    pub max_turns: Option<usize>,
    /// How many times an invalid model tool call may be retried with feedback.
    pub max_invalid_tool_call_retries: usize,
    /// JSON schema the final answer must satisfy, when structured output is
    /// requested.
    pub output_schema: Option<serde_json::Value>,
    /// How structured output is obtained.
    pub output_mode: OutputMode,
    /// Name of the synthetic output tool when output is collected through a
    /// tool call; `None` lets the driver pick.
    pub output_tool_name: Option<String>,
    /// Description of that synthetic tool.
    pub output_tool_description: Option<String>,
    /// Whether the driver may augment the preamble with structured-output
    /// instructions.
    pub augment_output_preamble: bool,
}

impl RunSpec {
    /// A spec with no preamble, a one-call budget, no tools choice and no
    /// structured output — the same defaults an `AgentRun::new` carries.
    pub fn new() -> Self {
        Self {
            augment_output_preamble: true,
            ..Self::default()
        }
    }

    /// The turn budget the protocol will use.
    pub fn effective_max_turns(&self) -> usize {
        self.max_turns.unwrap_or(1)
    }
}

impl AgentRun {
    /// Build a run from a [`RunSpec`], a prompt and an optional prior history.
    ///
    /// Applies the spec's budget, invalid-call retries, output validation and
    /// tool choice; everything else in the spec is request-shaping the driver
    /// reads when it prepares each model call.
    pub fn from_spec(
        spec: &RunSpec,
        prompt: impl Into<Message>,
        history: Option<Vec<Message>>,
    ) -> Self {
        let mut run = AgentRun::new(prompt)
            .max_turns(spec.effective_max_turns())
            .max_invalid_tool_call_retries(spec.max_invalid_tool_call_retries)
            .with_output_validation(spec.output_schema.clone(), DEFAULT_OUTPUT_RETRIES);
        if let Some(history) = history {
            run = run.with_history(history);
        }
        if let Some(tool_choice) = spec.tool_choice.clone() {
            run = run.with_tool_choice(tool_choice);
        }
        if let Some(name) = spec.output_tool_name.clone() {
            run = run.with_output_tool_name(name);
        }
        run
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_round_trips_through_json() {
        let spec = RunSpec {
            preamble: Some("be brief".into()),
            max_turns: Some(3),
            temperature: Some(0.2),
            output_schema: Some(serde_json::json!({"type": "object"})),
            ..RunSpec::new()
        };
        let json = serde_json::to_string(&spec).expect("serialize");
        let back: RunSpec = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, spec);
    }

    #[test]
    fn missing_fields_take_defaults() {
        let spec: RunSpec = serde_json::from_str("{}").expect("deserialize");
        assert_eq!(spec.effective_max_turns(), 1);
        assert!(spec.output_schema.is_none());
    }

    #[test]
    fn from_spec_matches_the_builder_chain() {
        let spec = RunSpec {
            max_turns: Some(4),
            max_invalid_tool_call_retries: 2,
            tool_choice: Some(ToolChoice::Auto),
            ..RunSpec::new()
        };
        let via_spec = AgentRun::from_spec(&spec, "hi", None);
        let via_chain = AgentRun::new("hi")
            .max_turns(4)
            .max_invalid_tool_call_retries(2)
            .with_output_validation(None, DEFAULT_OUTPUT_RETRIES)
            .with_tool_choice(ToolChoice::Auto);
        assert_eq!(
            serde_json::to_value(&via_spec).expect("serialize"),
            serde_json::to_value(&via_chain).expect("serialize")
        );
    }
}
