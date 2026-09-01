//! The protocol-facing configuration of a run, as plain data.
//!
//! [`RunSpec`] is the half of an agent definition that the protocol consumes:
//! prompt-shaping (preamble, static context, sampling parameters, additional
//! params), the turn budget, tool choice, structured-output policy. It carries
//! no model, no tools, no hooks, no memory — those are a driver's. Being plain
//! `Serialize + Deserialize` data it can be stored, diffed, loaded from a file,
//! or kept as an ECS component; `AgentRun::from_spec` (rig-agent) turns it into a run.

use rig_core::completion::Document;
use rig_core::message::ToolChoice;
use serde::{Deserialize, Serialize};

use super::output::OutputMode;

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

impl RunSpec {
    /// How many times a run re-prompts for structured output that failed
    /// validation before giving up. The default a run built from a spec
    /// carries; drivers that construct runs by hand pass it explicitly.
    pub const DEFAULT_OUTPUT_RETRIES: usize = 1;
}

#[cfg(test)]
mod tests;
