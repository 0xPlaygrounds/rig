//! Plain-data agent configuration.
//!
//! [`AgentConfig`] is the model-free half of an agent definition: everything
//! [`Agent`](super::Agent) holds except its four behavior slots (model,
//! hooks, memory, tool server). It is `Serialize + Deserialize` end to end,
//! so an agent's configuration can be persisted, diffed, and shipped across
//! processes alongside a suspended
//! [`AgentRun`](super::run::AgentRun) — and held as a component by hosts
//! (an ECS world, a job queue) that drive the run protocol directly.

use rig_core::message::ToolChoice;
use serde::{Deserialize, Serialize};

use super::run::OutputMode;
use crate::completion::Document;

/// Model-free agent configuration: plain data, no behavior slots.
///
/// Pair it with a provider selection (facade `ProviderConfig`) and a
/// [`ToolCatalog`](super::prepare::ToolCatalog) to drive an
/// [`AgentRun`](super::run::AgentRun) via
/// [`prepare_request`](super::prepare::prepare_request).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct AgentConfig {
    /// Agent name used for logging, debugging, and telemetry.
    pub name: Option<String>,
    /// Agent description, primarily useful for sub-agent workflows.
    pub description: Option<String>,
    /// System prompt.
    pub preamble: Option<String>,
    /// Context documents always provided to the model.
    pub static_context: Vec<Document>,
    /// Sampling temperature.
    pub temperature: Option<f64>,
    /// Maximum output-token count.
    pub max_tokens: Option<u64>,
    /// Provider-specific request parameters (genuinely schemaless
    /// passthrough).
    pub additional_params: Option<serde_json::Value>,
    /// Tool-choice policy.
    pub tool_choice: Option<ToolChoice>,
    /// Total model-call attempt budget, including retries, continuations, and
    /// provider-operation reissues after failure or cancellation.
    /// `None` uses the implicit budget of one.
    pub max_turns: Option<usize>,
    /// Retry budget for invalid tool-call recovery.
    pub max_invalid_tool_call_retries: usize,
    /// JSON Schema for structured output.
    pub output_schema: Option<schemars::Schema>,
    /// How `output_schema` is enforced (see [`OutputMode`] and #1928).
    pub output_mode: OutputMode,
    /// Name advertised for the synthetic output tool in Tool mode. `None`
    /// picks the collision-safe default (`final_result`); set it to express a
    /// bespoke extraction protocol (for example an output tool named
    /// `submit`). A run that has already committed a name on an earlier turn
    /// stays pinned to it.
    pub output_tool_name: Option<String>,
    /// Description advertised for the synthetic output tool in Tool mode.
    pub output_tool_description: Option<String>,
    /// Whether Tool output mode augments the preamble with calling
    /// instructions.
    pub augment_output_preamble: bool,
    /// Maximum tool calls executed concurrently within one turn's batch.
    pub tool_concurrency: usize,
    /// Whether to record sensitive request/response/tool content on GenAI
    /// telemetry spans.
    pub record_telemetry_content: bool,
}

impl AgentConfig {
    /// Create an empty configuration. `augment_output_preamble` defaults to
    /// true and `tool_concurrency` to 1, matching the classic builder's
    /// defaults; everything else starts unset.
    pub fn new() -> Self {
        Self {
            augment_output_preamble: true,
            tool_concurrency: 1,
            ..Self::default()
        }
    }

    /// Set the agent name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the system prompt.
    pub fn with_preamble(mut self, preamble: impl Into<String>) -> Self {
        self.preamble = Some(preamble.into());
        self
    }

    /// Set the sampling temperature.
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Set the maximum output-token count.
    pub fn with_max_tokens(mut self, max_tokens: u64) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Set the total model-call attempt budget.
    pub fn with_max_turns(mut self, max_turns: usize) -> Self {
        self.max_turns = Some(max_turns);
        self
    }

    /// Set the tool-choice policy.
    pub fn with_tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.tool_choice = Some(tool_choice);
        self
    }

    /// Pin the synthetic output tool's advertised name, description, and
    /// whether Tool output mode augments the preamble with calling
    /// instructions.
    pub fn with_output_tool(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        augment_preamble: bool,
    ) -> Self {
        self.output_tool_name = Some(name.into());
        self.output_tool_description = Some(description.into());
        self.augment_output_preamble = augment_preamble;
        self
    }

    /// Append a static context document.
    pub fn with_context(mut self, document: Document) -> Self {
        self.static_context.push(document);
        self
    }
}
