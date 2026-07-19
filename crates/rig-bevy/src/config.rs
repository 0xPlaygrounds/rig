//! Runtime and agent construction data.

use std::{fmt, time::Duration};

use rig_core::message::ToolChoice;
use schemars::{JsonSchema, Schema};
use serde::{Deserialize, Serialize};

use crate::{InvalidToolPolicy, MemoryId, ModelId, StructuredOutputPolicy, TenantId};

/// Whether model effects use blocking or provider-streaming completion.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub enum StreamingMode {
    /// Use `CompletionModel::completion`.
    #[default]
    Blocking,
    /// Use `CompletionModel::stream` and expose provisional events.
    Streaming,
}

/// ECS-native response acceptance and retry policy.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[non_exhaustive]
pub enum ResponseRetryPolicy {
    /// Accept every tool-free model response.
    #[default]
    Accept,
    /// Reject an empty tool-free text response and retry with feedback.
    RejectEmpty {
        /// Maximum response-retry attempts.
        max_retries: usize,
    },
}

/// Bounded operational configuration for one runtime world.
#[derive(Clone, Debug)]
pub struct RuntimeConfig {
    /// Capacity of the owned effect ingress queue.
    pub ingress_capacity: usize,
    /// Maximum owned effects queued before execution permits are available.
    pub effect_queue_capacity: usize,
    /// Maximum lifecycle events retained and broadcast per run.
    pub event_capacity: usize,
    /// Maximum rejected ingress audit records retained by the runtime.
    pub rejection_capacity: usize,
    /// Maximum effects executing at once.
    pub max_effects: usize,
    /// Maximum model calls executing at once.
    pub max_model_calls: usize,
    /// Maximum tool calls executing at once.
    pub max_tool_calls: usize,
    /// Maximum schedule passes without quiescence.
    pub max_schedule_passes: usize,
    /// Timeout applied independently to each effect.
    pub effect_timeout: Duration,
    /// Number of schedule ticks to retain an observed terminal run.
    pub terminal_retention_ticks: u64,
    /// Number of schedule ticks to retain a terminal run that was never observed.
    ///
    /// This bounds state held for abandoned handles; it should be generous enough
    /// that a caller stepping other runs can still observe a finished one.
    pub unobserved_terminal_retention_ticks: u64,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            ingress_capacity: 256,
            effect_queue_capacity: 256,
            event_capacity: 512,
            rejection_capacity: 512,
            max_effects: 32,
            max_model_calls: 8,
            max_tool_calls: 16,
            max_schedule_passes: 1_024,
            effect_timeout: Duration::from_secs(120),
            terminal_retention_ticks: 16,
            unobserved_terminal_retention_ticks: 1_024,
        }
    }
}

impl RuntimeConfig {
    pub(crate) fn validate(&self) -> Result<(), crate::RuntimeError> {
        let bounds = [
            ("ingress_capacity", self.ingress_capacity),
            ("effect_queue_capacity", self.effect_queue_capacity),
            ("event_capacity", self.event_capacity),
            ("rejection_capacity", self.rejection_capacity),
            ("max_effects", self.max_effects),
            ("max_model_calls", self.max_model_calls),
            ("max_tool_calls", self.max_tool_calls),
            ("max_schedule_passes", self.max_schedule_passes),
        ];
        if let Some((field, _)) = bounds.into_iter().find(|(_, value)| *value == 0) {
            return Err(crate::RuntimeError::InvalidConfiguration { field });
        }
        if self.unobserved_terminal_retention_ticks == 0 {
            return Err(crate::RuntimeError::InvalidConfiguration {
                field: "unobserved_terminal_retention_ticks",
            });
        }
        Ok(())
    }
}

/// Immutable agent configuration stored in ECS topology.
#[derive(Clone, Deserialize, Serialize)]
pub struct AgentSpec {
    /// Stable model binding selected for the agent.
    pub model_id: ModelId,
    /// Tenant that owns the agent and every run it spawns.
    pub tenant_id: TenantId,
    /// Optional observability name, recorded only when telemetry content is explicitly enabled.
    pub name: Option<String>,
    /// Optional system preamble.
    pub preamble: Option<String>,
    /// Total model-call budget including initial calls, retries, and continuations.
    pub max_model_calls: usize,
    /// Invalid-tool policy.
    pub invalid_tool_policy: InvalidToolPolicy,
    /// Tool-free response retry policy.
    pub response_retry_policy: ResponseRetryPolicy,
    /// Optional structured output schema and recovery policy.
    pub structured_output: Option<(Schema, StructuredOutputPolicy)>,
    /// Optional conversation memory binding.
    pub memory_id: Option<MemoryId>,
    /// Optional conversation identifier used by memory effects.
    pub conversation_id: Option<String>,
    /// Provider tool-choice request policy.
    pub tool_choice: Option<ToolChoice>,
    /// Sampling temperature forwarded to every model request.
    pub temperature: Option<f64>,
    /// Maximum generated tokens forwarded to every model request.
    pub max_tokens: Option<u64>,
    /// Provider-specific request parameters.
    pub additional_params: Option<serde_json::Value>,
    /// Whether sensitive provider telemetry content may be recorded.
    pub record_telemetry_content: bool,
    /// Default model execution surface for new runs.
    pub streaming: StreamingMode,
}

impl fmt::Debug for AgentSpec {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AgentSpec")
            .field("model_id", &self.model_id)
            .field("tenant_id", &"<redacted>")
            .field("name_configured", &self.name.is_some())
            .field("preamble_configured", &self.preamble.is_some())
            .field("max_model_calls", &self.max_model_calls)
            .field("invalid_tool_policy", &self.invalid_tool_policy)
            .field("response_retry_policy", &self.response_retry_policy)
            .field(
                "structured_output_configured",
                &self.structured_output.is_some(),
            )
            .field("memory_id", &self.memory_id)
            .field(
                "conversation_id_configured",
                &self.conversation_id.is_some(),
            )
            .field("tool_choice", &self.tool_choice)
            .field("temperature_configured", &self.temperature.is_some())
            .field("max_tokens_configured", &self.max_tokens.is_some())
            .field(
                "additional_params_configured",
                &self.additional_params.is_some(),
            )
            .field("record_telemetry_content", &self.record_telemetry_content)
            .field("streaming", &self.streaming)
            .finish()
    }
}

impl AgentSpec {
    /// Construct an agent specification for one rebound model and tenant.
    #[must_use]
    pub fn new(model_id: ModelId, tenant_id: TenantId) -> Self {
        Self {
            model_id,
            tenant_id,
            name: None,
            preamble: None,
            max_model_calls: 8,
            invalid_tool_policy: InvalidToolPolicy::Fail,
            response_retry_policy: ResponseRetryPolicy::Accept,
            structured_output: None,
            memory_id: None,
            conversation_id: None,
            tool_choice: None,
            temperature: None,
            max_tokens: None,
            additional_params: None,
            record_telemetry_content: false,
            streaming: StreamingMode::Blocking,
        }
    }

    /// Set an optional diagnostic name.
    #[must_use]
    pub fn name(mut self, value: impl Into<String>) -> Self {
        self.name = Some(value.into());
        self
    }

    /// Set the system preamble.
    #[must_use]
    pub fn preamble(mut self, value: impl Into<String>) -> Self {
        self.preamble = Some(value.into());
        self
    }

    /// Set the total model-call budget. Zero prevents the initial call.
    #[must_use]
    pub fn max_model_calls(mut self, value: usize) -> Self {
        self.max_model_calls = value;
        self
    }

    /// Select the invalid-tool policy.
    #[must_use]
    pub fn invalid_tool_policy(mut self, value: InvalidToolPolicy) -> Self {
        self.invalid_tool_policy = value;
        self
    }

    /// Select the response retry policy.
    #[must_use]
    pub fn response_retry_policy(mut self, value: ResponseRetryPolicy) -> Self {
        self.response_retry_policy = value;
        self
    }

    /// Configure typed structured output.
    #[must_use]
    pub fn structured_output<T>(mut self, policy: StructuredOutputPolicy) -> Self
    where
        T: JsonSchema,
    {
        self.structured_output = Some((schemars::schema_for!(T), policy));
        self
    }

    /// Configure a raw structured-output schema.
    #[must_use]
    pub fn structured_output_raw(mut self, schema: Schema, policy: StructuredOutputPolicy) -> Self {
        self.structured_output = Some((schema, policy));
        self
    }

    /// Configure conversation memory.
    #[must_use]
    pub fn memory(mut self, memory_id: MemoryId, conversation_id: impl Into<String>) -> Self {
        self.memory_id = Some(memory_id);
        self.conversation_id = Some(conversation_id.into());
        self
    }

    /// Select provider tool choice.
    #[must_use]
    pub fn tool_choice(mut self, value: ToolChoice) -> Self {
        self.tool_choice = Some(value);
        self
    }

    /// Set the sampling temperature used for model requests.
    #[must_use]
    pub fn temperature(mut self, value: f64) -> Self {
        self.temperature = Some(value);
        self
    }

    /// Set the maximum generated-token count used for model requests.
    #[must_use]
    pub fn max_tokens(mut self, value: u64) -> Self {
        self.max_tokens = Some(value);
        self
    }

    /// Set provider-specific parameters.
    #[must_use]
    pub fn additional_params(mut self, value: serde_json::Value) -> Self {
        self.additional_params = Some(value);
        self
    }

    /// Opt in to sensitive telemetry content for this agent.
    #[must_use]
    pub fn record_telemetry_content(mut self, value: bool) -> Self {
        self.record_telemetry_content = value;
        self
    }

    /// Select the default blocking or streaming model surface.
    #[must_use]
    pub fn streaming(mut self, value: StreamingMode) -> Self {
        self.streaming = value;
        self
    }
}
