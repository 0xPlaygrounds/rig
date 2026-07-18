//! Focused components for agent, run, operation, and terminal state.

use bevy_ecs::component::Component;
use rig_core::{completion::Usage, message::Message};
use serde::{Deserialize, Serialize};

use crate::topology::{
    AgentId, CapabilityId, EffectIdentity, Generation, OperationId, RunId, StoreOperationId,
    TenantId, ToolCallId,
};

/// Provider-agnostic model binding name. The implementation lives in a runtime registry.
#[derive(Component, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelBinding(pub String);

/// Explicit memory/store implementation and conversation binding for an agent.
#[derive(Component, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StoreBinding {
    /// Host-rebound store implementation name.
    pub implementation: String,
    /// Stable logical conversation name.
    pub conversation: String,
}

impl std::fmt::Debug for StoreBinding {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("StoreBinding")
            .field("implementation", &self.implementation)
            .field("conversation", &"<redacted>")
            .finish()
    }
}

/// Stable agent domain record.
#[derive(Component, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentNode {
    /// Stable identity.
    pub id: AgentId,
    /// Tenant that owns this agent.
    pub tenant: TenantId,
    /// Optional human-readable name.
    pub name: Option<String>,
    /// Optional system instructions.
    pub preamble: Option<String>,
}

impl std::fmt::Debug for AgentNode {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AgentNode")
            .field("id", &self.id)
            .field("tenant", &self.tenant)
            .field("name", &self.name.as_ref().map(|_| "<redacted>"))
            .field("preamble", &self.preamble.as_ref().map(|_| "<redacted>"))
            .finish()
    }
}

/// Authoritative lifecycle phase for a run.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum RunPhase {
    /// Domain state exists but memory has not been loaded.
    Created,
    /// Waiting for a memory load effect.
    LoadingMemory,
    /// Ready to prepare and dispatch another operation.
    Ready,
    /// A model or tool effect is in flight.
    Waiting,
    /// Accepted state is available to terminal observers.
    Terminal,
    /// Retention elapsed and cleanup may remove supporting entities.
    CleanupEligible,
}

/// Stable run domain record. Transcript, usage, and terminal facts are separate components.
#[derive(Component, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunNode {
    /// Stable run identity.
    pub id: RunId,
    /// Stable owning agent.
    pub agent: AgentId,
    /// Tenant boundary.
    pub tenant: TenantId,
    /// Generation used to reject stale completions.
    pub generation: Generation,
    /// Current lifecycle phase.
    pub phase: RunPhase,
}

/// Total model-call budget and accounting. Retries and continuations consume the same budget.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CallBudget {
    /// Maximum dispatched calls.
    pub limit: usize,
    /// Calls already dispatched.
    pub dispatched: usize,
}

impl CallBudget {
    /// Whether another model operation may be dispatched.
    pub fn can_dispatch(&self) -> bool {
        self.dispatched < self.limit
    }

    /// Consume one call if available.
    pub fn consume(&mut self) -> bool {
        if !self.can_dispatch() {
            return false;
        }
        self.dispatched += 1;
        true
    }
}

/// Canonical committed transcript. Provisional stream data never enters this component.
#[derive(Component, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CommittedTranscript(pub Vec<Message>);

impl std::fmt::Debug for CommittedTranscript {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_tuple("CommittedTranscript")
            .field(&format_args!("<redacted:{}>", self.0.len()))
            .finish()
    }
}

/// Cumulative usage from billed completed model operations.
#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct UsageLedger(pub Usage);

/// Run-scoped structured-response recovery count.
#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResponseRecovery {
    /// Corrective model turns already dispatched for this run.
    pub retries: usize,
}

/// One model operation and its immutable dispatch identity.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModelOperation {
    /// Stable operation identity.
    pub id: OperationId,
    /// Full correlation tuple.
    pub effect: EffectIdentity,
    /// Whether ingress already committed a completion.
    pub committed: bool,
    /// Whether cancellation or supersession makes later ingress stale.
    pub retired: bool,
}

/// Terminal reason selected once by deterministic priority.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum TerminalReason {
    /// An accepted response completed normally.
    Completed,
    /// Total model-call budget was exhausted.
    BudgetExhausted,
    /// A caller requested cancellation.
    Cancelled(String),
    /// ECS policy requested an orderly stop.
    Stopped(String),
    /// A provider/model operation failed.
    ProviderFailure(String),
    /// Tool execution or policy failed the run.
    ToolFailure(String),
    /// Structured output exhausted its bounded recovery policy.
    OutputFailure(String),
    /// Memory or store operation failed.
    StoreFailure(String),
    /// Progress exceeded the configured livelock guard.
    Livelock,
}

impl std::fmt::Debug for TerminalReason {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Completed => formatter.write_str("Completed"),
            Self::BudgetExhausted => formatter.write_str("BudgetExhausted"),
            Self::Cancelled(_) => formatter
                .debug_tuple("Cancelled")
                .field(&"<redacted>")
                .finish(),
            Self::Stopped(_) => formatter
                .debug_tuple("Stopped")
                .field(&"<redacted>")
                .finish(),
            Self::ProviderFailure(_) => formatter
                .debug_tuple("ProviderFailure")
                .field(&"<redacted>")
                .finish(),
            Self::ToolFailure(_) => formatter
                .debug_tuple("ToolFailure")
                .field(&"<redacted>")
                .finish(),
            Self::OutputFailure(_) => formatter
                .debug_tuple("OutputFailure")
                .field(&"<redacted>")
                .finish(),
            Self::StoreFailure(_) => formatter
                .debug_tuple("StoreFailure")
                .field(&"<redacted>")
                .finish(),
            Self::Livelock => formatter.write_str("Livelock"),
        }
    }
}

impl TerminalReason {
    /// Deterministic race priority. Higher values win before terminal commit.
    pub fn priority(&self) -> u8 {
        match self {
            Self::Cancelled(_) => 7,
            Self::Stopped(_) => 6,
            Self::ProviderFailure(_) | Self::StoreFailure(_) => 5,
            Self::ToolFailure(_) | Self::OutputFailure(_) => 4,
            Self::BudgetExhausted | Self::Livelock => 3,
            Self::Completed => 1,
        }
    }
}

/// Externally observable terminal fact retained independently from cleanup.
#[derive(Component, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TerminalState {
    /// Winning reason.
    pub reason: TerminalReason,
    /// Monotonic schedule tick when committed.
    pub committed_tick: u64,
}

/// Per-run progress guard and quiescence observation.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProgressState {
    /// Schedule passes that changed authoritative facts.
    pub changes: u64,
    /// Consecutive passes without progress.
    pub idle_passes: u32,
    /// Maximum idle passes before livelock termination.
    pub max_idle_passes: u32,
}

/// Immutable per-turn advertised capability snapshot.
#[derive(Component, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CapabilitySnapshot {
    /// Stable snapshot identity.
    pub id: CapabilityId,
    /// Exact model operation that advertised this snapshot.
    pub effect: EffectIdentity,
    /// Owning run and generation.
    pub run: RunId,
    /// Owning tenant.
    pub tenant: TenantId,
    /// Ordered advertised tool revisions.
    pub tools: Vec<AdvertisedTool>,
}

/// One immutable advertised tool revision.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AdvertisedTool {
    /// Provider-facing tool name.
    pub name: String,
    /// Stable implementation revision.
    pub revision: u64,
    /// Provider-facing definition captured for this turn.
    pub definition: rig_core::completion::ToolDefinition,
}

/// Tool call state correlated to one immutable capability snapshot.
#[derive(Component, Clone, Debug, PartialEq, Eq)]
pub struct ToolCallNode {
    /// Stable call identity.
    pub id: ToolCallId,
    /// Exact model operation and correlation that produced the call.
    pub effect: EffectIdentity,
    /// Stable owning run.
    pub run: RunId,
    /// Snapshot that authorized this call.
    pub capability: CapabilityId,
    /// Tool name and exact revision selected from the snapshot.
    pub name: String,
    /// Exact implementation revision.
    pub revision: u64,
    /// Model-call order used for deterministic commit.
    pub ordinal: usize,
    /// Whether the body was suppressed before execution.
    pub suppressed: bool,
    /// Whether a result has committed.
    pub committed: bool,
}

/// One correlated memory/store operation. Backend implementations remain outside ECS.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub struct StoreOperation {
    /// Stable store-operation identity.
    pub id: StoreOperationId,
    /// Full correlation tuple captured before dispatch.
    pub effect: EffectIdentity,
    /// Whether ingress committed this operation exactly once.
    pub committed: bool,
    /// Whether cancellation or failure retired this operation.
    pub retired: bool,
}

/// Cancellation is a distinct fact from terminal cleanup.
#[derive(Component, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CancellationRequested(pub String);

impl std::fmt::Debug for CancellationRequested {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_tuple("CancellationRequested")
            .field(&"<redacted>")
            .finish()
    }
}

/// Retention deadline expressed in runtime schedule ticks.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RetainUntil(pub u64);

/// Retention duration applied when a terminal fact commits.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RetentionWindow(pub u64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_counts_initial_call_and_every_continuation() {
        let mut budget = CallBudget {
            limit: 2,
            dispatched: 0,
        };
        assert!(budget.consume());
        assert!(budget.consume());
        assert!(!budget.consume());
    }

    #[test]
    fn cancellation_wins_terminal_races() {
        assert!(
            TerminalReason::Cancelled("caller".into()).priority()
                > TerminalReason::Completed.priority()
        );
    }

    #[test]
    fn sensitive_component_debug_output_is_redacted() {
        let secret = "component-secret-4b67";
        let values = [
            format!(
                "{:?}",
                AgentNode {
                    id: AgentId(1),
                    tenant: TenantId(2),
                    name: Some(secret.into()),
                    preamble: Some(secret.into()),
                }
            ),
            format!(
                "{:?}",
                StoreBinding {
                    implementation: "memory".into(),
                    conversation: secret.into(),
                }
            ),
            format!("{:?}", CommittedTranscript(vec![Message::user(secret)])),
            format!(
                "{:?}",
                TerminalState {
                    reason: TerminalReason::ProviderFailure(secret.into()),
                    committed_tick: 1,
                }
            ),
            format!("{:?}", CancellationRequested(secret.into())),
        ];
        for value in values {
            assert!(!value.contains(secret), "Debug leaked sensitive content");
            assert!(value.contains("<redacted"), "missing marker: {value}");
        }
    }
}
