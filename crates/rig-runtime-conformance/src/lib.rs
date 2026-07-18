//! Runtime-neutral scenario contracts used to test every Rig runtime.
//!
//! This crate is unpublished and is never re-exported by a production crate.
//! Adapters translate runtime-specific observations into [`Outcome`]; scenario
//! expectations deliberately make no assumptions about internal APIs.

use rig_core::completion::{Message, Usage};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// Every behavior that classic and ECS runtimes must exercise.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum Scenario {
    ModelCallBudgets,
    CanonicalTranscript,
    ToolPairing,
    UsageAccounting,
    InvalidToolRecovery,
    ResponseRetryRollback,
    StopAndCancellation,
    StructuredOutput,
    Memory,
    BlockingStreamingParity,
    ProviderFinalExposure,
    ProvisionalStreaming,
    ToolSuppression,
    Concurrency,
    StaleResultHandling,
}

impl Scenario {
    /// Complete ledger; adding a scenario requires updating both runtime adapters.
    pub const ALL: [Self; 15] = [
        Self::ModelCallBudgets,
        Self::CanonicalTranscript,
        Self::ToolPairing,
        Self::UsageAccounting,
        Self::InvalidToolRecovery,
        Self::ResponseRetryRollback,
        Self::StopAndCancellation,
        Self::StructuredOutput,
        Self::Memory,
        Self::BlockingStreamingParity,
        Self::ProviderFinalExposure,
        Self::ProvisionalStreaming,
        Self::ToolSuppression,
        Self::Concurrency,
        Self::StaleResultHandling,
    ];
}

/// Runtime-independent observable result of a scenario.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Outcome {
    pub committed_history: Vec<Message>,
    pub history_messages: usize,
    pub usage: Usage,
    pub model_calls: usize,
    pub tool_calls: Vec<String>,
    pub tool_call_count: usize,
    pub terminal_reason: String,
    pub provisional_events: usize,
    pub rejected_effects: usize,
    pub output_modes_observed: usize,
    /// Scenario-specific facts established by the private runtime adapter.
    pub observations: BTreeSet<Observation>,
}

/// Observable facts used by shared assertions without imposing a runtime API.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub enum Observation {
    ZeroBudgetRejected,
    ExactBudgetObserved,
    TranscriptValid,
    ProvisionalExcluded,
    CallsPaired,
    CommitOrderStable,
    AccountingIdempotent,
    AllInvalidPoliciesObserved,
    SuppressedToolNotExecuted,
    RetryRolledBack,
    RetryUsedTotalBudget,
    LateCommitRejected,
    CancellationRetained,
    AllOutputModesObserved,
    StructuredRecoveryBounded,
    MemoryLoadedBeforeModel,
    MemoryCommittedOnly,
    BlockingStreamingEqual,
    TypedProviderFinalObserved,
    ProviderFailureNotSuccess,
    ProvisionalRollbackObserved,
    AllSuppressionCausesObserved,
    ConcurrencyBounded,
    SiblingPolicyDeterministic,
    AllStaleClassesRejected,
}

/// Failure returned by the shared, runtime-neutral assertions.
#[derive(Debug, thiserror::Error)]
pub enum AssertionError {
    #[error("{scenario:?} missing conformance observation {missing:?}")]
    Missing {
        scenario: Scenario,
        missing: Observation,
    },
    #[error("{scenario:?} returned invalid concrete evidence: {details}")]
    Invalid {
        scenario: Scenario,
        details: &'static str,
    },
}

/// Validate one adapter result against the complete scenario contract.
pub fn assert_outcome(scenario: Scenario, outcome: &Outcome) -> Result<(), AssertionError> {
    let required: &[Observation] = match scenario {
        Scenario::ModelCallBudgets => &[
            Observation::ZeroBudgetRejected,
            Observation::ExactBudgetObserved,
        ],
        Scenario::CanonicalTranscript => &[
            Observation::TranscriptValid,
            Observation::ProvisionalExcluded,
        ],
        Scenario::ToolPairing => &[Observation::CallsPaired, Observation::CommitOrderStable],
        Scenario::UsageAccounting => &[Observation::AccountingIdempotent],
        Scenario::InvalidToolRecovery => &[
            Observation::AllInvalidPoliciesObserved,
            Observation::SuppressedToolNotExecuted,
        ],
        Scenario::ResponseRetryRollback => &[
            Observation::RetryRolledBack,
            Observation::RetryUsedTotalBudget,
        ],
        Scenario::StopAndCancellation => &[
            Observation::LateCommitRejected,
            Observation::CancellationRetained,
        ],
        Scenario::StructuredOutput => &[
            Observation::AllOutputModesObserved,
            Observation::StructuredRecoveryBounded,
        ],
        Scenario::Memory => &[
            Observation::MemoryLoadedBeforeModel,
            Observation::MemoryCommittedOnly,
        ],
        Scenario::BlockingStreamingParity => &[Observation::BlockingStreamingEqual],
        Scenario::ProviderFinalExposure => &[
            Observation::TypedProviderFinalObserved,
            Observation::ProviderFailureNotSuccess,
        ],
        Scenario::ProvisionalStreaming => &[
            Observation::ProvisionalExcluded,
            Observation::ProvisionalRollbackObserved,
        ],
        Scenario::ToolSuppression => &[
            Observation::AllSuppressionCausesObserved,
            Observation::SuppressedToolNotExecuted,
        ],
        Scenario::Concurrency => &[
            Observation::ConcurrencyBounded,
            Observation::CommitOrderStable,
            Observation::SiblingPolicyDeterministic,
        ],
        Scenario::StaleResultHandling => &[
            Observation::AllStaleClassesRejected,
            Observation::AccountingIdempotent,
        ],
    };
    for &missing in required {
        if !outcome.observations.contains(&missing) {
            return Err(AssertionError::Missing { scenario, missing });
        }
    }
    // Counters are checked only where both adapters expose the same concrete
    // artifact. Runtime-specific branches are established by the observations
    // above, which adapters emit only after their executable scenario passes.
    let valid = match scenario {
        Scenario::ModelCallBudgets => outcome.model_calls >= 1,
        Scenario::CanonicalTranscript => outcome.history_messages >= 2,
        Scenario::ToolPairing => outcome.tool_call_count >= 2,
        Scenario::UsageAccounting => outcome.model_calls >= 1 && outcome.usage.total_tokens > 0,
        Scenario::InvalidToolRecovery => outcome.tool_calls.is_empty(),
        Scenario::ResponseRetryRollback => outcome.history_messages >= 2,
        Scenario::StopAndCancellation => !outcome.terminal_reason.is_empty(),
        Scenario::StructuredOutput => {
            outcome.model_calls >= 1 && outcome.output_modes_observed == 4
        }
        Scenario::Memory => outcome.history_messages >= 2,
        Scenario::BlockingStreamingParity => {
            outcome.model_calls >= 1 && outcome.history_messages >= 2
        }
        Scenario::ProviderFinalExposure => outcome.terminal_reason == "completed",
        Scenario::ProvisionalStreaming => true,
        Scenario::ToolSuppression => outcome.tool_calls.is_empty(),
        Scenario::Concurrency => outcome.tool_call_count >= 2,
        Scenario::StaleResultHandling => outcome.model_calls >= 1,
    };
    if !valid {
        return Err(AssertionError::Invalid {
            scenario,
            details: "scenario counters/transcript do not establish the required behavior",
        });
    }
    Ok(())
}

/// Private runtime adapters implement this contract in their test modules.
pub trait Adapter {
    type Error: std::error::Error + 'static;

    fn run(&mut self, scenario: Scenario) -> Result<Outcome, Self::Error>;
}

#[cfg(test)]
mod tests {
    use super::{AssertionError, Observation, Outcome, Scenario, assert_outcome};

    #[test]
    fn ledger_contains_every_required_scenario_once() {
        let mut names = Scenario::ALL
            .into_iter()
            .map(|scenario| format!("{scenario:?}"))
            .collect::<Vec<_>>();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), 15);
    }

    #[test]
    fn assertions_reject_weakened_adapter_results() {
        let result = assert_outcome(Scenario::ModelCallBudgets, &Outcome::default());
        assert!(matches!(
            result,
            Err(AssertionError::Missing {
                missing: Observation::ZeroBudgetRejected,
                ..
            })
        ));
    }
}
