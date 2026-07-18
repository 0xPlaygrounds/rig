/// Stable identifier for one shared runtime conformance scenario.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ScenarioId {
    /// Total model-call budget includes retries and continuations.
    ModelCallBudgets,
    /// Only valid, accepted content enters canonical history.
    CanonicalTranscript,
    /// Calls and results pair exactly once in call order.
    ToolCallResultPairing,
    /// Billed operations and usage commit exactly once.
    UsageAccounting,
    /// Invalid calls support fail, retry, repair, skip, and stop.
    InvalidToolRecovery,
    /// Rejected model output rolls back before a fresh retry.
    ResponseRetryRollback,
    /// Stop, cancellation, observation, and cleanup remain distinct.
    StopAndCancellation,
    /// Native, tool, prompted, and automatic output modes recover consistently.
    StructuredOutput,
    /// Memory loads before dispatch and appends committed messages only.
    Memory,
    /// Blocking and streaming drivers reach the same canonical outcome.
    BlockingStreamingParity,
    /// Typed provider finals are exposed only on promised surfaces.
    ProviderFinalExposure,
    /// Provisional deltas never become accepted content after rollback.
    ProvisionalStreaming,
    /// Suppressed tools never execute.
    ToolSuppression,
    /// Execution is bounded while commits and events remain deterministic.
    Concurrency,
    /// Invalid, stale, duplicate, foreign, and superseded results cannot commit.
    StaleResultHandling,
}

/// One runtime-neutral scenario specification.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ScenarioDefinition {
    /// Stable scenario identifier.
    pub id: ScenarioId,
    /// Human-readable scenario name.
    pub name: &'static str,
    /// Observable invariants every runtime adapter must prove.
    pub observations: &'static [&'static str],
}

/// The complete shared scenario ledger.
pub const ALL_SCENARIOS: [ScenarioDefinition; 15] = [
    ScenarioDefinition {
        id: ScenarioId::ModelCallBudgets,
        name: "model-call budgets",
        observations: &[
            "zero rejects the initial call",
            "N permits exactly N total calls including retries and continuations",
        ],
    },
    ScenarioDefinition {
        id: ScenarioId::CanonicalTranscript,
        name: "canonical transcript validity",
        observations: &[
            "committed roles remain valid",
            "no empty synthetic turn is committed",
            "rejected and provisional content is excluded",
        ],
    },
    ScenarioDefinition {
        id: ScenarioId::ToolCallResultPairing,
        name: "tool-call/result pairing",
        observations: &[
            "each committed call has exactly one matching result",
            "parallel arrival commits in model-call order",
        ],
    },
    ScenarioDefinition {
        id: ScenarioId::UsageAccounting,
        name: "usage and completion accounting",
        observations: &[
            "one call record exists per billed completed model operation",
            "duplicate completion ingress never double counts",
        ],
    },
    ScenarioDefinition {
        id: ScenarioId::InvalidToolRecovery,
        name: "invalid-tool recovery",
        observations: &[
            "fail, retry, repair, skip, and stop are distinguishable",
            "suppressed calls never execute",
        ],
    },
    ScenarioDefinition {
        id: ScenarioId::ResponseRetryRollback,
        name: "response retry and rollback",
        observations: &[
            "only tool-free turns retry",
            "corrective feedback uses fresh preparation and policy",
            "rejected content is not accepted history",
        ],
    },
    ScenarioDefinition {
        id: ScenarioId::StopAndCancellation,
        name: "stop and cancellation",
        observations: &[
            "terminal state prevents later dispatch and commit",
            "cancellation is distinct from cleanup",
            "diagnostics remain observable before retention expires",
        ],
    },
    ScenarioDefinition {
        id: ScenarioId::StructuredOutput,
        name: "structured output",
        observations: &[
            "native, tool, prompted, and auto selection follow provider capability",
            "synthetic tool identity is collision-safe",
            "recovery and best-effort exhaustion are bounded",
        ],
    },
    ScenarioDefinition {
        id: ScenarioId::Memory,
        name: "memory",
        observations: &[
            "memory loads before the first model request",
            "only newly committed canonical messages append",
            "failed and stopped runs do not append",
        ],
    },
    ScenarioDefinition {
        id: ScenarioId::BlockingStreamingParity,
        name: "blocking/streaming parity",
        observations: &["history, usage, final content, errors, and terminal reason match"],
    },
    ScenarioDefinition {
        id: ScenarioId::ProviderFinalExposure,
        name: "provider-final exposure",
        observations: &[
            "typed final is available where promised",
            "a later provider error suppresses false success",
        ],
    },
    ScenarioDefinition {
        id: ScenarioId::ProvisionalStreaming,
        name: "provisional streaming",
        observations: &[
            "early deltas are observable but not committed",
            "retry, rejection, stop, and cancellation prevent promotion",
        ],
    },
    ScenarioDefinition {
        id: ScenarioId::ToolSuppression,
        name: "tool suppression",
        observations: &[
            "invalid peers and policy skips do not execute",
            "output finalization, stop, and cancellation do not execute",
        ],
    },
    ScenarioDefinition {
        id: ScenarioId::Concurrency,
        name: "concurrency",
        observations: &[
            "execution respects a configured bound",
            "commit and event order is deterministic",
            "sibling drain or cancellation behavior is defined",
        ],
    },
    ScenarioDefinition {
        id: ScenarioId::StaleResultHandling,
        name: "stale-result handling",
        observations: &[
            "duplicate, stale, superseded, and canceled results cannot commit",
            "wrong-generation, wrong-tenant, and foreign-world results cannot commit",
        ],
    },
];

/// Look up one complete scenario definition.
pub fn scenario(id: ScenarioId) -> Option<&'static ScenarioDefinition> {
    ALL_SCENARIOS.iter().find(|definition| definition.id == id)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    #[test]
    fn ledger_contains_fifteen_unique_nonempty_scenarios() {
        let ids = ALL_SCENARIOS
            .iter()
            .map(|definition| definition.id)
            .collect::<BTreeSet<_>>();

        assert_eq!(ids.len(), 15);
        assert!(ALL_SCENARIOS.iter().all(|definition| {
            !definition.name.is_empty() && !definition.observations.is_empty()
        }));
    }

    #[test]
    fn every_identifier_is_resolvable() {
        for definition in ALL_SCENARIOS {
            assert_eq!(scenario(definition.id), Some(&definition));
        }
    }
}
