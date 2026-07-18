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
    pub observations: &'static [ObservationDefinition],
}

/// One shared observable and its runtime-independent JSON expectation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ObservationDefinition {
    /// Human-readable invariant proved by runtime-specific evidence.
    pub description: &'static str,
    /// Canonical expected value encoded as JSON in the shared ledger.
    pub expected_json: &'static str,
    /// Whether a runtime may explicitly report this boundary as inapplicable.
    pub allows_not_applicable: bool,
}

macro_rules! observation {
    ($description:literal => $expected:literal) => {
        ObservationDefinition {
            description: $description,
            expected_json: $expected,
            allows_not_applicable: false,
        }
    };
}

macro_rules! runtime_specific_observation {
    ($description:literal => $expected:literal) => {
        ObservationDefinition {
            description: $description,
            expected_json: $expected,
            allows_not_applicable: true,
        }
    };
}

/// The complete shared scenario ledger.
pub const ALL_SCENARIOS: [ScenarioDefinition; 15] = [
    ScenarioDefinition {
        id: ScenarioId::ModelCallBudgets,
        name: "model-call budgets",
        observations: &[
            observation!("zero rejects the initial call" => "true"),
            observation!("N permits exactly N total calls including retries and continuations" => "[2,2]"),
        ],
    },
    ScenarioDefinition {
        id: ScenarioId::CanonicalTranscript,
        name: "canonical transcript validity",
        observations: &[
            observation!("committed roles remain valid" => "true"),
            observation!("no empty synthetic turn is committed" => "2"),
            observation!("rejected and provisional content is excluded" => "true"),
        ],
    },
    ScenarioDefinition {
        id: ScenarioId::ToolCallResultPairing,
        name: "tool-call/result pairing",
        observations: &[
            observation!("each committed call has exactly one matching result" => "2"),
            observation!("parallel arrival commits in model-call order" => "[\"first\",\"second\"]"),
        ],
    },
    ScenarioDefinition {
        id: ScenarioId::UsageAccounting,
        name: "usage and completion accounting",
        observations: &[
            observation!("one call record exists per billed completed model operation" => "true"),
            observation!("rejected provider operations remain billed exactly once" => "true"),
            observation!("duplicate completion ingress never double counts" => "true"),
        ],
    },
    ScenarioDefinition {
        id: ScenarioId::InvalidToolRecovery,
        name: "invalid-tool recovery",
        observations: &[
            observation!("fail, retry, repair, skip, and stop are distinguishable" => "true"),
            observation!("suppressed calls never execute" => "true"),
        ],
    },
    ScenarioDefinition {
        id: ScenarioId::ResponseRetryRollback,
        name: "response retry and rollback",
        observations: &[
            observation!("tool-free turns may retry" => "true"),
            observation!("tool-bearing turns never response-retry" => "true"),
            observation!("corrective feedback is prepared on a fresh request" => "true"),
            observation!("retry policy is re-evaluated on the retried response" => "true"),
            observation!("response retries consume the total model-call budget" => "true"),
            observation!("rejected content is not accepted history" => "true"),
        ],
    },
    ScenarioDefinition {
        id: ScenarioId::StopAndCancellation,
        name: "stop and cancellation",
        observations: &[
            observation!("stop prevents later dispatch and commit" => "true"),
            observation!("cancellation prevents late commit" => "true"),
            observation!("diagnostics remain observable before retention expires" => "true"),
            runtime_specific_observation!("retained terminal state expires only after observation and retention" => "true"),
        ],
    },
    ScenarioDefinition {
        id: ScenarioId::StructuredOutput,
        name: "structured output",
        observations: &[
            observation!("native, tool, prompted, and auto selection follow provider capability" => "true"),
            observation!("synthetic tool identity is collision-safe" => "true"),
            observation!("recovery and best-effort exhaustion are bounded" => "true"),
        ],
    },
    ScenarioDefinition {
        id: ScenarioId::Memory,
        name: "memory",
        observations: &[
            observation!("memory loads before the first model request" => "true"),
            observation!("only newly committed canonical messages append" => "true"),
            observation!("a multi-step run appends exactly once" => "true"),
            observation!("load failure is typed and prevents provider dispatch" => "true"),
            observation!("append failure does not erase the committed result" => "true"),
            observation!("failed and stopped runs do not append" => "true"),
        ],
    },
    ScenarioDefinition {
        id: ScenarioId::BlockingStreamingParity,
        name: "blocking/streaming parity",
        observations: &[
            observation!("successful history, usage, final content, and terminal reason match" => "true"),
            observation!("blocking and streaming provider failures produce matching error terminals" => "true"),
        ],
    },
    ScenarioDefinition {
        id: ScenarioId::ProviderFinalExposure,
        name: "provider-final exposure",
        observations: &[
            observation!("typed final is available where promised" => "true"),
            observation!("a later provider error suppresses false success" => "true"),
        ],
    },
    ScenarioDefinition {
        id: ScenarioId::ProvisionalStreaming,
        name: "provisional streaming",
        observations: &[
            observation!("early deltas are observable but not committed" => "true"),
            observation!("retry prevents provisional promotion" => "true"),
            observation!("provider rejection prevents provisional promotion" => "true"),
            observation!("stop prevents provisional promotion" => "true"),
            observation!("cancellation prevents provisional promotion" => "true"),
        ],
    },
    ScenarioDefinition {
        id: ScenarioId::ToolSuppression,
        name: "tool suppression",
        observations: &[
            observation!("invalid peer calls do not execute" => "true"),
            observation!("policy-skipped calls do not execute" => "true"),
            observation!("output finalization suppresses peer execution" => "true"),
            observation!("stop suppresses tool execution" => "true"),
            observation!("cancellation suppresses tool execution" => "true"),
        ],
    },
    ScenarioDefinition {
        id: ScenarioId::Concurrency,
        name: "concurrency",
        observations: &[
            observation!("execution respects a configured bound" => "true"),
            observation!("commit and event order is deterministic" => "[\"slow\",\"fast\"]"),
            observation!("sibling drain or cancellation behavior is defined" => "[\"fast\",\"slow\"]"),
        ],
    },
    ScenarioDefinition {
        id: ScenarioId::StaleResultHandling,
        name: "stale-result handling",
        observations: &[
            observation!("duplicate results cannot commit twice" => "true"),
            observation!("stale correlation or payload results cannot commit" => "true"),
            runtime_specific_observation!("superseded external results cannot commit" => "true"),
            observation!("canceled results cannot commit" => "true"),
            runtime_specific_observation!("wrong-generation, wrong-tenant, and foreign-world results cannot commit" => "true"),
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
            !definition.name.is_empty()
                && !definition.observations.is_empty()
                && definition.observations.iter().all(|observation| {
                    !observation.description.is_empty()
                        && serde_json::from_str::<serde_json::Value>(observation.expected_json)
                            .is_ok()
                })
        }));
    }

    #[test]
    fn every_identifier_is_resolvable() {
        for definition in ALL_SCENARIOS {
            assert_eq!(scenario(definition.id), Some(&definition));
        }
    }
}
