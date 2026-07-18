//! Shared test-only contracts for Rig runtime behavior.
//!
//! Runtime crates use these definitions from dev-dependencies and keep their
//! adapters private. This crate does not define a production runtime trait or
//! share an orchestration state machine.

use std::{
    collections::{BTreeMap, BTreeSet},
    future::Future,
    pin::Pin,
};

use thiserror::Error;

pub use rig_core::test_utils::{
    MockCompletionModel, MockError, MockResponse, MockStreamEvent, MockTurn,
};

/// Observable scenarios that every supported runtime must exercise.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum Scenario {
    /// Initial calls, retries, and continuations share one total budget.
    ModelCallBudgets,
    /// Only valid, accepted canonical messages enter committed history.
    CanonicalTranscript,
    /// Calls and results pair exactly once and commit in call order.
    ToolCallResultPairing,
    /// Billed completions and usage commit exactly once.
    UsageAccounting,
    /// All invalid-tool policy dispositions recover or terminate as specified.
    InvalidToolRecovery,
    /// Response retries roll back rejected content and consume total budget.
    ResponseRetryRollback,
    /// Stop and cancellation prevent later dispatch and remain observable.
    StopCancellation,
    /// Structured-output modes and bounded recovery are explicit.
    StructuredOutput,
    /// Memory loads before preparation and appends committed messages only.
    Memory,
    /// Blocking and streaming commit the same canonical outcome.
    BlockingStreamingParity,
    /// Typed provider finals are exposed only on genuine success.
    ProviderFinalExposure,
    /// Provisional deltas never become accepted after rollback or failure.
    ProvisionalStreaming,
    /// Suppressed tool calls never execute.
    ToolSuppression,
    /// Parallel work is bounded and commits deterministically.
    Concurrency,
    /// Invalid correlation, generation, tenant, and world results are inert.
    StaleResultHandling,
}

impl Scenario {
    /// Complete, stable scenario ledger in review order.
    pub const ALL: [Self; 15] = [
        Self::ModelCallBudgets,
        Self::CanonicalTranscript,
        Self::ToolCallResultPairing,
        Self::UsageAccounting,
        Self::InvalidToolRecovery,
        Self::ResponseRetryRollback,
        Self::StopCancellation,
        Self::StructuredOutput,
        Self::Memory,
        Self::BlockingStreamingParity,
        Self::ProviderFinalExposure,
        Self::ProvisionalStreaming,
        Self::ToolSuppression,
        Self::Concurrency,
        Self::StaleResultHandling,
    ];

    /// Canonical invariants an adapter must prove before recording this scenario.
    pub const fn invariants(self) -> &'static [&'static str] {
        match self {
            Self::ModelCallBudgets => &["zero-rejects", "n-total-calls"],
            Self::CanonicalTranscript => &["valid-role-order", "no-empty-turn", "accepted-only"],
            Self::ToolCallResultPairing => &["exact-pairing", "call-order-commit"],
            Self::UsageAccounting => &["one-record-per-completion", "duplicate-idempotent"],
            Self::InvalidToolRecovery => &[
                "fail",
                "retry",
                "repair",
                "skip",
                "stop",
                "suppressed-inert",
            ],
            Self::ResponseRetryRollback => &[
                "tool-free",
                "fresh-preparation",
                "budgeted",
                "accepted-only",
            ],
            Self::StopCancellation => &[
                "terminal-prevents-work",
                "diagnostics-retained",
                "cleanup-distinct",
            ],
            Self::StructuredOutput => {
                &["mode-selection", "collision-safe-tool", "bounded-recovery"]
            }
            Self::Memory => &[
                "load-before-call",
                "committed-only-append",
                "failure-mapped",
            ],
            Self::BlockingStreamingParity => &["history", "usage", "final", "error", "terminal"],
            Self::ProviderFinalExposure => &["typed-final", "no-false-success"],
            Self::ProvisionalStreaming => &["early-visible", "rollback-not-committed"],
            Self::ToolSuppression => &["no-dispatch"],
            Self::Concurrency => &["bounded", "deterministic", "sibling-policy"],
            Self::StaleResultHandling => &[
                "duplicate",
                "stale",
                "superseded",
                "cancelled",
                "wrong-generation",
                "wrong-tenant",
                "foreign-world",
            ],
        }
    }
}

/// Evidence recorded by one runtime-private conformance adapter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScenarioEvidence {
    /// Scenario exercised by the adapter.
    pub scenario: Scenario,
    verified: BTreeSet<&'static str>,
}

impl ScenarioEvidence {
    /// Construct evidence from invariant names proven by runtime-local assertions.
    pub fn new(scenario: Scenario, verified: impl IntoIterator<Item = &'static str>) -> Self {
        Self {
            scenario,
            verified: verified.into_iter().collect(),
        }
    }

    fn missing(&self) -> Vec<&'static str> {
        self.scenario
            .invariants()
            .iter()
            .copied()
            .filter(|invariant| !self.verified.contains(invariant))
            .collect()
    }
}

/// Complete evidence ledger for one runtime implementation.
#[derive(Clone, Debug, Default)]
pub struct SuiteEvidence {
    scenarios: BTreeMap<Scenario, ScenarioEvidence>,
}

/// Boxed scenario future used only by runtime-private test adapters.
pub type ScenarioFuture<'a, E> = Pin<Box<dyn Future<Output = Result<ScenarioEvidence, E>> + 'a>>;

/// Private-runtime adapter contract for running the shared observable ledger.
///
/// Runtime crates implement this only in `#[cfg(test)]` modules. It deliberately
/// exposes no orchestration state machine and is never a production dependency.
pub trait RuntimeScenarioAdapter {
    /// Adapter-specific typed failure.
    type Error: std::fmt::Display;

    /// Stable runtime name used in failure diagnostics.
    fn runtime_name(&self) -> &'static str;

    /// Exercise one scenario through the runtime's own public/private test API.
    fn exercise(&mut self, scenario: Scenario) -> ScenarioFuture<'_, Self::Error>;
}

/// Exercise all fifteen scenarios and reject missing or incomplete evidence.
pub async fn verify_runtime<A>(adapter: &mut A) -> Result<SuiteEvidence, ConformanceError>
where
    A: RuntimeScenarioAdapter,
{
    let mut suite = SuiteEvidence::default();
    for scenario in Scenario::ALL {
        let evidence =
            adapter
                .exercise(scenario)
                .await
                .map_err(|error| ConformanceError::AdapterFailure {
                    runtime: adapter.runtime_name(),
                    scenario,
                    message: error.to_string(),
                })?;
        if evidence.scenario != scenario {
            return Err(ConformanceError::WrongScenario {
                expected: scenario,
                found: evidence.scenario,
            });
        }
        suite.record(evidence)?;
    }
    suite.validate()?;
    Ok(suite)
}

impl SuiteEvidence {
    /// Add evidence, rejecting duplicate scenario records.
    pub fn record(&mut self, evidence: ScenarioEvidence) -> Result<(), ConformanceError> {
        let scenario = evidence.scenario;
        if self.scenarios.insert(scenario, evidence).is_some() {
            return Err(ConformanceError::DuplicateScenario(scenario));
        }
        Ok(())
    }

    /// Validate complete coverage and all required invariant names.
    pub fn validate(&self) -> Result<(), ConformanceError> {
        for scenario in Scenario::ALL {
            let evidence = self
                .scenarios
                .get(&scenario)
                .ok_or(ConformanceError::MissingScenario(scenario))?;
            let missing = evidence.missing();
            if !missing.is_empty() {
                return Err(ConformanceError::MissingInvariants { scenario, missing });
            }
        }
        Ok(())
    }
}

/// Shared conformance-ledger validation error.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum ConformanceError {
    /// A runtime-private adapter failed while exercising one scenario.
    #[error("{runtime} failed conformance scenario {scenario:?}: {message}")]
    AdapterFailure {
        /// Runtime adapter name.
        runtime: &'static str,
        /// Scenario being exercised.
        scenario: Scenario,
        /// Typed adapter failure rendered for the shared report.
        message: String,
    },
    /// An adapter returned evidence for a different scenario than requested.
    #[error("adapter returned {found:?} evidence while {expected:?} was requested")]
    WrongScenario {
        /// Requested scenario.
        expected: Scenario,
        /// Incorrect returned scenario.
        found: Scenario,
    },
    /// One adapter recorded the same scenario twice.
    #[error("duplicate conformance scenario: {0:?}")]
    DuplicateScenario(Scenario),
    /// One required scenario was not exercised.
    #[error("missing conformance scenario: {0:?}")]
    MissingScenario(Scenario),
    /// A scenario did not prove every canonical invariant.
    #[error("scenario {scenario:?} is missing invariants: {missing:?}")]
    MissingInvariants {
        /// Incomplete scenario.
        scenario: Scenario,
        /// Canonical invariant names not proven by the runtime adapter.
        missing: Vec<&'static str>,
    },
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use std::process::Command;

    #[test]
    fn ledger_requires_all_fifteen_scenarios() {
        let mut suite = SuiteEvidence::default();
        for scenario in Scenario::ALL {
            assert!(
                suite
                    .record(ScenarioEvidence::new(
                        scenario,
                        scenario.invariants().iter().copied()
                    ))
                    .is_ok()
            );
        }
        assert!(suite.validate().is_ok());
    }

    #[test]
    fn incomplete_evidence_is_rejected() {
        let mut suite = SuiteEvidence::default();
        assert!(
            suite
                .record(ScenarioEvidence::new(
                    Scenario::ModelCallBudgets,
                    ["zero-rejects"]
                ))
                .is_ok()
        );
        assert!(matches!(
            suite.validate(),
            Err(ConformanceError::MissingInvariants {
                scenario: Scenario::ModelCallBudgets,
                ..
            })
        ));
    }

    #[test]
    fn production_dependency_direction_has_no_runtime_cycles_or_leaks() {
        let workspace = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(std::path::Path::parent)
            .expect("conformance crate is nested under workspace/crates");
        let output = Command::new("cargo")
            .args(["metadata", "--format-version", "1", "--no-deps"])
            .current_dir(workspace)
            .output()
            .expect("cargo metadata should run");
        assert!(
            output.status.success(),
            "cargo metadata failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let metadata: serde_json::Value =
            serde_json::from_slice(&output.stdout).expect("metadata is JSON");
        let packages = metadata["packages"].as_array().expect("package array");

        let dependencies = |package_name: &str| {
            packages
                .iter()
                .find(|package| package["name"] == package_name)
                .and_then(|package| package["dependencies"].as_array())
                .expect("workspace package exists")
        };
        let has_any_edge = |from: &str, to: &str| {
            dependencies(from)
                .iter()
                .any(|dependency| dependency["name"] == to)
        };
        assert!(!has_any_edge("rig-core", "rig-agent"));
        assert!(!has_any_edge("rig-core", "rig-bevy"));
        assert!(!has_any_edge("rig-core", "bevy_ecs"));
        assert!(!has_any_edge("rig-agent", "rig-bevy"));
        assert!(!has_any_edge("rig-bevy", "rig-agent"));

        for package in packages {
            let name = package["name"].as_str().expect("package name");
            let manifest = package["manifest_path"].as_str().expect("manifest path");
            if !manifest.contains("/crates/rig-")
                || matches!(
                    name,
                    "rig-core" | "rig-agent" | "rig-bevy" | "rig-runtime-conformance"
                )
            {
                continue;
            }
            for dependency in package["dependencies"].as_array().expect("dependencies") {
                let kind = dependency["kind"].as_str().unwrap_or("normal");
                let dependency_name = dependency["name"].as_str().expect("dependency name");
                if kind != "dev" {
                    assert!(
                        !matches!(dependency_name, "rig-agent" | "rig-bevy"),
                        "{name} has forbidden {kind} dependency on {dependency_name}"
                    );
                }
            }
        }
    }
}
