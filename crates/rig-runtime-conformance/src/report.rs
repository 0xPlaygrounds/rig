use std::collections::BTreeSet;

use serde::Serialize;

use crate::{ScenarioId, scenario};

/// One observable conformance assertion with serializable evidence.
#[derive(Clone, Debug, PartialEq)]
pub struct ObservationAssertion {
    /// Exact observation text from the shared scenario ledger.
    pub observation: &'static str,
    /// Runtime-produced observable value.
    pub actual: serde_json::Value,
    /// Shared expected value.
    pub expected: serde_json::Value,
    /// Explicit rationale when the shared boundary does not exist on this runtime.
    pub not_applicable_reason: Option<&'static str>,
}

/// Complete runtime-specific evidence for one shared scenario.
#[derive(Clone, Debug, PartialEq)]
pub struct ScenarioReport {
    /// Runtime implementation that produced the evidence.
    pub runtime: &'static str,
    /// Shared scenario identity.
    pub scenario_id: ScenarioId,
    /// Exact assertion set for every required observation.
    pub assertions: Vec<ObservationAssertion>,
}

impl ScenarioReport {
    /// Start an empty report for one runtime and scenario.
    #[must_use]
    pub fn new(runtime: &'static str, scenario_id: ScenarioId) -> Self {
        Self {
            runtime,
            scenario_id,
            assertions: Vec::new(),
        }
    }

    fn definition_for(
        &self,
        observation: &'static str,
    ) -> Result<&'static crate::scenarios::ObservationDefinition, ReportBuildError> {
        scenario(self.scenario_id)
            .and_then(|definition| {
                definition
                    .observations
                    .iter()
                    .find(|candidate| candidate.description == observation)
            })
            .ok_or(ReportBuildError::UnknownObservation { observation })
    }

    /// Record one actual observation against the shared ledger expectation.
    pub fn observe<A>(
        &mut self,
        observation: &'static str,
        actual: A,
    ) -> Result<(), ReportBuildError>
    where
        A: Serialize,
    {
        let expected_json = self.definition_for(observation)?.expected_json;
        self.assertions.push(ObservationAssertion {
            observation,
            actual: serde_json::to_value(actual).map_err(ReportBuildError::Serialization)?,
            expected: serde_json::from_str(expected_json)
                .map_err(ReportBuildError::InvalidLedgerExpectation)?,
            not_applicable_reason: None,
        });
        Ok(())
    }

    /// Record that a ledger-declared runtime-specific boundary cannot exist here.
    pub fn observe_not_applicable(
        &mut self,
        observation: &'static str,
        reason: &'static str,
    ) -> Result<(), ReportBuildError> {
        let definition = self.definition_for(observation)?;
        if !definition.allows_not_applicable || reason.trim().is_empty() {
            return Err(ReportBuildError::NotApplicableForbidden { observation });
        }
        let expected: serde_json::Value = serde_json::from_str(definition.expected_json)
            .map_err(ReportBuildError::InvalidLedgerExpectation)?;
        self.assertions.push(ObservationAssertion {
            observation,
            actual: expected.clone(),
            expected,
            not_applicable_reason: Some(reason),
        });
        Ok(())
    }
}

/// Failure while constructing runtime evidence against the shared ledger.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ReportBuildError {
    /// Adapter named an observation outside its shared scenario.
    #[error("observation `{observation}` is not defined by the shared scenario")]
    UnknownObservation {
        /// Unknown observation text.
        observation: &'static str,
    },
    /// Runtime evidence could not be represented as JSON.
    #[error("runtime evidence serialization failed: {0}")]
    Serialization(serde_json::Error),
    /// A checked-in shared expectation was not valid JSON.
    #[error("shared ledger expectation is invalid JSON: {0}")]
    InvalidLedgerExpectation(serde_json::Error),
    /// An adapter attempted to skip a universally applicable observation.
    #[error("observation `{observation}` cannot be marked not applicable")]
    NotApplicableForbidden {
        /// Observation that may not be skipped.
        observation: &'static str,
    },
}

/// Structural or behavioral failure in one scenario report.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ConformanceFailure {
    /// The scenario identifier is not part of the shared ledger.
    #[error("unknown shared conformance scenario")]
    UnknownScenario,
    /// An expected observation is missing from the report.
    #[error("{runtime} report for {scenario:?} is missing observation `{observation}`")]
    MissingObservation {
        /// Runtime under test.
        runtime: &'static str,
        /// Scenario under test.
        scenario: ScenarioId,
        /// Missing observation.
        observation: &'static str,
    },
    /// The report contains an observation outside the shared scenario.
    #[error("{runtime} report for {scenario:?} contains unexpected observation `{observation}`")]
    UnexpectedObservation {
        /// Runtime under test.
        runtime: &'static str,
        /// Scenario under test.
        scenario: ScenarioId,
        /// Unexpected observation.
        observation: &'static str,
    },
    /// The same observation was reported more than once.
    #[error("{runtime} report for {scenario:?} duplicates observation `{observation}`")]
    DuplicateObservation {
        /// Runtime under test.
        runtime: &'static str,
        /// Scenario under test.
        scenario: ScenarioId,
        /// Duplicated observation.
        observation: &'static str,
    },
    /// Runtime evidence differs from the shared expected value.
    #[error(
        "{runtime} report for {scenario:?} failed `{observation}`: actual={actual}, expected={expected}"
    )]
    Mismatch {
        /// Runtime under test.
        runtime: &'static str,
        /// Scenario under test.
        scenario: ScenarioId,
        /// Failed observation.
        observation: &'static str,
        /// Serialized actual value.
        actual: Box<serde_json::Value>,
        /// Serialized expected value.
        expected: Box<serde_json::Value>,
    },
}

/// Verify that a runtime report proves every shared observation exactly once.
pub fn verify_report(report: &ScenarioReport) -> Result<(), ConformanceFailure> {
    let definition = scenario(report.scenario_id).ok_or(ConformanceFailure::UnknownScenario)?;
    let expected = definition
        .observations
        .iter()
        .map(|observation| observation.description)
        .collect::<BTreeSet<_>>();
    let mut seen = BTreeSet::new();
    for assertion in &report.assertions {
        if !expected.contains(assertion.observation) {
            return Err(ConformanceFailure::UnexpectedObservation {
                runtime: report.runtime,
                scenario: report.scenario_id,
                observation: assertion.observation,
            });
        }
        if !seen.insert(assertion.observation) {
            return Err(ConformanceFailure::DuplicateObservation {
                runtime: report.runtime,
                scenario: report.scenario_id,
                observation: assertion.observation,
            });
        }
        if assertion.not_applicable_reason.is_some()
            && definition
                .observations
                .iter()
                .find(|observation| observation.description == assertion.observation)
                .is_none_or(|observation| !observation.allows_not_applicable)
        {
            return Err(ConformanceFailure::Mismatch {
                runtime: report.runtime,
                scenario: report.scenario_id,
                observation: assertion.observation,
                actual: Box::new(serde_json::Value::String("not applicable".to_string())),
                expected: Box::new(assertion.expected.clone()),
            });
        }
        if assertion.actual != assertion.expected {
            return Err(ConformanceFailure::Mismatch {
                runtime: report.runtime,
                scenario: report.scenario_id,
                observation: assertion.observation,
                actual: Box::new(assertion.actual.clone()),
                expected: Box::new(assertion.expected.clone()),
            });
        }
    }
    if let Some(observation) = expected.difference(&seen).next().copied() {
        return Err(ConformanceFailure::MissingObservation {
            runtime: report.runtime,
            scenario: report.scenario_id,
            observation,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verifier_requires_every_observation_exactly_once() {
        let mut report = ScenarioReport::new("fixture", ScenarioId::ModelCallBudgets);
        assert!(
            report
                .observe("zero rejects the initial call", true)
                .is_ok()
        );
        assert!(matches!(
            verify_report(&report),
            Err(ConformanceFailure::MissingObservation { .. })
        ));

        assert!(
            report
                .observe(
                    "N permits exactly N total calls including retries and continuations",
                    (2, 2),
                )
                .is_ok()
        );
        assert!(verify_report(&report).is_ok());
    }
}
