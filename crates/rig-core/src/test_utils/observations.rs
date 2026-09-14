//! Helpers for comparing deterministic observation test fixtures.
//! These checks are not an execution-equivalence contract.

use crate::observe::{Action, Observation, ObservationTrace, Reason};
use serde::{Deserialize, Serialize};

/// Where two traces first disagree on a semantic field.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Divergence {
    /// The index of the first observation that differs (or the length of
    /// the shorter trace when one is a prefix of the other).
    pub index: usize,
    /// The expected fact, if any.
    pub expected: Option<Observation>,
    /// The actual fact, if any.
    pub actual: Option<Observation>,
}

/// The result of comparing two traces.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "comparison", rename_all = "snake_case")]
pub enum Comparison {
    /// Every semantic field agrees.
    Equal,
    /// The traces diverge here.
    Diverged(Box<Divergence>),
    /// The traces cannot be compared: one is incomplete.
    Incomparable {
        /// Why.
        reason: Reason,
    },
}

/// Compare two traces on their semantic fields: subject, stage, emitter,
/// action and sequence. Adapter analysis, measurements ([`Observation::at`]) and the session
/// name are ignored. An incomplete trace (dropped facts) is incomparable,
/// never equal: a missing fact is not evidence of agreement.
pub fn compare(expected: &ObservationTrace, actual: &ObservationTrace) -> Comparison {
    if !expected.is_complete() {
        return Comparison::Incomparable {
            reason: Reason::with_detail(
                "incomplete_expected",
                format!("the expected trace dropped {} facts", expected.dropped),
            ),
        };
    }
    if !actual.is_complete() {
        return Comparison::Incomparable {
            reason: Reason::with_detail(
                "incomplete_actual",
                format!("the actual trace dropped {} facts", actual.dropped),
            ),
        };
    }
    let semantic = |observation: &Observation| {
        let mut observation = observation.clone();
        observation.at = None;
        if let Action::Adapter { observation } = &mut observation.action {
            observation.analysis = None;
        }
        observation
    };
    let longest = expected.observations.len().max(actual.observations.len());
    for index in 0..longest {
        let left = expected.observations.get(index).map(semantic);
        let right = actual.observations.get(index).map(semantic);
        if left != right {
            return Comparison::Diverged(Box::new(Divergence {
                index,
                expected: left,
                actual: right,
            }));
        }
    }
    Comparison::Equal
}
