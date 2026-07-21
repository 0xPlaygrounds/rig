//! ECS-native policy values interpreted by ordered runtime systems.

use serde::{Deserialize, Serialize};

/// Policy applied when a model requests a tool outside its immutable turn
/// snapshot, or emits duplicate tool-call identities within one turn.
///
/// Duplicate identities are handled here because they can never commit to the
/// canonical transcript; under [`InvalidToolPolicy::Skip`] each repeat is
/// dropped and only the first occurrence executes. This deliberately diverges
/// from the classic runtime, which executes duplicate-identity calls as-is.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[non_exhaustive]
pub enum InvalidToolPolicy {
    /// Fail the run immediately.
    #[default]
    Fail,
    /// Re-prompt the model with corrective feedback up to the configured bound.
    ///
    /// Unadvertised-tool and duplicate-identity recoveries share one retry
    /// budget per run: a turn recovered for either defect kind consumes an
    /// attempt from the same counter.
    Retry {
        /// Maximum invalid-call recovery attempts.
        max_retries: usize,
    },
    /// Repair a case-insensitive, unambiguous tool-name mismatch.
    Repair,
    /// Commit a paired error result without executing the suppressed call.
    Skip,
    /// Stop the run without executing or committing the invalid call.
    Stop,
}

/// How structured output is requested from the provider.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[non_exhaustive]
pub enum OutputMode {
    /// Select native output when it composes with tools, otherwise use a synthetic tool.
    #[default]
    Auto,
    /// Apply the provider-native output schema.
    Native,
    /// Advertise a collision-safe synthetic terminal output tool.
    Tool,
    /// Add a schema instruction to the request without a provider constraint.
    Prompted,
}

/// Bounded validation and recovery policy for structured output.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct StructuredOutputPolicy {
    /// Selected output enforcement mode.
    pub mode: OutputMode,
    /// Maximum corrective model calls after invalid structured output.
    pub max_retries: usize,
    /// Commit the last canonical assistant response when recovery is exhausted.
    pub best_effort: bool,
}

impl Default for StructuredOutputPolicy {
    fn default() -> Self {
        Self {
            mode: OutputMode::Auto,
            max_retries: 2,
            best_effort: false,
        }
    }
}
