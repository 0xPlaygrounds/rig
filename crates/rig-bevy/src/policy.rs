//! ECS-native policy data and deterministic decisions.

use bevy_ecs::prelude::Component;
use serde::{Deserialize, Serialize};

/// Invalid-tool recovery decision. These are data, not hook callbacks.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum InvalidToolPolicy {
    Fail,
    Retry,
    Repair {
        replacement_name: String,
        replacement_arguments: String,
    },
    Skip {
        reason: String,
    },
    Stop {
        reason: String,
    },
}

/// Authoritative result of applying an invalid-tool policy.
#[derive(Clone, Debug)]
pub enum InvalidToolResolution {
    Failed,
    Retry { feedback: String },
    Repair { name: String, arguments: String },
    Skip { result: rig_core::tool::ToolResult },
    Stopped,
}

/// Structured-output transport policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum OutputMode {
    Native,
    Tool,
    Prompted,
    Auto,
}

/// Per-run bounded recovery configuration.
#[derive(Component, Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RecoveryPolicy {
    pub max_response_retries: usize,
    pub max_invalid_tool_retries: usize,
}

impl Default for RecoveryPolicy {
    fn default() -> Self {
        Self {
            max_response_retries: 1,
            max_invalid_tool_retries: 1,
        }
    }
}

/// Explicit bounded parallelism for owned effects.
#[derive(Component, Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ConcurrencyLimit(pub usize);

/// Deterministically resolve `Auto` without changing provider semantics.
pub fn select_output_mode(
    requested: OutputMode,
    native_supported: bool,
    has_tools: bool,
    native_composes_with_tools: bool,
) -> OutputMode {
    match requested {
        OutputMode::Auto if native_supported && (!has_tools || native_composes_with_tools) => {
            OutputMode::Native
        }
        OutputMode::Auto if has_tools => OutputMode::Tool,
        OutputMode::Auto => OutputMode::Prompted,
        explicit => explicit,
    }
}

/// Choose a deterministic synthetic structured-output tool name without
/// colliding with an advertised capability.
pub fn synthetic_output_tool_name<'a>(names: impl IntoIterator<Item = &'a str>) -> String {
    let names = names.into_iter().collect::<std::collections::BTreeSet<_>>();
    let base = "__rig_structured_output";
    if !names.contains(base) {
        return base.to_string();
    }
    let mut suffix = 2usize;
    loop {
        let candidate = format!("{base}_{suffix}");
        if !names.contains(candidate.as_str()) {
            return candidate;
        }
        suffix = suffix.saturating_add(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_output_selection_respects_tool_composition() {
        assert_eq!(
            select_output_mode(OutputMode::Auto, true, false, false),
            OutputMode::Native
        );
        assert_eq!(
            select_output_mode(OutputMode::Auto, true, true, false),
            OutputMode::Tool
        );
        assert_eq!(
            select_output_mode(OutputMode::Auto, true, true, true),
            OutputMode::Native
        );
        assert_eq!(
            select_output_mode(OutputMode::Auto, false, false, false),
            OutputMode::Prompted
        );
    }

    #[test]
    fn structured_output_modes_and_synthetic_names_are_deterministic() {
        for mode in [OutputMode::Native, OutputMode::Tool, OutputMode::Prompted] {
            assert_eq!(select_output_mode(mode, false, true, false), mode);
        }
        assert_eq!(synthetic_output_tool_name([]), "__rig_structured_output");
        assert_eq!(
            synthetic_output_tool_name(["__rig_structured_output", "__rig_structured_output_2"]),
            "__rig_structured_output_3"
        );
        let bounded = RecoveryPolicy::default();
        assert_eq!(bounded.max_response_retries, 1);
    }
}
