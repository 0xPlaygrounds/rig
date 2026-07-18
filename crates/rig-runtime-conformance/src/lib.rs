//! Shared, test-only behavioral specifications for Rig runtimes.
//!
//! This unpublished crate supplies scenario definitions and scripted portable
//! effects. It deliberately contains no production orchestration and no public
//! cross-runtime trait. `rig-agent` and `rig-bevy` consume these fixtures only
//! from their own tests and adapt them to their native APIs independently.

mod fixtures;
mod scenarios;

pub use fixtures::{
    CountingPortableTool, CountingPortableToolArgs, ScriptedCompletionModel,
    ScriptedCompletionTurn, ScriptedRawResponse, ScriptedStreamEvent, scripted_text_model,
};
pub use rig_core::test_utils::{AppendFailingMemory, CountingMemory, FailingMemory};
pub use scenarios::{ALL_SCENARIOS, ScenarioDefinition, ScenarioId, scenario};
