//! Shared, test-only behavioral specifications for Rig runtimes.
//!
//! This unpublished crate supplies scenario definitions and scripted portable
//! effects. It deliberately contains no production orchestration and no public
//! cross-runtime trait. `rig-agent` and `rig-bevy` consume these fixtures only
//! from their own tests and adapt them to their native APIs independently.

mod fixtures;
mod report;
mod scenarios;

pub use fixtures::{
    CountingPortableTool, CountingPortableToolArgs, PORTABLE_FIXTURE_IMAGE, PortableEmbeddingArgs,
    PortableEmbeddingContext, PortableEmbeddingFixture, PortableFixtureError,
    ScriptedCompletionModel, ScriptedCompletionTurn, ScriptedRawResponse, ScriptedStreamEvent,
    portable_dynamic_fixture, portable_fixture_output, scripted_text_model,
};
pub use report::{
    ConformanceFailure, ObservationAssertion, ReportBuildError, ScenarioReport, verify_report,
};
pub use rig_core::test_utils::{AppendFailingMemory, CountingMemory, FailingMemory};
pub use scenarios::{
    ALL_SCENARIOS, ObservationDefinition, ScenarioDefinition, ScenarioId, scenario,
};
