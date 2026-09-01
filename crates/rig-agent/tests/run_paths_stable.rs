//! Every run-protocol path that existed at 0.42.0 still resolves after the
//! protocol crate was dissolved (vocabulary → rig-core, `AgentRun` →
//! `rig_agent::run`). Compile-only: a missing path fails the build.

#![allow(unused_imports)]

use rig_agent::agent::hook::{
    InvalidToolCallAction, InvalidToolCallContext, RequestPatch, RetryRequest, RunEntry,
};
use rig_agent::agent::run::output_mode::OutputMode as _OutputModeViaRun;
use rig_agent::agent::run::streamed::{
    PartialStreamedTurn as _P, StreamedInvalidToolCall as _I, StreamedResolution as _R,
    StreamedTurn as _T, StreamedTurnAssembler as _A, StreamedTurnEvent as _E,
};
use rig_agent::agent::run::{
    AgentRun as _AgentRunViaRun, AgentRunStep as _StepViaRun, ModelTurn as _TurnViaRun,
    ModelTurnOutcome as _OutcomeViaRun, PendingToolCall as _PendingViaRun,
    RunEntry as _EntryViaRun, TurnTools as _ToolsViaRun,
};
use rig_agent::agent::{
    AgentRun, AgentRunStep, CompletionCall, ModelTurn, ModelTurnOutcome, OutputMode,
    PendingToolCall, PromptResponse, RunSpec, TurnTools,
};
use rig_agent::completion::PromptError;
use rig_agent::run::{AgentRun as _AgentRunDirect, RunEntry as _EntryDirect};
use rig_core::completion::{
    output::OutputMode as _CoreOutputMode,
    policy::RequestPatch as _CorePatch,
    prepare::{PreparedRequest, prepare_request},
    response::{CompletionCall as _CoreCall, PromptError as _CoreErr, PromptResponse as _CoreResp},
    spec::RunSpec as _CoreSpec,
};
use rig_core::streaming::assemble::StreamedTurnAssembler as _CoreAssembler;
use rig_core::transcript::{TranscriptError, validate_canonical};

#[test]
fn run_protocol_paths_resolve() {
    let _ = AgentRun::new("x").max_turns(1);
    let _ = RunSpec::new();
    let _: fn(&[rig_core::message::Message]) -> Result<(), TranscriptError> = validate_canonical;
}
