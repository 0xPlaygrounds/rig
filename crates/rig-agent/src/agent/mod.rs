//! Configurable agents, builders, hooks, and blocking or streaming run interfaces.
//!
//! [`AgentRunner`] owns execution and lifecycle policies. Tools and context can
//! be static or retrieved per turn; hooks can inject context through [`RequestPatch`].
//!
//! ```no_run
//! use rig_agent::prelude::*;
//! use rig_core::providers::openai::{self, OpenAI};
//! use rig_reqwest::prelude::*;
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let provider = OpenAI::from_env()?.bound()?;
//! let agent = provider.agent(openai::GPT_5_2).preamble("Be concise.").build();
//! let response = agent.prompt("Explain ownership.").await?;
//! # Ok(())
//! # }
//! ```
mod builder;
mod completion;
pub(crate) mod drive;
mod engine;
pub(crate) use engine::streaming_error_into_prompt;
pub mod hook;
pub mod run;
pub mod runner;
mod streaming;
mod telemetry;
mod tool;
mod typed;

/// Fallback display name used in telemetry spans and logs when an agent has no
/// configured name.
pub(crate) const UNKNOWN_AGENT_NAME: &str = "Unnamed Agent";

pub use crate::bus::ModelHandle;
pub use crate::run::response::{CompletionCall, MemoryAppend, PromptResponse};
pub use crate::run::spec::RunSpec;
pub use builder::{AgentBuilder, NoToolConfig, WithBuilderTools, WithToolServerHandle};
pub use completion::{Agent, AgentParts};
pub use hook::{
    AgentHook, CompletionCallAction, CompletionCallEvent, HookContext, HookStack,
    InvalidToolCallAction, InvalidToolCallContext, InvalidToolCallReason, ModelSelection,
    ModelSelectionAction, ModelTurnAction, ModelTurnFinished, ObservationAction, ReasoningDelta,
    RequestPatch, RetryRequest, RunEntry, RunHandle, RunId, RunSettled, RunStart, RunStartAction,
    Scratchpad, SettledOutcome, StepEventKind, TextDelta, ToolCallDelta,
};
pub use hook::{DispatchAction, DispatchEvent, OutcomeAction, OutcomeEvent};
pub use rig_core::completion::ModelRef;
/// Provider-neutral identity shared by core responses, stream terminals,
/// completion calls, and hook events.
pub use rig_core::completion::ResponseIdentity;
pub use rig_core::message::Text;
pub use run::TurnTools;
pub use run::{AgentRun, AgentRunStep, ModelTurn, ModelTurnOutcome, OutputMode, PendingToolCall};
pub use runner::AgentRunner;
pub use streaming::{
    MultiTurnStreamItem, RUN_EVENTS_CAPACITY, RunEvents, StreamingError, StreamingResult,
    stream_to_stdout,
};
pub use typed::{TypedPromptResponse, TypedRun};
