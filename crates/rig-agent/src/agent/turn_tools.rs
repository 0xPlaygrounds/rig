//! One turn's advertised tool sets and their dispatch target.
//!
//! [`TurnTools`] is produced when a model turn is prepared — by
//! [`AgentDriver`](super::AgentDriver) for hand-driven runs, and internally by
//! the runner — and carries everything the turn's tool handling needs, resolved
//! together at one instant: the executable and allowed tool-name sets, the
//! synthetic output-tool name, and dispatch pinned to the registry snapshot
//! whose definitions the provider received.
//!
//! The type splits along the durability line: [`TurnToolNames`] is data and is
//! recorded on the [`AgentRun`](super::run::AgentRun), while the dispatch
//! target is a live object rebuilt from the registry when a run resumes
//! elsewhere. [`TurnTools`] is the two halves paired back together.

use std::collections::BTreeSet;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use rig_core::message::{ToolChoice, UserContent};

use super::model::ModelHandle;
use super::run::{ModelTurn, PendingToolCall, StreamedTurnAssembler};
use crate::completion::{CompletionRequestBuilder, CompletionResponse};
use crate::tool::server::ToolRegistrySnapshot;
use crate::tool::{ToolContext, ToolExecutionError, ToolResult};

/// A prepared completion request: the configured builder plus the turn's tool
/// state, computed together from one registry snapshot.
pub(crate) struct PreparedCompletionRequest {
    /// Builder carrying the selected model handle: request preparation ran
    /// against this handle's captured capabilities, and the same handle
    /// executes the prepared request.
    pub(crate) builder: CompletionRequestBuilder<ModelHandle>,
    /// The turn's tool sets and dispatch target.
    pub(crate) tools: TurnTools,
    /// The tool choice the built request actually carries, after any per-turn
    /// patch. Returned rather than re-derived by callers: preparation is where
    /// the baseline and the patch are reconciled, so it is the only place that
    /// can answer without repeating the merge rule.
    pub(crate) tool_choice: Option<ToolChoice>,
}

impl PreparedCompletionRequest {
    /// What this turn resolved to, ready to commit on the run.
    pub(crate) fn turn_metadata(&self) -> PreparedTurnMetadata {
        PreparedTurnMetadata::new(self.tools.names(), self.tool_choice.clone())
            .with_output_tool_name(self.tools.output_tool_name.clone())
    }
}

/// The serializable half of [`TurnTools`]: the tool names a model call
/// advertised to the provider.
///
/// Names are data, so they travel with the run — [`AgentRun`](super::run::AgentRun)
/// records them when a model call is committed, and they survive
/// serialization. Implementations are live objects and cannot, so the
/// dispatch target — the turn's registry snapshot — is rebuilt from the
/// agent when a run resumes in another process. Pairing the two back
/// together yields a [`TurnTools`] in any run state, in any process, which is
/// what makes a suspended run resumable at *every* step boundary rather than
/// only while tool calls are pending.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct TurnToolNames {
    /// The real registry tools advertised to the provider this turn.
    pub executable: BTreeSet<String>,
    /// The tools the active `tool_choice` let the model call this turn,
    /// including the synthetic output tool when Tool output mode is active.
    pub allowed: BTreeSet<String>,
}

impl TurnToolNames {
    /// The names a turn advertised: the executable registry tools, and the
    /// tools the active `tool_choice` allowed the model to call.
    ///
    /// A hand-driver that prepares its own requests commits these with
    /// [`AgentRun::commit_model_call`](super::run::AgentRun::commit_model_call);
    /// this is how it builds them. The type is `#[non_exhaustive]` because it
    /// is serialized run state and will gain fields, which is why a struct
    /// literal will not do — the constructor is the stable way in. (Contrast
    /// [`ModelCallInputs`](super::run::ModelCallInputs), which is deliberately
    /// exhaustive so that destructuring it keeps working: the two types answer
    /// different questions and keep different answers.)
    pub fn new(
        executable: impl IntoIterator<Item = impl Into<String>>,
        allowed: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Self {
            executable: executable.into_iter().map(Into::into).collect(),
            allowed: allowed.into_iter().map(Into::into).collect(),
        }
    }

    /// Assemble the [`ModelTurn`] for a completion response received on the
    /// turn these names were advertised for. The single construction site for
    /// driver-facing turns, so the two name sets can never be transposed by a
    /// caller.
    pub(crate) fn model_turn(&self, response: &CompletionResponse) -> ModelTurn {
        ModelTurn::new(
            response.message_id.clone(),
            response.choice.clone(),
            response.usage,
            self.executable.clone(),
            self.allowed.clone(),
        )
    }
}

/// What a prepared turn resolved to, recorded when its model call is committed.
///
/// The turn's own answers, not the run's baseline. A per-turn
/// [`RequestPatch`](crate::agent::RequestPatch) may override the tool choice —
/// from a `CompletionCall` hook under the runner, or from the preparation
/// callback under [`AgentDriver`](super::AgentDriver) — and everything
/// downstream that reasons about *what this turn was allowed to do* must read
/// what was actually sent. Reading the run's baseline instead makes the state
/// machine disagree with the wire: a `Skip` resolution permitted under a
/// baseline of `Required` when the request that went out carried `None`, or an
/// invalid-tool-call hook told the choice was `None` when the request required
/// a tool.
///
/// Serialized with the run, so a resumed turn answers those questions the same
/// way the process that sent it would have.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct PreparedTurnMetadata {
    /// The tool names this turn advertised.
    pub tools: TurnToolNames,
    /// The tool choice the request actually carried, after any per-turn patch.
    pub tool_choice: Option<ToolChoice>,
    /// The synthetic structured-output tool advertised this turn, if any.
    pub output_tool_name: Option<String>,
}

impl PreparedTurnMetadata {
    /// The metadata for a turn prepared with these tools and choice.
    pub fn new(tools: TurnToolNames, tool_choice: Option<ToolChoice>) -> Self {
        Self {
            tools,
            tool_choice,
            output_tool_name: None,
        }
    }

    /// Record the synthetic structured-output tool advertised this turn.
    pub fn with_output_tool_name(mut self, name: Option<String>) -> Self {
        self.output_tool_name = name;
        self
    }
}

/// One turn's advertised tool sets and their dispatch target.
///
/// All four pieces were resolved together when the turn was prepared:
///
/// - [`executable_tool_names`](Self::executable_tool_names) — the real
///   registry tools advertised to the provider this turn.
/// - [`allowed_tool_names`](Self::allowed_tool_names) — the tools the active
///   `tool_choice` lets the model call this turn (including the synthetic
///   output tool, which is allowed but not executable).
/// - [`output_tool_name`](Self::output_tool_name) — the synthetic
///   structured-output tool, when Tool output mode is active.
/// - [`execute`](Self::execute) / [`execute_call`](Self::execute_call) —
///   dispatch against the registry **snapshot** taken when the turn was
///   prepared.
///
/// Cloning is cheap: the sets and the snapshot are shared behind `Arc`s.
///
/// # Snapshot semantics
///
/// The dispatch target is a per-turn snapshot, not the live registry. Tools
/// added to or removed from the agent after the turn was prepared (including
/// MCP refreshes) take effect on the *next* prepared turn; calls belonging to
/// this turn dispatch through the exact implementations whose definitions the
/// provider received. This is the same guarantee the agent runner gives its
/// own turns: advertising one implementation and dispatching another is the
/// skew this type exists to close.
#[derive(Clone)]
pub struct TurnTools {
    pub(crate) snapshot: Arc<ToolRegistrySnapshot>,
    pub(crate) executable_tool_names: Arc<BTreeSet<String>>,
    pub(crate) allowed_tool_names: Arc<BTreeSet<String>>,
    pub(crate) output_tool_name: Option<String>,
}

impl std::fmt::Debug for TurnTools {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TurnTools")
            .field("executable_tool_names", &self.executable_tool_names)
            .field("allowed_tool_names", &self.allowed_tool_names)
            .field("output_tool_name", &self.output_tool_name)
            .finish_non_exhaustive()
    }
}

impl TurnTools {
    /// Pair advertised names back with a dispatch target.
    ///
    /// The two halves come from different places once a run is resumed: the
    /// names from the serialized [`AgentRun`](super::run::AgentRun), the
    /// snapshot from the resuming process's registry.
    pub(crate) fn from_parts(
        snapshot: Arc<ToolRegistrySnapshot>,
        names: TurnToolNames,
        output_tool_name: Option<String>,
    ) -> Self {
        Self {
            snapshot,
            executable_tool_names: Arc::new(names.executable),
            allowed_tool_names: Arc::new(names.allowed),
            output_tool_name,
        }
    }

    /// The advertised names, detached from the dispatch target so they can be
    /// recorded on the run.
    pub(crate) fn names(&self) -> TurnToolNames {
        TurnToolNames {
            executable: (*self.executable_tool_names).clone(),
            allowed: (*self.allowed_tool_names).clone(),
        }
    }

    /// The executable registry tools advertised to the provider this turn.
    pub fn executable_tool_names(&self) -> &BTreeSet<String> {
        &self.executable_tool_names
    }

    /// The tools the active `tool_choice` lets the model call this turn,
    /// including the synthetic output tool when Tool output mode is active.
    pub fn allowed_tool_names(&self) -> &BTreeSet<String> {
        &self.allowed_tool_names
    }

    /// The synthetic structured-output tool advertised this turn, when the
    /// agent's output schema resolved to Tool output mode.
    ///
    /// This tool is **allowed but not executable**: a model call to it *is*
    /// the final structured answer. A driver never sees such a call as a
    /// pending tool — the run machine intercepts it and finalizes — so this
    /// is informational for [`AgentDriver`](super::AgentDriver) users.
    /// Dispatching it through [`execute`](Self::execute) is rejected with
    /// [`ToolErrorKind::NotExecutable`](crate::tool::ToolErrorKind::NotExecutable).
    pub fn output_tool_name(&self) -> Option<&str> {
        self.output_tool_name.as_deref()
    }

    /// Execute a tool call through the exact implementation advertised for
    /// this turn.
    ///
    /// Mirrors [`ToolServerHandle::execute`](crate::tool::server::ToolServerHandle::execute):
    /// the result carries success, failure, or refusal as a [`ToolResult`],
    /// and the tool's result metadata is published back to `context`. A name
    /// that was not advertised this turn — including one registered on the
    /// agent *after* the turn was prepared — resolves to a
    /// [`NotFound`](crate::tool::ToolErrorKind::NotFound) failure rather than
    /// reaching the live registry. The synthetic output tool is rejected with
    /// [`NotExecutable`](crate::tool::ToolErrorKind::NotExecutable): it has no
    /// implementation, and its call carries the final structured answer.
    pub async fn execute(
        &self,
        tool_name: &str,
        args: &str,
        context: &mut ToolContext,
    ) -> ToolResult {
        context.clear_dispatch_result();
        if self.output_tool_name.as_deref() == Some(tool_name) {
            return ToolResult::failed(ToolExecutionError::not_executable(format!(
                "`{tool_name}` is this turn's synthetic structured-output tool: it is advertised \
                 to the model but has no implementation. Its call carries the final structured \
                 answer; the driver must consume the call's arguments instead of dispatching it."
            )));
        }
        // The advertised names are the authority, not the snapshot. In-process
        // the two agree by construction (preparation narrows the snapshot to
        // the advertised set), but a resumed turn pairs names carried on the
        // run with a snapshot rebuilt here, so only this check makes the
        // guarantee above hold on both paths.
        if !self.executable_tool_names.contains(tool_name) {
            return ToolResult::failed(ToolExecutionError::not_found(format!(
                "`{tool_name}` was not advertised to the model on this turn"
            )));
        }
        self.snapshot.execute(tool_name, args, context).await
    }

    /// Execute one pending call and produce its tool-result content.
    ///
    /// Honors [`PendingToolCall::preresolved_result`] — a call suppressed by
    /// invalid tool-call recovery returns its pre-resolved content without
    /// executing anything. It still clears `context`'s dispatch result: the
    /// suppressed call published no metadata, so leaving the *previous* call's
    /// in place would attribute it to this one, and a loop over a turn's calls
    /// reads that metadata per call. Otherwise the call dispatches
    /// through this turn's snapshot via [`execute`](Self::execute) and the
    /// result is assembled into the [`UserContent`] tool-result value that
    /// [`AgentDriver::tool_results`](super::AgentDriver::tool_results) (or
    /// [`AgentRun::tool_results`](super::run::AgentRun::tool_results)) expects.
    ///
    /// Approval flows compose around this: execute approved calls here, and
    /// build denied calls' result content directly (see
    /// `examples/agent_with_durable_approval`).
    pub async fn execute_call(
        &self,
        call: &PendingToolCall,
        context: &mut ToolContext,
    ) -> UserContent {
        if let Some(result) = &call.preresolved_result {
            // Every dispatch surface clears first; this one dispatches nothing
            // but is still a call in the turn's sequence, so it must not
            // inherit the previous call's result metadata.
            context.clear_dispatch_result();
            return result.clone();
        }
        let name = &call.tool_call.function.name;
        let args = call.tool_call.function.arguments.to_string();
        let result = self.execute(name, &args, context).await;
        UserContent::tool_result_for(
            call.tool_call.id.clone(),
            call.tool_call.provider.clone(),
            name.clone(),
            result.output().clone().into_content(),
        )
    }

    /// Assemble the [`ModelTurn`] for a completion response received on this
    /// prepared turn. Delegates to [`TurnToolNames::model_turn`], the single
    /// construction site for driver-facing turns.
    pub(crate) fn model_turn(&self, response: &CompletionResponse) -> ModelTurn {
        self.names().model_turn(response)
    }

    /// The assembler for this turn's provider stream, carrying the names this
    /// turn advertised.
    ///
    /// The streaming counterpart of [`Self::model_turn`], and the way a
    /// [`AgentDriver`](super::AgentDriver) caller should build one: it takes
    /// the names off the turn the matching
    /// [`DriveStep::SendRequest`](super::DriveStep::SendRequest) carried, so
    /// there is no pair of same-typed sets for a caller to transpose or widen.
    /// Feed [`StreamedTurnAssembler::finish`]'s result to
    /// [`AgentDriver::accept_streamed_turn`](super::AgentDriver::accept_streamed_turn).
    pub fn streamed_turn_assembler(&self) -> StreamedTurnAssembler {
        StreamedTurnAssembler::new(&self.names())
    }
}

#[cfg(test)]
mod tests {
    use crate::agent::run::OutputMode;
    use crate::agent::{AgentBuilder, DriveStep};
    use crate::test_utils::MockCompletionModel;
    use crate::tool::{Tool, ToolContext, ToolErrorKind};
    use serde_json::json;

    /// A tool that publishes result metadata into the caller's context.
    struct MetadataTool;

    #[derive(Clone, Debug, PartialEq)]
    struct Marker(&'static str);

    #[derive(Debug, thiserror::Error)]
    #[error("metadata tool failed")]
    struct MetadataToolError;

    impl Tool for MetadataTool {
        const NAME: &'static str = "metadata";
        type Error = MetadataToolError;
        type Args = serde_json::Value;
        type Output = i32;

        fn description(&self) -> String {
            "publishes a result marker".to_string()
        }

        fn parameters(&self) -> serde_json::Value {
            json!({ "type": "object", "properties": {} })
        }

        async fn call(
            &self,
            context: &mut ToolContext,
            _args: Self::Args,
        ) -> Result<Self::Output, Self::Error> {
            context.insert_result(Marker("published"));
            Ok(1)
        }
    }

    /// Regression (finding 4): the output-tool rejection must clear the
    /// previous dispatch's result metadata, exactly like every other
    /// `execute` path.
    #[tokio::test]
    async fn rejected_output_tool_dispatch_clears_stale_context_metadata() {
        let agent = AgentBuilder::new(MockCompletionModel::text("unused"))
            .tool(MetadataTool)
            .output_schema_raw(
                serde_json::from_value(json!({
                    "type": "object",
                    "properties": { "value": { "type": "integer" } },
                    "required": ["value"]
                }))
                .expect("valid schema"),
            )
            .output_mode(OutputMode::Tool)
            .build();
        let mut driver = agent.drive("go");
        let tools = match driver.next_step().await.expect("prepare succeeds") {
            DriveStep::SendRequest { tools, .. } => tools,
            other => panic!("expected SendRequest, got {other:?}"),
        };
        let output_tool = tools
            .output_tool_name()
            .expect("Tool mode advertises an output tool")
            .to_owned();

        let mut context = ToolContext::new();
        let result = tools.execute("metadata", "{}", &mut context).await;
        assert!(result.is_success());
        assert_eq!(
            context.result::<Marker>(),
            Some(&Marker("published")),
            "the first dispatch publishes its metadata"
        );

        let result = tools.execute(&output_tool, "{}", &mut context).await;
        assert!(result.is_error_kind(ToolErrorKind::NotExecutable));
        assert_eq!(
            context.result::<Marker>(),
            None,
            "the rejection must not leave the previous dispatch's metadata behind"
        );
    }

    /// A suppressed call dispatches nothing, but it is still a call in the
    /// turn's sequence: a loop reading per-call metadata must not see the
    /// previous call's attributed to it.
    #[tokio::test]
    async fn preresolved_call_clears_stale_context_metadata() {
        use crate::agent::run::PendingToolCall;
        use rig_core::message::{ToolCall, ToolFunction, ToolResultContent, UserContent};

        let agent = AgentBuilder::new(MockCompletionModel::text("unused"))
            .tool(MetadataTool)
            .build();
        let mut driver = agent.drive("go");
        let tools = match driver.next_step().await.expect("prepare succeeds") {
            DriveStep::SendRequest { tools, .. } => tools,
            other => panic!("expected SendRequest, got {other:?}"),
        };

        let mut context = ToolContext::new();
        let result = tools.execute("metadata", "{}", &mut context).await;
        assert!(result.is_success());
        assert_eq!(context.result::<Marker>(), Some(&Marker("published")));

        // The next call in the same turn was suppressed by invalid tool-call
        // recovery, so it carries a pre-resolved result.
        let suppressed = PendingToolCall {
            tool_call: ToolCall::from_wire(
                "call_2",
                ToolFunction::new("metadata".to_string(), json!({})),
            ),
            preresolved_result: Some(UserContent::tool_result(
                "call_2",
                "metadata",
                vec![ToolResultContent::text("not executed")],
            )),
            internal_call_id: None,
        };
        let _ = tools.execute_call(&suppressed, &mut context).await;
        assert_eq!(
            context.result::<Marker>(),
            None,
            "a suppressed call must not inherit the previous call's metadata"
        );
    }

    /// `commit_model_call` is public and takes this type, so a hand-driver
    /// outside the crate must be able to build one. `#[non_exhaustive]` blocks
    /// a struct literal there, which is what the constructor is for.
    #[test]
    fn advertised_names_are_constructible_without_a_struct_literal() {
        use super::TurnToolNames;

        let names = TurnToolNames::new(["add", "subtract"], ["add"]);
        assert!(names.executable.contains("subtract"));
        assert!(names.allowed.contains("add") && !names.allowed.contains("subtract"));
    }

    /// The advertised names are the authority, not the snapshot. Pair a
    /// snapshot that *can* dispatch a tool with names that never advertised
    /// it — the shape a resumed turn produces — and dispatch must still
    /// refuse.
    #[tokio::test]
    async fn dispatch_refuses_a_name_the_turn_did_not_advertise() {
        use super::{TurnToolNames, TurnTools};
        use std::collections::BTreeSet;

        let agent = AgentBuilder::new(MockCompletionModel::text("unused"))
            .tool(MetadataTool)
            .build();
        let mut driver = agent.drive("go");
        let advertised = match driver.next_step().await.expect("prepare succeeds") {
            DriveStep::SendRequest { tools, .. } => tools,
            other => panic!("expected SendRequest, got {other:?}"),
        };
        assert!(advertised.executable_tool_names().contains("metadata"));

        // Same dispatch target, but a turn that advertised nothing.
        let disowned = TurnTools::from_parts(
            advertised.snapshot.clone(),
            TurnToolNames {
                executable: BTreeSet::new(),
                allowed: BTreeSet::new(),
            },
            None,
        );
        let mut context = ToolContext::new();
        let result = disowned.execute("metadata", "{}", &mut context).await;
        assert!(
            result.is_error_kind(ToolErrorKind::NotFound),
            "a reachable implementation is still not dispatchable if the turn never advertised it"
        );
    }

    /// The advertised-name gate is an `execute` early return like the
    /// output-tool rejection, so it owes the same guarantee: the previous
    /// dispatch's result metadata must not survive it.
    #[tokio::test]
    async fn unadvertised_name_rejection_clears_stale_context_metadata() {
        let agent = AgentBuilder::new(MockCompletionModel::text("unused"))
            .tool(MetadataTool)
            .build();
        let mut driver = agent.drive("go");
        let tools = match driver.next_step().await.expect("prepare succeeds") {
            DriveStep::SendRequest { tools, .. } => tools,
            other => panic!("expected SendRequest, got {other:?}"),
        };

        let mut context = ToolContext::new();
        let result = tools.execute("metadata", "{}", &mut context).await;
        assert!(result.is_success());
        assert_eq!(context.result::<Marker>(), Some(&Marker("published")));

        let result = tools.execute("never_advertised", "{}", &mut context).await;
        assert!(result.is_error_kind(ToolErrorKind::NotFound));
        assert_eq!(
            context.result::<Marker>(),
            None,
            "the rejection must not leave the previous dispatch's metadata behind"
        );
    }
}
