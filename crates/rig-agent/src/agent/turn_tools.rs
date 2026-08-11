//! One turn's advertised tool sets and their dispatch target.
//!
//! [`TurnTools`] is produced when a model turn is prepared — by
//! [`AgentDriver`](super::AgentDriver) for hand-driven runs, and internally by
//! the runner — and carries everything the turn's tool handling needs, resolved
//! together at one instant: the executable and allowed tool-name sets, the
//! synthetic output-tool name, and dispatch pinned to the registry snapshot
//! whose definitions the provider received.

use std::collections::BTreeSet;
use std::sync::Arc;

use rig_core::message::UserContent;

use super::model::ModelHandle;
use super::run::{ModelTurn, PendingToolCall};
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
        self.snapshot.execute(tool_name, args, context).await
    }

    /// Execute one pending call and produce its tool-result content.
    ///
    /// Honors [`PendingToolCall::preresolved_result`] — a call suppressed by
    /// invalid tool-call recovery returns its pre-resolved content without
    /// executing anything or touching `context`. Otherwise the call dispatches
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
    /// prepared turn. The single construction site for driver-facing turns,
    /// so the two name sets can never be transposed by a caller.
    pub(crate) fn model_turn(&self, response: &CompletionResponse) -> ModelTurn {
        ModelTurn::new(
            response.message_id.clone(),
            response.choice.clone(),
            response.usage,
            (*self.executable_tool_names).clone(),
            (*self.allowed_tool_names).clone(),
        )
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
}
