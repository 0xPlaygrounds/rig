//! One prepared model turn of a configured [`Agent`](super::Agent), for
//! hand-driven [`AgentRun`](super::run::AgentRun) loops.
//!
//! [`Agent::prepare_turn`](super::Agent::prepare_turn) reads the agent's
//! configuration — preamble, static context, temperature, `max_tokens`,
//! `additional_params`, `tool_choice`, output schema/mode, and the tool
//! registry — and resolves it into a [`PreparedTurn`]: the fully configured
//! completion request for one turn plus a [`TurnTools`] value that keeps the
//! turn's advertised tool set and its dispatch target consistent.
//!
//! This exists so a hand-driven `AgentRun` loop (custom provider transport,
//! suspend/resume across processes) can reuse the `Agent` it configured instead
//! of restating every piece of that configuration by hand. It is a
//! configuration read plus a dispatch target — not a second execution path:
//! [`Agent::runner`](super::Agent::runner) remains the only way to execute an
//! agent with hooks, memory, retrieval policy, and telemetry.

use std::collections::BTreeSet;
use std::sync::Arc;

use super::completion::PreparedCompletionRequest;
use super::model::ModelHandle;
use crate::completion::CompletionRequestBuilder;
use crate::tool::server::ToolRegistrySnapshot;
use crate::tool::{ToolContext, ToolDispatch, ToolExecutionError, ToolResult};

/// One prepared model turn: the configured completion request plus the turn's
/// tool sets and dispatch target, resolved together at one instant.
///
/// Produced by [`Agent::prepare_turn`](super::Agent::prepare_turn). The request
/// half and the tool half were computed from the same registry snapshot, so the
/// tools advertised in the request are exactly the tools [`TurnTools`]
/// dispatches — even if the agent's live registry changes afterwards.
///
/// A prepared turn reflects the agent's **baseline** configuration. No hooks
/// run: there is no `CompletionCall` request patch, no model-selection hook,
/// and no per-turn `active_tools` narrowing. Anyone who needs those wants
/// [`Agent::runner`](super::Agent::runner).
pub struct PreparedTurn {
    builder: CompletionRequestBuilder<ModelHandle>,
    tools: TurnTools,
}

impl std::fmt::Debug for PreparedTurn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedTurn")
            .field("tools", &self.tools)
            .finish_non_exhaustive()
    }
}

impl PreparedTurn {
    pub(crate) fn from_prepared(prepared: PreparedCompletionRequest) -> Self {
        let PreparedCompletionRequest {
            builder,
            tool_snapshot,
            executable_tool_names,
            allowed_tool_names,
            output_tool_name,
        } = prepared;
        Self {
            builder,
            tools: TurnTools {
                snapshot: tool_snapshot,
                executable_tool_names,
                allowed_tool_names,
                output_tool_name,
            },
        }
    }

    /// The turn's tool sets and dispatch target.
    ///
    /// [`TurnTools`] is cheap to clone; clone it before consuming the request
    /// with [`into_parts`](Self::into_parts) if you only need the tool half
    /// later.
    pub fn tools(&self) -> &TurnTools {
        &self.tools
    }

    /// Split the prepared turn into the configured completion request builder
    /// and the turn's [`TurnTools`].
    ///
    /// The builder already carries the agent's preamble (with any output-mode
    /// augmentation), static context, model parameters, `tool_choice`, and this
    /// turn's tool definitions. Call
    /// [`send`](crate::completion::CompletionRequestBuilder::send) on it to
    /// issue the request against the agent's configured model, or
    /// [`build`](crate::completion::CompletionRequestBuilder::build) it to hand
    /// the raw request to a custom transport.
    pub fn into_parts(self) -> (CompletionRequestBuilder<ModelHandle>, TurnTools) {
        (self.builder, self.tools)
    }
}

/// One turn's advertised tool sets and their dispatch target.
///
/// All four pieces were resolved together when the turn was prepared:
///
/// - [`executable_tool_names`](Self::executable_tool_names) — the real
///   registry tools advertised to the provider this turn. Feed this to
///   [`ModelTurn::new`](super::run::ModelTurn::new).
/// - [`allowed_tool_names`](Self::allowed_tool_names) — the tools the active
///   `tool_choice` lets the model call this turn (including the synthetic
///   output tool, which is allowed but not executable). Feed this to
///   [`ModelTurn::new`](super::run::ModelTurn::new) as well.
/// - [`output_tool_name`](Self::output_tool_name) — the synthetic
///   structured-output tool, when Tool output mode is active.
/// - [`execute`](Self::execute) — dispatch against the registry **snapshot**
///   taken when the turn was prepared.
///
/// # Snapshot semantics
///
/// The dispatch target is a per-turn snapshot, not the live registry. Tools
/// added to or removed from the agent after the turn was prepared (including
/// MCP refreshes) take effect on the *next* prepared turn; calls belonging to
/// this turn dispatch through the exact implementations whose definitions the
/// provider received. This is the same guarantee the agent runner gives its own
/// turns, and it is why this type exists instead of a handle to the live
/// registry: advertising one implementation and dispatching another is the skew
/// this closes.
#[derive(Clone)]
pub struct TurnTools {
    snapshot: Arc<ToolRegistrySnapshot>,
    executable_tool_names: BTreeSet<String>,
    allowed_tool_names: BTreeSet<String>,
    output_tool_name: Option<String>,
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
    /// This tool is **allowed but not executable**: a model call to it *is* the
    /// final structured answer. A hand-driven loop must intercept it by name —
    /// consume its arguments as the run's structured output instead of
    /// dispatching it (dispatching it through [`execute`](Self::execute) is
    /// rejected). Drivers that validate the arguments against the output schema
    /// should feed failures back to the model as an error tool result and
    /// re-prompt, mirroring
    /// [`AgentRun::with_output_validation`](super::run::AgentRun::with_output_validation).
    pub fn output_tool_name(&self) -> Option<&str> {
        self.output_tool_name.as_deref()
    }

    /// Execute a tool call through the exact implementation advertised for this
    /// turn.
    ///
    /// Mirrors [`ToolServerHandle::execute`](crate::tool::server::ToolServerHandle::execute):
    /// the result carries success, failure, or refusal as a [`ToolResult`], and
    /// the tool's result metadata is published back to `context`. A name that
    /// was not advertised this turn — including one registered on the agent
    /// *after* the turn was prepared — resolves to a not-found failure rather
    /// than reaching the live registry. Calling the synthetic output tool is
    /// rejected: it has no implementation, and its call is the final structured
    /// answer the driver must consume itself (see
    /// [`output_tool_name`](Self::output_tool_name)).
    pub async fn execute(
        &self,
        tool_name: &str,
        args: &str,
        context: &mut ToolContext,
    ) -> ToolResult {
        if self.output_tool_name.as_deref() == Some(tool_name) {
            return ToolResult::failed(ToolExecutionError::not_found(format!(
                "`{tool_name}` is this turn's synthetic structured-output tool: it is advertised \
                 to the model but has no implementation. Its call carries the final structured \
                 answer; the driver must consume the call's arguments instead of dispatching it."
            )));
        }
        context.clear_dispatch_result();
        let ToolDispatch {
            result,
            context: dispatch_context,
        } = self.snapshot.dispatch(tool_name, args, context).await;
        context.accept_dispatch_result(dispatch_context);
        result
    }
}

#[cfg(test)]
mod tests {
    use crate::agent::AgentBuilder;
    use crate::agent::run::OutputMode;
    use crate::completion::{CompletionError, Message};
    use crate::test_utils::{MockAddTool, MockCompletionModel, MockSubtractTool};
    use crate::tool::{ToolContext, ToolErrorKind};
    use rig_core::message::ToolChoice;
    use serde_json::json;

    fn schema(value: serde_json::Value) -> schemars::Schema {
        serde_json::from_value(value).expect("valid schema")
    }

    /// Criterion: advertise/dispatch consistency. A prepared turn keeps
    /// dispatching the implementation it advertised even after the agent's
    /// live registry mutates, and never sees tools registered afterwards.
    #[tokio::test]
    async fn prepared_turn_dispatches_the_implementation_it_advertised() {
        let agent = AgentBuilder::new(MockCompletionModel::text("unused"))
            .preamble("prepared preamble")
            .tool(MockAddTool)
            .build();

        let prepared = agent
            .prepare_turn("add 1 and 2", &[])
            .await
            .expect("prepare succeeds");
        let (request, tools) = prepared.into_parts();

        // The request carries the agent's configuration: the preamble leads
        // the history and the registered tool is advertised.
        let request = request.build();
        assert!(matches!(
            request.chat_history.first(),
            Some(Message::System { content }) if content == "prepared preamble"
        ));
        assert!(request.tools.iter().any(|tool| tool.name == "add"));
        assert!(tools.executable_tool_names().contains("add"));
        assert_eq!(tools.executable_tool_names(), tools.allowed_tool_names());

        // Mutate the live registry AFTER preparation: retire the advertised
        // tool and register a different one.
        agent.tool_server_handle.remove_tool("add").await;
        agent.tool_server_handle.add_tool(MockSubtractTool).await;

        // The prepared turn still dispatches the implementation it advertised...
        let mut context = ToolContext::new();
        let result = tools.execute("add", r#"{"x": 1, "y": 2}"#, &mut context).await;
        assert!(
            result.is_success(),
            "snapshot dispatch must reach the advertised implementation"
        );

        // ...and does not see tools registered after the snapshot was taken.
        let mut context = ToolContext::new();
        let result = tools
            .execute("subtract", r#"{"x": 1, "y": 2}"#, &mut context)
            .await;
        assert!(result.is_error_kind(ToolErrorKind::NotFound));

        // The live registry really did change underneath the snapshot.
        let mut context = ToolContext::new();
        let result = agent
            .tool_server_handle
            .execute("add", r#"{"x": 1, "y": 2}"#, &mut context)
            .await;
        assert!(result.is_error_kind(ToolErrorKind::NotFound));
    }

    /// Criterion: under Tool output mode the synthetic output tool is allowed
    /// and advertised but never executable, and dispatching it is rejected.
    #[tokio::test]
    async fn output_tool_is_allowed_but_never_executable() {
        let agent = AgentBuilder::new(MockCompletionModel::text("unused"))
            .tool(MockAddTool)
            .output_schema_raw(schema(json!({
                "type": "object",
                "properties": { "value": { "type": "integer" } },
                "required": ["value"]
            })))
            .output_mode(OutputMode::Tool)
            .build();

        let prepared = agent
            .prepare_turn("compute", &[])
            .await
            .expect("prepare succeeds");
        let (request, tools) = prepared.into_parts();
        let output_tool = tools
            .output_tool_name()
            .expect("Tool output mode advertises an output tool")
            .to_owned();

        assert!(tools.allowed_tool_names().contains(&output_tool));
        assert!(!tools.executable_tool_names().contains(&output_tool));
        // The provider sees the synthetic tool alongside the real one.
        let request = request.build();
        assert!(request.tools.iter().any(|tool| tool.name == output_tool));
        assert!(request.tools.iter().any(|tool| tool.name == "add"));

        let mut context = ToolContext::new();
        let result = tools.execute(&output_tool, "{}", &mut context).await;
        assert!(result.is_error_kind(ToolErrorKind::NotFound));
        let message = result.error().expect("rejected dispatch").message().to_owned();
        assert!(
            message.contains("structured-output"),
            "rejection must explain the synthetic tool: {message}"
        );
    }

    /// Criterion: an impossible `ToolChoice` fails at prepare time, locally,
    /// with no provider round-trip.
    #[tokio::test]
    async fn impossible_tool_choice_fails_locally_at_prepare_time() {
        let model = MockCompletionModel::text("unused");
        let agent = AgentBuilder::new(model.clone())
            .tool(MockAddTool)
            .tool_choice(ToolChoice::Specific {
                function_names: vec!["missing".to_string()],
            })
            .build();

        let err = agent
            .prepare_turn("go", &[])
            .await
            .expect_err("a tool_choice naming an unadvertised tool must fail at prepare time");
        assert!(matches!(
            err,
            CompletionError::RequestError(inner) if inner.to_string().contains("missing")
        ));
        assert_eq!(
            model.request_count(),
            0,
            "local validation must not cost a provider round-trip"
        );
    }

    /// `Required` forces a tool call, so an empty advertised set can never
    /// satisfy it — that must fail at prepare time, not degrade silently.
    #[tokio::test]
    async fn required_tool_choice_with_no_tools_fails_locally() {
        let model = MockCompletionModel::text("unused");
        let agent = AgentBuilder::new(model.clone())
            .tool_choice(ToolChoice::Required)
            .build();

        let err = agent
            .prepare_turn("go", &[])
            .await
            .expect_err("Required with no advertised tool must fail at prepare time");
        assert!(matches!(
            err,
            CompletionError::RequestError(inner) if inner.to_string().contains("Required")
        ));
        assert_eq!(model.request_count(), 0);
    }
}
