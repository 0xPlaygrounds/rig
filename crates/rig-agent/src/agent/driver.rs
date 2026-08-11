//! Hand-drive a configured [`Agent`] while owning every side effect yourself.
//!
//! [`Agent::drive`] pairs the sans-IO [`AgentRun`] state machine with the
//! agent's configuration: each [`DriveStep::SendRequest`] carries the fully
//! configured completion request (the caller sends it — or hands it to a
//! custom transport), and each [`DriveStep::ExecuteTools`] carries the
//! [`TurnTools`] of the turn that advertised those calls, paired by
//! construction rather than by caller discipline. The driver itself performs
//! **no** provider IO and **no** tool dispatch, and runs no hooks, memory,
//! retrieval policy, or telemetry — it is the run/turn pairing logic the
//! runner keeps internally, exposed for callers who own the IO. To *execute*
//! an agent with hooks, memory, and telemetry, use
//! [`Agent::runner`](super::Agent::runner); the driver is not a second
//! execution path.
//!
//! # Why a driver and not a bag of getters
//!
//! Hand-driving `AgentRun` requires state that must stay mutually consistent:
//! the request a turn sent, the tool sets that validate the model's calls, the
//! snapshot that dispatches them, and the run's committed structured-output
//! tool. Leaving that pairing to callers is how configuration drift and
//! advertise/dispatch skew happen; the driver owns it in one place while every
//! side effect stays with the caller.
//!
//! It owns that pairing without *holding* it. Everything durable lives on the
//! [`AgentRun`] — including the turn's advertised tool names — and the driver
//! keeps only a cache of the live registry snapshot, which cannot be
//! serialized in any design. That is what makes the durability guarantees
//! below hold at every step rather than at one of them.
//!
//! # Durability
//!
//! The serializable state is *all* of the state: the driver holds nothing it
//! could lose. Serialize [`AgentDriver::run`] at any step boundary — while
//! tool calls are pending, or while a model call is in flight with a
//! long-running or queued provider — and resume in another process with
//! [`Agent::drive_run`]. Every step is a resume point, including
//! [`DriveStep::SendRequest`]: the turn's advertised tool names travel with
//! the run, so the resuming process validates the model's reply against the
//! set the request actually carried rather than against whatever its registry
//! holds now.
//!
//! Preparation failures are equally survivable. Nothing advances until a
//! request exists, so a turn that fails to prepare — an unreachable tool
//! server, an impossible `tool_choice` — costs no turn from the budget and
//! leaves the run byte-identical, ready to retry.
//!
//! Tool *implementations* are live objects and cannot be serialized: the
//! resuming process rebuilds the same `Agent` and the driver takes a fresh
//! registry snapshot to dispatch pending calls through. If that snapshot no
//! longer contains a pending call's tool, the driver surfaces an error instead
//! of silently feeding a not-found result to the model (see
//! [`AgentDriver::allow_missing_resumed_tools`]) — the model chose that tool
//! from a registry this process no longer has, and re-prompting cannot fix
//! deployment drift. If you suspend runs across deploys, version your agent
//! definitions alongside the serialized run; the run's own format is versioned
//! by [`RUN_SCHEMA_VERSION`](super::run::RUN_SCHEMA_VERSION).

use std::sync::Arc;

use rig_core::message::UserContent;

use super::completion::{Agent, TurnBaseline, TurnRequest, build_prepared_completion_request};
use super::model::ModelHandle;
use super::run::{Advance, AgentRun, ModelCallInputs, ModelTurnOutcome, PendingToolCall};
use super::runner::build_agent_run;
use super::turn_tools::{PreparedCompletionRequest, TurnTools};
use crate::agent::hook::{InvalidToolCallAction, RequestPatch};
use crate::agent::prompt_request::PromptResponse;
use crate::completion::{
    CompletionError, CompletionRequestBuilder, CompletionResponse, Message, PromptError,
};
use crate::tool::server::ToolRegistrySnapshot;

impl Agent {
    /// Hand-drive this agent: build a driver whose run is seeded from the
    /// agent's configuration.
    ///
    /// Seeding mirrors [`Agent::runner`]: the model-call budget comes from
    /// `default_max_turns` (implicit budget of one when unset), and the run
    /// inherits the agent's `tool_choice` and output schema (with the default
    /// output-retry budget). Override per run with [`AgentDriver::max_turns`]
    /// and [`AgentDriver::history`], or construct a custom [`AgentRun`] and
    /// use [`Agent::drive_run`].
    pub fn drive(&self, prompt: impl Into<Message>) -> AgentDriver {
        self.drive_run(build_agent_run(
            prompt.into(),
            self.default_max_turns.unwrap_or(1),
            0,
            self.output_schema.as_ref(),
            None,
            self.tool_choice.clone(),
        ))
    }

    /// Hand-drive an existing [`AgentRun`] with this agent's configuration —
    /// the resume path for a run deserialized in a new process, or the entry
    /// point for a custom-configured run (which is taken as-is, not re-seeded).
    pub fn drive_run(&self, run: AgentRun) -> AgentDriver {
        AgentDriver {
            agent: self.clone(),
            run,
            snapshot: None,
            request_patch: RequestPatch::new(),
            allow_missing_resumed_tools: false,
        }
    }
}

/// What the caller must do next to advance an [`AgentDriver`].
///
/// Deliberately exhaustive, like [`AgentRunStep`]: a driver loop must handle
/// every step, so adding a variant is a breaking change by design.
pub enum DriveStep {
    /// Send this completion request to the model — via
    /// [`send`](CompletionRequestBuilder::send), or
    /// [`build`](CompletionRequestBuilder::build) it for a custom transport —
    /// then feed the response back through [`AgentDriver::model_response`].
    SendRequest {
        /// The fully configured request: the agent's preamble (with any
        /// output-mode augmentation), static context, model parameters,
        /// `tool_choice`, and this turn's tool definitions — with the
        /// driver's [`RequestPatch`](AgentDriver::request_patch) applied over
        /// that baseline, which is how a hand-driven turn gets the per-turn
        /// preamble, `tool_choice`, or `active_tools` narrowing the runner
        /// gets from its `CompletionCall` hooks. No hooks run on this path;
        /// the patch is the seam. The request still honors the agent's
        /// `record_telemetry_content` for provider-level spans; call
        /// `.record_content_telemetry(false)` on it to opt a hand-driven turn
        /// out.
        request: Box<CompletionRequestBuilder<ModelHandle>>,
        /// The turn's advertised tool sets — informational here (the driver
        /// assembles the model turn itself); the same value arrives on the
        /// following [`ExecuteTools`](Self::ExecuteTools) step for dispatch.
        tools: TurnTools,
        /// One-based index of this model call within the run.
        turn: usize,
    },
    /// Execute these tool calls — typically via
    /// [`TurnTools::execute_call`] — and feed the results back through
    /// [`AgentDriver::tool_results`]. `tools` is the dispatch target of the
    /// turn that advertised `calls`, paired by construction.
    ExecuteTools {
        /// The pending tool calls of the current assistant turn, in emission
        /// order.
        calls: Vec<PendingToolCall>,
        /// The advertising turn's tool sets and snapshot dispatch target.
        tools: TurnTools,
    },
    /// The run is complete.
    Done(PromptResponse),
}

impl std::fmt::Debug for DriveStep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SendRequest { tools, turn, .. } => f
                .debug_struct("SendRequest")
                .field("turn", turn)
                .field("tools", tools)
                .finish_non_exhaustive(),
            Self::ExecuteTools { calls, tools } => f
                .debug_struct("ExecuteTools")
                .field("calls", calls)
                .field("tools", tools)
                .finish(),
            Self::Done(response) => f.debug_tuple("Done").field(response).finish(),
        }
    }
}

/// Hand-drives one [`AgentRun`] with one [`Agent`]'s configuration. Built by
/// [`Agent::drive`] / [`Agent::drive_run`]; see the [module docs](self) for
/// the driving protocol and the boundary with [`AgentRunner`](super::AgentRunner).
pub struct AgentDriver {
    agent: Agent,
    run: AgentRun,
    /// Live dispatch target for the current turn — a **cache**, never state.
    ///
    /// Everything the driver must not lose lives on [`Self::run`]; a tool
    /// registry snapshot cannot, because implementations are live objects. It
    /// is held only so that a turn prepared in *this* process dispatches
    /// through the exact implementations the provider was shown, and is
    /// rebuilt on demand when a resumed run reaches its pending tool calls.
    snapshot: Option<Arc<ToolRegistrySnapshot>>,
    /// Per-turn request configuration applied to every turn this driver
    /// prepares. See [`Self::request_patch`].
    request_patch: RequestPatch,
    allow_missing_resumed_tools: bool,
}

impl AgentDriver {
    /// Set the input chat history preceding the prompt.
    pub fn history(mut self, history: Vec<Message>) -> Self {
        self.run = self.run.with_history(history);
        self
    }

    /// Override the seeded total model-call budget for this run.
    pub fn max_turns(mut self, max_turns: usize) -> Self {
        self.run = self.run.max_turns(max_turns);
        self
    }

    /// Set the per-turn request configuration for the turns this driver
    /// prepares.
    ///
    /// The driver runs no hooks, so this is how a hand-driven run gets what
    /// the runner gets from its `CompletionCall` hooks: a per-turn preamble,
    /// sampling parameters, `tool_choice`, `active_tools` narrowing, extra
    /// context, or a substituted history. Each set field replaces the agent's
    /// configured value for the turn; unset fields inherit it.
    ///
    /// The patch applies to every turn this driver prepares. Because the
    /// caller owns the loop, per-turn variation needs no callback — call
    /// [`Self::set_request_patch`] between steps.
    pub fn request_patch(mut self, patch: RequestPatch) -> Self {
        self.request_patch = patch;
        self
    }

    /// Replace the per-turn request configuration in place, so a driving loop
    /// can vary it from turn to turn. See [`Self::request_patch`].
    pub fn set_request_patch(&mut self, patch: RequestPatch) {
        self.request_patch = patch;
    }

    /// Opt out of the resumed-run drift check: dispatch pending calls whose
    /// tools are missing from this process's registry anyway, feeding the
    /// resulting not-found errors to the model instead of surfacing the drift
    /// to the caller. See the [module docs](self) on durability.
    pub fn allow_missing_resumed_tools(mut self) -> Self {
        self.allow_missing_resumed_tools = true;
        self
    }

    /// The sans-IO run state. Serialize this to suspend the run — for example
    /// while tool calls are pending approval — and resume it elsewhere with
    /// [`Agent::drive_run`].
    pub fn run(&self) -> &AgentRun {
        &self.run
    }

    /// Consume the driver, returning the run state.
    pub fn into_run(self) -> AgentRun {
        self.run
    }

    /// Advance to the next step the caller must perform.
    ///
    /// Preparing a model turn reads the agent's configuration and tool
    /// registry (one snapshot per turn) and maintains the run's committed
    /// structured-output tool exactly as the runner does: the committed name
    /// is re-advertised on every later turn, so Tool output mode cannot flip
    /// or re-pick a name mid-run. Fails locally — with no provider
    /// round-trip — when the configuration cannot produce a valid request
    /// (e.g. a `tool_choice` impossible against the advertised tool set).
    ///
    /// **Such a failure costs nothing.** Preparation runs entirely before the
    /// run advances: the turn is committed only once a request exists
    /// ([`AgentRun::commit_model_call`]), so an error here leaves the run
    /// exactly as it was — same state, same turn budget — and the step can be
    /// retried once the cause is fixed (a tool server that was briefly
    /// unreachable, say), here or in another process.
    pub async fn next_step(&mut self) -> Result<DriveStep, PromptError> {
        match self.run.advance()? {
            Advance::NeedsModelCall => {
                // Peek, prepare, *then* commit. Reading the inputs consumes
                // nothing, so everything fallible below happens while the run
                // is still fully intact.
                let ModelCallInputs {
                    prompt, history, ..
                } = self.run.peek_model_call()?;
                // Pin Tool output mode once committed (#1928), mirroring the
                // runner: read the run's committed name into preparation, and
                // store the resolved name back (fill-once).
                let committed = self.run.output_tool_name().map(str::to_owned);
                let patch = self.effective_request_patch();
                let prepared = build_prepared_completion_request(
                    TurnBaseline::from_agent(&self.agent),
                    TurnRequest {
                        prompt,
                        chat_history: &history,
                        committed_output_tool: committed.as_deref(),
                        patch: Some(&patch),
                    },
                )
                .await
                .map_err(PromptError::CompletionError)?;

                let PreparedCompletionRequest { builder, tools } = prepared;
                self.snapshot = Some(tools.snapshot.clone());
                let turn = self
                    .run
                    .commit_model_call(Some(tools.names()), tools.output_tool_name.clone());
                Ok(DriveStep::SendRequest {
                    request: Box::new(builder),
                    tools,
                    turn,
                })
            }
            Advance::CallTools(calls) => {
                let tools = self.dispatch_tools_for_turn(&calls).await?;
                Ok(DriveStep::ExecuteTools { calls, tools })
            }
            Advance::Done(response) => Ok(DriveStep::Done(response)),
        }
    }

    /// Feed one completion response back into the run. The turn's tool sets
    /// are supplied by the driver — the caller never assembles them.
    ///
    /// As with [`AgentRun::model_response`], a
    /// [`ModelTurnOutcome::NeedsResolution`] outcome must be answered via
    /// [`Self::resolve_invalid_tool_call`] before advancing.
    pub fn model_response(
        &mut self,
        response: &CompletionResponse,
    ) -> Result<ModelTurnOutcome, PromptError> {
        // The advertised names come from the run, not from this driver — which
        // is what lets a run serialized between `SendRequest` and the model's
        // reply be resumed in another process, and what guarantees the
        // response is validated against the set the request actually carried
        // rather than whatever the registry holds now.
        let Some(names) = self.run.advertised_tools().cloned() else {
            return Err(PromptError::CompletionError(CompletionError::RequestError(
                "model_response must follow a SendRequest step from this driver".into(),
            )));
        };
        self.run.model_response(names.model_turn(response))
    }

    /// Resolve a pending invalid tool call, exactly as
    /// [`AgentRun::resolve_invalid_tool_call`].
    pub fn resolve_invalid_tool_call(
        &mut self,
        action: InvalidToolCallAction,
    ) -> Result<ModelTurnOutcome, PromptError> {
        self.run.resolve_invalid_tool_call(action)
    }

    /// Feed the results for the pending tool calls back into the run, exactly
    /// as [`AgentRun::tool_results`].
    pub fn tool_results(&mut self, results: Vec<UserContent>) -> Result<(), PromptError> {
        self.run.tool_results(results)
    }

    /// The turn's tool sets paired with a dispatch target.
    ///
    /// The names always come from the run. The snapshot comes from this
    /// process: the one taken when the turn was prepared if the turn was
    /// prepared here, otherwise a fresh one — implementations are live
    /// objects, so a resumed process can only dispatch against its own
    /// registry.
    async fn dispatch_tools_for_turn(
        &mut self,
        calls: &[PendingToolCall],
    ) -> Result<TurnTools, PromptError> {
        let Some(names) = self.run.advertised_tools().cloned() else {
            return Err(PromptError::CompletionError(CompletionError::RequestError(
                "the run has no advertised tool set for the pending calls; drive the model turn \
                 through this driver so the turn's tools are recorded on the run"
                    .into(),
            )));
        };
        let output_tool_name = self.run.output_tool_name().map(str::to_owned);

        if let Some(snapshot) = &self.snapshot {
            return Ok(TurnTools::from_parts(
                snapshot.clone(),
                names,
                output_tool_name,
            ));
        }

        // Resumed run: rebuild the live half, then report deployment drift
        // rather than silently feeding not-found results to the model. The
        // model chose these tools from a registry this process no longer has,
        // and re-prompting cannot fix that.
        let snapshot = Arc::new(self.fresh_snapshot().await?);
        if !self.allow_missing_resumed_tools {
            let missing: Vec<&str> = calls
                .iter()
                .filter(|call| call.preresolved_result.is_none())
                .map(|call| call.tool_call.function.name.as_str())
                .filter(|name| {
                    output_tool_name.as_deref() != Some(*name)
                        && !snapshot
                            .definitions()
                            .iter()
                            .any(|tool| tool.name.as_str() == *name)
                })
                .collect();
            if !missing.is_empty() {
                return Err(PromptError::CompletionError(CompletionError::RequestError(
                    format!(
                        "resumed run has pending tool calls {missing:?} that are not registered \
                         in this process; register the tools on the agent before resuming, or \
                         call `allow_missing_resumed_tools()` to dispatch anyway and feed \
                         not-found results to the model"
                    )
                    .into(),
                )));
            }
        }

        self.snapshot = Some(snapshot.clone());
        Ok(TurnTools::from_parts(snapshot, names, output_tool_name))
    }

    /// Take a registry snapshot for a run resumed in this process.
    ///
    /// The retrieval query is re-derived from the run's history, matching what
    /// request preparation would have used.
    async fn fresh_snapshot(&self) -> Result<ToolRegistrySnapshot, PromptError> {
        let query = self
            .run
            .full_history()
            .iter()
            .rev()
            .find_map(|message| message.rag_text());
        self.agent
            .tool_server_handle
            .snapshot_tool_defs(query)
            .await
            .map_err(|_| {
                PromptError::CompletionError(CompletionError::RequestError(
                    "Failed to get tool definitions".into(),
                ))
            })
    }

    /// The patch actually applied to the next turn's request.
    ///
    /// The run's own `tool_choice` is the driver's baseline — a run built with
    /// [`AgentRun::with_tool_choice`] and handed to
    /// [`Agent::drive_run`](super::Agent::drive_run) is taken as-is, so its
    /// choice must reach the provider and not merely the run's internal
    /// decisions. An explicit [`Self::request_patch`] outranks it, exactly as
    /// a per-turn patch outranks the agent's baseline everywhere else.
    fn effective_request_patch(&self) -> RequestPatch {
        let mut patch = self.request_patch.clone();
        if patch.tool_choice.is_none() {
            patch.tool_choice = self.run.tool_choice().cloned();
        }
        patch
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::AgentBuilder;
    use crate::agent::run::OutputMode;
    use crate::completion::Message;
    use crate::test_utils::{MockAddTool, MockCompletionModel, MockSubtractTool, MockTurn};
    use crate::tool::{ToolContext, ToolErrorKind};
    use rig_core::message::ToolChoice;
    use serde_json::json;

    fn schema(value: serde_json::Value) -> schemars::Schema {
        serde_json::from_value(value).expect("valid schema")
    }

    fn value_schema() -> schemars::Schema {
        schema(json!({
            "type": "object",
            "properties": { "value": { "type": "integer" } },
            "required": ["value"]
        }))
    }

    /// Expect the next step to be `SendRequest`, panicking otherwise.
    macro_rules! expect_send {
        ($driver:expr) => {
            match $driver.next_step().await.expect("next_step succeeds") {
                DriveStep::SendRequest {
                    request,
                    tools,
                    turn,
                } => (request, tools, turn),
                other => panic!("expected SendRequest, got {other:?}"),
            }
        };
    }

    macro_rules! expect_execute_tools {
        ($driver:expr) => {
            match $driver.next_step().await.expect("next_step succeeds") {
                DriveStep::ExecuteTools { calls, tools } => (calls, tools),
                other => panic!("expected ExecuteTools, got {other:?}"),
            }
        };
    }

    /// Criterion: the full drive loop reuses the agent's configuration and the
    /// prepared turn keeps dispatching the implementation it advertised even
    /// after the live registry mutates.
    #[tokio::test]
    async fn drive_loop_dispatches_the_advertised_implementation() {
        let model = MockCompletionModel::new([
            MockTurn::tool_call("call_1", "add", json!({"x": 1, "y": 2})),
            MockTurn::text("3"),
        ]);
        let agent = AgentBuilder::new(model)
            .preamble("driver preamble")
            .default_max_turns(2)
            .tool(MockAddTool)
            .build();

        let mut driver = agent.drive("add 1 and 2");

        let (request, tools, turn) = expect_send!(driver);
        assert_eq!(turn, 1);
        assert!(tools.executable_tool_names().contains("add"));
        assert_eq!(tools.executable_tool_names(), tools.allowed_tool_names());
        let response = request.send().await.expect("scripted turn");
        driver.model_response(&response).expect("turn accepted");

        // Mutate the live registry AFTER the turn advertised its tools.
        agent.tool_server_handle.remove_tool("add").await;
        agent.tool_server_handle.add_tool(MockSubtractTool).await;

        let (calls, tools) = expect_execute_tools!(driver);
        assert_eq!(calls.len(), 1);
        let mut context = ToolContext::new();

        // The snapshot still dispatches the advertised implementation...
        let probe = tools
            .execute("add", r#"{"x": 1, "y": 2}"#, &mut context)
            .await;
        assert!(
            probe.is_success(),
            "snapshot dispatch must reach the advertised implementation"
        );
        // ...and does not see tools registered after it was taken.
        let probe = tools
            .execute("subtract", r#"{"x": 1, "y": 2}"#, &mut context)
            .await;
        assert!(probe.is_error_kind(ToolErrorKind::NotFound));

        let mut results = Vec::new();
        for call in &calls {
            results.push(tools.execute_call(call, &mut context).await);
        }
        driver.tool_results(results).expect("results accepted");

        let (request, _, turn) = expect_send!(driver);
        assert_eq!(turn, 2);
        let response = request.send().await.expect("scripted turn");
        driver.model_response(&response).expect("turn accepted");
        match driver.next_step().await.expect("next_step succeeds") {
            DriveStep::Done(response) => assert_eq!(response.output, "3"),
            other => panic!("expected Done, got {other:?}"),
        }
    }

    /// The prepared request carries the agent's configuration — the preamble
    /// leads the history and the registered tools are advertised.
    #[tokio::test]
    async fn prepared_request_carries_agent_configuration() {
        let agent = AgentBuilder::new(MockCompletionModel::text("unused"))
            .preamble("driver preamble")
            .tool(MockAddTool)
            .build();
        let mut driver = agent.drive("go");
        let (request, _, _) = expect_send!(driver);
        let request = request.build();
        assert!(matches!(
            request.chat_history.first(),
            Some(Message::System { content }) if content == "driver preamble"
        ));
        assert!(request.tools.iter().any(|tool| tool.name == "add"));
    }

    /// Criterion: under Tool output mode the synthetic output tool is allowed
    /// and advertised but never executable, and dispatching it is rejected
    /// with the machine-readable `NotExecutable` kind.
    #[tokio::test]
    async fn output_tool_is_allowed_but_never_executable() {
        let agent = AgentBuilder::new(MockCompletionModel::text("unused"))
            .tool(MockAddTool)
            .output_schema_raw(value_schema())
            .output_mode(OutputMode::Tool)
            .build();
        let mut driver = agent.drive("compute");
        let (request, tools, _) = expect_send!(driver);
        let output_tool = tools
            .output_tool_name()
            .expect("Tool output mode advertises an output tool")
            .to_owned();

        assert!(tools.allowed_tool_names().contains(&output_tool));
        assert!(!tools.executable_tool_names().contains(&output_tool));
        let request = request.build();
        assert!(request.tools.iter().any(|tool| tool.name == output_tool));

        let mut context = ToolContext::new();
        let result = tools.execute(&output_tool, "{}", &mut context).await;
        assert!(result.is_error_kind(ToolErrorKind::NotExecutable));
    }

    /// Criterion (finding 1): a Tool-output-mode agent driven by the
    /// documented pattern finalizes with its structured answer — the run's
    /// intercept is armed by the driver, so the output-tool call never
    /// surfaces as a pending tool.
    #[tokio::test]
    async fn tool_mode_run_finalizes_via_the_output_tool_intercept() {
        let model = MockCompletionModel::new([MockTurn::tool_call(
            "call_1",
            "final_result",
            json!({"value": 7}),
        )]);
        let agent = AgentBuilder::new(model)
            .tool(MockAddTool)
            .output_schema_raw(value_schema())
            .output_mode(OutputMode::Tool)
            .build();
        let mut driver = agent.drive("compute");
        let (request, _, _) = expect_send!(driver);
        let response = request.send().await.expect("scripted turn");
        driver.model_response(&response).expect("turn accepted");
        match driver.next_step().await.expect("next_step succeeds") {
            DriveStep::Done(response) => {
                assert!(
                    response.output.contains('7'),
                    "structured output should carry the answer: {}",
                    response.output
                );
            }
            other => panic!("the output-tool call must finalize the run, got {other:?}"),
        }
    }

    /// Criterion (finding 2): the run's committed output tool stays pinned
    /// across turns even when the tool set changes in between.
    #[tokio::test]
    async fn committed_output_tool_pins_across_turns() {
        let model = MockCompletionModel::new([
            MockTurn::text("not structured output"),
            MockTurn::tool_call("call_1", "final_result", json!({"value": 7})),
        ]);
        let agent = AgentBuilder::new(model)
            .default_max_turns(2)
            .tool(MockAddTool)
            .output_schema_raw(value_schema())
            .output_mode(OutputMode::Tool)
            .build();
        let mut driver = agent.drive("compute");
        let (request, tools, _) = expect_send!(driver);
        let committed = tools
            .output_tool_name()
            .expect("Tool mode commits a name on turn 1")
            .to_owned();
        let response = request.send().await.expect("scripted turn");
        driver.model_response(&response).expect("turn processed");

        // Change the tool set between turns: retire every executable tool.
        agent.tool_server_handle.remove_tool("add").await;

        // Tool-mode validation re-prompts; the committed name must survive.
        let (_, tools, turn) = expect_send!(driver);
        assert_eq!(turn, 2);
        assert_eq!(
            tools.output_tool_name(),
            Some(committed.as_str()),
            "the committed output tool must stay pinned when the tool set changes"
        );
        assert!(tools.allowed_tool_names().contains(&committed));
    }

    /// Criterion (finding 3): a run resumed from serialized state alone —
    /// fresh driver, fresh process semantics — re-emits its pending calls and
    /// dispatches them through a fresh snapshot.
    #[tokio::test]
    async fn resume_from_serialized_state_dispatches_via_a_fresh_snapshot() {
        let model = MockCompletionModel::new([
            MockTurn::tool_call("call_1", "add", json!({"x": 2, "y": 5})),
            MockTurn::text("7"),
        ]);
        let agent = AgentBuilder::new(model.clone())
            .default_max_turns(2)
            .tool(MockAddTool)
            .build();
        let mut driver = agent.drive("what is 2 + 5?");
        let (request, _, _) = expect_send!(driver);
        let response = request.send().await.expect("scripted turn");
        driver.model_response(&response).expect("turn accepted");
        let _ = expect_execute_tools!(driver);
        let serialized = serde_json::to_string(driver.run()).expect("run serializes");
        drop(driver);
        drop(agent);

        // "Fresh process": rebuild the same agent, deserialize the run.
        let agent = AgentBuilder::new(model)
            .default_max_turns(2)
            .tool(MockAddTool)
            .build();
        let run: AgentRun = serde_json::from_str(&serialized).expect("run deserializes");
        let mut driver = agent.drive_run(run);
        let (calls, tools) = expect_execute_tools!(driver);
        let mut context = ToolContext::new();
        let mut results = Vec::new();
        for call in &calls {
            results.push(tools.execute_call(call, &mut context).await);
        }
        driver.tool_results(results).expect("results accepted");
        let (request, _, _) = expect_send!(driver);
        let response = request.send().await.expect("scripted turn");
        driver.model_response(&response).expect("turn accepted");
        match driver.next_step().await.expect("next_step succeeds") {
            DriveStep::Done(response) => assert_eq!(response.output, "7"),
            other => panic!("expected Done, got {other:?}"),
        }
    }

    /// Criterion (finding 3, drift): a resumed pending call whose tool is
    /// missing from this process's registry is surfaced as an error before
    /// dispatch — and the opt-out downgrades it to a not-found tool result.
    #[tokio::test]
    async fn resumed_drift_is_loud_before_dispatch() {
        let model = MockCompletionModel::new([MockTurn::tool_call(
            "call_1",
            "add",
            json!({"x": 2, "y": 5}),
        )]);
        let agent = AgentBuilder::new(model)
            .default_max_turns(2)
            .tool(MockAddTool)
            .build();
        let mut driver = agent.drive("what is 2 + 5?");
        let (request, _, _) = expect_send!(driver);
        let response = request.send().await.expect("scripted turn");
        driver.model_response(&response).expect("turn accepted");
        let _ = expect_execute_tools!(driver);
        let serialized = serde_json::to_string(driver.run()).expect("run serializes");

        // Resume against an agent that no longer registers the pending tool.
        let bare_agent = AgentBuilder::new(MockCompletionModel::text("unused"))
            .default_max_turns(2)
            .build();
        let run: AgentRun = serde_json::from_str(&serialized).expect("run deserializes");
        let mut driver = bare_agent.drive_run(run);
        let err = driver
            .next_step()
            .await
            .expect_err("a missing pending tool must surface as drift");
        let message = err.to_string();
        assert!(message.contains("add"), "error names the tool: {message}");
        assert!(
            message.contains("allow_missing_resumed_tools"),
            "error names the opt-out: {message}"
        );

        // The opt-out dispatches anyway and yields a not-found result.
        let run: AgentRun = serde_json::from_str(&serialized).expect("run deserializes");
        let mut driver = bare_agent.drive_run(run).allow_missing_resumed_tools();
        let (calls, tools) = expect_execute_tools!(driver);
        let mut context = ToolContext::new();
        let result = tools
            .execute(
                &calls[0].tool_call.function.name,
                &calls[0].tool_call.function.arguments.to_string(),
                &mut context,
            )
            .await;
        assert!(result.is_error_kind(ToolErrorKind::NotFound));
    }

    /// Criterion (finding 11): the driven run inherits the agent's
    /// configuration instead of the driver restating it.
    #[tokio::test]
    async fn drive_seeds_the_run_from_agent_configuration() {
        let agent = AgentBuilder::new(MockCompletionModel::text("unused"))
            .default_max_turns(3)
            .tool_choice(ToolChoice::Auto)
            .output_schema_raw(value_schema())
            .tool(MockAddTool)
            .build();
        let driver = agent.drive("go");
        let state = serde_json::to_value(driver.run()).expect("run serializes");
        assert_eq!(state["max_turns"], 3, "seeded from default_max_turns");
        assert!(
            !state["tool_choice"].is_null(),
            "seeded from the agent's tool_choice"
        );
        assert!(
            state["output_schema"].is_object(),
            "output validation seeded from the agent's schema"
        );

        // No configured budget seeds the implicit budget of one, exactly as
        // `AgentRunner::from_agent` does.
        let unconfigured = AgentBuilder::new(MockCompletionModel::text("unused")).build();
        let state = serde_json::to_value(unconfigured.drive("go").run()).expect("run serializes");
        assert_eq!(state["max_turns"], 1);

        // Per-run overrides layer on top of the seeding, and the run state can
        // be reclaimed by value for suspension.
        let run = unconfigured
            .drive("go")
            .history(vec![Message::user("earlier context")])
            .max_turns(5)
            .into_run();
        let state = serde_json::to_value(&run).expect("run serializes");
        assert_eq!(state["max_turns"], 5);
        assert!(state["chat_history"].is_array());
    }

    /// Criterion: an impossible `ToolChoice` fails at prepare time, locally,
    /// with no provider round-trip.
    #[tokio::test]
    async fn impossible_tool_choice_fails_locally_at_prepare_time() {
        let model = MockCompletionModel::text("unused");
        let agent = AgentBuilder::new(model.clone())
            .tool_choice(ToolChoice::Specific {
                function_names: vec!["missing".to_string()],
            })
            .tool(MockAddTool)
            .build();
        let mut driver = agent.drive("go");
        let err = driver
            .next_step()
            .await
            .expect_err("a tool_choice naming an unadvertised tool must fail at prepare time");
        assert!(err.to_string().contains("missing"));
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
        let mut driver = agent.drive("go");
        let err = driver
            .next_step()
            .await
            .expect_err("Required with no advertised tool must fail at prepare time");
        assert!(err.to_string().contains("Required"));
        assert_eq!(model.request_count(), 0);
    }

    /// A preparation failure must cost nothing. Preparation runs before the
    /// run advances, so a turn that never reached the provider consumes no
    /// budget and leaves the run drivable — the caller fixes the cause (here,
    /// a tool the registry was missing; in practice a briefly unreachable tool
    /// server) and retries the same step.
    #[tokio::test]
    async fn failed_preparation_leaves_the_run_intact_and_retryable() {
        let model = MockCompletionModel::new([MockTurn::text("done")]);
        let agent = AgentBuilder::new(model.clone())
            .default_max_turns(1)
            .tool_choice(ToolChoice::Specific {
                function_names: vec!["add".to_string()],
            })
            .build();
        let mut driver = agent.drive("go");

        driver
            .next_step()
            .await
            .expect_err("a tool_choice naming an unregistered tool must fail at prepare time");
        assert_eq!(
            driver.run().turn(),
            0,
            "a request that never left the process must not consume a turn"
        );
        assert_eq!(model.request_count(), 0);

        // Fix the cause and drive the very same step again.
        agent.tool_server_handle.add_tool(MockAddTool).await;
        let (request, tools, turn) = expect_send!(driver);
        assert_eq!(turn, 1, "the retry takes the turn the failure did not");
        assert!(tools.executable_tool_names().contains("add"));
        let response = request.send().await.expect("scripted turn");
        driver.model_response(&response).expect("turn accepted");
    }

    /// The natural suspension point for a caller that owns the transport is
    /// *after* the request is sent and before the reply lands — a queued or
    /// long-running provider call. The turn's advertised tool names travel
    /// with the run, so the reply can be fed to a driver in another process.
    #[tokio::test]
    async fn run_suspended_awaiting_the_model_resumes_in_another_process() {
        let model = MockCompletionModel::new([
            MockTurn::tool_call("call_1", "add", json!({"x": 2, "y": 5})),
            MockTurn::text("7"),
        ]);
        let agent = AgentBuilder::new(model.clone())
            .default_max_turns(2)
            .tool(MockAddTool)
            .build();

        let mut driver = agent.drive("what is 2 + 5?");
        let (request, _, _) = expect_send!(driver);
        // Suspend with the model call in flight.
        let serialized = serde_json::to_string(driver.run()).expect("run serializes");
        let response = request.send().await.expect("scripted turn");
        drop(driver);
        drop(agent);

        // "Fresh process": rebuild the agent, deserialize, feed the reply.
        let agent = AgentBuilder::new(model)
            .default_max_turns(2)
            .tool(MockAddTool)
            .build();
        let run: AgentRun = serde_json::from_str(&serialized).expect("run deserializes");
        let mut driver = agent.drive_run(run);
        driver
            .model_response(&response)
            .expect("a run resumed mid-model-call accepts its reply");

        let (calls, tools) = expect_execute_tools!(driver);
        let mut context = ToolContext::new();
        let mut results = Vec::new();
        for call in &calls {
            results.push(tools.execute_call(call, &mut context).await);
        }
        driver.tool_results(results).expect("results accepted");
        let (request, _, _) = expect_send!(driver);
        let response = request.send().await.expect("scripted turn");
        driver.model_response(&response).expect("turn accepted");
        match driver.next_step().await.expect("next_step succeeds") {
            DriveStep::Done(response) => assert_eq!(response.output, "7"),
            other => panic!("expected Done, got {other:?}"),
        }
    }

    /// A custom run is taken as-is, so its own `tool_choice` must reach the
    /// provider — not merely the run's internal decisions. Forbidding tools on
    /// the run and having the model call one anyway is not a recoverable
    /// situation; the request has to carry the constraint.
    #[tokio::test]
    async fn a_custom_runs_tool_choice_reaches_the_request() {
        let agent = AgentBuilder::new(MockCompletionModel::text("unused"))
            .tool(MockAddTool)
            .build();
        let mut driver = agent.drive_run(AgentRun::new("go").with_tool_choice(ToolChoice::None));
        let (request, tools, _) = expect_send!(driver);
        assert_eq!(request.build().tool_choice, Some(ToolChoice::None));
        assert!(
            tools.allowed_tool_names().is_empty(),
            "ToolChoice::None allows nothing to be called"
        );
    }

    /// An explicit patch outranks the run's own choice, and reaches every
    /// other per-turn field the runner's `CompletionCall` hooks reach.
    #[tokio::test]
    async fn a_request_patch_overrides_the_baseline_for_the_turn() {
        let agent = AgentBuilder::new(MockCompletionModel::text("unused"))
            .preamble("baseline preamble")
            .tool(MockAddTool)
            .tool(MockSubtractTool)
            .build();
        let mut driver = agent
            .drive_run(AgentRun::new("go").with_tool_choice(ToolChoice::None))
            .request_patch(
                RequestPatch::new()
                    .preamble("patched preamble")
                    .tool_choice(ToolChoice::Required)
                    .active_tools(["add"]),
            );

        let (request, tools, _) = expect_send!(driver);
        assert!(
            tools.executable_tool_names().contains("add")
                && !tools.executable_tool_names().contains("subtract"),
            "active_tools narrows the advertised set: {:?}",
            tools.executable_tool_names()
        );
        let request = request.build();
        assert_eq!(request.tool_choice, Some(ToolChoice::Required));
        assert!(matches!(
            request.chat_history.first(),
            Some(Message::System { content }) if content == "patched preamble"
        ));
    }

    /// Resuming is about dispatch, not about validating a request that will
    /// never be built. A `tool_choice` the resuming process could not satisfy
    /// must not pre-empt the drift report, and must not defeat the opt-out
    /// that exists precisely for this situation.
    #[tokio::test]
    async fn resumed_dispatch_does_not_validate_an_unbuilt_requests_tool_choice() {
        let model = MockCompletionModel::new([MockTurn::tool_call(
            "call_1",
            "add",
            json!({"x": 2, "y": 5}),
        )]);
        let agent = AgentBuilder::new(model)
            .default_max_turns(2)
            .tool_choice(ToolChoice::Required)
            .tool(MockAddTool)
            .build();
        let mut driver = agent.drive("what is 2 + 5?");
        let (request, _, _) = expect_send!(driver);
        let response = request.send().await.expect("scripted turn");
        driver.model_response(&response).expect("turn accepted");
        let _ = expect_execute_tools!(driver);
        let serialized = serde_json::to_string(driver.run()).expect("run serializes");

        // Resume against a process whose registry lost the tool. `Required`
        // is unsatisfiable there, but no request is being built.
        let bare_agent = AgentBuilder::new(MockCompletionModel::text("unused"))
            .default_max_turns(2)
            .tool_choice(ToolChoice::Required)
            .build();

        let run: AgentRun = serde_json::from_str(&serialized).expect("run deserializes");
        let err = bare_agent
            .drive_run(run)
            .next_step()
            .await
            .expect_err("the missing pending tool must surface");
        let message = err.to_string();
        assert!(
            message.contains("add") && message.contains("allow_missing_resumed_tools"),
            "drift must be reported, not the unsatisfiable tool choice: {message}"
        );

        let run: AgentRun = serde_json::from_str(&serialized).expect("run deserializes");
        let mut driver = bare_agent.drive_run(run).allow_missing_resumed_tools();
        let (calls, tools) = expect_execute_tools!(driver);
        let mut context = ToolContext::new();
        let result = tools.execute_call(&calls[0], &mut context).await;
        assert!(
            matches!(result, UserContent::ToolResult(_)),
            "the opt-out must dispatch and produce a tool result"
        );
    }
}
