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
//! # Durability
//!
//! The serializable state is still [`AgentRun`] — serialize
//! [`AgentDriver::run`] while tool calls are pending, and resume in another
//! process with [`Agent::drive_run`]. Tool implementations are live objects
//! and cannot be serialized: the resuming process rebuilds the same `Agent`
//! and the driver takes a **fresh** registry snapshot for the pending calls.
//! If that snapshot no longer contains a pending call's tool, the driver
//! surfaces an error instead of silently feeding a not-found result to the
//! model (see [`AgentDriver::allow_missing_resumed_tools`]) — the model chose
//! that tool from a registry this process no longer has, and re-prompting
//! cannot fix deployment drift. If you suspend runs across deploys, version
//! your agent definitions alongside the serialized run.

use std::collections::BTreeSet;
use std::sync::Arc;

use rig_core::message::UserContent;

use super::completion::{Agent, allowed_tool_names_for_choice, build_prepared_completion_request};
use super::model::ModelHandle;
use super::run::{AgentRun, AgentRunStep, ModelTurnOutcome, PendingToolCall};
use super::runner::build_agent_run;
use super::turn_tools::{PreparedCompletionRequest, TurnTools};
use crate::agent::hook::InvalidToolCallAction;
use crate::agent::prompt_request::PromptResponse;
use crate::completion::{
    CompletionError, CompletionRequestBuilder, CompletionResponse, Message, PromptError,
};

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
            turn_tools: None,
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
        /// `tool_choice`, and this turn's tool definitions. It reflects the
        /// agent's **baseline** configuration — no `CompletionCall` hooks run
        /// on this path, so there is no per-turn request patch, model
        /// selection, or `active_tools` narrowing. The request still honors
        /// the agent's `record_telemetry_content` for provider-level spans;
        /// call `.record_content_telemetry(false)` on it to opt a hand-driven
        /// turn out.
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
    /// Tool state of the most recently prepared turn. `None` until the first
    /// `SendRequest` — or in a process that resumed a serialized run, where
    /// [`Self::resume_tools`] derives a fresh snapshot on demand.
    turn_tools: Option<TurnTools>,
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
    pub async fn next_step(&mut self) -> Result<DriveStep, PromptError> {
        match self.run.next_step()? {
            AgentRunStep::CallModel {
                prompt,
                history,
                turn,
            } => {
                // Pin Tool output mode once committed (#1928), mirroring the
                // runner: read the run's committed name into preparation, and
                // store the resolved name back (fill-once).
                let committed = self.run.output_tool_name().map(str::to_owned);
                let prepared = build_prepared_completion_request(
                    &self.agent.model,
                    prompt,
                    &history,
                    self.agent.preamble.as_deref(),
                    &self.agent.static_context,
                    self.agent.temperature,
                    self.agent.max_tokens,
                    self.agent.additional_params.as_ref(),
                    self.agent.record_telemetry_content,
                    self.agent.tool_choice.as_ref(),
                    &self.agent.tool_server_handle,
                    self.agent.output_schema.as_ref(),
                    &self.agent.output_mode,
                    committed.as_deref(),
                    None,
                    true,
                    None,
                )
                .await
                .map_err(PromptError::CompletionError)?;
                self.run
                    .set_output_tool_name(prepared.tools.output_tool_name.clone());
                let PreparedCompletionRequest { builder, tools } = prepared;
                self.turn_tools = Some(tools.clone());
                Ok(DriveStep::SendRequest {
                    request: Box::new(builder),
                    tools,
                    turn,
                })
            }
            AgentRunStep::CallTools { calls } => {
                let tools = match &self.turn_tools {
                    Some(tools) => tools.clone(),
                    // A process resuming a serialized run wakes here with no
                    // prepared turn: derive a fresh dispatch snapshot.
                    None => {
                        let tools = self.resume_tools().await?;
                        if !self.allow_missing_resumed_tools {
                            let missing: Vec<&str> = calls
                                .iter()
                                .filter(|call| call.preresolved_result.is_none())
                                .map(|call| call.tool_call.function.name.as_str())
                                .filter(|name| {
                                    !tools.executable_tool_names.contains(*name)
                                        && tools.output_tool_name() != Some(*name)
                                })
                                .collect();
                            if !missing.is_empty() {
                                return Err(PromptError::CompletionError(
                                    CompletionError::RequestError(
                                        format!(
                                            "resumed run has pending tool calls {missing:?} that \
                                             are not registered in this process; register the \
                                             tools on the agent before resuming, or call \
                                             `allow_missing_resumed_tools()` to dispatch anyway \
                                             and feed not-found results to the model"
                                        )
                                        .into(),
                                    ),
                                ));
                            }
                        }
                        self.turn_tools = Some(tools.clone());
                        tools
                    }
                };
                Ok(DriveStep::ExecuteTools { calls, tools })
            }
            AgentRunStep::Done(response) => Ok(DriveStep::Done(response)),
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
        let Some(tools) = &self.turn_tools else {
            return Err(PromptError::CompletionError(CompletionError::RequestError(
                "model_response must follow a SendRequest step from this driver".into(),
            )));
        };
        self.run.model_response(tools.model_turn(response))
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

    /// Derive a fresh dispatch target for a resumed run's pending tool calls.
    ///
    /// Necessarily a **new** snapshot: implementations are live objects, so a
    /// fresh process dispatches against its own registry state. The retrieval
    /// query is re-derived from the run's history (matching preparation), and
    /// the run's committed output tool — which is serialized with the run —
    /// stays non-executable.
    async fn resume_tools(&self) -> Result<TurnTools, PromptError> {
        let query = self
            .run
            .full_history()
            .iter()
            .rev()
            .find_map(|message| message.rag_text());
        let snapshot = self
            .agent
            .tool_server_handle
            .snapshot_tool_defs(query)
            .await
            .map_err(|_| {
                PromptError::CompletionError(CompletionError::RequestError(
                    "Failed to get tool definitions".into(),
                ))
            })?;
        let executable: BTreeSet<String> = snapshot
            .definitions()
            .iter()
            .map(|tool| tool.name.clone())
            .collect();
        let output_tool_name = self.run.output_tool_name().map(str::to_owned);
        let mut allowed = allowed_tool_names_for_choice(
            &executable,
            self.agent.tool_choice.as_ref(),
            output_tool_name.as_deref(),
            None,
        )
        .map_err(PromptError::CompletionError)?;
        if let Some(name) = &output_tool_name {
            allowed.insert(name.clone());
        }
        Ok(TurnTools {
            snapshot: Arc::new(snapshot),
            executable_tool_names: Arc::new(executable),
            allowed_tool_names: Arc::new(allowed),
            output_tool_name,
        })
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
}
