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
//! [`AgentRun`] — including what each committed turn resolved to — and the
//! driver's only other field is a cache of the live registry snapshot, which
//! cannot be serialized in any design. Per-turn configuration is an input to
//! [`AgentDriver::next_step_with`], never a field: policy for a turn that has
//! not happened yet is not run state, and a driver that stored it would resume
//! a serialized run with configuration the suspending process never recorded.
//! That is what makes the durability guarantees below hold at every step
//! rather than at one of them.
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
//! A turn can fail in two places, and both are recoverable:
//!
//! - **Preparing the request.** Nothing advances until a request exists, so an
//!   unreachable tool server or an impossible `tool_choice` costs no turn from
//!   the budget and leaves the run byte-identical. Call [`AgentDriver::next_step`]
//!   again once the cause is fixed.
//! - **Sending it.** The caller owns the send, so the caller is the only party
//!   that learns it failed. [`AgentDriver::rollback_model_call`] hands the turn
//!   back — refunding it and returning the run to preparing — and the next
//!   `next_step` yields a freshly prepared request. Deciding to use it takes
//!   two answers, not one:
//!   [`CompletionError::is_retryable`](crate::completion::CompletionError::is_retryable)
//!   says whether a retry *could succeed*, and only the caller can say whether
//!   one is *safe* — a request that reached the provider and lost only its
//!   reply will be billed twice. Bound the attempts yourself; the driver runs
//!   no IO and owns no clock.
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

use std::collections::BTreeSet;
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
use rig_core::wasm_compat::WasmBoxedFuture;

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
            allow_missing_resumed_tools: false,
        }
    }
}

/// What the caller must do next to advance an [`AgentDriver`].
///
/// Deliberately exhaustive, like [`AgentRunStep`](super::run::AgentRunStep): a
/// driver loop must handle every step, so adding a variant is a breaking change
/// by design.
pub enum DriveStep {
    /// Send this completion request to the model — via
    /// [`send`](CompletionRequestBuilder::send), or
    /// [`build`](CompletionRequestBuilder::build) it for a custom transport —
    /// then feed the response back through [`AgentDriver::model_response`].
    SendRequest {
        /// The fully configured request: the agent's preamble (with any
        /// output-mode augmentation), static context, model parameters,
        /// `tool_choice`, and this turn's tool definitions — with the
        /// [`RequestPatch`] from [`AgentDriver::next_step_with`]'s callback
        /// applied over that baseline, which is how a hand-driven turn gets
        /// the per-turn preamble, `tool_choice`, or `active_tools` narrowing
        /// the runner gets from its `CompletionCall` hooks. No hooks run on
        /// this path; the patch is the seam. The request still honors the
        /// agent's
        /// `record_telemetry_content` for provider-level spans; call
        /// `.record_content_telemetry(false)` on it to opt a hand-driven turn
        /// out.
        request: Box<CompletionRequestBuilder<ModelHandle>>,
        /// The turn's advertised tool sets.
        ///
        /// For a **blocking** send this is informational — the driver builds
        /// the model turn itself when you hand the response to
        /// [`AgentDriver::model_response`] — and the same value arrives on the
        /// following [`ExecuteTools`](Self::ExecuteTools) step for dispatch.
        ///
        /// For a **streamed** send it is required, and this is the only place
        /// a driven turn can get it:
        /// [`StreamedTurnAssembler::new`](super::run::StreamedTurnAssembler::new)
        /// takes the turn's executable and allowed name sets, so build the
        /// assembler from
        /// [`executable_tool_names`](TurnTools::executable_tool_names) and
        /// [`allowed_tool_names`](TurnTools::allowed_tool_names) here, then
        /// feed the assembled turn through
        /// [`AgentDriver::run_mut`]. Destructuring this step as
        /// `SendRequest { request, .. }` is fine for a blocking loop and will
        /// leave a streaming one with no way to validate the model's calls.
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

/// What a caller may decide about the turn that is about to be prepared.
///
/// Handed to the callback of [`AgentDriver::next_step_with`] before anything
/// advances, so a decision that fails costs no turn.
#[non_exhaustive]
pub struct TurnPreparationContext<'a> {
    /// The prompt this turn will send.
    pub prompt: &'a Message,
    /// The history preceding it.
    pub history: &'a [Message],
    /// One-based index this call *would* take, once committed.
    pub turn: usize,
}

/// A caller's decisions for one turn.
///
/// Per-turn configuration is an **input to preparation**, never state the
/// driver holds. Future policy is not run state: storing it would make a
/// resumed run silently prepare its next request differently from the process
/// that suspended it, and there would be nothing on the run to say so.
#[derive(Debug, Default)]
#[non_exhaustive]
pub struct TurnPreparation {
    /// Per-turn overrides layered over the agent's baseline. Each set field
    /// replaces the configured value for this turn; unset fields inherit it.
    pub patch: RequestPatch,
    /// The model to use for this turn. `None` uses the agent's.
    pub model: Option<ModelHandle>,
}

impl TurnPreparation {
    /// Prepare this turn with `patch` layered over the agent's baseline.
    pub fn with_patch(patch: RequestPatch) -> Self {
        Self { patch, model: None }
    }

    /// Use `model` for this turn instead of the agent's.
    pub fn using_model(mut self, model: ModelHandle) -> Self {
        self.model = Some(model);
        self
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

    /// Set the retry budget for [`InvalidToolCallAction::Retry`] resolutions,
    /// mirroring [`AgentRunner::max_invalid_tool_call_retries`](super::AgentRunner::max_invalid_tool_call_retries).
    ///
    /// [`Agent::drive`] seeds this at **zero**, so answering a
    /// [`ModelTurnOutcome::NeedsResolution`] with `Retry` fails until you raise
    /// it — the run has no budget to spend and reports the invalid call
    /// instead. Invalid tool-call retries also consume the total model-call
    /// budget, so raise [`Self::max_turns`] alongside it.
    pub fn max_invalid_tool_call_retries(mut self, retries: usize) -> Self {
        self.run = self.run.max_invalid_tool_call_retries(retries);
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

    /// Mutable access to the run, for the entry points the driver does not
    /// wrap.
    ///
    /// A streamed turn is fed through [`AgentRun::record_streamed_completion_call`],
    /// [`AgentRun::resolve_streamed_invalid_tool_call`] and
    /// [`AgentRun::streamed_turn`], all of which need `&mut AgentRun`. Driving
    /// a custom streaming transport is a headline use for this type, so the
    /// access has to exist; without it a streaming caller would have to
    /// [`Self::into_run`], drive the turn by hand, and rebuild the driver —
    /// which discards the per-turn snapshot cache and makes the driver treat a
    /// turn prepared in *this* process as a resume, drift check and all.
    ///
    /// # Do not commit or roll back a model call through this
    ///
    /// Use [`Self::next_step`] and [`Self::rollback_model_call`] for those.
    /// They keep the driver's cached dispatch target in step with the turn the
    /// run is on; committing a turn behind the driver's back would leave the
    /// previous turn's snapshot cached and dispatch this turn's calls through
    /// it. Feeding a *response* — streamed or otherwise — is safe, because it
    /// belongs to the turn the cache already holds.
    pub fn run_mut(&mut self) -> &mut AgentRun {
        &mut self.run
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
        self.next_step_with(|_| Box::pin(async { Ok(TurnPreparation::default()) }))
            .await
    }

    /// Advance to the next step, deciding this turn's configuration first.
    ///
    /// The callback runs **before anything advances**, is handed the prompt,
    /// history and prospective turn index, and returns the turn's
    /// [`RequestPatch`] and optionally a model. It is the hand-driven
    /// equivalent of the runner's `CompletionCall` and model-selection hooks,
    /// and it is where a caller does per-turn work that can fail: a callback
    /// that returns `Err` costs no turn, exactly like a preparation failure,
    /// because the commit is still ahead of it.
    ///
    /// Per-turn configuration is an input, never driver state. A driver that
    /// stored it would resume a serialized run with configuration the
    /// suspending process never recorded, and nothing on the run would say so.
    ///
    /// ```rust,ignore
    /// let step = driver
    ///     .next_step_with(|ctx| {
    ///         Box::pin(async move {
    ///             Ok(TurnPreparation::with_patch(
    ///                 RequestPatch::new().active_tools(tools_for(ctx.turn)),
    ///             ))
    ///         })
    ///     })
    ///     .await?;
    /// ```
    pub async fn next_step_with<F>(&mut self, prepare: F) -> Result<DriveStep, PromptError>
    where
        F: for<'a> FnOnce(
            TurnPreparationContext<'a>,
        ) -> WasmBoxedFuture<'a, Result<TurnPreparation, PromptError>>,
    {
        match self.run.advance()? {
            Advance::NeedsModelCall => {
                // Peek, decide, prepare, *then* commit. Reading the inputs
                // consumes nothing, so everything fallible below — including
                // the caller's own callback — happens while the run is still
                // fully intact.
                let ModelCallInputs { prompt, history } = self.run.peek_model_call()?;
                let turn = self.run.turn() + 1;
                let preparation = prepare(TurnPreparationContext {
                    prompt: &prompt,
                    history: &history,
                    turn,
                })
                .await?;

                // The run's own choice is the baseline for a hand-driven turn:
                // a custom run handed to `drive_run` is taken as-is, so its
                // choice must reach the provider. An explicit patch outranks
                // it, exactly as a per-turn patch outranks the agent's.
                let mut patch = preparation.patch;
                if patch.tool_choice.is_none() {
                    patch.tool_choice = self.run.tool_choice().cloned();
                }

                // Pin Tool output mode once committed (#1928), mirroring the
                // runner: read the run's committed name into preparation, and
                // store the resolved name back (fill-once).
                let committed = self.run.output_tool_name().map(str::to_owned);
                let mut baseline = TurnBaseline::from_agent(&self.agent);
                if let Some(model) = preparation.model.as_ref() {
                    baseline.model = model;
                }
                let prepared = build_prepared_completion_request(
                    baseline,
                    TurnRequest {
                        prompt,
                        chat_history: &history,
                        committed_output_tool: committed.as_deref(),
                        patch: Some(&patch),
                    },
                )
                .await
                .map_err(PromptError::CompletionError)?;

                let metadata = prepared.turn_metadata();
                let PreparedCompletionRequest { builder, tools, .. } = prepared;
                self.snapshot = Some(tools.snapshot.clone());
                let turn = self.run.commit_model_call(Some(metadata))?;
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

    /// Hand back a model call that never produced a response, so the turn can
    /// be prepared and sent again.
    ///
    /// [`DriveStep::SendRequest`] gives the caller a request and the caller
    /// owns the send, so the caller is also the only party that learns the
    /// send failed. This is how that news gets back into the run: the turn is
    /// refunded and the run returns to preparing, so the next
    /// [`Self::next_step`] yields a **freshly prepared** `SendRequest` — new
    /// registry snapshot, new patch — rather than a replay of a request whose
    /// tool snapshot has since gone stale.
    ///
    /// Two questions decide whether to roll back, and the library answers only
    /// the first: *could a retry succeed?* — which
    /// [`CompletionError::is_retryable`](crate::completion::CompletionError::is_retryable)
    /// classifies — and *is a retry safe?*, which nothing here can know. A
    /// stream that died after the request was written is retryable and not
    /// replay-safe: rolling back on the first question alone bills a second
    /// completion and repeats whatever the model already caused. Only the
    /// caller can establish the second, through provider-side idempotency, its
    /// own record of what was transmitted, or a transport that fails before
    /// the write.
    ///
    /// ```rust,ignore
    /// if let DriveStep::SendRequest { request, .. } = driver.next_step().await? {
    ///     match request.send().await {
    ///         Ok(response) => { driver.model_response(&response)?; }
    ///         // `nothing_was_sent` is the caller's own knowledge; the driver
    ///         // cannot supply it, and `is_retryable` does not answer it.
    ///         Err(err) if err.is_retryable() && nothing_was_sent => {
    ///             driver.rollback_model_call()?
    ///         }
    ///         Err(err) => return Err(err.into()),
    ///     }
    /// }
    /// ```
    ///
    /// See [`AgentRun::rollback_model_call`] for the full semantics. Bounding
    /// attempts is yours to do; the driver runs no IO and owns no clock.
    pub fn rollback_model_call(&mut self) -> Result<(), PromptError> {
        self.run.rollback_model_call()?;
        // Drop the cached dispatch target too: the retry is a new turn and
        // must advertise, and dispatch through, a snapshot taken for it.
        self.snapshot = None;
        Ok(())
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
        let mut snapshot = self.fresh_snapshot(&names.executable).await?;
        // Narrow to what the turn advertised, mirroring what preparation does
        // in-process. Without this the resumed snapshot is the whole current
        // registry, and `TurnTools`' two halves — advertised names and
        // dispatch target — could disagree, which is the skew the type exists
        // to prevent.
        snapshot.retain_names(&names.executable);
        let snapshot = Arc::new(snapshot);

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
                        "resumed run has pending tool calls {missing:?} that are no longer \
                         registered in this process; register the tools on the agent before \
                         resuming, or call `allow_missing_resumed_tools()` to dispatch anyway \
                         and feed not-found results to the model"
                    )
                    .into(),
                )));
            }
        }

        self.snapshot = Some(snapshot.clone());
        Ok(TurnTools::from_parts(snapshot, names, output_tool_name))
    }

    /// Take a registry snapshot for a run resumed in this process, containing
    /// exactly the names the turn advertised and still has.
    ///
    /// **No retrieval query is run**, deliberately. Retrieval selects dynamic
    /// tools by similarity, and this snapshot is narrowed to `required`
    /// immediately afterwards, so every retrieved name that is not in
    /// `required` is discarded and every name in `required` is resolved from
    /// the registry by name regardless of ranking. A query would therefore
    /// contribute nothing to the result while costing a vector search — and,
    /// worse, would make resuming a run *fail* when the index is unavailable,
    /// even if every pending call is a static tool.
    ///
    /// Resolving `required` by name is also what makes the caller's drift check
    /// meaningful: absence from this snapshot means the tool is gone, not
    /// merely that a query did not rank it.
    async fn fresh_snapshot(
        &self,
        required: &BTreeSet<String>,
    ) -> Result<ToolRegistrySnapshot, PromptError> {
        self.agent
            .tool_server_handle
            .snapshot_tool_defs_including(None, required)
            .await
            .map_err(|_| {
                PromptError::CompletionError(CompletionError::RequestError(
                    "Failed to get tool definitions".into(),
                ))
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

    /// Assert a model turn was accepted outright.
    ///
    /// `ModelTurnOutcome` is `#[must_use]`: `NeedsResolution` has to be
    /// answered before the run may advance, so these tests name an unexpected
    /// one rather than dropping it and hitting a protocol violation later.
    fn expect_continue(outcome: ModelTurnOutcome) {
        match outcome {
            ModelTurnOutcome::Continue { .. } => {}
            other => panic!("expected the turn to be accepted, got {other:?}"),
        }
    }

    /// Expect the next step to be `SendRequest` with a per-turn preparation.
    macro_rules! expect_send_with {
        ($driver:expr, $patch:expr) => {
            match $driver
                .next_step_with(|_| Box::pin(async { Ok(TurnPreparation::with_patch($patch)) }))
                .await
                .expect("next_step_with succeeds")
            {
                DriveStep::SendRequest {
                    request,
                    tools,
                    turn,
                } => (request, tools, turn),
                other => panic!("expected SendRequest, got {other:?}"),
            }
        };
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
        expect_continue(driver.model_response(&response).expect("turn accepted"));

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
        expect_continue(driver.model_response(&response).expect("turn accepted"));
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
        expect_continue(driver.model_response(&response).expect("turn accepted"));
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
        expect_continue(driver.model_response(&response).expect("turn processed"));

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
        expect_continue(driver.model_response(&response).expect("turn accepted"));
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
        expect_continue(driver.model_response(&response).expect("turn accepted"));
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
        expect_continue(driver.model_response(&response).expect("turn accepted"));
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
        expect_continue(driver.model_response(&response).expect("turn accepted"));
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
        expect_continue(
            driver
                .model_response(&response)
                .expect("a run resumed mid-model-call accepts its reply"),
        );

        let (calls, tools) = expect_execute_tools!(driver);
        let mut context = ToolContext::new();
        let mut results = Vec::new();
        for call in &calls {
            results.push(tools.execute_call(call, &mut context).await);
        }
        driver.tool_results(results).expect("results accepted");
        let (request, _, _) = expect_send!(driver);
        let response = request.send().await.expect("scripted turn");
        expect_continue(driver.model_response(&response).expect("turn accepted"));
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
        let mut driver = agent.drive_run(AgentRun::new("go").with_tool_choice(ToolChoice::None));

        let (request, tools, _) = expect_send_with!(
            driver,
            RequestPatch::new()
                .preamble("patched preamble")
                .tool_choice(ToolChoice::Required)
                .active_tools(["add"])
        );
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

    /// The caller owns the send, so the caller is the only party that learns
    /// it failed. Handing the turn back refunds it and returns the run to
    /// preparing, so the run survives a failure the driver never sees.
    #[tokio::test]
    async fn failed_send_rolls_back_and_re_prepares() {
        let model = MockCompletionModel::new([MockTurn::text("done")]);
        let agent = AgentBuilder::new(model.clone())
            .default_max_turns(1)
            .tool(MockAddTool)
            .build();
        let mut driver = agent.drive("go");

        let (request, _, turn) = expect_send!(driver);
        assert_eq!(turn, 1);
        // The send fails: drop the request without sending it. The provider
        // never saw anything, so nothing was produced.
        drop(request);
        driver
            .rollback_model_call()
            .expect("a call that produced nothing can be handed back");
        assert_eq!(driver.run().turn(), 0);
        assert_eq!(driver.run().model_call_rollbacks(), 1);
        assert_eq!(model.request_count(), 0);

        // The budget of one is intact, so the retry can happen at all.
        let (request, _, turn) = expect_send!(driver);
        assert_eq!(turn, 1, "the retry takes the turn the failure did not");
        let response = request.send().await.expect("scripted turn");
        expect_continue(driver.model_response(&response).expect("turn accepted"));
        match driver.next_step().await.expect("next_step succeeds") {
            DriveStep::Done(response) => assert_eq!(response.output, "done"),
            other => panic!("expected Done, got {other:?}"),
        }
    }

    /// The retry re-derives the request rather than replaying one. Replaying
    /// would pin the failed attempt's tool snapshot, advertising one set of
    /// implementations while a later turn dispatches another.
    #[tokio::test]
    async fn rollback_re_derives_against_the_current_registry() {
        let agent = AgentBuilder::new(MockCompletionModel::text("unused"))
            .default_max_turns(2)
            .tool(MockAddTool)
            .build();
        let mut driver = agent.drive("go");

        let (_, tools, _) = expect_send!(driver);
        assert!(!tools.executable_tool_names().contains("subtract"));
        driver.rollback_model_call().expect("rollback succeeds");

        agent.tool_server_handle.add_tool(MockSubtractTool).await;
        let (_, tools, _) = expect_send!(driver);
        assert!(
            tools.executable_tool_names().contains("subtract"),
            "the retry must advertise the registry as it is now: {:?}",
            tools.executable_tool_names()
        );
    }

    /// A run suspended mid-send is resumable, and the rollback travels with
    /// it: the process that discovers the reply is lost need not be the one
    /// that sent the request.
    #[tokio::test]
    async fn rollback_after_resume_recovers_a_lost_reply() {
        let model = MockCompletionModel::new([MockTurn::text("done")]);
        let agent = AgentBuilder::new(model.clone())
            .default_max_turns(1)
            .tool(MockAddTool)
            .build();
        let mut driver = agent.drive("go");
        let (request, _, _) = expect_send!(driver);
        let serialized = serde_json::to_string(driver.run()).expect("run serializes");
        drop(request);
        drop(driver);

        // "Fresh process": the reply never arrived, so hand the turn back.
        let run: AgentRun = serde_json::from_str(&serialized).expect("run deserializes");
        let mut driver = agent.drive_run(run);
        driver.rollback_model_call().expect("rollback succeeds");
        assert_eq!(driver.run().turn(), 0);

        let (request, _, turn) = expect_send!(driver);
        assert_eq!(turn, 1);
        let response = request.send().await.expect("scripted turn");
        expect_continue(driver.model_response(&response).expect("turn accepted"));
    }

    /// `TurnTools` promises that a name the turn did not advertise cannot
    /// reach the live registry. In-process the snapshot enforces that; on a
    /// resumed turn only the advertised names can, since the snapshot is
    /// rebuilt here.
    #[tokio::test]
    async fn unadvertised_name_is_not_found_even_on_a_resumed_turn() {
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
        expect_continue(driver.model_response(&response).expect("turn accepted"));
        let _ = expect_execute_tools!(driver);
        let serialized = serde_json::to_string(driver.run()).expect("run serializes");

        // Resume against a process that has since registered another tool.
        let resumed_agent = AgentBuilder::new(MockCompletionModel::text("unused"))
            .default_max_turns(2)
            .tool(MockAddTool)
            .tool(MockSubtractTool)
            .build();
        let run: AgentRun = serde_json::from_str(&serialized).expect("run deserializes");
        let mut driver = resumed_agent.drive_run(run);
        let (_, tools) = expect_execute_tools!(driver);

        assert!(
            !tools.executable_tool_names().contains("subtract"),
            "the advertised set is the turn's, not this process's"
        );
        let mut context = ToolContext::new();
        let result = tools
            .execute("subtract", r#"{"x": 1, "y": 2}"#, &mut context)
            .await;
        assert!(
            result.is_error_kind(ToolErrorKind::NotFound),
            "a tool registered after the turn must not be reachable"
        );
        // The advertised tool still dispatches.
        let result = tools
            .execute("add", r#"{"x": 2, "y": 5}"#, &mut context)
            .await;
        assert!(result.is_success());
    }

    /// `Retry` needs a budget, and `drive()` seeds zero — so the driver must
    /// expose the setter the runner has, or the documented resolution path is
    /// unreachable for every driver built the documented way.
    #[tokio::test]
    async fn a_retry_resolution_needs_a_budget_the_driver_can_set() {
        let model = MockCompletionModel::new([
            MockTurn::tool_call("call_1", "nonexistent", json!({})),
            MockTurn::text("ok"),
        ]);
        let agent = AgentBuilder::new(model)
            .default_max_turns(3)
            .tool(MockAddTool)
            .build();

        // Seeded at zero: the retry has nothing to spend.
        let mut driver = agent.drive("go");
        let (request, _, _) = expect_send!(driver);
        let response = request.send().await.expect("scripted turn");
        let outcome = driver
            .model_response(&response)
            .expect("the invalid call needs resolution");
        assert!(matches!(outcome, ModelTurnOutcome::NeedsResolution(_)));
        driver
            .resolve_invalid_tool_call(InvalidToolCallAction::Retry {
                feedback: "use a registered tool".to_string(),
            })
            .expect_err("a retry budget of zero cannot retry");
    }

    /// With a budget, the same resolution is accepted and the run re-prepares.
    #[tokio::test]
    async fn a_retry_resolution_succeeds_once_the_budget_is_set() {
        let model = MockCompletionModel::new([
            MockTurn::tool_call("call_1", "nonexistent", json!({})),
            MockTurn::text("recovered"),
        ]);
        let agent = AgentBuilder::new(model)
            .default_max_turns(3)
            .tool(MockAddTool)
            .build();

        let mut driver = agent.drive("go").max_invalid_tool_call_retries(1);
        let (request, _, _) = expect_send!(driver);
        let response = request.send().await.expect("scripted turn");
        let outcome = driver
            .model_response(&response)
            .expect("the invalid call needs resolution");
        assert!(matches!(outcome, ModelTurnOutcome::NeedsResolution(_)));

        let outcome = driver
            .resolve_invalid_tool_call(InvalidToolCallAction::Retry {
                feedback: "use a registered tool".to_string(),
            })
            .expect("a retry budget of one accepts the retry");
        assert!(matches!(outcome, ModelTurnOutcome::TurnRetried));

        // The retry re-prepares and the run completes.
        let (request, _, _) = expect_send!(driver);
        let response = request.send().await.expect("scripted turn");
        expect_continue(driver.model_response(&response).expect("turn accepted"));
        match driver.next_step().await.expect("next_step succeeds") {
            DriveStep::Done(response) => assert_eq!(response.output, "recovered"),
            other => panic!("expected Done, got {other:?}"),
        }
    }

    /// Retrieval picks dynamic tools by similarity to the turn's query, so a
    /// registered dynamic tool can be absent from a resumed snapshot purely
    /// because that query did not rank it. Reporting that as "no longer
    /// registered" would be false, and the advice it carries — register the
    /// tool before resuming — unfollowable.
    #[tokio::test]
    async fn resumed_dynamic_tool_outside_retrieval_is_not_drift() {
        use crate::test_utils::MockToolIndex;
        use crate::tool::ToolSet;
        use crate::tool::server::ToolServer;

        let model = MockCompletionModel::new([MockTurn::tool_call(
            "call_1",
            "subtract",
            json!({"x": 5, "y": 2}),
        )]);
        // Retrieval finds `subtract` while the turn is prepared.
        let handle = ToolServer::new()
            .tool(MockAddTool)
            .retrieved_tools(
                1,
                MockToolIndex::new(["subtract"]),
                ToolSet::from_tools(vec![MockSubtractTool]),
            )
            .run();
        let agent = AgentBuilder::new(model)
            .default_max_turns(2)
            .tool_server_handle(handle)
            .build();

        let mut driver = agent.drive("what is 5 - 2?");
        let (request, tools, _) = expect_send!(driver);
        assert!(tools.executable_tool_names().contains("subtract"));
        let response = request.send().await.expect("scripted turn");
        expect_continue(driver.model_response(&response).expect("turn accepted"));
        let _ = expect_execute_tools!(driver);
        let serialized = serde_json::to_string(driver.run()).expect("run serializes");

        // Resume where retrieval ranks nothing — the tool is still registered.
        let resumed_handle = ToolServer::new()
            .tool(MockAddTool)
            .retrieved_tools(
                1,
                MockToolIndex::new(Vec::<String>::new()),
                ToolSet::from_tools(vec![MockSubtractTool]),
            )
            .run();
        let resumed_agent = AgentBuilder::new(MockCompletionModel::text("unused"))
            .default_max_turns(2)
            .tool_server_handle(resumed_handle)
            .build();
        let run: AgentRun = serde_json::from_str(&serialized).expect("run deserializes");
        let mut driver = resumed_agent.drive_run(run);

        let (calls, tools) = expect_execute_tools!(driver);
        let mut context = ToolContext::new();
        let result = tools.execute_call(&calls[0], &mut context).await;
        match result {
            UserContent::ToolResult(result) => assert!(
                !format!("{:?}", result.content).contains("not found"),
                "a registered tool retrieval missed must still dispatch: {:?}",
                result.content
            ),
            other => panic!("expected a tool result, got {other:?}"),
        }
    }

    /// The two halves of a resumed `TurnTools` must agree: the snapshot is
    /// narrowed to what the turn advertised, not left as this process's whole
    /// registry.
    #[tokio::test]
    async fn resumed_snapshot_equals_the_advertised_set() {
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
        expect_continue(driver.model_response(&response).expect("turn accepted"));
        let _ = expect_execute_tools!(driver);
        let serialized = serde_json::to_string(driver.run()).expect("run serializes");

        let resumed_agent = AgentBuilder::new(MockCompletionModel::text("unused"))
            .default_max_turns(2)
            .tool(MockAddTool)
            .tool(MockSubtractTool)
            .build();
        let run: AgentRun = serde_json::from_str(&serialized).expect("run deserializes");
        let advertised = run
            .advertised_tools()
            .expect("the turn recorded its advertised names")
            .clone();
        let mut driver = resumed_agent.drive_run(run);
        let (_, tools) = expect_execute_tools!(driver);

        assert_eq!(tools.executable_tool_names(), &advertised.executable);
        assert!(
            !tools.executable_tool_names().contains("subtract"),
            "the resuming process's extra tool is not part of this turn"
        );
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
        expect_continue(driver.model_response(&response).expect("turn accepted"));
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
