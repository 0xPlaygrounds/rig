//! [`AgentRunner`]: the hook-aware driver that turns a sans-IO
//! [`AgentRun`] into a complete agent loop.
//!
//! [`AgentRun`] decides *what* to do next; it
//! performs no IO and carries no hooks. `AgentRunner` pairs that machine with
//! the side-effecting concerns — building and sending completion requests,
//! executing tools, loading/saving conversation memory — and fires an
//! [`AgentHook`] at every observable point. [`Agent::prompt`] returns an
//! `AgentRunner`; configure it and drive it with custom, composable hooks:
//!
//! ```rust,no_run
//! # use rig_agent::Agent;
//! # async fn example(agent: Agent) -> Result<(), Box<dyn std::error::Error>> {
//! let response = agent
//!     .prompt("What is 2 + 2?")
//!     .max_turns(3)
//!     .run()
//!     .await?;
//! println!("{}", response.output);
//! # Ok(())
//! # }
//! ```

use std::sync::{Arc, Mutex};

use futures::StreamExt;
use tracing_futures::Instrument;

use super::{
    completion::{Agent, AgentConfig},
    engine::{DriveItem, UnaryTurnSource, drive_agent, streaming_error_into_prompt},
    hook::{AgentHook, HookContext},
    run::{AgentRun, response::PromptResponse, spec::UnhandledInvalidToolCall},
    telemetry::acquire_agent_span,
};
use rig_core::{completion::ModelRef, message::ToolChoice};

use crate::{
    completion::{CompletionError, CompletionModel, Document, Message, PromptError, Usage},
    tool::{ToolContext, server::ToolServerHandle},
};

use super::UNKNOWN_AGENT_NAME;

/// A hook-aware driver over [`AgentRun`].
///
/// Construct one with [`Agent::prompt`] (a fresh run) or [`Agent::resume`]
/// (a persisted one), attach hooks with [`add_hook`](Self::add_hook), then
/// call [`run`](Self::run) (blocking), [`stream`](Self::stream)
/// (incremental) or [`run_channel`](Self::run_channel) (a future plus an
/// event feed). Hooks are held in a [`HookStack`](super::hook::HookStack), an ordered,
/// runtime-composable list; `run()` and `stream()` share the same loop and fire
/// the same events, so they behave identically apart from the streamed delta
/// events the medium adds.
#[derive(Clone)]
pub struct AgentRunner {
    /// The run's own copy of the agent's configuration, cloned as one unit by
    /// [`from_agent`](Self::from_agent). Per-run overrides mutate this copy and
    /// never the source [`Agent`]. `description` rides along unused during
    /// execution — an accepted tradeoff for a single shared config type.
    pub(crate) config: AgentConfig,
    /// Where the run starts: a prompt ([`Agent::prompt`]) or a persisted
    /// run to continue ([`Agent::resume`]).
    pub(crate) origin: RunOrigin,
    pub(crate) chat_history: Option<Vec<Message>>,
    pub(crate) max_invalid_tool_call_retries: usize,
    pub(crate) tool_server_handle: ToolServerHandle,
    /// Typed context cloned freshly for every tool dispatch.
    pub(crate) tool_context: ToolContext,
    pub(crate) output_tool_name: Option<String>,
    pub(crate) output_tool_description: Option<String>,
    pub(crate) augment_output_preamble: bool,
    pub(crate) unhandled_invalid_tool_call: UnhandledInvalidToolCall,
    pub(crate) concurrency: usize,
    pub(crate) error_usage: Option<Arc<Mutex<Usage>>>,
}

/// Where a run starts. A prompt builds a fresh [`AgentRun`] (after any
/// memory load); a persisted run is continued as it is, so neither a prompt
/// nor a history nor a memory load applies to it.
#[derive(Clone)]
pub(crate) enum RunOrigin {
    Prompt(Message),
    Resume(Box<AgentRun>),
}

/// The `(history_override, memory_handle)` pair resolved for one run by
/// [`AgentRunner::resolve_history_and_memory`].
pub(crate) type HistoryAndMemory = (
    Option<Vec<Message>>,
    Option<(crate::bus::MemoryHandle, rig_core::id::ConversationId)>,
);

impl AgentRunner {
    /// Build a runner from an agent, seeding it with the agent's default hook
    /// stack. The one construction site behind [`Agent::prompt`] and the
    /// typed and extractor runs.
    pub(crate) fn from_agent(agent: &Agent, prompt: impl Into<Message>) -> Self {
        Self::new(agent, RunOrigin::Prompt(prompt.into()))
    }

    /// Build a runner that continues `run` ([`Agent::resume`]).
    pub(crate) fn resuming(agent: &Agent, run: AgentRun) -> Self {
        Self::new(agent, RunOrigin::Resume(Box::new(run)))
    }

    fn new(agent: &Agent, origin: RunOrigin) -> Self {
        Self {
            config: agent.config.clone(),
            origin,
            chat_history: None,
            max_invalid_tool_call_retries: 0,
            tool_server_handle: agent.tool_server_handle.clone(),
            tool_context: ToolContext::new(),
            output_tool_name: None,
            output_tool_description: None,
            augment_output_preamble: true,
            unhandled_invalid_tool_call: UnhandledInvalidToolCall::Fail,
            concurrency: 1,
            error_usage: None,
        }
    }

    /// Append a hook to the stack (on top of any the agent already carries).
    /// Hooks run in registration order; how their results compose is
    /// event-dependent (model selections and `ToolCall`/`ToolResult` rewrites
    /// chain, `CompletionCall` request patches accumulate and merge, while
    /// model-turn steering and observe-only/recovery events use their
    /// event-specific terminal action). See the [`hook`](crate::agent::hook)
    /// module docs.
    pub fn add_hook<H>(mut self, hook: H) -> Self
    where
        H: AgentHook + 'static,
    {
        self.config.hooks.push(hook);
        self
    }
}

impl AgentRunner {
    /// Set the total model-call budget, including the initial call and every
    /// retry or continuation. Zero emits no model calls; one permits only the
    /// initial call. Exceeding the budget returns [`PromptError::MaxTurnsError`].
    pub fn max_turns(mut self, max_turns: usize) -> Self {
        self.config.max_turns = max_turns;
        self
    }

    /// Set the default model candidate for this run.
    ///
    /// This does not suppress registered model-selection hooks, which may
    /// replace the candidate before each model call (including retries).
    /// Append an unconditional selecting hook last when the run must always
    /// use one model.
    pub fn using_model(mut self, label: impl Into<ModelRef>) -> Self {
        self.config.model_key = self.config.bus.model_key(label.into().as_str());
        self.config.anonymous_model = None;
        self
    }

    /// Register `model` on the agent's bus under a generated label and use
    /// it as this run's default. The registration is scoped to this runner
    /// and the run it produces: it leaves the bus when they drop.
    pub fn using_model_value<M>(mut self, model: M) -> Self
    where
        M: CompletionModel + 'static,
    {
        let anonymous = self.config.bus.register_anonymous_model(model);
        self.config.model_key = anonymous.key().clone();
        self.config.anonymous_model = Some(anonymous);
        self
    }

    /// Set the typed context cloned for every tool dispatch in this run.
    pub fn tool_context(mut self, context: ToolContext) -> Self {
        self.tool_context = context;
        self
    }

    /// Set the chat history preceding the prompt. Passing explicit history
    /// bypasses conversation memory for this run.
    pub fn history<I, T>(mut self, history: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Into<Message>,
    {
        self.chat_history = Some(history.into_iter().map(Into::into).collect());
        self
    }

    /// Override the agent preamble for this run.
    pub fn preamble(mut self, preamble: impl Into<String>) -> Self {
        self.config.preamble = Some(preamble.into());
        self
    }

    /// Remove the agent's configured preamble for this run.
    pub fn without_preamble(mut self) -> Self {
        self.config.preamble = None;
        self
    }

    /// Append one static context document for this run.
    pub fn document(mut self, document: Document) -> Self {
        self.config.static_context.push(document);
        self
    }

    /// Append static context documents for this run.
    pub fn documents(mut self, documents: impl IntoIterator<Item = Document>) -> Self {
        self.config.static_context.extend(documents);
        self
    }

    /// Override the model temperature for this run.
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.config.temperature = Some(temperature);
        self
    }

    /// Remove the agent's configured temperature for this run.
    pub fn without_temperature(mut self) -> Self {
        self.config.temperature = None;
        self
    }

    /// Override the maximum completion token count for this run.
    pub fn max_tokens(mut self, max_tokens: u64) -> Self {
        self.config.max_tokens = Some(max_tokens);
        self
    }

    /// Remove the agent's configured maximum token count for this run.
    pub fn without_max_tokens(mut self) -> Self {
        self.config.max_tokens = None;
        self
    }

    /// Shallow-merge object fields into the provider-specific parameters for
    /// this run. Later fields win. A non-object baseline is replaced by the
    /// supplied object. A later completion-call hook patch has final
    /// precedence: object values shallow-merge, while a non-object on either
    /// side causes wholesale replacement by the hook value.
    pub fn merge_additional_params(
        mut self,
        params: serde_json::Map<String, serde_json::Value>,
    ) -> Self {
        let params = serde_json::Value::Object(params);
        self.config.additional_params = Some(match self.config.additional_params.take() {
            Some(baseline) if baseline.is_object() => crate::json_utils::merge(baseline, params),
            _ => params,
        });
        self
    }

    /// Replace all provider-specific parameters for this run. A later
    /// completion-call hook patch has final precedence: object values
    /// shallow-merge, while a non-object on either side causes wholesale
    /// replacement by the hook value.
    pub fn replace_additional_params(mut self, params: serde_json::Value) -> Self {
        self.config.additional_params = Some(params);
        self
    }

    /// Remove the agent's configured provider-specific parameters for this run.
    /// A later completion-call hook may still supply its own parameters.
    pub fn without_additional_params(mut self) -> Self {
        self.config.additional_params = None;
        self
    }

    /// Override the tool-choice policy for this run.
    pub fn tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.config.tool_choice = Some(tool_choice);
        self
    }

    /// Remove the agent's configured tool-choice policy for this run.
    pub fn without_tool_choice(mut self) -> Self {
        self.config.tool_choice = None;
        self
    }

    /// Configure the synthetic tool used by an internal Tool-output flow.
    pub(crate) fn output_tool(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        augment_preamble: bool,
    ) -> Self {
        self.output_tool_name = Some(name.into());
        self.output_tool_description = Some(description.into());
        self.augment_output_preamble = augment_preamble;
        self
    }

    /// Ignore invalid tool calls when every registered hook declines to act.
    ///
    /// Set what this run does with an invalid tool call no hook resolves.
    /// See [`UnhandledInvalidToolCall`].
    pub fn unhandled_invalid_tool_call(mut self, policy: UnhandledInvalidToolCall) -> Self {
        self.unhandled_invalid_tool_call = policy;
        self
    }

    /// Opt in or out of recording sensitive request, response, and tool content
    /// on GenAI telemetry spans for this run.
    ///
    /// Defaults to the agent's setting, which defaults to `false`. Enabling this
    /// can expose prompts, retrieved context, tool results, model responses, and
    /// other sensitive or high-cardinality data through OpenTelemetry span
    /// attributes, which can increase observability backend storage and query
    /// costs. Only enable it when content telemetry is acceptable for this run.
    /// Structural metadata and token usage remain available when disabled.
    pub fn record_content_telemetry(mut self, enabled: bool) -> Self {
        self.config.record_telemetry_content = enabled;
        self
    }

    /// Execute up to `concurrency` tools at once (1 by default). Applies to
    /// **both** the blocking [`run`](Self::run) and the streaming
    /// [`stream`](Self::stream) paths.
    ///
    /// The resulting message history is the same in both paths regardless of
    /// `concurrency`: final tool results are persisted in tool-call order. At
    /// the default `concurrency` of 1 the two paths are fully in lock-step; with
    /// `concurrency > 1` the tools run in parallel, so a `ToolCall`/`ToolResult`
    /// **hook may fire in completion order** rather than call order — the
    /// per-tool side effects interleave even though the final history does not.
    ///
    /// For the streaming path: the driver emits *all* of a turn's `ToolCall`
    /// stream items eagerly (in call order) when the model turn commits, then —
    /// only after the whole tool batch settles successfully — surfaces the
    /// per-tool `ToolExecutionCommitted` and `ToolResult` stream items in **call
    /// order** (never completion order), for the tools whose body actually ran.
    /// The persisted message history is unchanged.
    ///
    /// A `concurrency` of 0 is clamped to 1; at `1` the tools of a turn run
    /// strictly sequentially in call order, failing fast on the first
    /// terminating error.
    pub fn tool_concurrency(mut self, concurrency: usize) -> Self {
        self.concurrency = concurrency.max(1);
        self
    }

    /// Set the conversation id used to load and persist memory for this run.
    pub fn conversation(mut self, id: impl Into<rig_core::id::ConversationId>) -> Self {
        self.config.conversation_id = Some(id.into());
        self
    }

    /// Disable conversation memory for this run (no load, no save).
    pub fn without_memory(mut self) -> Self {
        self.config.memory_key = None;
        self.config.conversation_id = None;
        self
    }

    /// Set the retry budget for invalid tool-call recovery. Invalid tool-call
    /// retries also consume the total model-call budget.
    pub fn max_invalid_tool_call_retries(mut self, retries: usize) -> Self {
        self.max_invalid_tool_call_retries = retries;
        self
    }

    pub(crate) fn agent_name_or_default(&self) -> &str {
        self.config.name.as_deref().unwrap_or(UNKNOWN_AGENT_NAME)
    }

    /// The [`AgentRun`] this runner drives: the persisted run it continues,
    /// or a fresh one from its prompt and configuration. `history_override`
    /// replaces the configured chat history (e.g. with memory-loaded
    /// history) and applies only to a fresh run. Delegates to
    /// [`build_agent_run`] — the single construction site shared with the
    /// streaming driver.
    pub(crate) fn build_run(&self, history_override: Option<Vec<Message>>) -> AgentRun {
        let prompt = match &self.origin {
            // Cloned, not moved: `build_run` is shared by both drivers over
            // `&self`, and the runner is handed whole to the driver next. The
            // copy is one transcript per resumed run — the order of the runner
            // clone a typed run already makes per attempt — and it spares a
            // "taken" placeholder state in `RunOrigin`.
            RunOrigin::Resume(run) => return (**run).clone(),
            RunOrigin::Prompt(prompt) => prompt.clone(),
        };
        let run = build_agent_run(
            prompt,
            self.config.max_turns,
            self.max_invalid_tool_call_retries,
            self.unhandled_invalid_tool_call,
            self.config.output_schema.as_ref(),
            history_override.or_else(|| self.chat_history.clone()),
            self.config.tool_choice.clone(),
        );
        match &self.output_tool_name {
            Some(name) => run.with_output_tool_name(name.clone()),
            None => run,
        }
    }
}

/// Construct an [`AgentRun`] from explicit run configuration. The single place a
/// run is built, so the blocking and streaming drivers configure runs
/// identically.
pub(crate) fn build_agent_run(
    prompt: Message,
    max_turns: usize,
    max_invalid_tool_call_retries: usize,
    unhandled_invalid_tool_call: UnhandledInvalidToolCall,
    output_schema: Option<&schemars::Schema>,
    history: Option<Vec<Message>>,
    tool_choice: Option<ToolChoice>,
) -> AgentRun {
    let spec = crate::run::spec::RunSpec {
        max_turns: Some(max_turns),
        max_invalid_tool_call_retries,
        unhandled_invalid_tool_call,
        output_schema: output_schema.map(|schema| schema.as_value().clone()),
        tool_choice,
        ..crate::run::spec::RunSpec::new()
    };
    AgentRun::from_spec(&spec, prompt, history)
}

impl AgentRunner {
    /// [`run_under`](Self::run_under) that also reports the usage a failed
    /// run consumed; `ambient` is the span the typed run was started in.
    pub(crate) async fn run_with_error_usage(
        mut self,
        ambient: tracing::Span,
    ) -> (Result<PromptResponse, PromptError>, Usage) {
        let usage = Arc::new(Mutex::new(Usage::new()));
        self.error_usage = Some(usage.clone());
        let result = self.run_under(ambient).await;
        let observed = result.as_ref().map_or_else(
            |_| {
                *usage
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
            },
            |response| response.usage,
        );
        (result, observed)
    }

    /// Open the per-run agent span, recording the prompt when content
    /// telemetry is enabled. Shared by the blocking and streaming surfaces.
    pub(crate) fn open_agent_span(&self, ambient: tracing::Span) -> (tracing::Span, bool) {
        let (agent_span, created_agent_span) = acquire_agent_span(
            ambient,
            self.agent_name_or_default(),
            self.config.preamble.as_deref(),
            self.config.record_telemetry_content,
        );

        // A resumed run's prompt is inside its state: the span records none.
        if self.config.record_telemetry_content
            && let RunOrigin::Prompt(prompt) = &self.origin
            && let Some(text) = prompt.rag_text()
        {
            agent_span.record("gen_ai.prompt", text);
        }

        (agent_span, created_agent_span)
    }

    /// Resolve the history override and memory handle for this run.
    ///
    /// When the caller passes explicit history, memory is fully bypassed
    /// (no load AND no save). Otherwise, if a memory backend and conversation
    /// id are both configured, prior history is loaded. Each surface adapts a
    /// load failure to its own error channel.
    pub(crate) async fn resolve_history_and_memory(
        &self,
        hook_ctx: &HookContext,
    ) -> Result<HistoryAndMemory, rig_core::memory::MemoryError> {
        match &self.chat_history {
            Some(_) => Ok((None, None)),
            None => match (self.config.memory_handle(), &self.config.conversation_id) {
                (Some(memory), Some(id)) => {
                    let memory = memory.map_err(|report| {
                        rig_core::memory::MemoryError::Internal(report.to_string())
                    })?;
                    // The load is a `Memory` dispatch at the boundary:
                    // observe-only for hooks unless one opts in.
                    let loaded = crate::agent::engine::dispatch_effect(
                        &self.config.hooks,
                        hook_ctx,
                        self.config.bus.dispatcher(),
                        memory.key(),
                        rig_core::effect::EffectKind::Memory {
                            op: rig_core::effect::MemoryOp::Load {
                                conversation: id.clone(),
                            },
                        },
                    )
                    .await
                    .and_then(|outcome| match outcome {
                        rig_core::effect::Outcome::Memory(
                            rig_core::effect::MemoryOutcome::Loaded { messages },
                        ) => Ok(messages),
                        other => Err(crate::agent::engine::wrong_outcome("loaded memory", &other)),
                    })
                    .map_err(memory_error_from_report)?;
                    Ok((Some(loaded), Some((memory, id.clone()))))
                }
                _ => Ok((None, None)),
            },
        }
    }

    /// The run-scoped hook context: minted once per run, before the memory
    /// load (a dispatch too), shared by every hook event.
    pub(crate) fn hook_context(&self, is_streaming: bool) -> HookContext {
        HookContext::new(
            is_streaming,
            self.config.name.clone(),
            Some(self.config.bus.dispatcher().clone()),
        )
    }

    /// Drive the agent loop to completion, returning the aggregated
    /// [`PromptResponse`]. Hooks fire at every observable point; the first hook
    /// to terminate cancels the run. The run belongs to the span this method
    /// (or `.await`) was called in, wherever the future is polled.
    pub fn run(
        self,
    ) -> impl Future<Output = Result<PromptResponse, PromptError>> + rig_core::wasm_compat::WasmCompatSend
    {
        // Like `stream()`: the run belongs to the span it was started in,
        // not to whichever task first polls the future.
        let ambient = tracing::Span::current();
        let run_under = ambient.clone();
        async move { self.run_under(run_under).await }.instrument(ambient)
    }

    async fn run_under(self, ambient: tracing::Span) -> Result<PromptResponse, PromptError> {
        let (agent_span, created_agent_span) = self.open_agent_span(ambient);
        let bus = self.config.bus.clone();
        let hook_ctx = self.hook_context(false);
        // A resumed run brought its history with it: nothing is loaded and
        // nothing is saved — no `Memory` dispatch, no memory hook event, no
        // record in the log — so its continuation is exactly the reference
        // log's tail and a memory backend that is down cannot fail it.
        let (history_override, memory_handle) = match &self.origin {
            RunOrigin::Resume(_) => (None, None),
            RunOrigin::Prompt(_) => {
                // A memory load is a dispatch too: drive the bus while resolving.
                let resolve = self.resolve_history_and_memory(&hook_ctx);
                futures::pin_mut!(resolve);
                let mut driven = bus.drive(futures::stream::once(resolve));
                match driven.next().await {
                    Some(resolved) => resolved?,
                    None => (None, None),
                }
            }
        };
        let run = self.build_run(history_override);

        // Fold the shared engine to its final response. The blocking surface
        // uses a unary model transport and ignores the intermediate items the
        // engine yields; the engine is driven under the caller's ambient span
        // (no `instrument`), keeping the agent span detached and the chat/tool
        // spans on the blocking `follows_from` chain.
        let record_telemetry_content = self.config.record_telemetry_content;
        let driver = drive_agent(
            self,
            UnaryTurnSource::new(record_telemetry_content),
            run,
            agent_span,
            created_agent_span,
            memory_handle,
            hook_ctx,
        );
        let driver = bus.drive(Box::pin(driver));
        futures::pin_mut!(driver);

        let mut response = None;
        while let Some(item) = driver.next().await {
            match item {
                Ok(DriveItem::Done(done)) => response = Some(*done),
                Ok(DriveItem::Item(_)) => {}
                Err(err) => {
                    // The engine settles an error ending *after* yielding
                    // it (`on_run_settled` with `SettledOutcome::Error`),
                    // so the fold drains the engine before returning: a
                    // fold that returned here dropped the engine at the
                    // yield and the settled hook never fired for a
                    // blocking run that a hook stopped or a provider
                    // refused, while the streaming surface's consumer,
                    // polling to the end, saw it fire.
                    let error = streaming_error_into_prompt(err);
                    while driver.next().await.is_some() {}
                    return Err(error);
                }
            }
        }

        // The engine yields `Done` unless it errored (handled above).
        response.ok_or_else(|| {
            PromptError::CompletionError(CompletionError::ResponseError(
                "agent run ended without producing a final response".to_string(),
            ))
        })
    }
}

/// `.await`ing a runner is [`run`](AgentRunner::run).
impl std::future::IntoFuture for AgentRunner {
    type Output = Result<PromptResponse, PromptError>;
    type IntoFuture = rig_core::wasm_compat::WasmBoxedFuture<'static, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(self.run())
    }
}

#[cfg(test)]
mod entry_tests;
#[cfg(test)]
mod ignore_tests;
#[cfg(test)]
mod settled_tests;
#[cfg(test)]
mod tests;

#[cfg(test)]
mod prompt_tests;

/// A memory report back into the memory error the run surface names.
pub(crate) fn memory_error_from_report(
    report: rig_core::error::ErrorReport,
) -> rig_core::memory::MemoryError {
    match report.kind {
        rig_core::error::ErrorKind::MemoryPolicy => {
            rig_core::memory::MemoryError::Policy(report.message)
        }
        rig_core::error::ErrorKind::MemoryBackend => {
            rig_core::memory::MemoryError::Backend(Box::new(report))
        }
        // A layer on the memory key denied the load: policy, and the run
        // fails at the record.
        rig_core::error::ErrorKind::Denied => rig_core::memory::MemoryError::Policy(report.message),
        rig_core::error::ErrorKind::Http { .. }
        | rig_core::error::ErrorKind::Json
        | rig_core::error::ErrorKind::Url
        | rig_core::error::ErrorKind::Request
        | rig_core::error::ErrorKind::Response
        | rig_core::error::ErrorKind::Provider
        | rig_core::error::ErrorKind::ProviderResponse
        | rig_core::error::ErrorKind::Tool(_)
        | rig_core::error::ErrorKind::Internal
        | rig_core::error::ErrorKind::Cancelled
        | rig_core::error::ErrorKind::Timeout
        | rig_core::error::ErrorKind::BusClosed
        | rig_core::error::ErrorKind::HandlerUnavailable
        | rig_core::error::ErrorKind::Divergence
        | rig_core::error::ErrorKind::Other => {
            rig_core::memory::MemoryError::Internal(report.to_string())
        }
    }
}
