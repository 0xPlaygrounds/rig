//! Hook-aware execution of [`AgentRun`] with per-run overrides, completion
//! requests and tools.
//!
//! ```rust,no_run
//! # use rig_agent::Agent;
//! # async fn example(agent: Agent) -> Result<(), Box<dyn std::error::Error>> {
//! let response = agent
//!     .prompt("What is 2 + 2?")
//!     .max_turns(3)
//!     .run()
//!     .await?;
//! println!("{}", response.output());
//! # Ok(())
//! # }
//! ```

use std::sync::{Arc, Mutex};

use futures::StreamExt;
use tracing_futures::Instrument;

use super::{
    completion::{Agent, AgentConfig},
    engine::{DriveItem, UnaryTurnSource, drive_agent},
    hook::{AgentHook, HookContext},
    telemetry::acquire_agent_span,
};
use crate::run::{AgentRun, response::PromptResponse, spec::UnhandledInvalidToolCall};
use rig_core::error::ProviderError;
use rig_core::{completion::ModelRef, message::ToolChoice};

use crate::{
    completion::{Document, Message, PromptError, Usage},
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
///
/// `O` is what `.await` recovers: `()` for the raw [`PromptResponse`], or a
/// [`TypedOutput`](super::TypedOutput) for a [`TypedRun`](super::TypedRun).
#[derive(Clone)]
#[must_use = "a run does nothing until awaited or driven"]
pub struct AgentRunner<O = ()> {
    /// Run-local configuration; overrides never mutate the source agent.
    pub(crate) config: AgentConfig,
    /// Where the run starts: a prompt ([`Agent::prompt`]) or a persisted
    /// run to continue ([`Agent::resume`]).
    pub(crate) origin: RunOrigin,
    pub(crate) chat_history: Option<Vec<Arc<Message>>>,
    pub(crate) tool_server_handle: ToolServerHandle,
    /// Typed context cloned freshly for every tool dispatch.
    pub(crate) tool_context: ToolContext,
    pub(crate) concurrency: usize,
    pub(crate) error_usage: Option<Arc<Mutex<Usage>>>,
    pub(crate) output: O,
}

/// Where a run starts. A prompt builds a fresh [`AgentRun`]; a persisted run
/// is continued as it is, so neither a prompt nor a history applies to it.
#[derive(Clone)]
pub(crate) enum RunOrigin {
    Prompt(Message),
    Resume(AgentRun),
}

impl AgentRunner {
    /// Build a runner from an agent, seeding it with the agent's default hook
    /// stack. The one construction site behind [`Agent::prompt`] and the
    /// typed and extractor runs.
    pub(crate) fn from_agent(agent: &Agent, prompt: impl Into<Message>) -> Self {
        Self::new(agent, RunOrigin::Prompt(prompt.into()))
    }

    /// Build a runner that continues `run` ([`Agent::resume`]).
    pub(crate) fn resuming(agent: &Agent, run: AgentRun) -> Self {
        Self::new(agent, RunOrigin::Resume(run))
    }

    fn new(agent: &Agent, origin: RunOrigin) -> Self {
        Self {
            config: agent.config.clone(),
            origin,
            chat_history: None,
            tool_server_handle: agent.tool_server_handle.clone(),
            tool_context: ToolContext::new(),
            concurrency: 1,
            error_usage: None,
            output: (),
        }
    }

    /// Configure the synthetic tool used by an internal Tool-output flow.
    pub(crate) fn output_tool(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        augment_preamble: bool,
    ) -> Self {
        let spec = self.config.spec_mut();
        spec.output_tool_name = Some(name.into());
        spec.output_tool_description = Some(description.into());
        spec.augment_output_preamble = augment_preamble;
        self
    }

    /// Set the policy for invalid tool calls that no hook resolves. A resumed
    /// run keeps the policy it persisted.
    pub fn unhandled_invalid_tool_call(mut self, policy: UnhandledInvalidToolCall) -> Self {
        self.config.spec_mut().unhandled_invalid_tool_call = policy;
        self
    }
}

impl<O> AgentRunner<O> {
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

    /// Set the total model-call budget, including the initial call and every
    /// retry or continuation. Zero emits no model calls; one permits only the
    /// initial call. Exceeding the budget returns [`PromptError::MaxTurns`],
    /// which a typed run wraps in
    /// [`StructuredOutputError::Prompt`](crate::completion::StructuredOutputError::Prompt).
    pub fn max_turns(mut self, max_turns: usize) -> Self {
        self.config.spec_mut().max_turns = Some(max_turns);
        self
    }

    /// Keep this run's conversation in `store`, in place of the agent's
    /// ([`AgentBuilder::memory`](super::AgentBuilder::memory)), such as a
    /// store chosen per request; one store also holds every user's
    /// conversation, a log each ([`conversation`](Self::conversation)).
    pub fn memory(
        mut self,
        store: impl rig_memory::AsyncStore
        + rig_core::wasm_compat::WasmCompatSend
        + rig_core::wasm_compat::WasmCompatSync
        + 'static,
    ) -> Self {
        self.config.memory = Some(super::memory::Memory::new(store));
        self
    }

    /// The log of the memory store this run loads and saves
    /// ([`AgentBuilder::conversation`](super::AgentBuilder::conversation)).
    pub fn conversation(mut self, log: impl Into<String>) -> Self {
        self.config.conversation = Some(log.into());
        self
    }

    /// This run neither loads nor saves a conversation.
    pub fn without_memory(mut self) -> Self {
        self.config.memory = None;
        self
    }

    /// The store and log this run keeps its conversation in: none without a
    /// store, or with an explicit [`history`](AgentRunner::history).
    pub(crate) fn memory_log(&self) -> Option<(super::memory::Memory, String)> {
        let memory = self
            .config
            .memory
            .clone()
            .filter(|_| self.chat_history.is_none())?;
        let log = self
            .config
            .conversation
            .as_ref()
            .or(self.config.name.as_ref());
        Some((memory, log.map_or_else(|| "agent".to_owned(), Clone::clone)))
    }

    /// Set the default model candidate for this run.
    ///
    /// This does not suppress registered model-selection hooks, which may
    /// replace the candidate before each model call (including retries).
    /// Append an unconditional selecting hook last when the run must always
    /// use one model.
    pub fn using_model(mut self, label: impl Into<ModelRef>) -> Self {
        self.config.select_model_label(label.into());
        self
    }

    /// Register `model` in the agent's table under a generated label and use
    /// it as this run's default. The registration is scoped to this runner
    /// and the run it produces: it leaves the table when they drop.
    pub fn using_model_value(
        mut self,
        model: impl Into<rig_core::DynModel<rig_core::operation::Completion>>,
    ) -> Self {
        self.config.select_anonymous_model(model);
        self
    }

    /// Set the typed context cloned for every tool dispatch in this run.
    pub fn tool_context(mut self, context: ToolContext) -> Self {
        self.tool_context = context;
        self
    }

    /// Set the chat history preceding the prompt. The run then neither
    /// loads nor saves a memory store's conversation.
    pub fn history<I, T>(mut self, history: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Into<Message>,
    {
        let history = history.into_iter().map(|message| Arc::new(message.into()));
        self.chat_history = Some(history.collect());
        self
    }

    /// Override the agent preamble for this run.
    pub fn preamble(mut self, preamble: impl Into<String>) -> Self {
        self.config.spec_mut().preamble = Some(preamble.into());
        self
    }

    /// Remove the agent's configured preamble for this run.
    pub fn without_preamble(mut self) -> Self {
        self.config.spec_mut().preamble = None;
        self
    }

    /// Append one static context document for this run.
    pub fn document(mut self, document: Document) -> Self {
        self.config.spec_mut().static_context.push(document);
        self
    }

    /// Append static context documents for this run.
    pub fn documents(mut self, documents: impl IntoIterator<Item = Document>) -> Self {
        self.config.spec_mut().static_context.extend(documents);
        self
    }

    /// Override the model temperature for this run.
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.config.spec_mut().temperature = Some(temperature);
        self
    }

    /// Remove the agent's configured temperature for this run.
    pub fn without_temperature(mut self) -> Self {
        self.config.spec_mut().temperature = None;
        self
    }

    /// Override the maximum completion token count for this run.
    pub fn max_tokens(mut self, max_tokens: u64) -> Self {
        self.config.spec_mut().max_tokens = Some(max_tokens);
        self
    }

    /// Remove the agent's configured maximum token count for this run.
    pub fn without_max_tokens(mut self) -> Self {
        self.config.spec_mut().max_tokens = None;
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
        self.config.spec_mut().additional_params =
            Some(match self.config.spec_mut().additional_params.take() {
                Some(baseline) if baseline.is_object() => {
                    crate::json_utils::merge(baseline, params)
                }
                _ => params,
            });
        self
    }

    /// Replace all provider-specific parameters for this run. A later
    /// completion-call hook patch has final precedence: object values
    /// shallow-merge, while a non-object on either side causes wholesale
    /// replacement by the hook value.
    pub fn replace_additional_params(mut self, params: serde_json::Value) -> Self {
        self.config.spec_mut().additional_params = Some(params);
        self
    }

    /// Remove the agent's configured provider-specific parameters for this run.
    /// A later completion-call hook may still supply its own parameters.
    pub fn without_additional_params(mut self) -> Self {
        self.config.spec_mut().additional_params = None;
        self
    }

    /// Overlay `options` on the agent's generation options for this run:
    /// each field `options` sets wins, every other field keeps the agent's
    /// ([`GenerationOptions::overlay`](rig_core::completion::GenerationOptions::overlay)).
    /// A policy `options` sets wins too, so a run can restore
    /// [`OnUnsupported::Error`](rig_core::completion::OnUnsupported::Error)
    /// over an agent's `Ignore`.
    ///
    /// Calls apply in order: this overlays only the fields `options` sets,
    /// so it wins over a shortcut such as [`Self::seed`] called before it
    /// for those fields, and a shortcut called after it sets its one field
    /// on top.
    pub fn options(mut self, options: rig_core::completion::GenerationOptions) -> Self {
        self.config.spec_mut().options = self.config.spec.options.clone().overlay(&options);
        self
    }

    generation_option_setters!("this run's");

    /// Set the stop sequences in this run's generation options, keeping its
    /// other fields: the same as
    /// `.options(GenerationOptions::new().stop(stop))`. As with that overlay,
    /// an empty list means "not set" and keeps the agent's stop sequences;
    /// it does not clear them. See [`Self::options`] for the order of calls.
    pub fn stop<S: Into<String>>(self, stop: impl IntoIterator<Item = S>) -> Self {
        self.options(rig_core::completion::GenerationOptions::new().stop(stop))
    }

    /// Overlay `options` on the agent's provider options for this run: each
    /// provider entry `options` holds replaces the agent's, every other
    /// entry keeps the agent's
    /// ([`ProviderOptions::overlay`](rig_core::completion::ProviderOptions::overlay)).
    ///
    /// Unlike [`AgentBuilder::provider_options`](crate::agent::AgentBuilder::provider_options),
    /// it merges rather than replaces, so calls apply in order without
    /// loss: an entry set by an earlier [`Self::provider_option`] is kept
    /// unless `options` holds an entry for the same provider, and a later
    /// [`Self::provider_option`] replaces its provider's entry on top.
    pub fn provider_options(mut self, options: rig_core::completion::ProviderOptions) -> Self {
        self.config.spec_mut().provider_options =
            self.config.spec.provider_options.clone().overlay(&options);
        self
    }

    /// Store `options` as the entry of their provider
    /// ([`ExtensionOptions::Ext`](rig_core::completion::ExtensionOptions::Ext))
    /// for this run, in place of the agent's entry for that provider; every
    /// other entry keeps the agent's. The same as
    /// `.provider_options(ProviderOptions::new().set(options))`, so options
    /// that write no field hold no entry and keep the agent's entry for
    /// that provider. Options that do not serialize fail each request's
    /// encode. See [`Self::provider_options`] for the order of calls. A
    /// third-party provider reusing a built-in options type needs
    /// [`ProviderOptions::with::<P>`](rig_core::completion::ProviderOptions::with)
    /// and [`Self::provider_options`]: this stores under the built-in key.
    pub fn provider_option<E: rig_core::completion::ExtensionOptions>(self, options: E) -> Self {
        self.provider_options(rig_core::completion::ProviderOptions::new().set(options))
    }

    /// Override the tool-choice policy for this run.
    pub fn tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.config.spec_mut().tool_choice = Some(tool_choice);
        self
    }

    /// Remove the agent's configured tool-choice policy for this run.
    pub fn without_tool_choice(mut self) -> Self {
        self.config.spec_mut().tool_choice = None;
        self
    }

    /// Opt in or out of recording sensitive request, response, and tool content
    /// on GenAI telemetry spans for this run.
    ///
    /// Defaults to the agent's setting, which defaults to `false`. Enabling this
    /// can expose prompts, retrieved context, tool results, model responses,
    /// function tool definitions (including descriptions and schemas), and
    /// other sensitive or high-cardinality data through OpenTelemetry span
    /// attributes, which can increase observability backend storage and query
    /// costs. Only enable it when content telemetry is acceptable for this run.
    /// Structural metadata and token usage remain available when disabled.
    pub fn record_content_telemetry(mut self, enabled: bool) -> Self {
        self.config.record_telemetry_content = enabled;
        self
    }

    /// Accept finish reasons outside the normalized vocabulary
    /// ([`FinishReason::Other`](rig_core::completion::FinishReason::Other))
    /// as a normal stop, for this run. Off by default: such a turn fails the run,
    /// and history replay leaves it out. When on, the turn succeeds, its
    /// tool calls run, and it replays.
    ///
    /// `Other` also holds genuine failures, such as Gemini's
    /// `MALFORMED_FUNCTION_CALL` and `RECITATION` or Bedrock's
    /// `malformed_tool_use`, so tool calls from a malformed reply may run.
    /// Filtered content still fails the turn.
    pub fn accept_unknown_finish_reasons(mut self, accept: bool) -> Self {
        self.config.spec_mut().accept_unknown_finish_reasons = accept;
        self
    }

    /// Limit concurrent tool calls in awaited and streamed runs. Defaults to one;
    /// zero is clamped to one. Sequential execution follows call order and stops
    /// at the first terminating error; concurrent hooks and side effects may interleave.
    /// Committed history always retains call order.
    ///
    /// Streams emit model tool calls when the turn commits. Execution confirmations
    /// and results appear in call order only after the whole batch succeeds;
    /// execution confirmations exclude calls whose bodies did not run.
    pub fn tool_concurrency(mut self, concurrency: usize) -> Self {
        self.concurrency = concurrency.max(1);
        self
    }

    /// Set the retry budget for invalid tool-call recovery. Invalid tool-call
    /// retries also consume the total model-call budget.
    pub fn max_invalid_tool_call_retries(mut self, retries: usize) -> Self {
        self.config.spec_mut().max_invalid_tool_call_retries = retries;
        self
    }

    /// Set how many consecutive turns may call a tool with arguments that are
    /// not a JSON object. Each such call is answered with feedback, or as the
    /// invalid-call hook decides; a turn whose calls all parse resets the
    /// count. Past the limit the run fails. `None`, the default, sets no
    /// limit. A resumed run keeps the limit it persisted.
    pub fn max_consecutive_malformed_tool_calls(mut self, limit: impl Into<Option<usize>>) -> Self {
        self.config.spec_mut().max_consecutive_malformed_tool_calls = limit.into();
        self
    }

    /// This runner recovering `output` instead, and the output it replaced.
    pub(crate) fn replace_output<P>(self, output: P) -> (AgentRunner<P>, O) {
        let Self {
            config,
            origin,
            chat_history,
            tool_server_handle,
            tool_context,
            concurrency,
            error_usage,
            output: replaced,
        } = self;
        let runner = AgentRunner {
            config,
            origin,
            chat_history,
            tool_server_handle,
            tool_context,
            concurrency,
            error_usage,
            output,
        };
        (runner, replaced)
    }

    pub(crate) fn agent_name_or_default(&self) -> &str {
        self.config.name.as_deref().unwrap_or(UNKNOWN_AGENT_NAME)
    }

    /// The [`AgentRun`] this runner drives: the persisted run it continues,
    /// or a fresh one from its prompt and configuration. The one construction
    /// site, so the blocking and streaming drivers configure runs identically.
    pub(crate) fn build_run(&self) -> AgentRun {
        let prompt = match &self.origin {
            RunOrigin::Resume(run) => return run.clone(),
            RunOrigin::Prompt(prompt) => prompt.clone(),
        };
        AgentRun::from_spec(&self.config.spec, prompt, self.chat_history.clone())
    }
}

impl AgentRunner {
    /// [`run_under`](Self::run_under) that also reports the usage a failed
    /// run consumed; `ambient` is the span the typed run was started in.
    pub(crate) async fn run_with_error_usage(
        mut self,
        ambient: tracing::Span,
    ) -> (Result<PromptResponse, PromptError>, Usage) {
        let usage = Arc::new(Mutex::new(Usage::default()));
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
            self.config.spec.preamble.as_deref(),
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

    /// The run-scoped hook context: minted once per run, shared by every
    /// hook event.
    pub(crate) fn hook_context(&self, is_streaming: bool) -> HookContext {
        HookContext::new(
            is_streaming,
            self.config.name.clone(),
            self.config.handlers.clone(),
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
        let hook_ctx = self.hook_context(false);
        let run = self.build_run();

        // Instrument the fold under the captured agent span, not the poller's span.
        let record_telemetry_content = self.config.record_telemetry_content;
        // Boxed, so the futures awaiting the run stay small.
        let mut driver = Box::pin(drive_agent(
            self,
            UnaryTurnSource::new(record_telemetry_content),
            run,
            agent_span.clone(),
            created_agent_span,
            hook_ctx,
        ));
        let fold = async move {
            let mut response = None;
            while let Some(item) = driver.next().await {
                match item {
                    Ok(DriveItem::Done(done)) => response = Some(done),
                    Ok(DriveItem::Surfaced(_) | DriveItem::Projected(_)) => {}
                    Err(err) => {
                        // Drain through termination so engine teardown completes
                        // rather than being dropped at the error yield.
                        while driver.next().await.is_some() {}
                        return Err(err);
                    }
                }
            }
            response.ok_or_else(|| {
                PromptError::Provider(ProviderError::Response(
                    "agent run ended without producing a final response".to_string(),
                ))
            })
        };
        fold.instrument(agent_span).await
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
mod empty_turn_tests;
#[cfg(test)]
mod entry_tests;
#[cfg(test)]
mod ignore_tests;
#[cfg(test)]
mod settled_tests;
#[cfg(test)]
mod tests;

#[cfg(test)]
pub(crate) mod prompt_tests;
