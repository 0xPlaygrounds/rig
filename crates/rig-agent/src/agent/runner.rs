//! [`AgentRunner`]: the hook-aware driver that turns a sans-IO
//! [`AgentRun`] into a complete agent loop.
//!
//! [`AgentRun`] decides *what* to do next; it
//! performs no IO and carries no hooks. `AgentRunner` pairs that machine with
//! the side-effecting concerns — building and sending completion requests,
//! executing tools, loading/saving conversation memory — and fires an
//! [`AgentHook`] at every observable point. [`Agent::prompt`] and
//! [`Agent::stream_prompt`] both return an `AgentRunner`, and you can build
//! one directly to drive an agent with custom, composable hooks:
//!
//! ```rust,no_run
//! # use rig_agent::Agent;
//! # async fn example(agent: Agent) -> Result<(), Box<dyn std::error::Error>> {
//! let response = agent
//!     .runner("What is 2 + 2?")
//!     .max_turns(3)
//!     .run()
//!     .await?;
//! println!("{}", response.output);
//! # Ok(())
//! # }
//! ```

use std::sync::{Arc, Mutex};

use futures::StreamExt;

use super::{
    ModelHandle,
    completion::{Agent, AgentConfig},
    engine::{DriveItem, UnaryTurnSource, drive_agent, streaming_error_into_prompt},
    hook::AgentHook,
    run::{AgentRun, response::PromptResponse, spec::UnhandledInvalidToolCall},
    telemetry::acquire_agent_span,
};
use rig_core::{memory::ConversationMemory, message::ToolChoice};

use crate::{
    completion::{CompletionError, CompletionModel, Document, Message, PromptError, Usage},
    tool::{ToolContext, server::ToolServerHandle},
};

use super::UNKNOWN_AGENT_NAME;

/// A hook-aware driver over [`AgentRun`].
///
/// Construct one from an [`Agent`] with [`Agent::runner`], attach hooks with
/// [`add_hook`](Self::add_hook), then call
/// [`run`](Self::run) (blocking) or
/// [`stream`](Self::stream)
/// (incremental). Hooks are held in a [`HookStack`](super::hook::HookStack), an ordered,
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
    pub(crate) prompt: Message,
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

/// The `(history_override, memory_handle)` pair resolved for one run by
/// [`AgentRunner::resolve_history_and_memory`].
pub(crate) type HistoryAndMemory = (
    Option<Vec<Message>>,
    Option<(Arc<dyn ConversationMemory>, rig_core::id::ConversationId)>,
);

impl AgentRunner {
    /// Build a runner from an agent, seeding it with the agent's default hook
    /// stack. Prefer [`Agent::runner`].
    pub fn from_agent(agent: &Agent, prompt: impl Into<Message>) -> Self {
        Self {
            config: agent.config.clone(),
            prompt: prompt.into(),
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
    pub fn using_model(mut self, model: ModelHandle) -> Self {
        self.config.model = model;
        self
    }

    /// Erase and set a typed default model for this run.
    pub fn using_model_value<M>(self, model: M) -> Self
    where
        M: CompletionModel + 'static,
    {
        self.using_model(ModelHandle::new(model))
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
        self.config.memory = None;
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

    /// Build the sans-IO [`AgentRun`] for this runner's configuration.
    /// `history_override` replaces the configured chat history (e.g. with
    /// memory-loaded history). Delegates to [`build_agent_run`] — the single
    /// construction site shared with the streaming driver.
    pub(crate) fn build_run(&self, history_override: Option<Vec<Message>>) -> AgentRun {
        let run = build_agent_run(
            self.prompt.clone(),
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
    pub(crate) async fn run_with_error_usage(
        mut self,
    ) -> (Result<PromptResponse, PromptError>, Usage) {
        let usage = Arc::new(Mutex::new(Usage::new()));
        self.error_usage = Some(usage.clone());
        let result = self.run().await;
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
    pub(crate) fn open_agent_span(&self) -> (tracing::Span, bool) {
        let (agent_span, created_agent_span) = acquire_agent_span(
            self.agent_name_or_default(),
            self.config.preamble.as_deref(),
            self.config.record_telemetry_content,
        );

        if self.config.record_telemetry_content
            && let Some(text) = self.prompt.rag_text()
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
    ) -> Result<HistoryAndMemory, rig_core::memory::MemoryError> {
        match &self.chat_history {
            Some(_) => Ok((None, None)),
            None => match (&self.config.memory, &self.config.conversation_id) {
                (Some(memory), Some(id)) => {
                    let loaded = memory.load(id).await?;
                    Ok((Some(loaded), Some((memory.clone(), id.clone()))))
                }
                _ => Ok((None, None)),
            },
        }
    }

    /// Drive the agent loop to completion, returning the aggregated
    /// [`PromptResponse`]. Hooks fire at every observable point; the first hook
    /// to terminate cancels the run.
    pub async fn run(self) -> Result<PromptResponse, PromptError> {
        let (agent_span, created_agent_span) = self.open_agent_span();
        let (history_override, memory_handle) = self.resolve_history_and_memory().await?;
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
            false,
        );
        futures::pin_mut!(driver);

        let mut response = None;
        while let Some(item) = driver.next().await {
            match item {
                Ok(DriveItem::Done(done)) => response = Some(*done),
                Ok(DriveItem::Item(_)) => {}
                Err(err) => return Err(streaming_error_into_prompt(err)),
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
mod tests;

#[cfg(test)]
mod prompt_tests;
