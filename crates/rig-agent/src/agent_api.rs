//! The thin, forward-looking agent: plain data plus inherent methods over
//! the session drivers.
//!
//! [`SessionAgent`] pairs an [`AgentConfig`] with a [`ProviderConfig`], an
//! optional [`ToolExecutor`], and attach-and-forget [`Hooks`], and drives
//! [`AgentSession`] / [`AgentStream`] internally. There are no behavior
//! slots beyond those fields: tools are data ([`ToolCatalog`] +
//! executor records), hooks are data (named callback records folded with
//! the classic semantics), and conversation memory inverts to explicit
//! host calls (see below). It is named `SessionAgent` while the classic
//! [`Agent`](crate::agent::Agent) still exists.
//!
//! ```no_run
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! use rig_agent::agent::AgentConfig;
//! use rig_agent::agent_api::SessionAgent;
//! use rig_agent::provider::ProviderConfig;
//!
//! let agent = SessionAgent::new(
//!     AgentConfig::new().with_preamble("You are terse."),
//!     ProviderConfig::OpenAi(
//!         rig_agent::core::providers::openai::functions::Config::new("gpt-4o"),
//!     ),
//! );
//! println!("{}", agent.prompt("Hello!").await?);
//! # Ok(())
//! # }
//! ```
//!
//! # Conversation memory as host calls
//!
//! The classic driver owned a `ConversationMemory` slot; [`SessionAgent`]
//! deliberately does not. The classic semantics translate to two explicit
//! host calls around a run:
//!
//! - **load-before**: history is loaded from memory before the prompt, and a
//!   load failure is **fatal** (the run never starts);
//! - **append-after**: the finished run's new messages
//!   ([`PromptResponse::messages`]) are appended after a successful run, and
//!   an append failure **warns and proceeds** (the response is still
//!   surfaced);
//! - **explicit history bypasses memory entirely** — a caller that passes
//!   its own history (classic `.history(...)` / [`SessionAgent::chat`] /
//!   [`SessionAgent::run_with_history`]) performs neither the load nor the
//!   append against a memory backend.
//!
//! ```no_run
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! use rig_agent::agent::AgentConfig;
//! use rig_agent::agent_api::SessionAgent;
//! use rig_agent::provider::ProviderConfig;
//! use rig_core::memory::{ConversationMemory, InMemoryConversationMemory};
//!
//! let memory = InMemoryConversationMemory::new();
//! let conversation_id = "user-42";
//! let agent = SessionAgent::new(
//!     AgentConfig::new().with_preamble("You are terse."),
//!     ProviderConfig::OpenAi(
//!         rig_agent::core::providers::openai::functions::Config::new("gpt-4o"),
//!     ),
//! );
//!
//! // Load-before: a load failure is fatal, exactly like the classic driver.
//! let history = memory.load(conversation_id).await?;
//! let response = agent.run_with_history("Hello!", history).await?;
//!
//! // Append-after: warn and proceed on failure, exactly like the classic
//! // driver — the response is still surfaced.
//! if let Some(messages) = &response.messages
//!     && let Err(error) = memory.append(conversation_id, messages.clone()).await
//! {
//!     tracing::warn!(
//!         %error,
//!         conversation_id,
//!         "conversation memory append failed; surfacing final response anyway"
//!     );
//! }
//! println!("{}", response.output);
//! # Ok(())
//! # }
//! ```

use std::sync::Arc;

use rig_core::message::UserContent;

use crate::agent::hook::{InvalidToolCallAction, ModelTurnAction, ObservationAction};
use crate::agent::prepare::ToolCatalog;
use crate::agent::{AgentConfig, PromptResponse};
use crate::completion::{CompletionError, Message, PromptError};
use crate::executor::ToolExecutor;
use crate::hooks::Hooks;
use crate::provider::{ProviderConfig, Runtime};
use crate::session::{AgentSession, SessionEvent, SessionPolicy};
use crate::stream::AgentStream;
use crate::tool::{ToolOutput, ToolResult};

/// The forward-looking concrete agent: plain data plus inherent methods over
/// the session layer. See the [module docs](self).
#[derive(Clone)]
pub struct SessionAgent {
    /// The agent's model-free configuration.
    pub config: AgentConfig,
    /// The provider fulfilling model calls.
    pub provider: ProviderConfig,
    /// The runtime handle model calls execute on.
    pub rt: Arc<Runtime>,
    /// Tool definitions advertised each turn.
    pub tools: ToolCatalog,
    /// Executes [`SessionEvent::ToolCallsReady`] batches when present; with
    /// no executor a run that produces an executable tool call fails.
    pub executor: Option<ToolExecutor>,
    /// Attach-and-forget hooks dispatched at every surfaced event.
    pub hooks: Hooks,
}

impl std::fmt::Debug for SessionAgent {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SessionAgent")
            .field("config", &self.config)
            .field("tools", &self.tools)
            .field("executor", &self.executor)
            .field("hooks", &self.hooks)
            .finish_non_exhaustive()
    }
}

impl SessionAgent {
    /// Create an agent over a fresh default [`Runtime`].
    pub fn new(config: AgentConfig, provider: ProviderConfig) -> Self {
        Self {
            config,
            provider,
            rt: Arc::new(Runtime::new()),
            tools: ToolCatalog::default(),
            executor: None,
            hooks: Hooks::default(),
        }
    }

    /// Use this runtime handle instead of a fresh default one.
    pub fn with_runtime(mut self, rt: Arc<Runtime>) -> Self {
        self.rt = rt;
        self
    }

    /// Advertise these tool definitions each turn (the executor, when
    /// present, answers the calls it knows; unknown names produce the
    /// registry's not-found result).
    pub fn with_tools(mut self, catalog: ToolCatalog) -> Self {
        self.tools = catalog;
        self
    }

    /// Attach a tool executor. When no catalog was set explicitly, the
    /// executor's own [`ToolExecutor::catalog`] is adopted so the agent
    /// advertises exactly what it can run.
    pub fn with_executor(mut self, executor: ToolExecutor) -> Self {
        if self.tools == ToolCatalog::default() {
            self.tools = executor.catalog();
        }
        self.executor = Some(executor);
        self
    }

    /// Attach hooks, dispatched at every surfaced event with the classic
    /// fold semantics (see [`Hooks`]).
    pub fn with_hooks(mut self, hooks: Hooks) -> Self {
        self.hooks = hooks;
        self
    }

    /// The [`SessionPolicy`] this agent drives sessions under: the default
    /// policy when no hooks are attached; with hooks, every decision point
    /// is surfaced so each one can be dispatched through the hook list.
    fn session_policy(&self) -> SessionPolicy {
        if self.hooks.is_empty() {
            SessionPolicy::default()
        } else {
            SessionPolicy {
                surface_model_turns: true,
                surface_completion_calls: true,
                surface_tool_calls: true,
                surface_tool_results: true,
            }
        }
    }

    /// Build the configured session for one prompt.
    fn session(&self, prompt: impl Into<Message>, history: Vec<Message>) -> AgentSession {
        let session = AgentSession::new(
            self.config.clone(),
            self.provider.clone(),
            self.rt.clone(),
            prompt,
        )
        .with_tools(self.tools.clone())
        .with_policy(self.session_policy());
        if history.is_empty() {
            session
        } else {
            session.with_history(history)
        }
    }

    /// Run one prompt to completion, dispatching [`Hooks`] at every surfaced
    /// event and answering tool batches through the executor when present.
    ///
    /// # Errors
    /// Provider/protocol errors; a hook `Stop` cancels the run; an
    /// executable tool call with no executor attached fails the run.
    pub async fn run(&self, prompt: impl Into<Message>) -> Result<PromptResponse, PromptError> {
        self.run_with_history(prompt, Vec::new()).await
    }

    /// [`SessionAgent::run`] with explicit input history preceding the
    /// prompt. Explicit history means conversation memory (a host concern —
    /// see the [module docs](self)) is bypassed entirely.
    pub async fn run_with_history(
        &self,
        prompt: impl Into<Message>,
        history: Vec<Message>,
    ) -> Result<PromptResponse, PromptError> {
        let mut session = self.session(prompt, history);
        loop {
            match session.advance().await? {
                SessionEvent::BeforeModelCall {
                    turn,
                    prompt,
                    history,
                } => {
                    let action = self
                        .hooks
                        .dispatch_completion_call(turn, &prompt, &history)
                        .await;
                    session.reply_before_call(action)?;
                }
                SessionEvent::TurnFinished {
                    turn,
                    content,
                    usage,
                } => {
                    // Observation over the full provider response first
                    // (classic order: the response hook fires before the
                    // model-turn verdict).
                    if let Some(response) = session.last_response().cloned()
                        && let ObservationAction::Stop(reason) = self
                            .hooks
                            .dispatch_completion_response(turn, &response)
                            .await
                    {
                        session.reply_turn(ModelTurnAction::Stop(reason))?;
                        continue;
                    }
                    let action = self.hooks.dispatch_model_turn(turn, &content, usage).await;
                    session.reply_turn(action)?;
                }
                SessionEvent::InvalidToolCall(context) => {
                    let action = self
                        .hooks
                        .dispatch_invalid_tool_call(&context)
                        .await
                        // Preserve the classic default: fail fast.
                        .unwrap_or_else(InvalidToolCallAction::fail);
                    session.resolve_invalid(action)?;
                }
                SessionEvent::ToolCallPending { call } => {
                    // The effective action already carries any chained
                    // rewrite; a terminal `Skip`'s salvaged rewrite cannot be
                    // applied through the single-action inbox and only
                    // affects reporting, never execution (the body does not
                    // run).
                    let (action, _salvaged) = self.hooks.dispatch_tool_call(&call.tool_call).await;
                    session.reply_tool_call(action)?;
                }
                SessionEvent::ToolCallsReady(calls) => {
                    let results = match &self.executor {
                        Some(executor) => executor.execute_batch(&calls).await.results,
                        None => {
                            // Preresolved-only batches (hook skips, invalid
                            // call recovery) pass through; an executable call
                            // with no executor is a host configuration error.
                            let mut results = Vec::with_capacity(calls.len());
                            for call in &calls {
                                match &call.preresolved_result {
                                    Some(content) => results.push(content.clone()),
                                    None => {
                                        return Err(PromptError::CompletionError(
                                            CompletionError::ResponseError(
                                                "model called a tool but this SessionAgent has \
                                                 no executor; attach one with with_executor"
                                                    .to_string(),
                                            ),
                                        ));
                                    }
                                }
                            }
                            results
                        }
                    };
                    session.provide_tool_results(results)?;
                }
                SessionEvent::ToolResultReady { call, result } => {
                    let raw = raw_tool_result(&result);
                    let action = self.hooks.dispatch_tool_result(&call, &raw).await;
                    session.reply_tool_result(action)?;
                }
                SessionEvent::Done(response) => return Ok(response),
            }
        }
    }

    /// Run one prompt and return the final output text.
    ///
    /// # Errors
    /// As [`SessionAgent::run`].
    pub async fn prompt(&self, prompt: impl Into<Message>) -> Result<String, PromptError> {
        Ok(self.run(prompt).await?.output)
    }

    /// Run one prompt against the caller's history and extend it with the
    /// run's new messages, mirroring the classic
    /// [`Chat`](crate::completion::Chat) implementation. Explicit history
    /// means conversation memory is bypassed (see the [module docs](self)).
    ///
    /// # Errors
    /// As [`SessionAgent::run`]; on error the caller's history is unchanged.
    pub async fn chat(
        &self,
        prompt: impl Into<Message>,
        chat_history: &mut Vec<Message>,
    ) -> Result<String, PromptError> {
        let response = self.run_with_history(prompt, chat_history.clone()).await?;
        if let Some(messages) = response.messages {
            chat_history.extend(messages);
        }
        Ok(response.output)
    }

    /// Build a pre-configured [`AgentStream`] for one prompt: this agent's
    /// tools and session policy are already applied.
    ///
    /// The streaming surface stays host-driven: pull items with
    /// [`AgentStream::next_item_with_tools`] to answer
    /// tool batches through this agent's executor, and dispatch [`Hooks`]
    /// per surfaced item via the event-specific dispatchers
    /// ([`Hooks::dispatch_completion_call`],
    /// [`Hooks::dispatch_model_turn`], [`Hooks::dispatch_tool_call`],
    /// [`Hooks::dispatch_tool_result`],
    /// [`Hooks::dispatch_invalid_tool_call`], and
    /// [`Hooks::dispatch_stream_finish`] for the terminal record).
    pub fn stream(&self, prompt: impl Into<Message>) -> AgentStream {
        AgentStream::new(
            self.config.clone(),
            self.provider.clone(),
            self.rt.clone(),
            prompt,
        )
        .with_tools(self.tools.clone())
        .with_policy(self.session_policy())
    }
}

/// Reconstruct the raw-result view the tool-result hook receives from the
/// committed model-visible content: tool-result content blocks map back onto
/// a successful [`ToolResult`] verbatim; any other content is rendered as
/// text.
fn raw_tool_result(result: &UserContent) -> ToolResult {
    match result {
        UserContent::ToolResult(tool_result) => {
            ToolResult::success(ToolOutput::content(tool_result.content.clone()))
        }
        other => ToolResult::success(ToolOutput::text(
            serde_json::to_string(other).unwrap_or_default(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::hook::{CompletionCallAction, RequestPatch, ToolResultAction};
    use crate::completion::Chat;
    use crate::hooks::{HookDecision, HookEntry, HookEvent};
    use crate::provider::MockScript;
    use crate::tool::PortableDynamicTool;
    use rig_core::OneOrMany;
    use rig_core::completion::{CompletionResponse, FinishReason, Usage};
    use rig_core::message::AssistantContent;

    fn usage(total: u64) -> Usage {
        let mut usage = Usage::new();
        usage.total_tokens = total;
        usage
    }

    fn text_response(text: &str) -> CompletionResponse {
        CompletionResponse::new(
            OneOrMany::one(AssistantContent::text(text)),
            usage(5),
            "mock",
        )
        .with_finish_reason(FinishReason::Stop)
    }

    fn tool_call_response(id: &str, name: &str, args: serde_json::Value) -> CompletionResponse {
        CompletionResponse::new(
            OneOrMany::one(AssistantContent::tool_call(id, name, args)),
            usage(3),
            "mock",
        )
        .with_finish_reason(FinishReason::ToolCalls)
    }

    fn adder() -> PortableDynamicTool {
        PortableDynamicTool::new(
            "add",
            "Adds two numbers",
            serde_json::json!({
                "type": "object",
                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                "required": ["a", "b"]
            }),
            |args| {
                Box::pin(async move {
                    let a = args
                        .get("a")
                        .and_then(serde_json::Value::as_i64)
                        .unwrap_or(0);
                    let b = args
                        .get("b")
                        .and_then(serde_json::Value::as_i64)
                        .unwrap_or(0);
                    Ok(ToolOutput::json(serde_json::json!(a + b)))
                })
            },
        )
    }

    fn entry(
        name: &str,
        decide: impl Fn(HookEvent) -> HookDecision + Send + Sync + 'static,
    ) -> HookEntry {
        HookEntry::new(name, move |event| {
            let decision = decide(event);
            Box::pin(async move { decision })
        })
    }

    #[tokio::test]
    async fn prompt_round_trips_a_text_run() {
        let script = MockScript::from_responses(vec![text_response("hello there")]);
        let agent = SessionAgent::new(AgentConfig::new(), ProviderConfig::Mock(script));
        let output = agent.prompt("hi").await.expect("prompt");
        assert_eq!(output, "hello there");
    }

    #[tokio::test]
    async fn run_reports_usage_and_messages() {
        let script = MockScript::from_responses(vec![text_response("ok")]);
        let agent = SessionAgent::new(AgentConfig::new(), ProviderConfig::Mock(script));
        let response = agent.run("hi").await.expect("run");
        assert_eq!(response.output, "ok");
        assert_eq!(response.usage.total_tokens, 5);
        let messages = response.messages.expect("messages recorded");
        assert_eq!(messages.len(), 2); // prompt + assistant turn
    }

    #[tokio::test]
    async fn executor_drives_a_tool_loop_to_completion() {
        let mut config = AgentConfig::new();
        config.max_turns = Some(3);
        let script = MockScript::from_responses(vec![
            tool_call_response("call_1", "add", serde_json::json!({"a": 1, "b": 2})),
            text_response("the sum is 3"),
        ]);
        let probe = script.clone();
        let agent = SessionAgent::new(config, ProviderConfig::Mock(script))
            .with_executor(ToolExecutor::new().register(adder()));

        let response = agent.run("what is 1 + 2?").await.expect("run");
        assert_eq!(response.output, "the sum is 3");
        assert_eq!(response.completion_calls.len(), 2);
        // The second request carried the executed tool result.
        let second =
            serde_json::to_string(&probe.requests().get(1).expect("two calls")).expect("serialize");
        assert!(second.contains('3'), "got {second}");
    }

    #[tokio::test]
    async fn run_without_executor_fails_on_an_executable_tool_call() {
        let mut config = AgentConfig::new();
        config.max_turns = Some(3);
        let script = MockScript::from_responses(vec![tool_call_response(
            "call_1",
            "add",
            serde_json::json!({"a": 1, "b": 2}),
        )]);
        let agent = SessionAgent::new(config, ProviderConfig::Mock(script))
            .with_tools(ToolExecutor::new().register(adder()).catalog());
        let error = agent.run("add").await.expect_err("no executor");
        assert!(error.to_string().contains("no executor"), "got {error}");
    }

    #[tokio::test]
    async fn hook_patch_is_visible_in_the_provider_request() {
        let script = MockScript::from_responses(vec![text_response("patched run")]);
        let probe = script.clone();
        let hooks = Hooks::new().with(entry("patcher", |event| match event {
            HookEvent::BeforeModelCall { .. } => HookDecision::CompletionCall(
                CompletionCallAction::patch(RequestPatch::new().temperature(0.42)),
            ),
            _ => HookDecision::Continue,
        }));
        let agent =
            SessionAgent::new(AgentConfig::new(), ProviderConfig::Mock(script)).with_hooks(hooks);

        let output = agent.prompt("hi").await.expect("prompt");
        assert_eq!(output, "patched run");
        let request = probe.requests().first().cloned().expect("one request");
        assert_eq!(request.temperature, Some(0.42));
    }

    #[tokio::test]
    async fn hook_tool_result_rewrite_reaches_the_committed_history() {
        let mut config = AgentConfig::new();
        config.max_turns = Some(3);
        let script = MockScript::from_responses(vec![
            tool_call_response("call_1", "add", serde_json::json!({"a": 1, "b": 2})),
            text_response("done"),
        ]);
        let probe = script.clone();
        let hooks = Hooks::new().with(entry("redactor", |event| match event {
            HookEvent::ToolResult { .. } => {
                HookDecision::ToolResult(ToolResultAction::rewrite("redacted"))
            }
            _ => HookDecision::Continue,
        }));
        let agent = SessionAgent::new(config, ProviderConfig::Mock(script))
            .with_executor(ToolExecutor::new().register(adder()))
            .with_hooks(hooks);

        let response = agent.run("what is 1 + 2?").await.expect("run");
        assert_eq!(response.output, "done");
        // The second request carries the rewritten presentation, not the raw
        // tool output.
        let second =
            serde_json::to_string(&probe.requests().get(1).expect("two calls")).expect("serialize");
        assert!(second.contains("redacted"), "got {second}");
    }

    #[tokio::test]
    async fn chat_extends_history_in_parity_with_the_classic_agent() {
        use crate::agent::AgentBuilder;
        use crate::agent::runner::test_support::{MockCompletionModel, MockTurn};

        // Two identical scripts (a MockScript clone shares its cursor).
        let session_script = MockScript::from_responses(vec![text_response("nice to meet you")]);
        let classic_model = MockCompletionModel::from_turns([MockTurn::text("nice to meet you")]);

        let agent = SessionAgent::new(AgentConfig::new(), ProviderConfig::Mock(session_script));
        let mut session_history = vec![Message::user("earlier"), Message::assistant("noted")];
        let session_output = agent
            .chat("hello", &mut session_history)
            .await
            .expect("session chat");

        let classic = AgentBuilder::new(classic_model.provider()).build();
        let mut classic_history = vec![Message::user("earlier"), Message::assistant("noted")];
        let classic_output = classic
            .chat("hello", &mut classic_history)
            .await
            .expect("classic chat");

        assert_eq!(session_output, classic_output);
        assert_eq!(
            serde_json::to_value(&session_history).expect("serialize"),
            serde_json::to_value(&classic_history).expect("serialize"),
        );
    }

    #[tokio::test]
    async fn stream_is_preconfigured_with_tools_and_policy() {
        let script = MockScript::from_responses(vec![]);
        let agent = SessionAgent::new(AgentConfig::new(), ProviderConfig::Mock(script))
            .with_executor(ToolExecutor::new().register(adder()))
            .with_hooks(Hooks::new().with(entry("noop", |_| HookDecision::Continue)));
        let stream = agent.stream("hi");
        assert!(stream.tools.executable.contains("add"));
        assert!(stream.policy.surface_completion_calls);
        assert!(stream.policy.surface_tool_calls);
    }

    /// Telemetry smoke: a `SessionAgent` run creates the classic
    /// `invoke_agent` and per-call `chat` spans.
    #[tokio::test]
    async fn run_emits_invoke_agent_and_chat_spans() {
        use std::sync::{Arc as StdArc, Mutex};
        use tracing_subscriber::layer::SubscriberExt;

        #[derive(Default)]
        struct SpanNames(StdArc<Mutex<Vec<String>>>);
        impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for SpanNames {
            fn on_new_span(
                &self,
                attrs: &tracing::span::Attributes<'_>,
                _id: &tracing::span::Id,
                _ctx: tracing_subscriber::layer::Context<'_, S>,
            ) {
                self.0
                    .lock()
                    .expect("names")
                    .push(attrs.metadata().name().to_string());
            }
        }

        let names = StdArc::new(Mutex::new(Vec::new()));
        let layer = SpanNames(names.clone());
        let subscriber = tracing_subscriber::registry().with(layer);
        let _guard = tracing::subscriber::set_default(subscriber);
        // Parallel tests without a subscriber can cache `Interest::never`
        // for these callsites (see the runner span tests' warm-up); force a
        // re-registration against the scoped subscriber before running.
        tracing::callsite::rebuild_interest_cache();

        let script = MockScript::from_responses(vec![text_response("traced")]);
        let mut config = AgentConfig::new();
        config.record_telemetry_content = true;
        let agent = SessionAgent::new(config, ProviderConfig::Mock(script));
        let output = agent.prompt("hi").await.expect("prompt");
        assert_eq!(output, "traced");

        let names = names.lock().expect("names").clone();
        assert!(
            names.iter().any(|name| name == "invoke_agent"),
            "expected an invoke_agent span, got {names:?}"
        );
        assert!(
            names.iter().any(|name| name == "chat"),
            "expected a chat span, got {names:?}"
        );
    }
}
