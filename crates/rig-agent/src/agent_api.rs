//! Host-owned conversation memory: the recipe, and the transitional
//! `SessionAgent` alias.
//!
//! The forward-looking session agent this module introduced in R1 **is** now
//! [`Agent`](crate::Agent) — the two types merged, so `SessionAgent` is only a
//! deprecated alias. What remains genuinely documented here is the memory
//! recipe every agent method assumes.
//!
//! # Conversation memory as host calls
//!
//! No agent in this crate owns a memory slot. Memory is host-owned data, and
//! the semantics the classic driver used to apply internally translate to two
//! explicit host calls around a run:
//!
//! - **load-before**: history is loaded from memory before the prompt, and a
//!   load failure is **fatal** (the run never starts). The classic driver
//!   loaded only when a memory backend *and* a conversation id were both set;
//!   with a host call, "no id" simply means "don't call load";
//! - **append-after**: the finished run's new messages
//!   ([`PromptResponse::messages`](crate::agent::PromptResponse::messages)) are appended in **one** call after a
//!   successful run — one append per run, not per model call, so a multi-step
//!   tool round-trip persists its committed transcript exactly once. An
//!   append failure **warns and proceeds** (the response is still surfaced),
//!   and a run that errors or is stopped by a hook never reaches the append;
//! - **explicit history bypasses memory entirely** — a caller that passes its
//!   own history (`.runner(p).history(...)` / [`Agent::chat`](crate::Agent::chat) /
//!   [`Agent::run_with_history`](crate::Agent::run_with_history)) performs neither the load nor the
//!   append. There is no `without_memory()` switch any more: not calling the
//!   store *is* the switch.
//!
//! The conversation id is a flat string key chosen by the host, and nothing
//! ever clears a conversation on your behalf — call the store's `clear` when
//! a thread ends.
//!
//! ```no_run
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! use rig_agent::agent::AgentConfig;
//! use rig_agent::Agent;
//! use rig_agent::provider::ProviderConfig;
//! use rig_core::memory::InMemoryConversationMemory;
//!
//! let memory = InMemoryConversationMemory::new();
//! let conversation_id = "user-42";
//! let agent = Agent::new(
//!     AgentConfig::new().with_preamble("You are terse."),
//!     ProviderConfig::OpenAi(
//!         rig_agent::core::providers::openai::functions::Config::new("gpt-4o"),
//!     ),
//! );
//!
//! // Load-before: a load failure is fatal, exactly like the classic driver.
//! let history = memory.load(conversation_id)?;
//! let response = agent.run_with_history("Hello!", history).await?;
//!
//! // Append-after: warn and proceed on failure, exactly like the classic
//! // driver — the response is still surfaced.
//! if let Some(messages) = &response.messages
//!     && let Err(error) = memory.append(conversation_id, messages.clone())
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
//!
//! Streaming is the same recipe: load before `Agent::stream_prompt` (the
//! loaded history goes into the session's history) and append the final
//! response's `messages` once the stream ends. A load
//! failure that used to arrive as a one-item error stream is now just an
//! error returned before the stream is created.
//!
//! # Shaping history
//!
//! History-shaping policies live in the `rig-memory` companion crate as data:
//! `MemoryPolicy` (sliding window, token window), `Compactor` (rolling
//! summaries), and a concrete `PolicyMemory` whose `append` returns an
//! `AppendOutcome { stored, demoted, compaction }`. The host acts on
//! `demoted` (archive it) and `compaction` (fold it into a summary) instead
//! of a hook firing behind the run.

/// The single agent type, re-exported under its transitional name.
///
/// `SessionAgent` and the classic `Agent` merged in the single-architecture
/// migration: there is exactly one agent type now, and this alias only exists
/// so code written against the R1 session agent keeps compiling. Use
/// [`Agent`](crate::Agent).
#[deprecated(
    since = "0.22.0",
    note = "`SessionAgent` and `Agent` merged into one type; use `rig_agent::Agent`"
)]
pub type SessionAgent = crate::agent::Agent;

#[cfg(test)]
mod tests {
    use crate::agent::hook::{CompletionCallAction, RequestPatch, ToolResultAction};
    use crate::agent::{Agent, AgentConfig};
    use crate::completion::Message;
    use crate::executor::ToolExecutor;
    use crate::hooks::Hooks;
    use crate::hooks::{HookDecision, HookEntry, HookEvent};
    use crate::provider::MockScript;
    use crate::provider::ProviderConfig;
    use crate::test_utils::{AppendFailingMemory, CountingMemory, FailingMemory};
    use crate::tool::PortableDynamicTool;
    use crate::tool::ToolOutput;
    use rig_core::OneOrMany;
    use rig_core::completion::{CompletionResponse, FinishReason, Usage};
    use rig_core::message::AssistantContent;
    use rig_core::message::UserContent;

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
        let agent = Agent::new(AgentConfig::new(), ProviderConfig::Mock(script));
        let output = agent.prompt("hi").await.expect("prompt");
        assert_eq!(output, "hello there");
    }

    #[tokio::test]
    async fn run_reports_usage_and_messages() {
        let script = MockScript::from_responses(vec![text_response("ok")]);
        let agent = Agent::new(AgentConfig::new(), ProviderConfig::Mock(script));
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
        let agent = Agent::new(config, ProviderConfig::Mock(script))
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
        let agent = Agent::new(config, ProviderConfig::Mock(script))
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
        let agent = Agent::new(AgentConfig::new(), ProviderConfig::Mock(script)).with_hooks(hooks);

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
        let agent = Agent::new(config, ProviderConfig::Mock(script))
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
    async fn chat_extends_the_callers_history_with_the_committed_turn() {
        let script = MockScript::from_responses(vec![text_response("nice to meet you")]);
        let agent = Agent::new(AgentConfig::new(), ProviderConfig::Mock(script));
        let mut history = vec![Message::user("earlier"), Message::assistant("noted")];
        let output = agent.chat("hello", &mut history).await.expect("chat");

        assert_eq!(output, "nice to meet you");
        // The caller's history gained exactly the run's committed transcript:
        // the prompt and the assistant turn.
        assert_eq!(history.len(), 4);
        assert!(matches!(
            history.get(2),
            Some(Message::User { content })
                if matches!(content.first(), UserContent::Text(t) if t.text == "hello")
        ));
    }

    #[tokio::test]
    async fn stream_is_preconfigured_with_tools_and_policy() {
        let script = MockScript::from_responses(vec![]);
        let agent = Agent::new(AgentConfig::new(), ProviderConfig::Mock(script))
            .with_executor(ToolExecutor::new().register(adder()))
            .with_hooks(Hooks::new().with(entry("noop", |_| HookDecision::Continue)));
        let stream = agent.stream_prompt("hi");
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

        // Subscriber-scoped assertions serialize against each other, and a
        // parallel test without a subscriber can otherwise cache
        // `Interest::never` for these callsites.
        let _isolation = crate::test_utils::scoped_tracing_subscriber_guard().await;
        let names = StdArc::new(Mutex::new(Vec::new()));
        let layer = SpanNames(names.clone());
        let subscriber = tracing_subscriber::registry().with(layer);
        let _guard = tracing::subscriber::set_default(subscriber);

        // Warm up: register both span callsites against this subscriber,
        // then drop what the warm-up recorded.
        let warmup = Agent::new(
            AgentConfig::new(),
            ProviderConfig::Mock(MockScript::from_responses(vec![text_response("warm")])),
        );
        let _ = warmup.prompt("warm").await;
        tracing::callsite::rebuild_interest_cache();
        names.lock().expect("names").clear();

        let script = MockScript::from_responses(vec![text_response("traced")]);
        let mut config = AgentConfig::new();
        config.record_telemetry_content = true;
        let agent = Agent::new(config, ProviderConfig::Mock(script));
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

    // ----- host-owned conversation memory recipe -----
    //
    // These replace the classic driver's built-in memory orchestration tests.
    // The behaviors they pin (load feeds the request, one append per run, an
    // append failure never drops the response, a failed or stopped run never
    // appends, explicit history bypasses the store) are now properties of the
    // documented host recipe rather than of the agent.

    #[tokio::test]
    async fn host_recipe_load_before_feeds_the_request_and_append_after_persists_the_turn() {
        let memory = CountingMemory::default();
        memory
            .inner()
            .append(
                "t1",
                vec![Message::user("old-q"), Message::assistant("old-a")],
            )
            .expect("seed");

        let script = MockScript::from_responses(vec![text_response("new-a")]);
        let probe = script.clone();
        let agent = Agent::new(AgentConfig::new(), ProviderConfig::Mock(script));

        let history = memory.load("t1").expect("load");
        let response = agent.run_with_history("new-q", history).await.expect("run");
        let messages = response.messages.clone().expect("committed transcript");
        memory.append("t1", messages).expect("append");

        let sent = probe
            .requests()
            .first()
            .expect("one call")
            .chat_history
            .len();
        assert_eq!(sent, 3, "loaded history (2) + current prompt");
        assert_eq!(memory.load_count(), 1, "one load for the run");
        assert_eq!(memory.append_count(), 1, "one append for the run");

        let stored = memory.inner().load("t1").expect("stored");
        assert_eq!(
            stored.len(),
            4,
            "only the new turn is appended; loaded history is not duplicated: {stored:?}"
        );
    }

    #[tokio::test]
    async fn host_recipe_appends_a_multi_step_tool_run_exactly_once() {
        let memory = CountingMemory::default();
        let mut config = AgentConfig::new();
        config.max_turns = Some(3);
        let script = MockScript::from_responses(vec![
            tool_call_response("call_1", "add", serde_json::json!({"a": 2, "b": 3})),
            text_response("sum is 5"),
        ]);
        let agent = Agent::new(config, ProviderConfig::Mock(script))
            .with_executor(ToolExecutor::new().register(adder()));

        let response = agent.run("add 2 and 3").await.expect("run");
        let messages = response.messages.clone().expect("committed transcript");
        memory.append("t1", messages).expect("append");

        assert_eq!(
            memory.append_count(),
            1,
            "one append for the whole run, not one per model call"
        );
        let stored = memory.inner().load("t1").expect("stored");
        // user prompt + assistant tool call + tool result + final assistant text.
        assert_eq!(
            stored.len(),
            4,
            "the full committed turn is persisted once: {stored:?}"
        );
        assert!(
            matches!(
                stored.last(),
                Some(Message::Assistant { content, .. })
                    if content.iter().any(|item|
                        matches!(item, AssistantContent::Text(t) if t.text == "sum is 5"))
            ),
            "final assistant text is persisted: {stored:?}"
        );
        // The committed transcript is a well-formed role sequence: it starts
        // with the user prompt, never commits two assistant turns back to
        // back, and pairs each tool call with a following tool result.
        assert!(matches!(stored.first(), Some(Message::User { .. })));
        assert!(
            !stored
                .windows(2)
                .any(|pair| matches!(pair, [Message::Assistant { .. }, Message::Assistant { .. }])),
            "no two assistant messages back to back: {stored:?}"
        );
        for (index, message) in stored.iter().enumerate() {
            let has_tool_call = matches!(
                message,
                Message::Assistant { content, .. }
                    if content.iter().any(|item| matches!(item, AssistantContent::ToolCall(_)))
            );
            if has_tool_call {
                assert!(
                    matches!(stored.get(index + 1), Some(Message::User { content })
                        if content.iter().any(|item| matches!(item, UserContent::ToolResult(_)))),
                    "assistant tool call at {index} is followed by a tool result: {stored:?}"
                );
            }
        }
    }

    #[tokio::test]
    async fn host_recipe_append_failure_does_not_drop_the_response() {
        let memory = AppendFailingMemory::default();
        let script = MockScript::from_responses(vec![text_response("ack")]);
        let agent = Agent::new(AgentConfig::new(), ProviderConfig::Mock(script));

        let history = memory.load("t1").expect("load");
        let response = agent.run_with_history("hello", history).await.expect("run");
        let append = memory.append("t1", response.messages.clone().unwrap_or_default());

        assert!(append.is_err(), "the store fails on append");
        assert_eq!(
            response.output, "ack",
            "warn-and-proceed: the response is still surfaced"
        );
    }

    #[tokio::test]
    async fn host_recipe_load_failure_is_fatal_before_the_run_starts() {
        let memory = FailingMemory::default();
        let script = MockScript::from_responses(vec![text_response("unreached")]);
        let probe = script.clone();
        let agent = Agent::new(AgentConfig::new(), ProviderConfig::Mock(script));

        let error = memory.load("t1").expect_err("load fails");
        assert!(error.to_string().contains("load boom"), "got {error}");
        // The host never reaches the run, so the provider is never called.
        assert!(probe.requests().is_empty());
        let _ = agent;
    }

    #[tokio::test]
    async fn host_recipe_failed_run_appends_nothing() {
        let memory = CountingMemory::default();
        // An exhausted script errors on the first call, standing in for any
        // provider failure.
        let script = MockScript::from_responses(vec![]);
        let agent = Agent::new(AgentConfig::new(), ProviderConfig::Mock(script));

        let history = memory.load("t1").expect("load");
        let result = agent.run_with_history("hello", history).await;
        assert!(result.is_err(), "the provider failed");
        // The append only happens on the success path of the recipe.
        assert_eq!(memory.append_count(), 0, "failed runs do not append");
        assert!(memory.inner().load("t1").expect("stored").is_empty());
    }

    #[tokio::test]
    async fn host_recipe_stopped_run_appends_nothing() {
        let memory = CountingMemory::default();
        let hooks = Hooks::new().with(entry("stop-on-completion", |event| match event {
            HookEvent::BeforeModelCall { .. } => {
                HookDecision::CompletionCall(CompletionCallAction::stop("stop"))
            }
            _ => HookDecision::Continue,
        }));
        let script = MockScript::from_responses(vec![text_response("unreached")]);
        let agent = Agent::new(AgentConfig::new(), ProviderConfig::Mock(script)).with_hooks(hooks);

        let history = memory.load("t1").expect("load");
        let result = agent.run_with_history("hello", history).await;
        assert!(result.is_err(), "a stop hook terminates the run");
        assert_eq!(memory.append_count(), 0, "stopped runs do not append");
        assert!(memory.inner().load("t1").expect("stored").is_empty());
    }

    #[tokio::test]
    async fn host_recipe_explicit_history_bypasses_the_store() {
        // Passing caller history is how memory is bypassed: the host simply
        // performs neither the load nor the append.
        let memory = CountingMemory::default();
        memory
            .inner()
            .append("t1", vec![Message::user("from-memory")])
            .expect("seed");

        let script = MockScript::from_responses(vec![text_response("ack")]);
        let probe = script.clone();
        let agent = Agent::new(AgentConfig::new(), ProviderConfig::Mock(script));

        let response = agent
            .run_with_history("hello", vec![Message::user("from-caller")])
            .await
            .expect("run");
        assert_eq!(response.output, "ack");

        assert_eq!(memory.load_count(), 0, "load skipped");
        assert_eq!(memory.append_count(), 0, "append skipped");
        let sent = probe
            .requests()
            .first()
            .expect("one call")
            .chat_history
            .clone();
        assert_eq!(sent.len(), 2, "caller history (1) + current prompt");
        assert!(matches!(
            sent.first(),
            Message::User { content }
                if matches!(content.first(), UserContent::Text(t) if t.text == "from-caller")
        ));
    }
}
