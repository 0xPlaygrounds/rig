//! [`SessionRunner`]: the fluent per-request surface over the session layer.
//!
//! An [`Agent`] holds the defaults; a `SessionRunner` is one prompt plus the
//! per-request overrides applied on top of them. It is the successor of the
//! deleted `AgentRunner`/`StreamingPromptRequest` pair — same setters, but it
//! drives [`crate::session::AgentSession`] / [`crate::stream::AgentStream`]
//! instead of a second engine.
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

use std::sync::Arc;

use rig_core::message::ToolChoice;

use super::prepare::ToolCatalog;
use super::response::PromptResponse;
use super::run::OutputMode;
use super::{Agent, AgentConfig};
use crate::completion::{Document, Message, PromptError, StructuredOutputError};
use crate::executor::ToolExecutor;
use crate::hooks::{HookEntry, Hooks};
use crate::provider::{ProviderConfig, Runtime};
use crate::session::AgentSession;
use crate::stream::{AgentRunStream, AgentStream};

/// One configured agent request. See the [module docs](self).
#[non_exhaustive]
pub struct SessionRunner {
    /// This request's effective configuration (the agent's, plus overrides).
    pub config: AgentConfig,
    /// The provider fulfilling model calls.
    pub provider: ProviderConfig,
    /// Live transport handles shared with the owning agent.
    pub rt: Arc<Runtime>,
    /// Tool definitions advertised each turn.
    pub tools: ToolCatalog,
    /// Executes the model's tool calls.
    pub executor: Option<ToolExecutor>,
    /// Hooks dispatched at every surfaced event.
    pub hooks: Hooks,
    /// The prompt this request sends.
    pub prompt: Message,
    /// History preceding the prompt.
    pub history: Option<Vec<Message>>,
}

impl SessionRunner {
    /// Build a request from an agent, seeding it with the agent's
    /// configuration and hooks. Prefer [`Agent::runner`].
    pub fn from_agent(agent: &Agent, prompt: impl Into<Message>) -> Self {
        Self {
            config: agent.config.clone(),
            provider: agent.provider.clone(),
            rt: agent.rt.clone(),
            tools: agent.tools.clone(),
            executor: agent.executor.clone(),
            hooks: agent.hooks.clone(),
            prompt: prompt.into(),
            history: None,
        }
    }

    /// Append a hook to the stack (on top of any the agent already carries).
    /// Hooks run in registration order; how their results compose is
    /// event-dependent (`CompletionCall` request patches accumulate and
    /// merge, `ToolCall`/`ToolResult` rewrites chain, while model-turn
    /// steering and observe-only/recovery events use their event-specific
    /// terminal action). See the [`hook`](crate::agent::hook) module docs.
    pub fn add_hook(mut self, hook: HookEntry) -> Self {
        self.hooks.add(hook);
        self
    }

    /// Set the total model-call budget, including the initial call and every
    /// retry or continuation. Zero emits no model calls; one permits only the
    /// initial call. Exceeding the budget returns
    /// [`PromptError::MaxTurnsError`].
    pub fn max_turns(mut self, max_turns: usize) -> Self {
        self.config.max_turns = Some(max_turns);
        self
    }

    /// Set the chat history preceding the prompt.
    pub fn history<I, T>(mut self, history: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Into<Message>,
    {
        self.history = Some(history.into_iter().map(Into::into).collect());
        self
    }

    /// Override the agent preamble for this request.
    pub fn preamble(mut self, preamble: impl Into<String>) -> Self {
        self.config.preamble = Some(preamble.into());
        self
    }

    /// Remove the agent's configured preamble for this request.
    pub fn without_preamble(mut self) -> Self {
        self.config.preamble = None;
        self
    }

    /// Append one static context document for this request.
    pub fn document(mut self, document: Document) -> Self {
        self.config.static_context.push(document);
        self
    }

    /// Append static context documents for this request.
    pub fn documents(mut self, documents: impl IntoIterator<Item = Document>) -> Self {
        self.config.static_context.extend(documents);
        self
    }

    /// Override the model temperature for this request.
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.config.temperature = Some(temperature);
        self
    }

    /// Remove the agent's configured temperature for this request.
    pub fn without_temperature(mut self) -> Self {
        self.config.temperature = None;
        self
    }

    /// Override the maximum completion token count for this request.
    pub fn max_tokens(mut self, max_tokens: u64) -> Self {
        self.config.max_tokens = Some(max_tokens);
        self
    }

    /// Remove the agent's configured maximum token count for this request.
    pub fn without_max_tokens(mut self) -> Self {
        self.config.max_tokens = None;
        self
    }

    /// Shallow-merge object fields into the provider-specific parameters for
    /// this request. Later fields win. A non-object baseline is replaced by
    /// the supplied object. A later completion-call hook patch has final
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

    /// Replace all provider-specific parameters for this request.
    pub fn replace_additional_params(mut self, params: serde_json::Value) -> Self {
        self.config.additional_params = Some(params);
        self
    }

    /// Remove the agent's configured provider-specific parameters for this
    /// request. A later completion-call hook may still supply its own.
    pub fn without_additional_params(mut self) -> Self {
        self.config.additional_params = None;
        self
    }

    /// Override the tool-choice policy for this request.
    pub fn tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.config.tool_choice = Some(tool_choice);
        self
    }

    /// Remove the agent's configured tool-choice policy for this request.
    pub fn without_tool_choice(mut self) -> Self {
        self.config.tool_choice = None;
        self
    }

    /// Pin the synthetic output tool used by [`OutputMode::Tool`] structured
    /// output: its advertised `name` and `description`, and whether Tool mode
    /// augments the preamble with calling instructions.
    ///
    /// Without this the run picks a collision-safe default (`final_result`)
    /// and augments the preamble. Pinning it is how a bespoke extraction
    /// protocol (an output tool literally named `submit`, with its own
    /// preamble already describing the call) is expressed.
    pub fn output_tool(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        augment_preamble: bool,
    ) -> Self {
        self.config.output_tool_name = Some(name.into());
        self.config.output_tool_description = Some(description.into());
        self.config.augment_output_preamble = augment_preamble;
        self
    }

    /// Set the JSON Schema constraining this request's structured output.
    pub fn output_schema(mut self, schema: schemars::Schema) -> Self {
        self.config.output_schema = Some(schema);
        self
    }

    /// Set how `output_schema` is enforced for this request.
    pub fn output_mode(mut self, mode: OutputMode) -> Self {
        self.config.output_mode = mode;
        self
    }

    /// Opt in or out of recording sensitive request, response, and tool
    /// content on GenAI telemetry spans for this request.
    ///
    /// Defaults to the agent's setting, which defaults to `false`. Enabling
    /// this can expose prompts, retrieved context, tool results, model
    /// responses, and other sensitive or high-cardinality data through
    /// OpenTelemetry span attributes, which can increase observability
    /// backend storage and query costs. Structural metadata and token usage
    /// remain available when disabled.
    pub fn record_content_telemetry(mut self, enabled: bool) -> Self {
        self.config.record_telemetry_content = enabled;
        self
    }

    /// Execute up to `concurrency` of this request's tool calls at once (1 by
    /// default, i.e. sequential). The committed message history is the same
    /// at any concurrency — results are persisted in tool-call order — but
    /// per-tool hook side effects interleave above 1. A `concurrency` of 0 is
    /// clamped to 1.
    pub fn tool_concurrency(mut self, concurrency: usize) -> Self {
        let concurrency = concurrency.max(1);
        self.config.tool_concurrency = concurrency;
        self.executor = self
            .executor
            .map(|executor| executor.tool_concurrency(concurrency));
        self
    }

    /// Set the retry budget for invalid tool-call recovery. Invalid tool-call
    /// retries also consume the total model-call budget.
    pub fn max_invalid_tool_call_retries(mut self, retries: usize) -> Self {
        self.config.max_invalid_tool_call_retries = retries;
        self
    }

    /// The surfacing policy this request drives under: everything is
    /// surfaced as soon as one hook is attached, so each decision point can
    /// be dispatched (matching [`Agent`]).
    fn policy(&self) -> crate::session::SessionPolicy {
        if self.hooks.is_empty() {
            crate::session::SessionPolicy::default()
        } else {
            crate::session::SessionPolicy {
                surface_model_turns: true,
                surface_completion_calls: true,
                surface_tool_calls: true,
                surface_tool_results: true,
            }
        }
    }

    /// Build the blocking session this request drives.
    pub fn into_session(self) -> (AgentSession, Hooks, Option<ToolExecutor>) {
        let policy = self.policy();
        let record_content = self.config.record_telemetry_content;
        let defer_result = !self.hooks.is_empty();
        let executor = self.executor.map(|executor| {
            executor
                .record_content_telemetry(record_content)
                .defer_result_telemetry(defer_result)
        });
        let session = AgentSession::new(self.config, self.provider, self.rt, self.prompt)
            .with_tools(self.tools)
            .with_policy(policy);
        let session = match self.history {
            Some(history) if !history.is_empty() => session.with_history(history),
            _ => session,
        };
        (session, self.hooks, executor)
    }

    /// Drive the agent loop to completion, returning the aggregated
    /// [`PromptResponse`]. Hooks fire at every observable point; the first
    /// hook to terminate cancels the run.
    pub async fn run(self) -> Result<PromptResponse, PromptError> {
        let (mut session, hooks, executor) = self.into_session();
        session.drive(&hooks, executor.as_ref()).await
    }

    /// Drive the agent loop and deserialize the accepted output as `T`.
    ///
    /// `T`'s JSON schema is generated automatically and pinned as the
    /// provider's **native** structured-output constraint (no synthetic
    /// output tool), so the typed surface behaves identically across
    /// providers (#1928); the final text is parsed with a balanced-JSON
    /// fallback so prose or markdown fences around the JSON still parse. For
    /// tool-composing structured output use the untyped
    /// `output_schema`/`output_mode` surface, and for retry-on-parse-failure
    /// extraction use [`crate::extract`].
    pub async fn run_typed<T>(mut self) -> Result<T, StructuredOutputError>
    where
        T: schemars::JsonSchema + serde::de::DeserializeOwned,
    {
        self.config.output_schema = Some(schemars::schema_for!(T));
        self.config.output_mode = OutputMode::Native;

        let response = self.run().await.map_err(Box::new)?;
        if response.output.is_empty() {
            return Err(StructuredOutputError::EmptyResponse);
        }
        Ok(crate::extract::deserialize_structured_output(
            &response.output,
        )?)
    }

    /// Build the host-driven [`AgentStream`] for this request. Answer its
    /// decision inboxes yourself, or use [`SessionRunner::stream_run`] for
    /// the fully driven stream.
    pub fn stream(self) -> AgentStream {
        let policy = self.policy();
        let stream = AgentStream::new(self.config, self.provider, self.rt, self.prompt)
            .with_tools(self.tools)
            .with_policy(policy);
        match self.history {
            Some(history) if !history.is_empty() => stream.with_history(history),
            _ => stream,
        }
    }

    /// Drive the agent loop, streaming assistant content, tool activity, and
    /// a final response, with this request's hooks dispatched and its
    /// executor answering tool batches. Its item type is
    /// [`AgentRunItem`](crate::stream::AgentRunItem), which contains only
    /// observations; host decision requests remain exclusive to
    /// [`AgentStreamItem`](crate::stream::AgentStreamItem).
    ///
    /// The concrete [`AgentRunStream`] is pinned internally, so callers can
    /// use its inherent `.next().await` without importing `StreamExt` or
    /// pinning it first:
    ///
    /// ```ignore
    /// let mut stream = agent.runner(prompt).max_turns(3).stream_run();
    /// while let Some(item) = stream.next().await { /* … */ }
    /// ```
    pub fn stream_run(self) -> AgentRunStream {
        let hooks = self.hooks.clone();
        let record_content = self.config.record_telemetry_content;
        let defer_result = !hooks.is_empty();
        let executor = self.executor.clone().map(|executor| {
            executor
                .record_content_telemetry(record_content)
                .defer_result_telemetry(defer_result)
        });
        self.stream().drive(hooks, executor)
    }
}

#[cfg(test)]
mod tests {
    use super::SessionRunner;
    use crate::agent::AgentBuilder;
    use crate::agent::mock_support::{MockCompletionModel, MockTurn};
    use crate::completion::{Document, Message, PromptError};
    use crate::hooks::{HookDecision, HookEntry, HookEvent};
    use crate::test_utils::MockAddTool;
    use rig_core::message::ToolChoice;
    use schemars::JsonSchema;
    use serde::Deserialize;
    use serde_json::json;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    #[derive(Debug, Deserialize, JsonSchema, PartialEq)]
    struct TypedAnswer {
        value: String,
    }

    /// Named hook entry over a synchronous decision function.
    fn hook_entry(
        name: &str,
        decide: impl Fn(HookEvent) -> HookDecision + Send + Sync + 'static,
    ) -> HookEntry {
        HookEntry::sync(name, decide)
    }

    /// Counts `BeforeModelCall` dispatches.
    fn counting_hook(counter: Arc<AtomicUsize>) -> HookEntry {
        hook_entry("count-completion-calls", move |event| {
            if matches!(event, HookEvent::BeforeModelCall { .. }) {
                counter.fetch_add(1, Ordering::SeqCst);
            }
            HookDecision::Continue
        })
    }

    #[test]
    fn deserialize_structured_output_tolerates_fences_and_prose() {
        // Clean JSON (native / output-tool path).
        assert_eq!(
            crate::extract::deserialize_structured_output::<TypedAnswer>(r#"{"value":"x"}"#)
                .expect("clean json parses"),
            TypedAnswer { value: "x".into() }
        );
        // Markdown-fenced JSON (weak Prompted-mode models).
        assert_eq!(
            crate::extract::deserialize_structured_output::<TypedAnswer>(
                "```json\n{\"value\":\"y\"}\n```"
            )
            .expect("fenced json parses"),
            TypedAnswer { value: "y".into() }
        );
        // Prose around the JSON object.
        assert_eq!(
            crate::extract::deserialize_structured_output::<TypedAnswer>(
                "Here you go: {\"value\":\"z\"} — hope that helps!"
            )
            .expect("embedded json parses"),
            TypedAnswer { value: "z".into() }
        );
        // No JSON at all still errors.
        assert!(
            crate::extract::deserialize_structured_output::<TypedAnswer>("no json here").is_err()
        );
    }

    #[test]
    fn agent_exposes_read_only_name_and_description() {
        let named = AgentBuilder::new(MockCompletionModel::text("done").provider())
            .name("researcher")
            .description("Finds evidence")
            .build();
        assert_eq!(named.name(), Some("researcher"));
        assert_eq!(named.description(), Some("Finds evidence"));

        let unnamed = AgentBuilder::new(MockCompletionModel::text("done").provider()).build();
        assert_eq!(unnamed.name(), None);
        assert_eq!(unnamed.description(), None);
    }

    #[test]
    fn one_hook_instance_attaches_to_distinct_agents() {
        let hook = hook_entry("provider-independent", |_| HookDecision::Continue);
        let _mock_agent = AgentBuilder::new(MockCompletionModel::default().provider())
            .add_hook(hook.clone())
            .build();
        let _other_agent = AgentBuilder::new(MockCompletionModel::text("other").provider())
            .add_hook(hook)
            .build();
    }

    #[tokio::test]
    async fn runner_applies_per_run_request_overrides() {
        let model = MockCompletionModel::text("done");
        AgentBuilder::new(model.clone().provider())
            .preamble("baseline preamble")
            .context("baseline document")
            .temperature(0.1)
            .max_tokens(10)
            .additional_params(json!({"baseline": true}))
            .build()
            .runner("go")
            .preamble("run preamble")
            .document(Document {
                id: "run-one".into(),
                text: "first run document".into(),
                additional_props: Default::default(),
            })
            .documents([Document {
                id: "run-two".into(),
                text: "second run document".into(),
                additional_props: Default::default(),
            }])
            .temperature(0.7)
            .max_tokens(42)
            .replace_additional_params(json!({"override": true}))
            .tool_choice(ToolChoice::None)
            .run()
            .await
            .expect("runner request should succeed");

        let requests = model.requests();
        let request = requests.first().expect("one request");
        assert!(request.chat_history.iter().any(
            |message| matches!(message, Message::System { content } if content == "run preamble")
        ));
        assert!(
            request
                .documents
                .iter()
                .any(|document| document.text == "baseline document")
        );
        assert!(
            request
                .documents
                .iter()
                .any(|document| document.id == "run-one")
        );
        assert!(
            request
                .documents
                .iter()
                .any(|document| document.id == "run-two")
        );
        assert_eq!(request.temperature, Some(0.7));
        assert_eq!(request.max_tokens, Some(42));
        assert_eq!(request.additional_params, Some(json!({"override": true})));
        assert_eq!(request.tool_choice, Some(ToolChoice::None));
    }

    #[tokio::test]
    async fn runner_can_merge_additional_params_into_the_baseline() {
        let model = MockCompletionModel::text("done");
        AgentBuilder::new(model.clone().provider())
            .additional_params(json!({"baseline": true, "winner": "baseline"}))
            .build()
            .runner("go")
            .merge_additional_params(
                json!({"override": true, "winner": "runner"})
                    .as_object()
                    .expect("object")
                    .clone(),
            )
            .run()
            .await
            .expect("runner request should succeed");

        assert_eq!(
            model
                .requests()
                .first()
                .expect("one request")
                .additional_params,
            Some(json!({"baseline": true, "override": true, "winner": "runner"}))
        );
    }

    #[tokio::test]
    async fn runner_can_replace_additional_params_wholesale() {
        let model = MockCompletionModel::text("done");
        AgentBuilder::new(model.clone().provider())
            .additional_params(json!({"baseline": true}))
            .build()
            .runner("go")
            .replace_additional_params(json!({"replacement": true}))
            .run()
            .await
            .expect("runner request should succeed");

        let requests = model.requests();
        let request = requests.first().expect("one request");
        assert_eq!(
            request.additional_params,
            Some(json!({"replacement": true}))
        );
    }

    #[tokio::test]
    async fn runner_can_clear_configured_request_defaults() {
        let model = MockCompletionModel::text("done");
        AgentBuilder::new(model.clone().provider())
            .preamble("baseline")
            .temperature(0.1)
            .max_tokens(10)
            .additional_params(json!({"baseline": true}))
            .tool_choice(ToolChoice::Required)
            .build()
            .runner("go")
            .without_preamble()
            .without_temperature()
            .without_max_tokens()
            .without_additional_params()
            .without_tool_choice()
            .run()
            .await
            .expect("runner request should succeed");

        let requests = model.requests();
        let request = requests.first().expect("one request");
        assert!(
            !request
                .chat_history
                .iter()
                .any(|message| matches!(message, Message::System { .. }))
        );
        assert_eq!(request.temperature, None);
        assert_eq!(request.max_tokens, None);
        assert_eq!(request.additional_params, None);
        assert_eq!(request.tool_choice, None);
    }

    /// `SessionRunner::from_agent` preserves the distinction between an absent
    /// agent default (the implicit one-call budget) and an explicit zero budget.
    #[tokio::test]
    async fn from_agent_preserves_implicit_one_and_explicit_zero_budgets() {
        let implicit_model = MockCompletionModel::from_turns([
            MockTurn::tool_call("tc1", "add", json!({"x": 2, "y": 3})),
            MockTurn::text("the answer is 5"),
        ]);
        let implicit_recorded = implicit_model.clone();
        let implicit_agent = AgentBuilder::new(implicit_model.provider())
            .tool(MockAddTool)
            .build();
        let implicit_runner = SessionRunner::from_agent(&implicit_agent, "add 2 and 3");
        assert_eq!(implicit_runner.config.max_turns, None);

        let implicit_err = implicit_runner
            .run()
            .await
            .expect_err("implicit budget should reject the second model call");
        assert!(matches!(
            implicit_err,
            PromptError::MaxTurnsError { max_turns: 1, .. }
        ));
        assert_eq!(implicit_recorded.request_count(), 1);

        let zero_model = MockCompletionModel::text("should not be requested");
        let zero_recorded = zero_model.clone();
        let zero_agent = AgentBuilder::new(zero_model.provider())
            .default_max_turns(0)
            .build();
        let zero_runner = SessionRunner::from_agent(&zero_agent, "do not call");
        assert_eq!(zero_runner.config.max_turns, Some(0));

        let zero_err = zero_runner
            .run()
            .await
            .expect_err("explicit zero budget should reject the initial model call");
        assert!(matches!(
            zero_err,
            PromptError::MaxTurnsError { max_turns: 0, .. }
        ));
        assert_eq!(zero_recorded.request_count(), 0);
    }

    /// A runner-level `add_hook` APPENDS to the agent's default hooks rather
    /// than replacing them: a hook registered on the builder and one registered
    /// on the runner both observe the same run.
    #[tokio::test]
    async fn runner_add_hook_appends_to_agent_default_hooks() {
        let agent_calls = Arc::new(AtomicUsize::new(0));
        let runner_calls = Arc::new(AtomicUsize::new(0));

        AgentBuilder::new(
            MockCompletionModel::from_turns([
                MockTurn::tool_call("tc1", "add", json!({"x": 2, "y": 3})),
                MockTurn::text("the answer is 5"),
            ])
            .provider(),
        )
        .tool(MockAddTool)
        .add_hook(counting_hook(agent_calls.clone()))
        .build()
        .runner("add 2 and 3")
        .max_turns(3)
        .add_hook(counting_hook(runner_calls.clone()))
        .run()
        .await
        .expect("run should succeed");

        assert!(
            agent_calls.load(Ordering::SeqCst) >= 1,
            "the agent-default hook must still observe the run after a runner-level add_hook"
        );
        assert!(
            runner_calls.load(Ordering::SeqCst) >= 1,
            "the runner-level hook must also observe the run"
        );
        assert_eq!(
            agent_calls.load(Ordering::SeqCst),
            runner_calls.load(Ordering::SeqCst),
            "add_hook appends (both hooks observe every turn); it does not replace"
        );
    }

    #[tokio::test]
    async fn run_typed_deserializes_output_and_untyped_run_reports_completion_calls() {
        use crate::agent::CompletionCall;
        use crate::completion::Usage;

        let call_usage = Usage {
            input_tokens: 4,
            output_tokens: 6,
            total_tokens: 10,
            cached_input_tokens: 0,
            cache_creation_input_tokens: 0,
            tool_use_prompt_tokens: 0,
            reasoning_tokens: 0,
        };
        let typed_agent = AgentBuilder::new(
            MockCompletionModel::new([MockTurn::text(r#"{"value":"ok"}"#).with_usage(call_usage)])
                .provider(),
        )
        .build();

        let value = typed_agent
            .runner("return typed json")
            .run_typed::<TypedAnswer>()
            .await
            .expect("typed prompt should succeed");
        assert_eq!(
            value,
            TypedAnswer {
                value: "ok".to_string()
            }
        );

        let untyped_agent = AgentBuilder::new(
            MockCompletionModel::new([MockTurn::text(r#"{"value":"ok"}"#).with_usage(call_usage)])
                .provider(),
        )
        .build();
        let response = untyped_agent
            .runner("return typed json")
            .run()
            .await
            .expect("prompt should succeed");
        assert_eq!(response.usage, call_usage);
        assert_eq!(
            response.completion_calls(),
            &[CompletionCall::new(0, call_usage)]
        );
    }

    #[tokio::test]
    async fn prompt_response_records_completion_call_without_reported_usage() {
        use crate::agent::CompletionCall;
        use crate::completion::Usage;

        let agent =
            AgentBuilder::new(MockCompletionModel::new([MockTurn::text("ok")]).provider()).build();

        let response = agent
            .runner("say ok")
            .run()
            .await
            .expect("prompt should succeed");

        assert_eq!(response.output, "ok");
        assert_eq!(response.usage, Usage::new());
        assert_eq!(
            response.completion_calls(),
            &[CompletionCall::new(0, Usage::new())]
        );
    }
}
