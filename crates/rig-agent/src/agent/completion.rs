//! The single agent type: plain data plus inherent methods over the session
//! drivers.
//!
//! [`Agent`] pairs an [`AgentConfig`](super::config::AgentConfig) with a
//! [`ProviderConfig`](crate::provider::ProviderConfig), an advertised
//! [`ToolCatalog`](super::prepare::ToolCatalog), an optional
//! [`ToolExecutor`], and attach-and-forget [`Hooks`]. There are no behavior
//! slots beyond those fields: tools are data (catalog + executor records),
//! hooks are data (named callback records folded with the classic
//! semantics), and conversation memory inverts to explicit host calls.
//!
//! Every method drives [`AgentSession`] or [`AgentStream`] — there is no
//! second driver. Per-request configuration lives on
//! [`SessionRunner`](super::request::SessionRunner), built with
//! [`Agent::runner`].

use super::prepare::ToolCatalog;
use super::request::SessionRunner;
use super::response::PromptResponse;
use super::run::OutputMode;
use crate::completion::{
    CompletionError, Message, PromptError, StructuredOutputError, ToolDefinition,
};
use crate::executor::ToolExecutor;
use crate::hooks::Hooks;
use crate::provider::{ProviderConfig, Runtime};
use crate::session::{AgentSession, SessionPolicy};
use crate::stream::{AgentRunStream, AgentStream};
use crate::tool::{PortableDynamicTool, ToolExecutionError, ToolOutput};
use rig_core::{message::ToolChoice, wasm_compat::WasmCompatSend};
use std::{collections::BTreeSet, sync::Arc};

use super::AgentConfig;
use super::UNKNOWN_AGENT_NAME;

/// Base name of the synthetic output tool used by [`OutputMode::Tool`].
const DEFAULT_OUTPUT_TOOL_NAME: &str = "final_result";

#[derive(serde::Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
struct AgentToolArgs {
    /// The task or question delegated to the inner agent.
    prompt: String,
}

/// Whether the active [`ToolChoice`] lets the model call the synthetic output
/// tool. Tool output mode finalizes via that call, so when the choice forbids it
/// (`None`, or a `Specific` allow-list that lists only the caller's real tools)
/// Tool mode cannot work and must fall back to native structured output.
pub(crate) fn tool_choice_permits_output_tool(tool_choice: Option<&ToolChoice>) -> bool {
    matches!(
        tool_choice,
        None | Some(ToolChoice::Auto | ToolChoice::Required)
    )
}

/// Whether the active [`ToolChoice`] can call the *named* synthetic output tool.
///
/// Unlike [`tool_choice_permits_output_tool`] — which runs during output-mode
/// resolution, before the output-tool name is known, and so conservatively
/// treats every `Specific` set as forbidding the call — this knows the committed
/// output-tool name, so a `Specific` set that names it counts as callable. That
/// matches [`allowed_tool_names_for_choice`], which advertises the output tool
/// for exactly that choice. Only a `None` choice or a `Specific` set that omits
/// the output tool genuinely cannot finalize a pinned Tool-mode turn.
pub(crate) fn output_tool_callable(
    tool_choice: Option<&ToolChoice>,
    output_tool_name: &str,
) -> bool {
    match tool_choice {
        Some(ToolChoice::Specific { function_names }) => function_names
            .iter()
            .any(|name| name.as_str() == output_tool_name),
        other => tool_choice_permits_output_tool(other),
    }
}

/// Resolve the caller-facing [`OutputMode`] to a concrete mode for one request.
///
/// With no schema there is nothing to enforce, so the result is always `Native`
/// (the synthetic tool and prompt injection only make sense with a schema).
/// `Auto` becomes `Tool` only when a real executable tool is present, the tool
/// choice permits the output-tool call, AND the provider does *not* compose
/// native structured output with tools — i.e. only where the native constraint
/// would actually suppress tool calls (#1928). On providers that compose them
/// (OpenAI, Anthropic), `Auto` keeps guaranteed native structured output.
/// `Tool` (explicit or via `Auto`) requires that the active [`ToolChoice`]
/// permit the output-tool call; when it does not, it degrades to `Native` so
/// structured output is still enforced rather than silently dropped. Explicit
/// `Prompted`/`Native` are honored when a schema is present. The returned mode is
/// never `Auto`.
pub(crate) fn resolve_output_mode(
    has_schema: bool,
    has_executable_tools: bool,
    output_tool_callable: bool,
    provider_composes_native: bool,
    requested: &OutputMode,
) -> OutputMode {
    if !has_schema {
        return OutputMode::Native;
    }
    match requested {
        OutputMode::Native => OutputMode::Native,
        OutputMode::Prompted => OutputMode::Prompted,
        OutputMode::Tool if output_tool_callable => OutputMode::Tool,
        OutputMode::Tool => OutputMode::Native,
        OutputMode::Auto
            if has_executable_tools && output_tool_callable && !provider_composes_native =>
        {
            OutputMode::Tool
        }
        OutputMode::Auto => OutputMode::Native,
    }
}

/// Pick a collision-safe name for the synthetic output tool, never shadowing a
/// real executable tool (which would make the model's output call dispatchable).
pub(crate) fn pick_output_tool_name(executable_tool_names: &BTreeSet<String>) -> String {
    let mut name = DEFAULT_OUTPUT_TOOL_NAME.to_string();
    let mut suffix = 1u32;
    while executable_tool_names.contains(&name) {
        name = format!("{DEFAULT_OUTPUT_TOOL_NAME}_{suffix}");
        suffix += 1;
    }
    name
}

/// Compute the allowed tool names for a `tool_choice` **and** validate the
/// effective request locally (no provider round-trip).
///
/// The effective advertised tool set for a turn is the executable tools (after
/// any per-turn `active_tools` filtering) plus the synthetic output tool
/// (`output_tool_name`) when structured output runs in Tool mode. Validation:
///
/// - [`ToolChoice::Required`] with **no** advertised tool (no executable tool and
///   no output tool) is a local error — the model is forced to call a tool but
///   none is advertised.
/// - [`ToolChoice::Specific`] must name only advertised tools (executable tools
///   or the output tool); an empty specific set is also an error.
///
/// `pre_filter_tool_names` is the full executable tool set *before* any per-turn
/// `active_tools` filtering — `Some` only when an `active_tools` allow-list was
/// applied. When the incompatibility was actually **caused** by that filter (a
/// tool that would otherwise satisfy the choice was dropped), the error says so
/// and suggests setting a compatible `tool_choice` in the same `RequestPatch`.
/// A plain typo naming a tool that never existed is *not* blamed on the filter.
pub(crate) fn allowed_tool_names_for_choice(
    executable_tool_names: &BTreeSet<String>,
    tool_choice: Option<&ToolChoice>,
    output_tool_name: Option<&str>,
    pre_filter_tool_names: Option<&BTreeSet<String>>,
) -> Result<BTreeSet<String>, CompletionError> {
    let has_advertised_tool = !executable_tool_names.is_empty() || output_tool_name.is_some();
    let hint = |active_tools_caused: bool| {
        if active_tools_caused {
            " A per-turn `active_tools` allow-list narrowed the advertised tools this turn; \
             set a compatible `tool_choice` in the same `RequestPatch`, or widen `active_tools`."
        } else {
            ""
        }
    };
    // The advertised tools the model may call: executable tools + the output tool.
    let advertised = || {
        executable_tool_names
            .iter()
            .map(String::as_str)
            .chain(output_tool_name)
            .collect::<Vec<_>>()
    };

    let allowed = match tool_choice {
        None | Some(ToolChoice::Auto) => executable_tool_names.clone(),
        Some(ToolChoice::Required) => {
            if !has_advertised_tool {
                // The filter caused this only if there *were* tools before it ran.
                let active_tools_caused = pre_filter_tool_names.is_some_and(|pf| !pf.is_empty());
                return Err(CompletionError::RequestError(
                    format!(
                        "ToolChoice::Required forces the model to call a tool, but no tools are \
                         advertised this turn.{}",
                        hint(active_tools_caused)
                    )
                    .into(),
                ));
            }
            executable_tool_names.clone()
        }
        Some(ToolChoice::None) => BTreeSet::new(),
        Some(ToolChoice::Specific { function_names }) => {
            if function_names.is_empty() {
                return Err(CompletionError::RequestError(
                    "ToolChoice::Specific requires at least one function name".into(),
                ));
            }

            let requested = function_names.iter().cloned().collect::<BTreeSet<String>>();
            let missing = function_names
                .iter()
                .map(String::as_str)
                .filter(|name| {
                    !executable_tool_names.contains(*name) && Some(*name) != output_tool_name
                })
                .collect::<Vec<_>>();

            if !missing.is_empty() {
                // The filter caused this only if a missing name existed pre-filter
                // (i.e. `active_tools` dropped it) — not for a plain typo.
                let active_tools_caused = pre_filter_tool_names
                    .is_some_and(|pf| missing.iter().any(|name| pf.contains(*name)));
                return Err(CompletionError::RequestError(
                    format!(
                        "ToolChoice::Specific requested tool names not advertised this turn: \
                         {missing:?}. Advertised: {:?}.{}",
                        advertised(),
                        hint(active_tools_caused)
                    )
                    .into(),
                ));
            }

            requested
        }
    };

    Ok(allowed)
}

/// Rig's agent: an LLM provider selection combined with a preamble (system
/// prompt), static context documents, advertised tools, and hooks.
///
/// Build one with [`AgentBuilder`](super::AgentBuilder) (the ergonomic path)
/// or construct it directly from plain data with [`Agent::new`]. Either way the
/// provider is a `functions::Config` wrapped in the
/// [`ProviderConfig`](crate::provider::ProviderConfig) arm that names it — there
/// is no client object. Every execution method drives the session layer:
/// [`AgentSession`] for the blocking methods and [`AgentStream`] for the
/// streaming ones.
///
/// # Example
/// ```no_run
/// use rig_agent::prelude::*;
/// use rig_core::providers::openai;
///
/// # async fn run() -> Result<(), Box<dyn std::error::Error>> {
/// let comedian_agent = AgentBuilder::new(ProviderConfig::OpenAi(
///         openai::functions::Config::from_env(openai::GPT_5_2)?,
///     ))
///     .preamble("You are a comedian here to entertain the user using humour and jokes.")
///     .temperature(0.9)
///     .build();
///
/// let response = comedian_agent.prompt("Entertain me!").await?;
/// # Ok(())
/// # }
/// ```
///
/// # Conversation memory
///
/// An `Agent` owns no memory slot: conversation history is host-owned data.
/// Wrap a run with two explicit calls against a store such as
/// [`InMemoryConversationMemory`](rig_core::memory::InMemoryConversationMemory)
/// (or `rig_memory::PolicyMemory` for windowing and rolling summaries):
///
/// - **load-before** — `let history = memory.load(id)?;` then
///   [`Agent::run_with_history`] (or `.runner(p).history(history)`). A load
///   failure is fatal: the run never starts.
/// - **append-after** — take [`PromptResponse::messages`] from the finished
///   run and `memory.append(id, messages)` once per run (a multi-step tool
///   round-trip commits its transcript exactly once). Log and proceed on
///   failure so a store hiccup never drops a reply, and skip the append
///   entirely when the run errored or a hook stopped it.
///
/// Passing your own history is how memory is bypassed, and the conversation
/// id is a flat string key you choose. The full recipe, including the
/// streaming variant, lives in the [`agent_api`](crate::agent_api) module
/// docs.
#[derive(Clone)]
#[non_exhaustive]
pub struct Agent {
    /// The agent's model-free configuration.
    pub config: AgentConfig,
    /// Provider selection as plain configuration (which provider, base URL,
    /// credential location, and model identifier).
    pub provider: ProviderConfig,
    /// Live transport handles requests are fulfilled with.
    pub rt: Arc<Runtime>,
    /// Tool definitions advertised each turn.
    pub tools: ToolCatalog,
    /// Executes the model's tool calls. With no executor a run that produces
    /// an executable tool call fails.
    pub executor: Option<ToolExecutor>,
    /// Hooks dispatched at every surfaced event, in registration order.
    pub hooks: Hooks,
}

impl std::fmt::Debug for Agent {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Agent")
            .field("config", &self.config)
            .field("tools", &self.tools)
            .field("executor", &self.executor)
            .field("hooks", &self.hooks)
            .finish_non_exhaustive()
    }
}

impl Agent {
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

    /// Returns the configured agent name.
    pub fn name(&self) -> Option<&str> {
        self.config.name.as_deref()
    }

    /// Returns the configured agent description.
    pub fn description(&self) -> Option<&str> {
        self.config.description.as_deref()
    }

    /// Convert this agent into one concrete portable tool record.
    ///
    /// The returned tool accepts exactly one required string field, `prompt`,
    /// and delegates it through this agent's ordinary session driver. This is
    /// a data-to-data conversion: it does not restore the removed classic tool
    /// context, server, or erased-dispatch architecture.
    pub fn into_tool(
        self,
        name: impl Into<String>,
        description: impl Into<String>,
    ) -> PortableDynamicTool {
        let parameters = schemars::schema_for!(AgentToolArgs).as_value().clone();
        let agent = Arc::new(self);

        PortableDynamicTool::new(name, description, parameters, move |arguments| {
            let agent = Arc::clone(&agent);
            async move {
                let arguments: AgentToolArgs =
                    serde_json::from_value(arguments).map_err(|error| {
                        ToolExecutionError::invalid_args(format!(
                            "failed to parse agent tool arguments: {error}"
                        ))
                        .with_source(error)
                    })?;
                agent
                    .prompt(arguments.prompt)
                    .await
                    .map(ToolOutput::text)
                    .map_err(ToolExecutionError::from_error)
            }
        })
    }

    pub(crate) fn name_or_default(&self) -> &str {
        self.config.name.as_deref().unwrap_or(UNKNOWN_AGENT_NAME)
    }

    /// The [`SessionPolicy`] this agent drives sessions under: the default
    /// policy when no hooks are attached; with hooks, every decision point
    /// is surfaced so each one can be dispatched through the hook list.
    pub(crate) fn session_policy(&self) -> SessionPolicy {
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
    pub(crate) fn session(
        &self,
        prompt: impl Into<Message>,
        history: Vec<Message>,
    ) -> AgentSession {
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

    /// This agent's executor with content telemetry aligned to the agent's
    /// `record_telemetry_content` setting (the classic driver passed the same
    /// flag down to tool execution).
    pub(crate) fn telemetry_aware_executor(&self) -> Option<ToolExecutor> {
        // With hooks attached the driver surfaces the post-execution decision
        // point and records the post-hook presentation itself.
        let defer_result = !self.hooks.is_empty();
        self.executor.clone().map(|executor| {
            executor
                .record_content_telemetry(self.config.record_telemetry_content)
                .defer_result_telemetry(defer_result)
        })
    }

    /// Build a per-request [`SessionRunner`] for this agent, seeded with the
    /// agent's configuration and hooks.
    ///
    /// This is **the** fluent per-request surface: it carries every
    /// per-request knob (history, preamble, documents, temperature, token
    /// limits, additional params, tool choice, tool concurrency, telemetry
    /// content, turn and invalid-tool-call budgets, extra hooks) and
    /// terminates in [`SessionRunner::run`], [`SessionRunner::run_typed`], or
    /// [`SessionRunner::stream`].
    pub fn runner(&self, prompt: impl Into<Message>) -> SessionRunner {
        SessionRunner::from_agent(self, prompt)
    }

    /// A fluent structured-extraction run over this agent's configuration.
    ///
    /// The returned [`ExtractionRunner`](crate::extract::ExtractionRunner) is
    /// not generic: pick the extracted type at the terminal with
    /// `.run::<T>()` or `.run_with_usage::<T>()`.
    ///
    /// ```no_run
    /// # async fn run(agent: &rig_agent::Agent) -> Result<(), Box<dyn std::error::Error>> {
    /// #[derive(serde::Deserialize, schemars::JsonSchema)]
    /// struct Person {
    ///     name: String,
    /// }
    ///
    /// let person: Person = agent.extractor("Alice is 30.").run().await?;
    /// # let _ = person;
    /// # Ok(())
    /// # }
    /// ```
    pub fn extractor(&self, prompt: impl Into<Message>) -> crate::extract::ExtractionRunner {
        crate::extract::ExtractionRunner::new(
            self.config.clone(),
            self.provider.clone(),
            Arc::clone(&self.rt),
            prompt,
        )
    }

    /// Send a prompt and return the accepted assistant text after full
    /// runtime orchestration (tools, hooks, retries, telemetry).
    ///
    /// For per-request configuration use [`Agent::runner`].
    pub async fn prompt(
        &self,
        prompt: impl Into<Message> + WasmCompatSend,
    ) -> Result<String, PromptError> {
        Ok(self.run(prompt).await?.output)
    }

    /// Send a prompt and return the full [`PromptResponse`] — accepted
    /// output, aggregated usage, per-call details, and the committed message
    /// transcript.
    pub async fn run(
        &self,
        prompt: impl Into<Message> + WasmCompatSend,
    ) -> Result<PromptResponse, PromptError> {
        // Convert eagerly: threading the generic prompt through the driver
        // chain would make the returned future's `Send`-ness depend on the
        // caller's lifetime, which breaks higher-ranked `Send` bounds (see
        // the Discord integration's `async_trait` handlers).
        self.drive_run(prompt.into(), Vec::new()).await
    }

    /// [`Agent::run`] with explicit input history preceding the prompt.
    /// Explicit history means conversation memory (a host concern — see the
    /// [`agent_api`](crate::agent_api) module docs) is bypassed entirely.
    pub async fn run_with_history(
        &self,
        prompt: impl Into<Message> + WasmCompatSend,
        history: Vec<Message>,
    ) -> Result<PromptResponse, PromptError> {
        self.drive_run(prompt.into(), history).await
    }

    /// The one blocking drive: non-generic, so the future it returns is
    /// `Send` for every caller lifetime.
    async fn drive_run(
        &self,
        prompt: Message,
        history: Vec<Message>,
    ) -> Result<PromptResponse, PromptError> {
        let mut session = self.session(prompt, history);
        let executor = self.telemetry_aware_executor();
        session.drive(&self.hooks, executor.as_ref()).await
    }

    /// Execute one turn against caller-owned canonical history, appending only
    /// committed messages to `chat_history`.
    #[tracing::instrument(skip(self, prompt, chat_history), fields(agent_name = self.name_or_default()))]
    pub async fn chat(
        &self,
        prompt: impl Into<Message> + WasmCompatSend,
        chat_history: &mut Vec<Message>,
    ) -> Result<String, PromptError> {
        let response = self.drive_run(prompt.into(), chat_history.clone()).await?;
        if let Some(messages) = response.messages {
            chat_history.extend(messages);
        }
        Ok(response.output)
    }

    /// Send a prompt and deserialize the accepted structured response as `T`.
    ///
    /// The JSON schema for `T` is generated automatically and passed as the
    /// provider's **native** structured-output constraint (see
    /// [`SessionRunner::run_typed`], which this delegates to). For a fluent
    /// per-request variant, use `agent.runner(prompt).run_typed::<T>()`.
    ///
    /// # Example
    /// ```rust,ignore
    /// #[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
    /// struct WeatherForecast {
    ///     city: String,
    ///     temperature_f: f64,
    /// }
    ///
    /// let forecast = agent
    ///     .prompt_typed::<WeatherForecast>("What's the weather in NYC?")
    ///     .await?;
    /// ```
    pub async fn prompt_typed<T>(
        &self,
        prompt: impl Into<Message> + WasmCompatSend,
    ) -> Result<T, StructuredOutputError>
    where
        T: schemars::JsonSchema + serde::de::DeserializeOwned + WasmCompatSend,
    {
        self.runner(prompt).run_typed::<T>().await
    }

    /// Build a pre-configured [`AgentStream`] for one prompt: this agent's
    /// tools and session policy are already applied.
    ///
    /// This is the **host-driven** streaming surface: pull items with
    /// [`AgentStream::next_item`] (or
    /// [`AgentStream::next_item_with_tools`] to answer tool batches through
    /// an executor) and answer the decision inboxes yourself. For the classic
    /// fire-and-forget behavior — hooks dispatched and tools executed for you
    /// — use [`Agent::stream_run`].
    pub fn stream_prompt(&self, prompt: impl Into<Message> + WasmCompatSend) -> AgentStream {
        AgentStream::new(
            self.config.clone(),
            self.provider.clone(),
            self.rt.clone(),
            prompt,
        )
        .with_tools(self.tools.clone())
        .with_policy(self.session_policy())
    }

    /// [`Agent::stream_prompt`] seeded with canonical history.
    pub fn stream_chat<I, T>(
        &self,
        prompt: impl Into<Message> + WasmCompatSend,
        chat_history: I,
    ) -> AgentStream
    where
        I: IntoIterator<Item = T>,
        T: Into<Message>,
    {
        let history: Vec<Message> = chat_history.into_iter().map(Into::into).collect();
        let stream = self.stream_prompt(prompt);
        if history.is_empty() {
            stream
        } else {
            stream.with_history(history)
        }
    }

    /// The fully driven streaming surface: this agent's hooks are dispatched
    /// and its executor answers tool batches, so the returned stream is a
    /// pure observation stream of assistant deltas, tool activity, per-call
    /// records, and the terminal [`AgentStreamItem::Final`].
    ///
    /// [`AgentStreamItem::Final`]: crate::stream::AgentStreamItem::Final
    ///
    /// The concrete [`AgentRunStream`] is pinned internally, so callers can
    /// use its inherent `.next().await` without importing `StreamExt` or
    /// pinning it first.
    pub fn stream_run(&self, prompt: impl Into<Message> + WasmCompatSend) -> AgentRunStream {
        self.stream_prompt(prompt)
            .drive(self.hooks.clone(), self.telemetry_aware_executor())
    }

    /// Resolve the provider-facing tool definitions registered on the agent.
    ///
    /// This read-only view does not expose tool dispatch. Per-turn narrowing
    /// of the advertised set happens through
    /// [`RequestPatch::active_tools`](super::hook::RequestPatch::active_tools).
    pub async fn tool_definitions(&self) -> Vec<ToolDefinition> {
        self.tools.definitions.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::MockScript;
    use rig_core::OneOrMany;
    use rig_core::completion::{CompletionResponse, FinishReason, Usage};
    use rig_core::message::AssistantContent;

    fn tool_names(names: &[&str]) -> BTreeSet<String> {
        names.iter().map(|name| (*name).to_string()).collect()
    }

    fn text_response(text: &str) -> CompletionResponse {
        CompletionResponse::new(
            OneOrMany::one(AssistantContent::text(text)),
            Usage::new(),
            "mock",
        )
        .with_finish_reason(FinishReason::Stop)
    }

    #[tokio::test]
    async fn into_tool_uses_one_strict_argument_record() {
        let tool = Agent::new(
            AgentConfig::new(),
            ProviderConfig::Mock(MockScript::from_responses(Vec::new())),
        )
        .into_tool("delegate", "Delegate a task");

        let definition = tool.definition();
        assert_eq!(definition.name, "delegate");
        assert_eq!(definition.parameters["type"], "object");
        assert_eq!(definition.parameters["additionalProperties"], false);
        assert_eq!(
            definition.parameters["required"],
            serde_json::json!(["prompt"])
        );
        assert_eq!(
            definition.parameters["properties"]["prompt"]["type"],
            "string"
        );

        for invalid in [
            serde_json::json!({}),
            serde_json::json!({"prompt": 1}),
            serde_json::json!([]),
            serde_json::json!({"prompt": "hello", "unknown": true}),
        ] {
            let error = tool
                .execute(invalid)
                .await
                .expect_err("invalid arguments must not reach the agent");
            assert_eq!(error.kind(), crate::tool::ToolErrorKind::InvalidArgs);
            assert!(
                std::error::Error::source(&error)
                    .and_then(|source| source.downcast_ref::<serde_json::Error>())
                    .is_some(),
                "the concrete deserialization error should be retained"
            );
        }
    }

    #[tokio::test]
    async fn into_tool_forwards_prompts_concurrently() {
        let script = MockScript::from_responses(vec![text_response("one"), text_response("two")]);
        let probe = script.clone();
        let tool = Agent::new(AgentConfig::new(), ProviderConfig::Mock(script))
            .into_tool("delegate", "Delegate a task");

        let first = {
            let tool = tool.clone();
            tokio::spawn(async move { tool.execute(serde_json::json!({"prompt": "first"})).await })
        };
        let second = {
            let tool = tool.clone();
            tokio::spawn(async move { tool.execute(serde_json::json!({"prompt": "second"})).await })
        };

        let first = first
            .await
            .expect("first task joins")
            .expect("first tool call");
        let second = second
            .await
            .expect("second task joins")
            .expect("second tool call");
        let mut outputs = [
            first.as_text().expect("text output"),
            second.as_text().expect("text output"),
        ];
        outputs.sort_unstable();
        assert_eq!(outputs, ["one", "two"]);

        let mut prompts: Vec<Message> = probe
            .requests()
            .into_iter()
            .map(|request| request.chat_history.last())
            .collect();
        prompts.sort_by_key(|message| format!("{message:?}"));
        let mut expected = [Message::user("first"), Message::user("second")];
        expected.sort_by_key(|message| format!("{message:?}"));
        assert_eq!(prompts, expected);
    }

    #[tokio::test]
    #[cfg(not(target_family = "wasm"))]
    async fn into_tool_retains_prompt_error_as_native_source() {
        let script = MockScript::from_responses(Vec::new())
            .with_errors(vec![Some("provider failed".to_owned())]);
        let tool = Agent::new(AgentConfig::new(), ProviderConfig::Mock(script))
            .into_tool("delegate", "Delegate a task");

        let error = tool
            .execute(serde_json::json!({"prompt": "fail"}))
            .await
            .expect_err("the scripted provider failure should surface");
        let source = std::error::Error::source(&error).expect("native source retained");
        assert!(source.downcast_ref::<PromptError>().is_some());
    }

    #[test]
    fn allowed_tool_names_defaults_to_all_executable_tools() {
        let executable = tool_names(&["add", "subtract"]);

        assert_eq!(
            allowed_tool_names_for_choice(&executable, None, None, None).unwrap(),
            executable
        );
    }

    #[test]
    fn allowed_tool_names_auto_and_required_allow_all_executable_tools() {
        let executable = tool_names(&["add", "subtract"]);

        assert_eq!(
            allowed_tool_names_for_choice(&executable, Some(&ToolChoice::Auto), None, None)
                .unwrap(),
            executable
        );
        assert_eq!(
            allowed_tool_names_for_choice(&executable, Some(&ToolChoice::Required), None, None)
                .unwrap(),
            executable
        );
    }

    #[test]
    fn allowed_tool_names_none_allows_no_tools() {
        let executable = tool_names(&["add", "subtract"]);

        assert!(
            allowed_tool_names_for_choice(&executable, Some(&ToolChoice::None), None, None)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn allowed_tool_names_specific_allows_requested_executable_tools() {
        let executable = tool_names(&["add", "subtract"]);
        let choice = ToolChoice::Specific {
            function_names: vec!["add".to_string()],
        };

        assert_eq!(
            allowed_tool_names_for_choice(&executable, Some(&choice), None, None).unwrap(),
            tool_names(&["add"])
        );
    }

    #[test]
    fn allowed_tool_names_specific_rejects_missing_tools() {
        let executable = tool_names(&["add"]);
        let choice = ToolChoice::Specific {
            function_names: vec!["missing".to_string()],
        };

        let err = allowed_tool_names_for_choice(&executable, Some(&choice), None, None)
            .expect_err("missing specific tool should fail before provider request");

        assert!(matches!(
            err,
            CompletionError::RequestError(err)
                if err.to_string().contains("missing")
                    && err.to_string().contains("add")
        ));
    }

    #[test]
    fn allowed_tool_names_specific_rejects_empty_names() {
        let executable = tool_names(&["add"]);
        let choice = ToolChoice::Specific {
            function_names: vec![],
        };

        let err = allowed_tool_names_for_choice(&executable, Some(&choice), None, None)
            .expect_err("empty specific tool choice should fail before provider request");

        assert!(matches!(
            err,
            CompletionError::RequestError(err)
                if err.to_string().contains("requires at least one function name")
        ));
    }

    #[test]
    fn output_tool_callable_honors_specific_naming_the_output_tool() {
        // Auto / Required / no explicit choice all permit the output-tool call.
        assert!(output_tool_callable(None, "final_result"));
        assert!(output_tool_callable(
            Some(&ToolChoice::Auto),
            "final_result"
        ));
        assert!(output_tool_callable(
            Some(&ToolChoice::Required),
            "final_result"
        ));
        // A `Specific` set that NAMES the output tool can call it — the case the
        // pinned Tool-mode stall warning must not flag (it is accepted by
        // `allowed_tool_names_for_choice`, which advertises the output tool).
        assert!(output_tool_callable(
            Some(&ToolChoice::Specific {
                function_names: vec!["final_result".to_string()],
            }),
            "final_result",
        ));
        // A `Specific` set that omits it — or `ToolChoice::None` — genuinely cannot
        // finalize a pinned Tool-mode turn, so the warning should still fire there.
        assert!(!output_tool_callable(
            Some(&ToolChoice::Specific {
                function_names: vec!["search".to_string()],
            }),
            "final_result",
        ));
        assert!(!output_tool_callable(
            Some(&ToolChoice::None),
            "final_result"
        ));
    }

    #[test]
    fn required_with_no_advertised_tool_is_local_error() {
        let empty = tool_names(&[]);
        let err = allowed_tool_names_for_choice(&empty, Some(&ToolChoice::Required), None, None)
            .expect_err("Required with no advertised tool must fail locally");
        assert!(matches!(
            err,
            CompletionError::RequestError(err) if err.to_string().contains("Required")
        ));
    }

    #[test]
    fn required_with_only_the_output_tool_is_allowed() {
        // Structured-output Tool mode with no real tools: the model can still be
        // forced to call the synthetic output tool, so Required is valid.
        let empty = tool_names(&[]);
        let allowed = allowed_tool_names_for_choice(
            &empty,
            Some(&ToolChoice::Required),
            Some("final_result"),
            None,
        )
        .expect("Required is satisfiable by the output tool");
        // The output tool is added to the allowed set by the caller, so the
        // executable-derived allowed set is empty here.
        assert!(allowed.is_empty());
    }

    #[test]
    fn required_with_active_tools_filter_names_the_filter_in_the_error() {
        let empty = tool_names(&[]);
        let err = allowed_tool_names_for_choice(
            &empty,
            Some(&ToolChoice::Required),
            None,
            Some(&tool_names(&["add"])),
        )
        .expect_err("Required after active_tools filtered everything must fail locally");
        let msg = err.to_string();
        assert!(
            msg.contains("active_tools"),
            "error should name active_tools: {msg}"
        );
        assert!(
            msg.contains("RequestPatch"),
            "error should suggest RequestPatch: {msg}"
        );
    }

    #[test]
    fn specific_naming_a_filtered_out_tool_is_a_local_error_with_hint() {
        // active_tools narrowed the advertised set to {add}; Specific still names
        // the now-filtered-out `subtract`.
        let executable = tool_names(&["add"]);
        let choice = ToolChoice::Specific {
            function_names: vec!["subtract".to_string()],
        };
        let err = allowed_tool_names_for_choice(
            &executable,
            Some(&choice),
            None,
            Some(&tool_names(&["add", "subtract"])),
        )
        .expect_err("Specific naming a filtered-out tool must fail locally");
        let msg = err.to_string();
        assert!(
            msg.contains("subtract"),
            "error should name the missing tool: {msg}"
        );
        assert!(
            msg.contains("active_tools"),
            "error should name active_tools: {msg}"
        );
    }

    #[test]
    fn specific_may_name_the_output_tool() {
        // The effective advertised set includes the synthetic output tool.
        let empty = tool_names(&[]);
        let choice = ToolChoice::Specific {
            function_names: vec!["final_result".to_string()],
        };
        let allowed =
            allowed_tool_names_for_choice(&empty, Some(&choice), Some("final_result"), None)
                .expect("Specific naming the output tool is valid");
        assert_eq!(allowed, tool_names(&["final_result"]));
    }

    #[test]
    fn specific_typo_is_not_blamed_on_active_tools() {
        // Specific names a tool that never existed (a typo), even though an
        // active_tools filter was applied. The error must NOT blame active_tools,
        // because the filter never had that tool to drop.
        let executable = tool_names(&["add"]);
        let choice = ToolChoice::Specific {
            function_names: vec!["nonexistent".to_string()],
        };
        let err = allowed_tool_names_for_choice(
            &executable,
            Some(&choice),
            None,
            Some(&tool_names(&["add"])),
        )
        .expect_err("Specific naming a non-existent tool must fail locally");
        let msg = err.to_string();
        assert!(msg.contains("nonexistent"), "error names the typo: {msg}");
        assert!(
            !msg.contains("active_tools"),
            "a plain typo must not be blamed on active_tools: {msg}"
        );
    }

    #[test]
    fn resolve_output_mode_without_schema_is_always_native() {
        // No schema => nothing to enforce, regardless of the requested mode or tools.
        for requested in [
            OutputMode::Auto,
            OutputMode::Tool,
            OutputMode::Native,
            OutputMode::Prompted,
        ] {
            assert_eq!(
                resolve_output_mode(false, true, true, false, &requested),
                OutputMode::Native,
                "no schema should force Native for {requested:?}"
            );
            assert_eq!(
                resolve_output_mode(false, false, true, false, &requested),
                OutputMode::Native,
            );
        }
    }

    #[test]
    fn resolve_output_mode_auto_picks_tool_only_when_tools_present() {
        // This is the #1928 fix: with tools on a provider that does NOT compose
        // native output with tools, the schema must not be a native `format`
        // constraint on every turn, so Auto routes to Tool.
        assert_eq!(
            resolve_output_mode(true, true, true, false, &OutputMode::Auto),
            OutputMode::Tool,
        );
        // No tools => native structured output is safe and preferred.
        assert_eq!(
            resolve_output_mode(true, false, true, false, &OutputMode::Auto),
            OutputMode::Native,
        );
    }

    #[test]
    fn resolve_output_mode_auto_keeps_native_when_provider_composes() {
        // On providers that compose native structured output with tools (OpenAI,
        // Anthropic), Auto keeps guaranteed native output even with tools present.
        assert_eq!(
            resolve_output_mode(true, true, true, true, &OutputMode::Auto),
            OutputMode::Native,
        );
    }

    #[test]
    fn resolve_output_mode_honors_explicit_choice_with_schema() {
        for (requested, expected) in [
            (OutputMode::Tool, OutputMode::Tool),
            (OutputMode::Native, OutputMode::Native),
            (OutputMode::Prompted, OutputMode::Prompted),
        ] {
            // Explicit modes are honored regardless of tools or provider support.
            assert_eq!(
                resolve_output_mode(true, true, true, false, &requested),
                expected
            );
            assert_eq!(
                resolve_output_mode(true, false, true, true, &requested),
                expected
            );
        }
    }

    #[test]
    fn resolve_output_mode_degrades_to_native_when_output_tool_not_callable() {
        // Tool mode finalizes via the output-tool call; when the tool choice
        // forbids it (None / Specific), structured output must still be enforced
        // via Native rather than silently dropped (#1928 regression guard).
        assert_eq!(
            resolve_output_mode(true, true, false, false, &OutputMode::Auto),
            OutputMode::Native,
        );
        assert_eq!(
            resolve_output_mode(true, true, false, false, &OutputMode::Tool),
            OutputMode::Native,
        );
        // Prompted does not rely on tools, so it is unaffected.
        assert_eq!(
            resolve_output_mode(true, true, false, false, &OutputMode::Prompted),
            OutputMode::Prompted,
        );
    }

    #[test]
    fn tool_choice_permits_output_tool_only_for_auto_required_or_unset() {
        assert!(tool_choice_permits_output_tool(None));
        assert!(tool_choice_permits_output_tool(Some(&ToolChoice::Auto)));
        assert!(tool_choice_permits_output_tool(Some(&ToolChoice::Required)));
        assert!(!tool_choice_permits_output_tool(Some(&ToolChoice::None)));
        assert!(!tool_choice_permits_output_tool(Some(
            &ToolChoice::Specific {
                function_names: vec!["add".to_string()],
            }
        )));
    }

    #[test]
    fn pick_output_tool_name_defaults_when_unused() {
        let executable = tool_names(&["add", "subtract"]);
        assert_eq!(pick_output_tool_name(&executable), DEFAULT_OUTPUT_TOOL_NAME);
    }

    #[test]
    fn pick_output_tool_name_avoids_collision_with_real_tools() {
        // A user tool literally named `final_result` must not be shadowed, or
        // the model's output call would be dispatched to the tool server.
        let executable = tool_names(&["final_result"]);
        assert_eq!(pick_output_tool_name(&executable), "final_result_1");

        let executable = tool_names(&["final_result", "final_result_1"]);
        assert_eq!(pick_output_tool_name(&executable), "final_result_2");
    }
}
