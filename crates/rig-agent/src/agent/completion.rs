use super::hook::{HookStack, RequestPatch};
use super::run::OutputMode;
use super::runner::AgentRunner;
use super::typed::TypedRun;
use crate::{
    completion::{
        CompletionError, CompletionModel, CompletionRequestBuilder, Document, Message, PromptError,
        ToolDefinition,
    },
    run::response::PromptResponse,
    tool::server::{ToolRegistrySnapshot, ToolServerError, ToolServerHandle},
};
use rig_core::completion::ModelHandle;
use rig_core::id::ConversationId;
use rig_core::{message::ToolChoice, wasm_compat::WasmCompatSend};
use std::{collections::BTreeSet, sync::Arc};

use super::UNKNOWN_AGENT_NAME;

/// A prepared completion request plus the executable Rig tool names advertised
/// to the provider for this turn.
pub(crate) struct PreparedCompletionRequest {
    /// Builder carrying the selected model handle: request preparation ran
    /// against this handle's captured capabilities, and the same handle
    /// executes the prepared request.
    pub(crate) builder: CompletionRequestBuilder<ModelHandle>,
    /// Exact implementations behind this turn's provider definitions.
    pub(crate) tool_snapshot: Arc<ToolRegistrySnapshot>,
    /// The definitions the request carries (executable tools plus, in Tool
    /// output mode, the synthetic output tool), for the run's `TurnTools`.
    pub(crate) advertised_tools: Vec<ToolDefinition>,
    pub(crate) executable_tool_names: BTreeSet<String>,
    pub(crate) allowed_tool_names: BTreeSet<String>,
    /// When Tool output mode is active, the name of the synthetic output tool
    /// advertised to the model (allowed but not executable). See #1928.
    pub(crate) output_tool_name: Option<String>,
    /// The output-token cap this exact attempt was prepared with — the agent's
    /// configured value after the runner/request overrides and after the merged
    /// completion-call [`RequestPatch`](crate::agent::hook::RequestPatch), i.e.
    /// the structured cap that reaches the provider. A cap smuggled through
    /// `additional_params` passthrough is not reflected here, by design: this
    /// reports the field the request actually set.
    ///
    /// Carried here rather than read back off the builder because the builder is
    /// consumed by `send`/`stream` before a turn's hooks fire, and because
    /// provenance matters: this is the same binding applied to the request, so
    /// it cannot drift from what was sent. Both surfaces receive this struct, so
    /// neither can report a different number for the same attempt.
    pub(crate) max_tokens: Option<u64>,
}

/// Helper function to build a completion request from the runner's configured
/// baseline while preserving the executable Rig tool names sent to the
/// provider. Only the per-turn inputs — the selected model, prompt, history,
/// committed output tool, and hook patch — arrive as parameters; everything
/// else is read off the runner.
///
/// The driver's share is the IO around the protocol: retrieve this turn's
/// tools (the one `.await`), hand them with the spec and patch to
/// [`crate::run::prepare::prepare_request`], then bind the prepared data to the selected
/// model's request builder and pin the snapshot to the executable set.
pub(crate) async fn build_prepared_completion_request(
    runner: &crate::agent::AgentRunner,
    model: &ModelHandle,
    prompt: Message,
    chat_history: &[Message],
    committed_output_tool: Option<&str>,
    request_patch: Option<&RequestPatch>,
) -> Result<PreparedCompletionRequest, CompletionError> {
    let record_telemetry_content = runner.config.record_telemetry_content;
    let tool_server_handle = &runner.tool_server_handle;

    // Retrieved tools keep their existing query-selection behavior: prefer the
    // current prompt's RAG text, then the latest matching history message.
    let retrieval_query = prompt.rag_text().or_else(|| {
        chat_history
            .iter()
            .rev()
            .find_map(rig_core::completion::Message::rag_text)
    });

    let mut tool_snapshot = tool_server_handle
        .snapshot_tool_defs(retrieval_query)
        .await
        .map_err(|_| CompletionError::RequestError("Failed to get tool definitions".into()))?;

    let mut spec = runner.config.run_spec();
    spec.output_tool_description
        .clone_from(&runner.output_tool_description);
    spec.augment_output_preamble = runner.augment_output_preamble;

    let prepared = crate::run::prepare::prepare_request(
        &spec,
        &model.capabilities(),
        chat_history,
        tool_snapshot.take_definitions(),
        committed_output_tool,
        request_patch,
    )?;

    // Narrow dispatch to the tools actually advertised this turn (a per-turn
    // `active_tools` allow-list), so the implementation behind every definition
    // the provider received is the one that runs.
    tool_snapshot.retain_names(&prepared.executable_tool_names);

    let advertised_tools = prepared.tools.clone();
    let executable_tool_names = prepared.executable_tool_names.clone();
    let allowed_tool_names = prepared.allowed_tool_names.clone();
    let output_tool_name = prepared.output_tool_name.clone();
    let max_tokens = prepared.max_tokens;
    let completion_request = prepared
        .apply(model.completion_request(prompt))
        .record_content_telemetry(record_telemetry_content);

    Ok(PreparedCompletionRequest {
        builder: completion_request,
        tool_snapshot: Arc::new(tool_snapshot),
        advertised_tools,
        executable_tool_names,
        allowed_tool_names,
        output_tool_name,
        // The post-patch binding from above — the one `.max_tokens_opt(..)`
        // put on the request.
        max_tokens,
    })
}

/// Struct representing an LLM agent. An agent is an LLM model combined with a preamble
/// (i.e.: system prompt) and a static set of context documents and tools.
/// All context documents and tools are always provided to the agent when prompted.
///
/// Default hooks attached with [`AgentBuilder::add_hook`](crate::agent::AgentBuilder::add_hook)
/// are used for every prompt request, plus any added on the request or runner.
///
/// # Example
/// ```no_run
/// use rig_agent::prelude::*;
/// use rig_core::providers::openai;
/// use rig_reqwest::prelude::*;
///
/// # async fn run() -> Result<(), Box<dyn std::error::Error>> {
/// let openai = openai::Client::from_env()?;
///
/// let comedian_agent = openai
///     .agent(openai::GPT_5_2)
///     .preamble("You are a comedian here to entertain the user using humour and jokes.")
///     .temperature(0.9)
///     .build();
///
/// let response = comedian_agent.prompt("Entertain me!").await?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct Agent {
    pub(crate) config: AgentConfig,
    pub(crate) tool_server_handle: ToolServerHandle,
}

/// Everything an [`AgentBuilder`](crate::agent::AgentBuilder) configures and the
/// built [`Agent`] carries unchanged.
///
/// Building only moves this across and resolves the tool state into a
/// [`ToolServerHandle`], so a new setting is declared once here instead of in
/// two parallel field lists.
#[derive(Clone)]
pub(crate) struct AgentConfig {
    /// Name of the agent used for logging and debugging
    pub(crate) name: Option<String>,
    /// Agent description. Primarily useful when using sub-agents as part of an agent workflow and converting agents to other formats.
    pub(crate) description: Option<String>,
    /// Completion model (e.g.: OpenAI's gpt-3.5-turbo-1106, Cohere's command-r)
    pub(crate) model: ModelHandle,
    /// System prompt
    pub(crate) preamble: Option<String>,
    /// Context documents always available to the agent
    pub(crate) static_context: Vec<Document>,
    /// Additional parameters to be passed to the model
    pub(crate) additional_params: Option<serde_json::Value>,
    /// Whether to record sensitive request, response, and tool content on GenAI spans.
    ///
    /// Defaults to `false`. Enabling this can expose prompts, retrieved context,
    /// tool results, model responses, and other sensitive or high-cardinality data
    /// through OpenTelemetry span attributes, which can increase observability
    /// backend storage and query costs.
    pub(crate) record_telemetry_content: bool,
    /// Maximum number of tokens for the completion
    pub(crate) max_tokens: Option<u64>,
    /// Temperature of the model
    pub(crate) temperature: Option<f64>,
    /// Whether or not the underlying LLM should be forced to use a tool before providing a response.
    pub(crate) tool_choice: Option<ToolChoice>,
    /// Total model-call budget, including the initial call and every retry or
    /// continuation. Defaults to `1` (the initial call only).
    pub(crate) max_turns: usize,
    /// Default hook stack applied to every prompt request and runner created
    /// from this agent. Empty by default.
    pub(crate) hooks: HookStack,
    /// Optional JSON Schema for structured output. When set, providers that support
    /// native structured outputs will constrain the model's response to match this schema.
    pub(crate) output_schema: Option<schemars::Schema>,
    /// How `output_schema` is enforced — tool call, native structured output, or
    /// prompt injection (see [`OutputMode`] and issue #1928).
    pub(crate) output_mode: OutputMode,
    /// Optional conversation memory backend that loads/saves history per conversation id.
    pub(crate) memory: Option<Arc<dyn rig_core::memory::ConversationMemory>>,
    /// Optional conversation id used when none is set per-request.
    pub(crate) conversation_id: Option<ConversationId>,
}

impl AgentConfig {
    /// The unconfigured starting point for a builder over `model`.
    pub(crate) fn new(model: ModelHandle) -> Self {
        Self {
            name: None,
            description: None,
            model,
            preamble: None,
            static_context: vec![],
            additional_params: None,
            record_telemetry_content: false,
            max_tokens: None,
            temperature: None,
            tool_choice: None,
            max_turns: 1,
            hooks: HookStack::new(),
            output_schema: None,
            output_mode: OutputMode::default(),
            memory: None,
            conversation_id: None,
        }
    }
}

impl AgentConfig {
    /// The protocol-facing half of this configuration as plain data: what a
    /// driver needs to shape requests and budget a run, without the model,
    /// hooks, memory or identity this config also carries.
    pub(crate) fn run_spec(&self) -> crate::run::spec::RunSpec {
        crate::run::spec::RunSpec {
            preamble: self.preamble.clone(),
            static_context: self.static_context.clone(),
            additional_params: self.additional_params.clone(),
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            tool_choice: self.tool_choice.clone(),
            max_turns: Some(self.max_turns),
            max_invalid_tool_call_retries: 0,
            output_schema: self
                .output_schema
                .as_ref()
                .map(|schema| schema.as_value().clone()),
            output_mode: self.output_mode.clone(),
            output_tool_name: None,
            output_tool_description: None,
            augment_output_preamble: true,
            unhandled_invalid_tool_call: crate::run::spec::UnhandledInvalidToolCall::Fail,
        }
    }

    /// Overwrite the protocol-facing fields from `spec`, leaving model, hooks,
    /// memory and identity untouched. Fails only if `spec.output_schema` is
    /// not a valid JSON schema.
    pub(crate) fn apply_run_spec(
        &mut self,
        spec: &crate::run::spec::RunSpec,
    ) -> Result<(), serde_json::Error> {
        self.preamble.clone_from(&spec.preamble);
        self.static_context.clone_from(&spec.static_context);
        self.additional_params.clone_from(&spec.additional_params);
        self.max_tokens = spec.max_tokens;
        self.temperature = spec.temperature;
        self.tool_choice.clone_from(&spec.tool_choice);
        self.max_turns = spec.effective_max_turns();
        self.output_schema = spec
            .output_schema
            .clone()
            .map(schemars::Schema::try_from)
            .transpose()?;
        self.output_mode = spec.output_mode.clone();
        Ok(())
    }
}

impl Agent {
    /// The protocol-facing configuration of this agent as plain data
    /// ([`RunSpec`](crate::run::spec::RunSpec)): preamble, static context, sampling
    /// parameters, turn budget, tool choice and structured-output policy —
    /// everything a run needs that is not a model, a tool, a hook or a memory.
    pub fn run_spec(&self) -> crate::run::spec::RunSpec {
        self.config.run_spec()
    }
}

impl Agent {
    /// The tool server this agent dispatches tool calls through. Cloning the
    /// handle lets external tool sources (an MCP client handler, for example)
    /// register and refresh tools the agent will see on its next turn.
    pub fn tool_server_handle(&self) -> &crate::tool::server::ToolServerHandle {
        &self.tool_server_handle
    }

    /// Returns the configured agent name.
    pub fn name(&self) -> Option<&str> {
        self.config.name.as_deref()
    }

    /// Returns the configured agent description.
    pub fn description(&self) -> Option<&str> {
        self.config.description.as_deref()
    }

    pub(crate) fn name_or_default(&self) -> &str {
        self.name().unwrap_or(UNKNOWN_AGENT_NAME)
    }

    /// Build a hook-aware [`AgentRunner`] for this agent, seeded with the
    /// agent's default hook stack. Attach more hooks with
    /// [`AgentRunner::add_hook`], then call [`AgentRunner::run`].
    pub fn runner(&self, prompt: impl Into<Message>) -> AgentRunner {
        AgentRunner::from_agent(self, prompt)
    }

    /// Returns the agent's current default model handle.
    pub fn model_handle(&self) -> &ModelHandle {
        &self.config.model
    }

    /// Replace the default model used by runners created after this call.
    ///
    /// Existing runners retain their model snapshot, and replacing one cloned
    /// agent does not mutate another clone. Model-selection hooks may replace
    /// the captured default at each model-call boundary.
    pub fn set_model_handle(&mut self, model: ModelHandle) {
        self.config.model = model;
    }

    /// Erase and install a typed completion model as this agent's new default.
    pub fn set_model<M>(&mut self, model: M)
    where
        M: CompletionModel + 'static,
    {
        self.set_model_handle(ModelHandle::new(model));
    }

    /// Return this agent with a replacement default model handle.
    ///
    /// Model-selection hooks may replace this default for individual calls.
    pub fn with_model_handle(mut self, model: ModelHandle) -> Self {
        self.set_model_handle(model);
        self
    }

    /// Return this agent with an erased typed model as its new default.
    pub fn with_model<M>(mut self, model: M) -> Self
    where
        M: CompletionModel + 'static,
    {
        self.set_model(model);
        self
    }

    /// Resolve the provider-facing tool definitions available for a prompt.
    ///
    /// This read-only view does not expose tool dispatch. Agent execution and
    /// tool lifecycle hooks remain owned by [`Self::runner`].
    pub async fn tool_definitions(
        &self,
        prompt: Option<String>,
    ) -> Result<Vec<ToolDefinition>, ToolServerError> {
        self.tool_server_handle.tool_defs(prompt).await
    }
}

impl Agent {
    /// Run `prompt` through the agent loop. The returned [`AgentRunner`] is the
    /// run: configure it (history, turn budget, tool context, hooks, …) and
    /// `.await` it for the [`PromptResponse`], whose `output` is the accepted
    /// assistant text.
    ///
    /// ```rust,no_run
    /// # use rig_agent::Agent;
    /// # async fn example(agent: Agent) -> Result<(), Box<dyn std::error::Error>> {
    /// let response = agent.prompt("What is 2 + 2?").max_turns(3).await?;
    /// println!("{}", response.output);
    /// # Ok(())
    /// # }
    /// ```
    pub fn prompt(&self, prompt: impl Into<Message>) -> AgentRunner {
        AgentRunner::from_agent(self, prompt)
    }

    /// Run one turn against caller-owned history, appending only the messages
    /// the run committed. Returns the same [`PromptResponse`] as
    /// [`prompt`](Self::prompt).
    #[tracing::instrument(skip(self, prompt, chat_history), fields(agent_name = self.name_or_default()))]
    pub async fn chat(
        &self,
        prompt: impl Into<Message> + WasmCompatSend,
        chat_history: &mut Vec<Message>,
    ) -> Result<PromptResponse, PromptError> {
        let mut response = AgentRunner::from_agent(self, prompt)
            .history(chat_history.clone())
            .await?;
        if let Some(messages) = response.messages.take() {
            chat_history.extend(messages);
        }
        Ok(response)
    }

    /// Run `prompt` as a stream: configure the returned runner, then call
    /// [`AgentRunner::stream`] or [`AgentRunner::run_channel`].
    pub fn stream_prompt(&self, prompt: impl Into<Message>) -> AgentRunner {
        AgentRunner::from_agent(self, prompt)
    }

    /// [`stream_prompt`](Self::stream_prompt) with canonical chat history.
    pub fn stream_chat<I, T>(&self, prompt: impl Into<Message>, chat_history: I) -> AgentRunner
    where
        I: IntoIterator<Item = T>,
        T: Into<Message>,
    {
        AgentRunner::from_agent(self, prompt).history(chat_history)
    }

    /// Run `prompt` and deserialize the accepted structured response as `T`.
    ///
    /// The JSON schema for `T` is generated and sent to the provider as the
    /// run's structured-output schema. Providers that support native structured
    /// outputs constrain the model's response to match it.
    ///
    /// ```rust,ignore
    /// #[derive(Debug, Deserialize, JsonSchema)]
    /// struct WeatherForecast { city: String, temperature_f: f64 }
    ///
    /// let forecast = agent
    ///     .prompt_typed::<WeatherForecast>("What's the weather in NYC?")
    ///     .max_turns(3)
    ///     .await?
    ///     .output;
    /// ```
    pub fn prompt_typed<T>(&self, prompt: impl Into<Message>) -> TypedRun<T>
    where
        T: JsonSchema + DeserializeOwned + WasmCompatSend,
    {
        TypedRun::native(self, prompt)
    }
}

use schemars::JsonSchema;
use serde::de::DeserializeOwned;

#[cfg(test)]
mod request_identity_tests;
