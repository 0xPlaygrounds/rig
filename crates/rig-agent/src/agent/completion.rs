use super::hook::{HookStack, RequestPatch};
use super::run::OutputMode;
use super::runner::AgentRunner;
use super::typed::TypedRun;
use crate::{
    completion::{
        CompletionError, CompletionModel, CompletionRequest, CompletionRequestBuilder, Document,
        Message, PromptError, ToolDefinition,
    },
    run::response::PromptResponse,
    tool::server::{ToolRegistrySnapshot, ToolServerError, ToolServerHandle},
};
use rig_core::bus::{BusDriver, Dispatcher, ModelHandle};
use rig_core::completion::ModelRef;
use rig_core::effect::{EffectLog, HandlerDescriptor, Key, family};
use rig_core::id::ConversationId;

use super::bus::AgentBus;
use rig_core::{message::ToolChoice, wasm_compat::WasmCompatSend};
use std::{collections::BTreeSet, sync::Arc};

use super::UNKNOWN_AGENT_NAME;

/// A prepared completion request plus the executable Rig tool names advertised
/// to the provider for this turn.
pub(crate) struct PreparedCompletionRequest {
    /// Builder carrying the selected model handle: request preparation ran
    /// against this handle's captured capabilities, and the same handle
    /// executes the prepared request.
    pub(crate) request: CompletionRequest,
    /// The messages telemetry records for this attempt.
    pub(crate) telemetry_messages: Vec<Message>,
    /// The typed view the request is dispatched to.
    pub(crate) model: ModelHandle,
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
    ctx: &crate::agent::HookContext,
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

    // Tool retrieval is a `Retrieve` dispatch at the boundary (observe-only
    // for hooks unless one opts in), recorded like every other effect.
    let mut dynamic_tool_ids = Vec::new();
    for (key, kind) in tool_server_handle.retrieval_effects(retrieval_query) {
        let ids = crate::agent::engine::dispatch_effect(
            &runner.config.hooks,
            ctx,
            runner.config.bus.dispatcher(),
            &key,
            kind,
        )
        .await
        .and_then(|outcome| match outcome {
            rig_core::effect::Outcome::Documents(rig_core::effect::RetrievedDocuments::Ids(
                ids,
            )) => Ok(ids.into_iter().map(|(_, id)| id).collect::<Vec<String>>()),
            other => Err(crate::agent::engine::wrong_outcome("retrieved ids", &other)),
        })
        .map_err(|report| {
            CompletionError::RequestError(
                format!("Failed to get tool definitions: {report}").into(),
            )
        })?;
        dynamic_tool_ids.extend(ids);
    }
    let mut tool_snapshot = tool_server_handle.snapshot_with_dynamic(&dynamic_tool_ids);

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
    let builder = prepared
        .apply(CompletionRequestBuilder::unbound(prompt))
        .record_content_telemetry(record_telemetry_content);
    let telemetry_messages = if record_telemetry_content {
        builder.messages_for_telemetry()
    } else {
        Vec::new()
    };
    // The agent records the input itself, so the request the provider sees
    // carries the flag off: one span, no double recording.
    let request = builder.record_content_telemetry(false).build();

    Ok(PreparedCompletionRequest {
        request,
        telemetry_messages,
        model: model.clone(),
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
    /// The bus every run dispatches through.
    pub(crate) bus: AgentBus,
    /// The key of the default model on the bus.
    pub(crate) model_key: Key<family::Completion>,
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
    pub(crate) memory_key: Option<Key<family::Memory>>,
    /// Optional conversation id used when none is set per-request.
    pub(crate) conversation_id: Option<ConversationId>,
    /// The anonymous model this value selected ([`Agent::set_model`],
    /// [`AgentRunner::using_model_value`]); its registration lives as long
    /// as the values sharing it.
    pub(crate) anonymous_model: Option<std::sync::Arc<super::bus::AnonymousModel>>,
}

impl AgentConfig {
    /// The unconfigured starting point for a builder over `model`.
    pub(crate) fn new(bus: AgentBus, model_key: Key<family::Completion>) -> Self {
        Self {
            name: None,
            description: None,
            bus,
            model_key,
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
            memory_key: None,
            conversation_id: None,
            anonymous_model: None,
        }
    }

    /// Bind the default model's typed view.
    pub(crate) fn model_handle(&self) -> Result<ModelHandle, rig_core::error::ErrorReport> {
        self.bus.dispatcher().bind(&self.model_key)
    }

    /// The default model's label as registered now (the key's tail when
    /// nothing serves it).
    pub(crate) fn model_ref(&self) -> ModelRef {
        match self
            .bus
            .dispatcher()
            .descriptor(self.model_key.raw())
            .map(|descriptor| descriptor.family)
        {
            Some(rig_core::effect::FamilyDescriptor::Completion { model, .. }) => model,
            Some(rig_core::effect::FamilyDescriptor::Tool { .. })
            | Some(rig_core::effect::FamilyDescriptor::Embed { .. })
            | Some(rig_core::effect::FamilyDescriptor::Memory {})
            | Some(rig_core::effect::FamilyDescriptor::Retrieve {})
            | Some(rig_core::effect::FamilyDescriptor::Rerank { .. })
            | Some(rig_core::effect::FamilyDescriptor::Custom { .. })
            | None => ModelRef::new(
                self.bus
                    .model_label(self.model_key.raw())
                    .unwrap_or(self.model_key.as_str()),
            ),
        }
    }

    /// Bind the model registered under `label`.
    pub(crate) fn model_by_ref(
        &self,
        label: &ModelRef,
    ) -> Result<ModelHandle, rig_core::error::ErrorReport> {
        self.bus
            .dispatcher()
            .bind(&self.bus.model_key(label.as_str()))
    }

    /// Bind the memory handle, when memory is configured.
    pub(crate) fn memory_handle(
        &self,
    ) -> Option<Result<rig_core::bus::MemoryHandle, rig_core::error::ErrorReport>> {
        self.memory_key
            .as_ref()
            .map(|key| self.bus.dispatcher().bind(key))
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

    /// The key of this agent's default model on its bus (a completion key;
    /// `.raw()` for the wire string).
    pub fn model_key(&self) -> &Key<family::Completion> {
        &self.config.model_key
    }

    /// The descriptor of this agent's default model, as registered now.
    pub fn model_descriptor(&self) -> Option<HandlerDescriptor> {
        self.config
            .bus
            .dispatcher()
            .descriptor(self.config.model_key.raw())
    }

    /// The label of this agent's default model, as registered now.
    pub fn model_ref(&self) -> Option<ModelRef> {
        self.model_descriptor()
            .and_then(|descriptor| match descriptor.family {
                rig_core::effect::FamilyDescriptor::Completion { model, .. } => Some(model),
                rig_core::effect::FamilyDescriptor::Tool { .. }
                | rig_core::effect::FamilyDescriptor::Embed { .. }
                | rig_core::effect::FamilyDescriptor::Memory {}
                | rig_core::effect::FamilyDescriptor::Retrieve {}
                | rig_core::effect::FamilyDescriptor::Rerank { .. }
                | rig_core::effect::FamilyDescriptor::Custom { .. } => None,
            })
    }

    /// Register `model` on this agent's bus under `label` and return the
    /// label a run selects it by.
    pub fn register_model<M>(&self, label: impl Into<ModelRef>, model: M) -> ModelRef
    where
        M: CompletionModel + 'static,
    {
        let label = label.into();
        self.config.bus.register_model(&label, model);
        label
    }

    /// Make the model registered under `label` this agent value's default.
    /// Value semantics: clones of the agent keep their own default.
    pub fn set_model_ref(&mut self, label: impl Into<ModelRef>) {
        self.config.model_key = self.config.bus.model_key(label.into().as_str());
        self.config.anonymous_model = None;
    }

    /// Register `model` under a generated label and make it this agent
    /// value's default. The registration is scoped to the values that
    /// select it (this agent, its clones, the runners it produces): it
    /// leaves the bus when the last of them drops or selects another model.
    pub fn set_model<M>(&mut self, model: M)
    where
        M: CompletionModel + 'static,
    {
        let anonymous = self.config.bus.register_anonymous_model(model);
        self.config.model_key = anonymous.key().clone();
        self.config.anonymous_model = Some(anonymous);
    }

    /// The owner segment of the keys this agent minted
    /// (`<owner>/model:<label>`, `<owner>/memory`, ...): the label given to
    /// [`AgentBuilder::owner`](crate::agent::AgentBuilder::owner), else
    /// `agent#<n>`.
    pub fn owner(&self) -> &str {
        self.config.bus.owner()
    }

    /// [`Agent::set_model_ref`] by value.
    pub fn with_model_ref(mut self, label: impl Into<ModelRef>) -> Self {
        self.set_model_ref(label);
        self
    }

    /// [`Agent::set_model`] by value.
    pub fn with_model<M>(mut self, model: M) -> Self
    where
        M: CompletionModel + 'static,
    {
        self.set_model(model);
        self
    }

    /// The effect log recorded so far, when the agent was built with
    /// [`AgentBuilder::record_effects`](super::AgentBuilder::record_effects).
    pub fn effect_log(&self) -> Option<EffectLog> {
        self.config.bus.effect_log().map(|log| self.stamp(log))
    }

    /// Take the recorded effect log, leaving the recorder empty.
    pub fn take_effect_log(&self) -> Option<EffectLog> {
        self.config.bus.take_effect_log().map(|log| self.stamp(log))
    }

    /// A stable hash of this agent's run spec, what a log it records
    /// carries in its header and what [`check_replayable`](Self::check_replayable)
    /// compares.
    pub fn run_spec_hash(&self) -> u64 {
        rig_core::effect::stable_hash(&self.config.run_spec()).unwrap_or_default()
    }

    fn stamp(&self, mut log: EffectLog) -> EffectLog {
        log.header.run_spec = Some(self.run_spec_hash());
        log
    }

    /// Whether `log` can be replayed by this agent: the log's format is this
    /// rig's, its run spec hash is this agent's, and every key its
    /// signature names is served on this agent's bus by a handler of the
    /// recorded family. Refused up front, with both sides in the message,
    /// rather than at the record where the run would have diverged.
    pub fn check_replayable(&self, log: &EffectLog) -> Result<(), rig_core::error::ErrorReport> {
        rig_core::bus::EffectLogReplayer::check_header(log)?;
        if let Some(recorded) = log.header.run_spec {
            let mine = self.run_spec_hash();
            if recorded != mine {
                return Err(rig_core::error::ErrorReport::new(
                    rig_core::error::ErrorKind::Internal,
                    format!(
                        "replay refused: the log was recorded under run spec {recorded:#018x}, this agent runs under {mine:#018x}"
                    ),
                ));
            }
        }
        for (key, family) in &log.header.signature {
            match self.config.bus.dispatcher().descriptor(key) {
                Some(descriptor) if descriptor.family.family() == *family => {}
                Some(descriptor) => {
                    return Err(rig_core::error::ErrorReport::new(
                        rig_core::error::ErrorKind::HandlerUnavailable,
                        format!(
                            "replay refused: `{key}` serves {} on this bus, the log needs {family}",
                            descriptor.family.family()
                        ),
                    ));
                }
                None => {
                    return Err(rig_core::error::ErrorReport::new(
                        rig_core::error::ErrorKind::HandlerUnavailable,
                        format!("replay refused: nothing serves `{key}`, which the log needs"),
                    ));
                }
            }
        }
        Ok(())
    }

    /// Whether this agent owns and drives its own bus driver.
    pub fn owns_bus(&self) -> bool {
        self.config.bus.owns_driver()
    }

    /// Take the agent apart: the dispatcher and the driver leave together,
    /// so whoever gets the dispatcher also gets the duty to drive. Fails
    /// when the agent was built over a host's bus or when another clone of
    /// it still shares the driver.
    pub fn into_parts(self) -> Result<AgentParts, Box<Agent>> {
        let Agent {
            mut config,
            tool_server_handle,
        } = self;
        let detached = config.bus.detached();
        let bus = std::mem::replace(&mut config.bus, detached);
        let registrar = bus.registrar().clone();
        match bus.try_into_parts() {
            Ok((dispatcher, driver)) => Ok(AgentParts {
                dispatcher,
                registrar,
                driver,
                agent: Agent {
                    config,
                    tool_server_handle,
                },
            }),
            Err(bus) => {
                config.bus = bus;
                Err(Box::new(Agent {
                    config,
                    tool_server_handle,
                }))
            }
        }
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

/// An agent taken apart by [`Agent::into_parts`]: the same agent, now over
/// the bus as a host would hold it, plus the dispatcher and the driver.
pub struct AgentParts {
    /// The bus's client half.
    pub dispatcher: Dispatcher,
    /// The bus's registration handle: register on the bus once the driver
    /// is spawned.
    pub registrar: rig_core::bus::Registrar,
    /// The bus's serving half: spawn it or drive it, or the agent hangs.
    pub driver: BusDriver,
    /// The agent, now over the bus rather than owning it.
    pub agent: Agent,
}
