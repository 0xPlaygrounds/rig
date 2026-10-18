//! Agent configuration, request preparation, and prompting entry points.
//!
//! ```no_run
//! # async fn example(agent: rig_agent::Agent) -> Result<(), rig_agent::completion::PromptError> {
//! let response = agent.prompt("Explain ownership.").await?;
//! println!("{}", response.output());
//! # Ok(())
//! # }
//! ```

use super::hook::{HookStack, RequestPatch};
use super::runner::AgentRunner;
use super::typed::TypedRun;
use crate::handlers::{Handlers, ModelHandle};
use crate::run::spec::RunSpec;
use crate::run::{AgentRun, OutputMode};
use crate::{
    completion::{CompletionRequest, Message, PromptError, ToolDefinition},
    run::response::PromptResponse,
    tool::{
        ToolCatalog,
        server::{ToolServerError, ToolServerHandle},
    },
};
use rig_core::completion::ModelRef;
use rig_core::effect::{HandlerDescriptor, HandlerKey, Key, family};
use rig_core::error::ProviderError;

use rig_core::{message::ToolChoice, wasm_compat::WasmCompatSend};
use std::sync::{
    Arc,
    atomic::{AtomicU64, AtomicUsize, Ordering},
};

use super::UNKNOWN_AGENT_NAME;

/// What a completion request was prepared with, including the executable Rig
/// tool names advertised to the provider for this turn.
pub(crate) struct PreparedCompletionRequest {
    /// The messages telemetry records for this attempt.
    pub(crate) telemetry_messages: Vec<Arc<Message>>,
    /// The typed view the request is dispatched to.
    pub(crate) model: ModelHandle,
    /// Exact implementations behind this turn's provider definitions.
    pub(crate) tool_snapshot: Arc<ToolCatalog>,
    /// The definitions the request carries (executable tools plus, in Tool
    /// output mode, the synthetic output tool), for the run's `TurnTools`.
    pub(crate) advertised_tools: Vec<Arc<ToolDefinition>>,
    /// The turn's tool policy, fed back to the run with the response.
    pub(crate) policy: crate::run::TurnPolicy,
    /// Effective structured output-token cap after runner overrides and request
    /// patches. Does not interpret caps passed through additional parameters.
    pub(crate) max_tokens: Option<u64>,
}

/// Retrieve tools and prepare a request from runner defaults and per-turn inputs.
/// Pins executable registrations and the selected model for this attempt.
/// Returns retrieval, preparation, and message-validation failures.
pub(crate) async fn build_prepared_completion_request(
    runner: &crate::agent::AgentRunner,
    ctx: &crate::agent::HookContext,
    model: &ModelHandle,
    prompt: Arc<Message>,
    chat_history: &[Arc<Message>],
    committed_output_tool: Option<&str>,
    request_patch: Option<&RequestPatch>,
) -> Result<(CompletionRequest, PreparedCompletionRequest), ProviderError> {
    let record_telemetry_content = runner.config.record_telemetry_content;
    let tool_server_handle = &runner.tool_server_handle;

    let retrieval_query = prompt.rag_text().or_else(|| {
        chat_history
            .iter()
            .rev()
            .find_map(|message| message.rag_text())
    });

    // Tool retrieval is a `Retrieve` dispatch at the boundary (observe-only
    // for hooks unless one opts in), recorded like every other effect.
    let mut dynamic_tool_ids = Vec::new();
    for (key, kind) in tool_server_handle.retrieval_effects(retrieval_query) {
        let ids = crate::agent::engine::dispatch_effect(runner, ctx, &key, kind)
            .await
            .and_then(|outcome| match outcome {
                rig_core::effect::Answered::Documents(
                    rig_core::effect::RetrievedDocuments::Ids(ids),
                ) => Ok(ids.into_iter().map(|(_, id)| id).collect::<Vec<String>>()),
                other => Err(crate::agent::engine::wrong_outcome("retrieved ids", &other)),
            })
            .map_err(|report| {
                ProviderError::request(format!("Failed to get tool definitions: {report}"))
            })?;
        dynamic_tool_ids.extend(ids);
    }
    let mut tool_snapshot = tool_server_handle.snapshot_with_dynamic(&dynamic_tool_ids);

    let capabilities = model.capabilities();
    let unforced = unforce_output_tool(&runner.config.spec, &capabilities);
    let (spec, committed_output_tool) = match &unforced {
        Some((spec, true)) => (spec, None),
        Some((spec, false)) => (spec, committed_output_tool),
        None => (&*runner.config.spec, committed_output_tool),
    };

    let prepared = crate::run::prepare::prepare_request(
        spec,
        &capabilities,
        chat_history,
        tool_snapshot.take_definitions(),
        committed_output_tool,
        request_patch,
    )?;

    // Narrow dispatch to the tools actually advertised this turn (a per-turn
    // `active_tools` allow-list), so the implementation behind every definition
    // the provider received is the one that runs.
    tool_snapshot.retain_names(prepared.policy.executable());

    let advertised_tools = prepared.tools.clone();
    let policy = prepared.policy.clone();
    let max_tokens = prepared.max_tokens;
    // The agent records the input itself, so the request the provider sees
    // carries the telemetry flag off: one span, no double recording.
    let request = prepared.apply(CompletionRequest::new(prompt));
    let telemetry_messages = if record_telemetry_content {
        request.messages_for_telemetry()
    } else {
        Vec::new()
    };
    let prepared = PreparedCompletionRequest {
        telemetry_messages,
        model: model.clone(),
        tool_snapshot: Arc::new(tool_snapshot),
        advertised_tools,
        policy,
        max_tokens,
    };
    Ok((request, prepared))
}

/// Rig's own structured-output flow (the extractor) forces its output tool.
/// For a model that rejects a forced tool choice, drop that force: use native
/// structured output where the wire composes it with tools, else keep the
/// output tool at `auto`, and say so in the preamble either way. A caller's
/// own forced choice outside that flow is left alone. Returns the spec
/// unforced, and whether the output tool was replaced by native output.
fn unforce_output_tool(
    spec: &RunSpec,
    capabilities: &rig_core::completion::ProviderCapabilities,
) -> Option<(RunSpec, bool)> {
    let name = spec.output_tool_name.as_deref()?;
    let forced = matches!(
        spec.tool_choice,
        Some(ToolChoice::Required | ToolChoice::Specific { .. })
    );
    if !capabilities.rejects_forced_tool_choice || !forced || spec.output_schema.is_none() {
        return None;
    }
    let mut spec = spec.clone();
    spec.tool_choice = None;
    let native = capabilities.composes_native_output_with_tools;
    let line = if native {
        format!(
            "The `{name}` tool is not available for this model. Reply with only the JSON \
             object you would have passed to it."
        )
    } else {
        format!("Call the `{name}` tool with your answer instead of replying in text.")
    };
    spec.preamble = Some(match spec.preamble.take() {
        Some(preamble) => format!("{preamble}\n\n{line}"),
        None => line,
    });
    if native {
        spec.output_mode = OutputMode::Native;
    }
    Some((spec, native))
}

/// Configured model, request settings, hooks, and tool registry for runs.
/// Per-run overrides and dynamic retrieval determine each turn's effective request.
///
/// Default hooks attached with [`AgentBuilder::add_hook`](crate::agent::AgentBuilder::add_hook)
/// are used for every prompt request, plus any added on the request or runner.
///
/// # Example
/// ```no_run
/// use rig_agent::prelude::*;
/// use rig_core::{Model, providers::openai::{self, OpenAI}};
///
/// # async fn run() -> Result<(), Box<dyn std::error::Error>> {
/// let model = OpenAI::from_env()?.completion(openai::GPT_5_2);
///
/// let comedian_agent = AgentBuilder::new(model)
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

/// Configuration shared by the builder and built agent, separate from tool state.
#[derive(Clone)]
pub(crate) struct AgentConfig {
    /// Name of the agent used for logging and debugging
    pub(crate) name: Option<String>,
    /// Agent description. Primarily useful when using sub-agents as part of an agent workflow and converting agents to other formats.
    pub(crate) description: Option<String>,
    /// The handler table every run dispatches through.
    pub(crate) handlers: Handlers,
    /// The owner segment of every key this agent mints
    /// (`<owner>/model:<label>`, ...).
    pub(crate) owner: Arc<str>,
    /// How many anonymous models this agent's values registered.
    pub(crate) anonymous_models: Arc<AtomicUsize>,
    /// The key of the default model in the table.
    pub(crate) model_key: Key<family::Completion>,
    /// The request settings and run policy: the agent's
    /// ([`Agent::run_spec`]), and a runner's overrides on a copy of it.
    pub(crate) spec: Arc<RunSpec>,
    /// Whether to record sensitive request, response, and tool content on GenAI spans.
    ///
    /// Defaults to `false`. Enabling this can expose prompts, retrieved context,
    /// tool results, model responses, and other sensitive or high-cardinality data
    /// through OpenTelemetry span attributes, which can increase observability
    /// backend storage and query costs.
    pub(crate) record_telemetry_content: bool,
    /// Default hook stack applied to every prompt request and runner created
    /// from this agent. Empty by default.
    pub(crate) hooks: HookStack,
    /// The models registered as routes at build (`model_route`), part of
    /// the required row: a program that can select them needs them served.
    pub(crate) route_keys: Vec<Key<family::Completion>>,
    /// The dynamic-context indexes registered at build
    /// (`dynamic_context`), part of the required row: every model call
    /// queries them.
    pub(crate) context_keys: Vec<Key<family::Retrieve>>,
    /// The anonymous model this value selected ([`Agent::set_model`],
    /// [`AgentRunner::using_model_value`]); its registration lives as long
    /// as the values sharing it.
    pub(crate) anonymous_model: Option<Arc<AnonymousModel>>,
    /// Where runs load their conversation from and save it to.
    pub(crate) memory: Option<super::memory::Memory>,
    /// The log of [`memory`](Self::memory) runs use; the agent's name, or
    /// `agent`, when `None`.
    pub(crate) conversation: Option<String>,
}

/// The per-process counter behind an agent's default owner label.
static NEXT_AGENT: AtomicU64 = AtomicU64::new(0);

/// The owner label an agent gets when its builder names none:
/// `agent#<n>`, distinct per process.
pub(crate) fn default_owner() -> String {
    format!("agent#{}", NEXT_AGENT.fetch_add(1, Ordering::Relaxed))
}

/// The registration of a model under a generated label, scoped to the
/// values that selected it: the last clone dropping deregisters the key.
pub(crate) struct AnonymousModel {
    key: Key<family::Completion>,
    handlers: Handlers,
}

impl Drop for AnonymousModel {
    fn drop(&mut self) {
        self.handlers.deregister(self.key.raw());
    }
}

/// Log a registration under a key this agent minted that the table refused:
/// the key changed family.
#[track_caller]
pub(crate) fn register_generated(registered: Result<(), rig_core::error::ErrorReport>) {
    if let Err(report) = registered {
        let caller = std::panic::Location::caller();
        tracing::error!(%report, %caller, "a generated key changed family; the registration was refused");
    }
}

impl AgentConfig {
    /// The unconfigured starting point for a builder over `model`.
    pub(crate) fn new(handlers: Handlers, model_key: Key<family::Completion>) -> Self {
        Self {
            name: None,
            description: None,
            handlers,
            owner: Arc::from(""),
            anonymous_models: Arc::default(),
            model_key,
            spec: Arc::new(RunSpec {
                max_turns: Some(1),
                ..RunSpec::new()
            }),
            record_telemetry_content: false,
            hooks: HookStack::new(),
            route_keys: Vec::new(),
            context_keys: Vec::new(),
            anonymous_model: None,
            memory: None,
            conversation: None,
        }
    }

    /// The request settings to change, copied first while shared.
    pub(crate) fn spec_mut(&mut self) -> &mut RunSpec {
        Arc::make_mut(&mut self.spec)
    }

    /// The model-call budget.
    pub(crate) fn max_turns(&self) -> usize {
        self.spec.effective_max_turns()
    }

    /// Bind the default model's typed view.
    pub(crate) fn model_handle(&self) -> Result<ModelHandle, rig_core::error::ErrorReport> {
        self.handlers.bind(&self.model_key)
    }

    /// The default model's label as registered now (the key's tail when
    /// nothing serves it).
    pub(crate) fn model_label(&self) -> ModelRef {
        match self
            .handlers
            .descriptor(self.model_key.raw())
            .map(|descriptor| descriptor.family)
        {
            Some(rig_core::effect::FamilyDescriptor::Completion { model, .. }) => model,
            _ => {
                let key = self.model_key.as_str();
                let label = key
                    .strip_prefix(&*self.owner)
                    .and_then(|rest| rest.strip_prefix("/model:"));
                ModelRef::new(label.unwrap_or(key))
            }
        }
    }

    /// Bind the model registered under `label`.
    pub(crate) fn model_by_ref(
        &self,
        label: &ModelRef,
    ) -> Result<ModelHandle, rig_core::error::ErrorReport> {
        self.handlers.bind(&self.model_key_for(label.as_str()))
    }

    /// The key this agent mints for `suffix` (`model:<label>`,
    /// `retrieve:context#<n>`), typed by the family registered under it.
    pub(crate) fn key<F: rig_core::effect::Family>(&self, suffix: &str) -> Key<F> {
        Key::new_unchecked(self.raw_key(suffix))
    }

    /// The wire key this agent mints for `suffix`.
    pub(crate) fn raw_key(&self, suffix: &str) -> HandlerKey {
        HandlerKey::from(format!("{}/{suffix}", self.owner))
    }

    /// The key this agent mints for the model labelled `label`.
    pub(crate) fn model_key_for(&self, label: &str) -> Key<family::Completion> {
        self.key(rig_core::effect::model_key(label).as_str())
    }

    /// Register `model` under `label` (replacing any model under it) and
    /// return the key a run selects it by.
    pub(crate) fn register_model(
        &self,
        label: &ModelRef,
        model: impl Into<rig_core::DynModel<rig_core::operation::Completion>>,
    ) -> Key<family::Completion> {
        let key = self.model_key_for(label.as_str());
        let adapter = rig_core::serve::adapters::ModelAdapter::new(label.clone(), model.into());
        register_generated(self.handlers.register(key.raw().clone(), adapter));
        key
    }

    /// Register `model` under a fresh generated label and make it this
    /// value's default; the key leaves the table when the last value
    /// selecting it drops.
    pub(crate) fn select_anonymous_model(
        &mut self,
        model: impl Into<rig_core::DynModel<rig_core::operation::Completion>>,
    ) {
        let n = self.anonymous_models.fetch_add(1, Ordering::SeqCst);
        let key = self.register_model(&ModelRef::new(format!("anonymous#{n}")), model);
        self.model_key = key.clone();
        self.anonymous_model = Some(Arc::new(AnonymousModel {
            key,
            handlers: self.handlers.clone(),
        }));
    }

    /// Make the model registered under `label` this value's default.
    pub(crate) fn select_model_label(&mut self, label: ModelRef) {
        self.model_key = self.model_key_for(label.as_str());
        self.anonymous_model = None;
    }
}

impl Agent {
    /// Return serializable request settings and run policy, excluding live models,
    /// tools, and hooks.
    pub fn run_spec(&self) -> RunSpec {
        RunSpec::clone(&self.config.spec)
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

    /// The key of this agent's default model in its table (a completion
    /// key; `.raw()` for the wire string).
    pub fn model_key(&self) -> &Key<family::Completion> {
        &self.config.model_key
    }

    /// The descriptor of this agent's default model, as registered now.
    pub fn model_descriptor(&self) -> Option<HandlerDescriptor> {
        self.config.handlers.descriptor(self.config.model_key.raw())
    }

    /// The label of this agent's default model, as registered now.
    pub fn model_label(&self) -> Option<ModelRef> {
        self.model_descriptor()
            .and_then(|descriptor| match descriptor.family {
                rig_core::effect::FamilyDescriptor::Completion { model, .. } => Some(model),
                rig_core::effect::FamilyDescriptor::Tool { .. }
                | rig_core::effect::FamilyDescriptor::Embed { .. }
                | rig_core::effect::FamilyDescriptor::Retrieve {}
                | rig_core::effect::FamilyDescriptor::Rerank { .. }
                | rig_core::effect::FamilyDescriptor::Custom { .. } => None,
            })
    }

    /// Register `model` in this agent's table under `label` and return the
    /// label a run selects it by.
    pub fn register_model(
        &self,
        label: impl Into<ModelRef>,
        model: impl Into<rig_core::DynModel<rig_core::operation::Completion>>,
    ) -> ModelRef {
        let label = label.into();
        self.config.register_model(&label, model);
        label
    }

    /// Make the model registered under `label` this agent value's default.
    /// Value semantics: clones of the agent keep their own default.
    pub fn set_model_label(&mut self, label: impl Into<ModelRef>) {
        self.config.select_model_label(label.into());
    }

    /// Register `model` under a generated label and make it this agent
    /// value's default. The registration is scoped to the values that
    /// select it (this agent, its clones, the runners it produces): it
    /// leaves the table when the last of them drops or selects another model.
    pub fn set_model(
        &mut self,
        model: impl Into<rig_core::DynModel<rig_core::operation::Completion>>,
    ) {
        self.config.select_anonymous_model(model);
    }

    /// The owner segment of the keys this agent minted
    /// (`<owner>/model:<label>`, ...): the label given to
    /// [`AgentBuilder::owner`](crate::agent::AgentBuilder::owner), else
    /// `agent#<n>`.
    pub fn owner(&self) -> &str {
        &self.config.owner
    }

    /// The handler table this agent dispatches through.
    pub fn handlers(&self) -> &Handlers {
        &self.config.handlers
    }

    /// [`Agent::set_model_label`] by value.
    pub fn with_model_label(mut self, label: impl Into<ModelRef>) -> Self {
        self.set_model_label(label);
        self
    }

    /// [`Agent::set_model`] by value.
    pub fn with_model(
        mut self,
        model: impl Into<rig_core::DynModel<rig_core::operation::Completion>>,
    ) -> Self {
        self.set_model(model);
        self
    }

    /// The effect row this program can dispatch to: its model, every tool
    /// the registry serves and its retrieval indexes,
    /// each with the family it needs, from the registry rather than what a
    /// run happened to dispatch.
    pub fn required_row(&self) -> rig_core::effect::EffectRow {
        use rig_core::effect::EffectFamily;
        let mut row = rig_core::effect::EffectRow::new();
        row.insert(
            self.config.model_key.raw().clone(),
            EffectFamily::Completion,
        );
        for route in &self.config.route_keys {
            row.insert(route.raw().clone(), EffectFamily::Completion);
        }
        for (_, tool) in self.tool_server_handle.toolset().iter() {
            row.insert(tool.key().raw().clone(), EffectFamily::Tool);
        }
        for key in self.tool_server_handle.retrieval_keys() {
            row.insert(key, EffectFamily::Retrieve);
        }
        for context in &self.config.context_keys {
            row.insert(context.raw().clone(), EffectFamily::Retrieve);
        }
        row
    }

    /// What decides on this agent's behalf: its hook stack's names, then
    /// every layer on its handlers (the handler table's order, outermost
    /// in). The log's `hooks`; a replay under another stack of either is
    /// refused.
    pub fn program_names(&self) -> Vec<String> {
        let mut names = self.config.hooks.names();
        for descriptor in self.config.handlers.descriptors() {
            names.extend(descriptor.layers);
        }
        names
    }

    /// The descriptor currently registered under `key`.
    pub fn handler_descriptor(&self, key: &HandlerKey) -> Option<HandlerDescriptor> {
        self.config.handlers.descriptor(key)
    }

    /// Resolve the provider-facing tool definitions available for a prompt.
    ///
    /// This read-only view does not expose tool dispatch. Agent execution and
    /// tool lifecycle hooks remain owned by [`Self::prompt`].
    pub async fn tool_definitions(
        &self,
        prompt: Option<String>,
    ) -> Result<Vec<Arc<ToolDefinition>>, ToolServerError> {
        self.tool_server_handle.tool_defs(prompt).await
    }
}

impl Agent {
    /// Start a run from `prompt`. The returned [`AgentRunner`] is the run:
    /// configure it (history, turn budget, tool context, hooks, …), then pick
    /// how to drive it. Nothing happens until it is driven.
    ///
    /// - `.await` (or [`run`](AgentRunner::run)) folds the whole loop to a
    ///   [`PromptResponse`], whose `output` is the accepted assistant text.
    /// - [`stream`](AgentRunner::stream) yields every provider delta, tool
    ///   event and the final response as a stream.
    /// - [`run_channel`](AgentRunner::run_channel) splits the run into a
    ///   future and an event feed for a host with its own executor or tick.
    ///
    /// The medium is chosen by the terminal call alone: an awaited runner asks
    /// the provider for a complete response, a streamed one for a stream.
    /// Hooks, tools and the resulting history are the same in each.
    ///
    /// ```rust,no_run
    /// # use rig_agent::Agent;
    /// # use futures::StreamExt;
    /// # async fn example(agent: Agent) -> Result<(), Box<dyn std::error::Error>> {
    /// let response = agent.prompt("What is 2 + 2?").max_turns(3).await?;
    /// println!("{}", response.output());
    ///
    /// let mut stream = agent.prompt("And 3 + 3?").stream();
    /// while let Some(item) = stream.next().await {
    ///     let item = item?;
    ///     // text deltas, tool calls and results, then `FinalResponse`
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn prompt(&self, prompt: impl Into<Message>) -> AgentRunner {
        AgentRunner::from_agent(self, prompt)
    }

    /// Continue a persisted run instead of starting one from a prompt.
    ///
    /// The state a driver serialized between steps (see [`AgentRun`]) is
    /// resumed under this agent's hooks, tools, and handlers, executing pending tool
    /// calls or requesting the next model turn.
    /// The run is authoritative for what it persisted: its prompt, its
    /// history, its turn budget, its invalid-tool-call retry budget and its
    /// unhandled-invalid-tool-call policy, so
    /// [`history`](AgentRunner::history), [`max_turns`](AgentRunner::max_turns)
    /// and [`unhandled_invalid_tool_call`](AgentRunner::unhandled_invalid_tool_call)
    /// on the returned runner have no effect. Everything else still comes
    /// from the agent and the runner: the request shape (preamble, documents,
    /// sampling parameters, additional params, tool choice, output mode),
    /// [`add_hook`](AgentRunner::add_hook),
    /// [`tool_context`](AgentRunner::tool_context),
    /// [`tool_concurrency`](AgentRunner::tool_concurrency), the model
    /// selection and telemetry settings (inbound tool context is driver
    /// state, never part of the persisted run). Invalid-call hooks see, and a
    /// `Skip` is gated by, the tool choice each turn was sent with. One
    /// run-side value still shows through: the output tool the run pinned
    /// stays pinned even though the schema and mode advertising it are the
    /// runner's.
    /// Unanswered tool
    /// calls re-execute on resume: a tool that ran before the suspension but
    /// whose result was not fed to the run runs again (at least once), while
    /// answered calls keep their results. Drive it like any runner: `.await`,
    /// [`stream`](AgentRunner::stream) or
    /// [`run_channel`](AgentRunner::run_channel). The run must have been
    /// suspended by the same rig version.
    ///
    /// ```rust,no_run
    /// # use rig_agent::{Agent, AgentRun};
    /// # async fn example(agent: Agent, saved: &str) -> Result<(), Box<dyn std::error::Error>> {
    /// let run: AgentRun = serde_json::from_str(saved)?;
    /// let response = agent.resume(run).tool_concurrency(4).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn resume(&self, run: AgentRun) -> AgentRunner {
        AgentRunner::resuming(self, run)
    }

    /// Run `prompt` against caller-owned history with the agent's defaults,
    /// then append the messages the run committed to `chat_history`. This is
    /// [`prompt`](Self::prompt) with [`history`](AgentRunner::history) plus
    /// the write-back; the runner form is the one to configure or stream.
    /// The returned [`PromptResponse`] keeps its `messages`.
    ///
    /// The caller's history is the run's input history and the caller owns
    /// persistence. A run that fails appends nothing: the error carries the
    /// history the run had, the caller's vector is unchanged.
    #[tracing::instrument(skip(self, prompt, chat_history), fields(agent_name = self.name_or_default()))]
    pub async fn chat(
        &self,
        prompt: impl Into<Message> + WasmCompatSend,
        chat_history: &mut Vec<Message>,
    ) -> Result<PromptResponse, PromptError> {
        let response = AgentRunner::from_agent(self, prompt)
            .history(chat_history.iter().cloned())
            .await?;
        chat_history.extend(response.messages.iter().cloned());
        Ok(response)
    }

    /// Run `prompt` and deserialize the accepted structured response as `T`.
    ///
    /// The JSON schema for `T` is generated and sent to the provider as the
    /// run's structured-output schema. Providers that support native structured
    /// outputs constrain the model's response to match it.
    ///
    /// ```no_run
    /// # async fn example(agent: rig_agent::Agent) -> Result<(), Box<dyn std::error::Error>> {
    /// #[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
    /// struct WeatherForecast { city: String, temperature_f: f64 }
    /// let forecast = agent
    ///     .prompt_typed::<WeatherForecast>("What's the weather in NYC?")
    ///     .max_turns(3)
    ///     .await?
    ///     .output;
    /// # Ok(())
    /// # }
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
