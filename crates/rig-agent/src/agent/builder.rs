//! Agent construction.
//!
//! `AgentBuilder::new(model)` creates the agent's bus and registers the
//! model on it; every tool, memory backend and retrieval index added to the
//! builder is registered as a handler under a generated key, and the agent
//! keeps the [`BusDriver`](rig_bus::BusDriver) and drives it inline
//! while a run is awaited. `AgentBuilder::over_bus` builds an agent over a
//! host's bus instead: the host drives.
//!
//! The typestate tracks where tools come from:
//! - `NoToolConfig`: no tools yet;
//! - `WithBuilderTools`: tools added through the builder API;
//! - `WithToolServerHandle`: a pre-existing shared [`ToolServerHandle`].
//!
//! Use one or the other, not both.

use std::sync::{Arc, OnceLock};

use rig_bus::{Bus, BusConfig, Dispatcher, Registrar};
use rig_core::serve::ErasedHandler;
use rig_core::serve::adapters::{CompletionAdapter, MemoryAdapter, RetrieveAdapter};
use rig_core::{
    completion::{CompletionModel, Document, ModelRef},
    effect::{HandlerKey, Key, family},
    memory::ConversationMemory,
    vector_store::{VectorSearchRequest, VectorStoreIndex, request::DynamicSearchFilter},
    wasm_compat::{WasmCompatSend, WasmCompatSync},
};
use schemars::{JsonSchema, Schema, schema_for};

use crate::{
    agent::{
        AgentHook, CompletionCallAction, CompletionCallEvent as CompletionCall, HookContext,
        RequestPatch,
    },
    completion::message::ToolChoice,
    tool::{
        DynamicTool, PortableDynamicTool, Tool, ToolSet,
        server::{ToolServer, ToolServerHandle},
    },
};

use super::{Agent, OutputMode, bus::AgentBus, completion::AgentConfig};

/// The `dynamic_context` hook: retrieves documents for the prompt through
/// the agent's bus (an `IndexHandle` bound to the index registered at
/// build) and patches them into the request as extra context.
struct DynamicContext {
    samples: usize,
    /// The index's key, minted at build once the agent's owner is known.
    key: Arc<OnceLock<Key<family::Retrieve>>>,
}

impl AgentHook for DynamicContext {
    async fn on_completion_call(
        &self,
        ctx: &HookContext,
        event: CompletionCall<'_>,
    ) -> CompletionCallAction {
        let query = event.prompt.rag_text().or_else(|| {
            event
                .history
                .iter()
                .rev()
                .find_map(rig_core::completion::Message::rag_text)
        });
        let Some(query) = query else {
            return CompletionCallAction::continue_run();
        };
        let Some(key) = self.key.get() else {
            return CompletionCallAction::stop(
                "dynamic context is keyed at build; this hook was not built",
            );
        };
        let index = match ctx.bind(key) {
            Ok(index) => index,
            Err(report) => {
                return CompletionCallAction::stop(format!(
                    "failed to bind the dynamic context index: {report}"
                ));
            }
        };

        let request = VectorSearchRequest::builder()
            .query(query)
            .samples(self.samples as u64)
            .build();
        match index.top_n::<serde_json::Value>(request).await {
            Ok(results) => CompletionCallAction::patch(RequestPatch::new().extra_context(
                results.into_iter().map(|(_, id, value)| Document {
                    id,
                    text:
                        serde_json::to_string_pretty(&value).unwrap_or_else(|_| value.to_string()),
                    additional_props: Default::default(),
                }),
            )),
            Err(error) => {
                CompletionCallAction::stop(format!("failed to retrieve dynamic context: {error}"))
            }
        }
    }
}

/// Typestate: no tools configured.
#[derive(Default)]
pub struct NoToolConfig;

/// Typestate: a pre-existing shared registry supplies the tools.
pub struct WithToolServerHandle {
    handle: ToolServerHandle,
}

/// Typestate: tools added through the builder.
pub struct WithBuilderTools(ToolServer);

/// Where the built agent's bus comes from.
enum BusSource {
    /// The agent's own bus, created at build with this sizing.
    Owned(BusConfig),
    /// A host's bus; the host drives it. The agent registers on it through
    /// the host's registrar.
    Host(Dispatcher, Registrar),
}

/// The default model: a model the builder registers under a label, or the
/// key of one already registered on a host's bus.
enum DefaultModel {
    Labelled(ModelRef, ErasedHandler),
    /// An explicit key and the host's line that asserted it.
    Key(HandlerKey, &'static std::panic::Location<'static>),
}

/// Builds an [`Agent`].
///
/// Every handler the builder registers (the default model, model routes,
/// memory, dynamic-context indexes) is minted a key under the agent's
/// owner label at build — `<owner>/model:<label>`, `<owner>/memory`,
/// `<owner>/retrieve:context#<n>` — so two agents on one host bus never
/// overwrite each other's handlers. The owner is [`AgentBuilder::owner`]'s
/// label, else `agent#<n>` from a per-process counter; an agent over a
/// host's bus names its owner up front ([`AgentBuilder::over_bus`]).
pub struct AgentBuilder<ToolState = NoToolConfig> {
    config: AgentConfig,
    tool_state: ToolState,
    bus: BusSource,
    owner: Option<String>,
    model: DefaultModel,
    /// Handlers to register at build, by key suffix.
    pending: Vec<(String, ErasedHandler)>,
    /// The dynamic-context hooks' key slots, by key suffix.
    dynamic_contexts: Vec<(String, Arc<OnceLock<Key<family::Retrieve>>>)>,
    memory: bool,
    record_effects: bool,
    retrieval_indexes: usize,
}

impl<ToolState> AgentBuilder<ToolState> {
    /// Name the agent.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.config.name = Some(name.into());
        self
    }

    /// Describe the agent.
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.config.description = Some(description.into());
        self
    }

    /// Set the system prompt.
    pub fn preamble(mut self, preamble: impl Into<String>) -> Self {
        self.config.preamble = Some(preamble.into());
        self
    }

    /// Clear the system prompt.
    pub fn without_preamble(mut self) -> Self {
        self.config.preamble = None;
        self
    }

    /// Append a paragraph to the system prompt.
    pub fn append_preamble(mut self, doc: &str) -> Self {
        self.config.preamble = Some(format!(
            "{}\n{}",
            self.config.preamble.unwrap_or_default(),
            doc
        ));
        self
    }

    /// Add a static context document.
    pub fn context(mut self, doc: impl Into<String>) -> Self {
        self.config.static_context.push(Document {
            id: format!("static_doc_{}", self.config.static_context.len()),
            text: doc.into(),
            additional_props: Default::default(),
        });
        self
    }

    /// Retrieve `samples` documents from `index` for every prompt and add
    /// them as context. The index is registered on the agent's bus.
    pub fn dynamic_context<I, F>(mut self, samples: usize, index: I) -> Self
    where
        I: VectorStoreIndex<Filter = F> + 'static,
        F: DynamicSearchFilter + WasmCompatSend + WasmCompatSync + 'static,
    {
        let n = self.retrieval_indexes;
        self.retrieval_indexes += 1;
        let suffix = format!("retrieve:context#{n}");
        let key = Arc::new(OnceLock::new());
        self.pending.push((
            suffix.clone(),
            ErasedHandler::new(RetrieveAdapter::new(index)),
        ));
        self.dynamic_contexts.push((suffix, key.clone()));
        self.add_hook(DynamicContext { samples, key })
    }

    /// Set the tool choice.
    pub fn tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.config.tool_choice = Some(tool_choice);
        self
    }

    /// Set the default maximum number of turns.
    pub fn default_max_turns(mut self, default_max_turns: usize) -> Self {
        self.config.max_turns = default_max_turns;
        self
    }

    /// Set the sampling temperature.
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.config.temperature = Some(temperature);
        self
    }

    /// Set the output-token cap.
    pub fn max_tokens(mut self, max_tokens: u64) -> Self {
        self.config.max_tokens = Some(max_tokens);
        self
    }

    /// Set provider passthrough parameters.
    pub fn additional_params(mut self, params: serde_json::Value) -> Self {
        self.config.additional_params = Some(params);
        self
    }

    /// Record message content into telemetry spans.
    pub fn record_content_telemetry(mut self, enabled: bool) -> Self {
        self.config.record_telemetry_content = enabled;
        self
    }

    /// Constrain the output to `T`'s JSON schema.
    pub fn output_schema<T>(mut self) -> Self
    where
        T: JsonSchema,
    {
        self.config.output_schema = Some(schema_for!(T));
        self
    }

    /// Constrain the output to a raw JSON schema.
    pub fn output_schema_raw(mut self, schema: Schema) -> Self {
        self.config.output_schema = Some(schema);
        self
    }

    /// Apply a run spec's settings.
    pub fn apply_spec(
        mut self,
        spec: &crate::run::spec::RunSpec,
    ) -> Result<Self, serde_json::Error> {
        self.config.apply_run_spec(spec)?;
        Ok(self)
    }

    /// Set the structured-output mode.
    pub fn output_mode(mut self, mode: OutputMode) -> Self {
        self.config.output_mode = mode;
        self
    }

    /// Persist and load conversation history through `memory`, registered
    /// on the agent's bus.
    pub fn memory<B>(mut self, memory: B) -> Self
    where
        B: ConversationMemory + 'static,
    {
        self.pending.push((
            "memory".to_owned(),
            ErasedHandler::new(MemoryAdapter::new(memory)),
        ));
        self.memory = true;
        self
    }

    /// The conversation id memory loads and saves under.
    pub fn conversation(mut self, id: impl Into<rig_core::id::ConversationId>) -> Self {
        self.config.conversation_id = Some(id.into());
        self
    }

    /// Register another model the run can select by label
    /// (`ModelSelectionAction::select(label)`, `using_model(label)`).
    pub fn model_route<M>(mut self, label: impl Into<ModelRef>, model: M) -> Self
    where
        M: CompletionModel + 'static,
    {
        let label = label.into();
        self.pending.push((
            rig_core::effect::model_key(label.as_str()).to_string(),
            ErasedHandler::new(CompletionAdapter::new(label, model)),
        ));
        self
    }

    /// Name the agent's keys: `<owner>/model:<label>`, `<owner>/memory`,
    /// `<owner>/retrieve:context#<n>`, and its own tools' `<owner>/tool:…`.
    /// The default is the agent's [`name`](Self::name) when one is set —
    /// so a named agent's keys are the same in every process, which a log
    /// meant for replay elsewhere needs — else `agent#<n>` from a
    /// per-process counter.
    pub fn owner(mut self, label: impl Into<String>) -> Self {
        self.owner = Some(label.into());
        self
    }

    /// The bus sizing and serving policy this agent's bus is created with.
    /// The default serves concurrently; the agent's tool concurrency is
    /// governed by the runner, which the cassette corpus was recorded with
    /// at its default of one. An agent over a host's bus reports the
    /// default: the host sized its bus.
    pub fn bus_config(&self) -> BusConfig {
        match &self.bus {
            BusSource::Owned(config) => *config,
            BusSource::Host(..) => BusConfig::default(),
        }
    }

    /// Size the agent's own bus. No effect on an agent over a host's bus.
    pub fn configure_bus(mut self, bus_config: BusConfig) -> Self {
        if let BusSource::Owned(config) = &mut self.bus {
            *config = bus_config;
        }
        self
    }

    /// Record every dispatch into the agent's effect log
    /// ([`Agent::effect_log`]).
    pub fn record_effects(mut self) -> Self {
        self.record_effects = true;
        self
    }

    /// Add a hook.
    pub fn add_hook<H>(mut self, hook: H) -> Self
    where
        H: AgentHook + 'static,
    {
        self.config.hooks.push(hook);
        self
    }

    fn with_tool_state<S>(self, tool_state: S) -> AgentBuilder<S> {
        AgentBuilder {
            config: self.config,
            tool_state,
            bus: self.bus,
            owner: self.owner,
            model: self.model,
            pending: self.pending,
            dynamic_contexts: self.dynamic_contexts,
            memory: self.memory,
            record_effects: self.record_effects,
            retrieval_indexes: self.retrieval_indexes,
        }
    }

    fn build_agent(self, handle: impl FnOnce(ToolState, &str) -> ToolServerHandle) -> Agent {
        /// A host's `over_bus` key that serves another family: the host's
        /// programming error, reported at the host's line.
        #[allow(
            clippy::panic,
            reason = "a wrong-family host key is a programming error at the host's call site, not a runtime condition; `build` stays infallible for every other case"
        )]
        fn host_key_of_another_family(
            key: &HandlerKey,
            caller: &'static std::panic::Location<'static>,
            family: rig_core::effect::EffectFamily,
        ) -> ! {
            panic!(
                "the model key `{key}` handed to `over_bus` at {caller} serves the {family} family, not a completion model"
            )
        }

        let Self {
            mut config,
            tool_state,
            bus,
            owner,
            model,
            mut pending,
            dynamic_contexts,
            memory,
            record_effects,
            retrieval_indexes: _,
        } = self;
        // The owner: the label given, else the agent's name (so a named
        // agent's keys are the same in every process — what a log meant
        // for replay elsewhere needs), else a per-process counter.
        let owner = owner
            .or_else(|| config.name.clone())
            .unwrap_or_else(crate::agent::bus::default_owner);
        config.bus = match bus {
            BusSource::Owned(bus_config) => {
                let (dispatcher, registrar, driver) = Bus::channel_with(bus_config);
                AgentBus::owned(dispatcher, registrar, driver, owner)
            }
            BusSource::Host(dispatcher, registrar) => AgentBus::over(dispatcher, registrar, owner),
        };
        config.model_key = match model {
            DefaultModel::Labelled(label, handler) => {
                let suffix = rig_core::effect::model_key(label.as_str()).to_string();
                pending.insert(0, (suffix, handler));
                config.bus.model_key(label.as_str())
            }
            DefaultModel::Key(key, caller) => {
                // A host's key is asserted, not minted: check what it serves
                // now, and fail at build — at the host's line — rather than
                // at the first run. A key that serves another family is the
                // host's programming error, not a runtime condition, so it
                // is a panic (decided: `build()` stays infallible for every
                // other case); a key nothing serves *yet* is legal — the
                // host may register after building — and fails at the
                // first run as `HandlerUnavailable`.
                if let Some(descriptor) = config.bus.dispatcher().descriptor(&key)
                    && descriptor.family.family()
                        != <family::Completion as rig_core::effect::Family>::FAMILY
                {
                    host_key_of_another_family(&key, caller, descriptor.family.family());
                }
                Key::new_unchecked(key)
            }
        };
        for (suffix, handler) in pending {
            let key = config.bus.raw_key(&suffix);
            crate::agent::bus::register_generated(config.bus.register_erased(key, handler));
        }
        for (suffix, slot) in dynamic_contexts {
            // The slot is this builder's own, filled exactly once.
            let _ = slot.set(config.bus.key(&suffix));
        }
        if memory {
            config.memory_key = Some(config.bus.key("memory"));
        }
        if record_effects {
            crate::agent::bus::register_generated(config.bus.enable_recording());
        }
        let tool_server_handle = handle(tool_state, config.bus.owner());
        tool_server_handle.attach(config.bus.registrar());
        Agent {
            tool_server_handle,
            config,
        }
    }
}

impl AgentBuilder<NoToolConfig> {
    /// An agent over its own bus, with `model` registered as the default
    /// model (label `default`).
    pub fn new<M>(model: M) -> Self
    where
        M: CompletionModel + 'static,
    {
        Self::named_model("default", model)
    }

    /// An agent over its own bus, with `model` registered under `label`.
    pub fn named_model<M>(label: impl Into<ModelRef>, model: M) -> Self
    where
        M: CompletionModel + 'static,
    {
        Self::with_bus_config(BusConfig::default(), label, model)
    }

    /// An agent over its own bus created with `bus_config`, with `model`
    /// registered under `label`.
    pub fn with_bus_config<M>(bus_config: BusConfig, label: impl Into<ModelRef>, model: M) -> Self
    where
        M: CompletionModel + 'static,
    {
        let label = label.into();
        let handler = ErasedHandler::new(CompletionAdapter::new(label.clone(), model));
        Self::start(
            BusSource::Owned(bus_config),
            None,
            DefaultModel::Labelled(label, handler),
        )
    }

    /// An agent over a host's bus, named `owner` on it: the model under
    /// `model` must be registered on the bus (the key is used as given),
    /// and the host drives it. Everything else the builder registers
    /// (memory, routes, tools) goes through `registrar`, keyed under
    /// `owner`.
    #[track_caller]
    pub fn over_bus(
        dispatcher: Dispatcher,
        registrar: Registrar,
        owner: impl Into<String>,
        model: HandlerKey,
    ) -> Self {
        Self::start(
            BusSource::Host(dispatcher, registrar),
            Some(owner.into()),
            DefaultModel::Key(model, std::panic::Location::caller()),
        )
    }

    fn start(bus: BusSource, owner: Option<String>, model: DefaultModel) -> Self {
        // The config's bus and model key are placeholders until build mints
        // the real ones under the owner.
        let (placeholder, placeholder_registrar, _driver) = Bus::channel_with(BusConfig::default());
        let key = Key::new_unchecked(match &model {
            DefaultModel::Labelled(label, _) => HandlerKey::from(label.as_str()),
            DefaultModel::Key(key, _) => key.clone(),
        });
        Self {
            config: AgentConfig::new(
                AgentBus::over(placeholder, placeholder_registrar, String::new()),
                key,
            ),
            tool_state: NoToolConfig,
            bus,
            owner,
            model,
            pending: Vec::new(),
            dynamic_contexts: Vec::new(),
            memory: false,
            record_effects: false,
            retrieval_indexes: 0,
        }
    }

    /// Use a pre-existing shared registry.
    pub fn tool_server_handle(
        self,
        handle: ToolServerHandle,
    ) -> AgentBuilder<WithToolServerHandle> {
        self.with_tool_state(WithToolServerHandle { handle })
    }

    fn into_tool_builder(self) -> AgentBuilder<WithBuilderTools> {
        self.with_tool_state(WithBuilderTools(ToolServer::new()))
    }

    /// Add a typed tool.
    pub fn tool<T>(self, tool: T) -> AgentBuilder<WithBuilderTools>
    where
        T: Tool + 'static,
    {
        self.into_tool_builder().tool(tool)
    }

    /// Build the agent with no tools.
    pub fn build(self) -> Agent {
        self.build_agent(|_, owner| ToolServer::new().owner(owner).run())
    }
}

impl AgentBuilder<NoToolConfig> {
    /// Add a runtime-defined tool.
    pub fn dynamic_tool(self, tool: DynamicTool) -> AgentBuilder<WithBuilderTools> {
        self.into_tool_builder().dynamic_tool(tool)
    }

    /// Add a portable tool.
    pub fn portable_dynamic_tool(
        self,
        tool: PortableDynamicTool,
    ) -> AgentBuilder<WithBuilderTools> {
        self.into_tool_builder().portable_dynamic_tool(tool)
    }

    /// Add runtime-defined tools.
    pub fn dynamic_tools(self, tools: Vec<DynamicTool>) -> AgentBuilder<WithBuilderTools> {
        self.into_tool_builder().dynamic_tools(tools)
    }

    /// Add retrievable tools chosen per request by `index`.
    pub fn retrieved_tools<I, F>(
        self,
        sample: usize,
        index: I,
        toolset: ToolSet,
    ) -> AgentBuilder<WithBuilderTools>
    where
        I: VectorStoreIndex<Filter = F> + 'static,
        F: DynamicSearchFilter + WasmCompatSend + WasmCompatSync + 'static,
    {
        self.into_tool_builder()
            .retrieved_tools(sample, index, toolset)
    }
}

impl AgentBuilder<WithToolServerHandle> {
    /// Build the agent over the shared registry.
    pub fn build(self) -> Agent {
        self.build_agent(|state, _| state.handle)
    }
}

impl AgentBuilder<WithBuilderTools> {
    fn map_server(self, register: impl FnOnce(ToolServer) -> ToolServer) -> Self {
        let Self {
            config,
            tool_state,
            bus,
            owner,
            model,
            pending,
            dynamic_contexts,
            memory,
            record_effects,
            retrieval_indexes,
        } = self;
        Self {
            config,
            tool_state: WithBuilderTools(register(tool_state.0)),
            bus,
            owner,
            model,
            pending,
            dynamic_contexts,
            memory,
            record_effects,
            retrieval_indexes,
        }
    }

    /// Add a typed tool.
    pub fn tool<T>(self, tool: T) -> Self
    where
        T: Tool + 'static,
    {
        self.map_server(|server| server.tool(tool))
    }

    /// Add a runtime-defined tool.
    pub fn dynamic_tool(self, tool: DynamicTool) -> Self {
        self.map_server(|server| server.dynamic_tool(tool))
    }

    /// Add a portable tool.
    pub fn portable_dynamic_tool(self, tool: PortableDynamicTool) -> Self {
        self.map_server(|server| server.portable_dynamic_tool(tool))
    }

    /// Add runtime-defined tools.
    pub fn dynamic_tools(self, tools: Vec<DynamicTool>) -> Self {
        self.map_server(|server| server.dynamic_tools(tools))
    }

    /// Add retrievable tools chosen per request by `index`.
    pub fn retrieved_tools<I, F>(self, sample: usize, index: I, toolset: ToolSet) -> Self
    where
        I: VectorStoreIndex<Filter = F> + 'static,
        F: DynamicSearchFilter + WasmCompatSend + WasmCompatSync + 'static,
    {
        self.map_server(|server| server.retrieved_tools(sample, index, toolset))
    }

    /// Build the agent with the builder's tools.
    pub fn build(self) -> Agent {
        // The builder's own registry takes the agent's owner, so a named
        // agent's tool keys are stable too.
        self.build_agent(|state, owner| state.0.owner(owner).run())
    }
}

#[cfg(test)]
mod tests;
