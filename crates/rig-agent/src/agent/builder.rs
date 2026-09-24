//! Agent construction over an owned or host-driven bus. Typestate permits either
//! builder-supplied tools or a shared tool server, not both.
//!
//! ```
//! use rig_agent::{Agent, AgentBuilder, core::completion::CompletionModel};
//! fn assistant(model: impl CompletionModel + 'static) -> Agent {
//!     AgentBuilder::new(model).preamble("Be concise.").build()
//! }
//! ```

use std::sync::{Arc, OnceLock};

use crate::bus::{Bus, Dispatcher, Recording, Registrar};
use rig_core::serve::ServingPolicy;
use rig_core::serve::adapters::{CompletionAdapter, MemoryAdapter, RetrieveAdapter};
use rig_core::serve::{ErasedHandler, Recorder};
use rig_core::{
    completion::{CompletionModel, Document, ModelRef},
    effect::{HandlerKey, Key, family},
    memory::ConversationMemory,
    vector_store::{VectorSearchRequest, VectorStoreIndex, request::DynamicSearchFilter},
    wasm_compat::{WasmCompatSend, WasmCompatSync},
};
use schemars::{JsonSchema, Schema, schema_for};

use crate::{
    agent::{AgentHook, CompletionCallAction, CompletionCallEvent, HookContext, RequestPatch},
    completion::message::ToolChoice,
    tool::{
        DynamicTool, Tool, ToolSet,
        server::{ToolServer, ToolServerHandle},
    },
};

use super::{Agent, OutputMode, completion::AgentConfig, drive::AgentBus};

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
        event: CompletionCallEvent<'_>,
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
    Owned(ServingPolicy),
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
/// owner label at build: `<owner>/model:<label>`, `<owner>/memory`, and
/// `<owner>/retrieve:context#<n>`. Hosts sharing a bus must choose distinct
/// owners to avoid replacing each other's bindings. The explicit owner takes
/// precedence over the agent name, then a generated `agent#<n>` label.
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
    recorder: Option<Recording>,
    retrieval_indexes: usize,
    /// The labels `model_route` registered, in order.
    routes: Vec<String>,
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

    /// Retrieve `samples` documents through a retrieval-family handler registered
    /// under the next context key, using the dynamic-context hook lifecycle.
    pub fn dynamic_context_handler(
        mut self,
        samples: usize,
        handler: impl rig_core::serve::Serve + 'static,
    ) -> Self {
        let n = self.retrieval_indexes;
        self.retrieval_indexes += 1;
        let suffix = format!("retrieve:context#{n}");
        let key = Arc::new(OnceLock::new());
        self.pending
            .push((suffix.clone(), ErasedHandler::new(handler)));
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

    /// Enable or disable sensitive message content on telemetry spans.
    /// Disabled by default; enabling may expose prompts, responses, and tool data.
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

    /// Register a memory-family handler under the agent's memory key.
    pub fn memory_handler(mut self, handler: impl rig_core::serve::Serve + 'static) -> Self {
        self.pending
            .push(("memory".to_owned(), ErasedHandler::new(handler)));
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
        self.routes.push(label.as_str().to_owned());
        self.pending.push((
            rig_core::effect::model_key(label.as_str()).to_string(),
            ErasedHandler::new(CompletionAdapter::new(label, model)),
        ));
        self
    }

    /// Register a completion-family handler under the route key for `label`.
    /// Includes that route in the program's required effect row.
    pub fn model_route_handler(
        mut self,
        label: impl Into<ModelRef>,
        handler: impl rig_core::serve::Serve + 'static,
    ) -> Self {
        let label = label.into();
        self.routes.push(label.as_str().to_owned());
        self.pending.push((
            rig_core::effect::model_key(label.as_str()).to_string(),
            ErasedHandler::new(handler),
        ));
        self
    }

    /// Name the agent's keys: `<owner>/model:<label>`, `<owner>/memory`,
    /// `<owner>/retrieve:context#<n>`, and its own tools' `<owner>/tool:…`.
    /// Defaults to the agent's name, or a process-local `agent#<n>` counter.
    /// Use stable, distinct owners for replay and agents sharing a bus.
    pub fn owner(mut self, label: impl Into<String>) -> Self {
        self.owner = Some(label.into());
        self
    }

    /// Configure the owned bus, which serves concurrently by default.
    /// Runner tool concurrency is independent. Has no effect on a host-owned
    /// bus and emits a warning in that case.
    pub fn configure_bus(mut self, bus_config: ServingPolicy) -> Self {
        match &mut self.bus {
            BusSource::Owned(config) => *config = bus_config,
            BusSource::Host(..) => tracing::warn!(
                "AgentBuilder::configure_bus has no effect over a host's bus; the host sized it"
            ),
        }
        self
    }

    /// Observe every dispatch with a caller-owned recorder.
    ///
    /// Keep a clone of a shared recorder to inspect its observations. The
    /// driver retains the installed instance even after [`Agent::into_parts`].
    /// Like [`BusDriver::record_to`](crate::bus::BusDriver::record_to), this
    /// requires a thread-safe recorder on every target because reply observers
    /// cross the bus's thread-safe channels, including on browser WASM.
    ///
    /// An agent over a host's bus cannot install a recorder; the host owns
    /// that driver's recording policy. Asking here fails at build.
    pub fn record_to(mut self, recorder: impl Recorder + Send + Sync) -> Self {
        self.recorder = Some(Recording::new(recorder));
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
            recorder: self.recorder,
            retrieval_indexes: self.retrieval_indexes,
            routes: self.routes,
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

        /// Panic at the host's construction site when an agent tries to replace
        /// recording policy on a driver it does not own.
        #[allow(
            clippy::panic,
            reason = "recording over a host's bus is a programming error at the host's call site, not a runtime condition; `build` stays infallible for every other case"
        )]
        fn recording_over_a_hosts_bus(caller: &'static std::panic::Location<'static>) -> ! {
            panic!(
                "the agent built over a host's bus at {caller} cannot record: the host records through its driver (`BusDriver::record_to`)"
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
            recorder,
            retrieval_indexes: _,
            routes,
        } = self;
        let host = match &model {
            DefaultModel::Key(_, caller) => Some(*caller),
            DefaultModel::Labelled(..) => None,
        };
        // Named agents need stable keys across processes for replay.
        let owner = owner
            .or_else(|| config.name.clone())
            .unwrap_or_else(crate::agent::drive::default_owner);
        config.bus = match bus {
            BusSource::Owned(bus_config) => {
                let (dispatcher, registrar, driver) = Bus::channel_with(bus_config);
                AgentBus::owned(dispatcher, registrar, driver, owner, bus_config)
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
                // Reject an incompatible host binding now, but allow missing keys
                // because hosts may register them after construction.
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
            crate::agent::drive::register_generated(config.bus.register_erased(key, handler));
        }
        for (suffix, slot) in dynamic_contexts {
            // The slot is this builder's own, filled exactly once.
            let key = config.bus.key(&suffix);
            config.context_keys.push(key.clone());
            let _ = slot.set(key);
        }
        config.route_keys = routes
            .iter()
            .map(|label| config.bus.model_key(label))
            .collect();
        if memory {
            config.memory_key = Some(config.bus.key("memory"));
        } else if let Some(conversation) = &config.conversation_id {
            // Warn once at construction rather than silently ignoring memory on each run.
            tracing::warn!(
                %conversation,
                "AgentBuilder::conversation set without memory(..) or memory_handler(..): nothing is loaded or saved"
            );
        }
        if let Some(recorder) = recorder {
            match (host, config.bus.record_to(recorder)) {
                (Some(caller), Err(_)) => recording_over_a_hosts_bus(caller),
                (None, registered) => crate::agent::drive::register_generated(registered),
                (Some(_), Ok(())) => {}
            }
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
    /// Size the bus with [`configure_bus`](Self::configure_bus).
    pub fn named_model<M>(label: impl Into<ModelRef>, model: M) -> Self
    where
        M: CompletionModel + 'static,
    {
        let label = label.into();
        let handler = ErasedHandler::new(CompletionAdapter::new(label.clone(), model));
        Self::start(
            BusSource::Owned(ServingPolicy::default()),
            None,
            DefaultModel::Labelled(label, handler),
        )
    }

    /// Build over a host-driven bus using `model` verbatim and owner-qualified
    /// keys for additional handlers. `dispatcher` and `registrar` must name the
    /// same bus; register the completion model before running the agent.
    /// Building panics for an existing wrong-family model binding or a requested
    /// recorder, since the host owns recording policy.
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
        let (placeholder, placeholder_registrar, _driver) =
            Bus::channel_with(ServingPolicy::default());
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
            recorder: None,
            retrieval_indexes: 0,
            routes: Vec::new(),
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

    /// Add retrievable tools chosen per request by `handler`, a
    /// retrieval-family handler such as a replayer answering a recorded
    /// index; see [`ToolServer::retrieved_tools_handler`].
    pub fn retrieved_tools_handler(
        self,
        sample: usize,
        handler: impl rig_core::serve::Serve + 'static,
        toolset: ToolSet,
    ) -> AgentBuilder<WithBuilderTools> {
        self.into_tool_builder()
            .retrieved_tools_handler(sample, handler, toolset)
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
            recorder,
            retrieval_indexes,
            routes,
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
            recorder,
            retrieval_indexes,
            routes,
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

    /// Add retrievable tools chosen per request by `handler`, a
    /// retrieval-family handler such as a replayer answering a recorded
    /// index; see [`ToolServer::retrieved_tools_handler`].
    pub fn retrieved_tools_handler(
        self,
        sample: usize,
        handler: impl rig_core::serve::Serve + 'static,
        toolset: ToolSet,
    ) -> Self {
        self.map_server(|server| server.retrieved_tools_handler(sample, handler, toolset))
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
