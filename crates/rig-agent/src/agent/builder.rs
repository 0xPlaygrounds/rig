//! Agent construction.
//!
//! `AgentBuilder::new(model)` creates the agent's bus and registers the
//! model on it; every tool, memory backend and retrieval index added to the
//! builder is registered as a handler under a generated key, and the agent
//! keeps the [`BusDriver`](rig_core::bus::BusDriver) and drives it inline
//! while a run is awaited. `AgentBuilder::over_bus` builds an agent over a
//! host's bus instead: the host drives.
//!
//! The typestate tracks where tools come from:
//! - `NoToolConfig`: no tools yet;
//! - `WithBuilderTools`: tools added through the builder API;
//! - `WithToolServerHandle`: a pre-existing shared [`ToolServerHandle`].
//!
//! Use one or the other, not both.

use rig_core::{
    bus::{
        Bus, BusConfig, Dispatcher,
        adapters::{CompletionAdapter, MemoryAdapter, RetrieveAdapter},
        model_key,
    },
    completion::{CompletionModel, Document, ModelRef},
    effect::HandlerKey,
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
    key: HandlerKey,
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
        let Some(dispatcher) = ctx.dispatcher() else {
            return CompletionCallAction::stop(
                "dynamic context needs the agent's bus, which this run has none of",
            );
        };
        let index = match dispatcher.handle::<rig_core::effect::family::Retrieve>(&self.key) {
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

/// Builds an [`Agent`].
pub struct AgentBuilder<ToolState = NoToolConfig> {
    config: AgentConfig,
    tool_state: ToolState,
    /// Handlers the builder registers on the driver before it is spawned:
    /// the bus is created at `new`, so registration is immediate.
    bus_config: BusConfig,
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
        let key = HandlerKey::from(format!("retrieve:context#{n}"));
        crate::agent::bus::register_generated(
            self.config
                .bus
                .dispatcher()
                .register(key.clone(), RetrieveAdapter::new(index)),
        );
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
        let key = HandlerKey::from("memory");
        crate::agent::bus::register_generated(
            self.config
                .bus
                .dispatcher()
                .register(key.clone(), MemoryAdapter::new(memory)),
        );
        self.config.memory_key = Some(key);
        self
    }

    /// The conversation id memory loads and saves under.
    pub fn conversation(mut self, id: impl Into<rig_core::id::ConversationId>) -> Self {
        self.config.conversation_id = Some(id.into());
        self
    }

    /// Register another model the run can select by label
    /// (`ModelSelectionAction::select(label)`, `using_model(label)`).
    pub fn model_route<M>(self, label: impl Into<ModelRef>, model: M) -> Self
    where
        M: CompletionModel + 'static,
    {
        self.config.bus.register_model(&label.into(), model);
        self
    }

    /// The bus sizing and serving policy this agent's bus was created with
    /// (`AgentBuilder::with_bus_config`). The default serves concurrently;
    /// the agent's tool concurrency is governed by the runner, which the
    /// cassette corpus was recorded with at its default of one.
    pub fn bus_config(&self) -> BusConfig {
        self.bus_config
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
            bus_config: self.bus_config,
            record_effects: self.record_effects,
            retrieval_indexes: self.retrieval_indexes,
        }
    }

    fn build_agent(mut self, handle: impl FnOnce(ToolState) -> ToolServerHandle) -> Agent {
        let tool_server_handle = handle(self.tool_state);
        tool_server_handle.attach(self.config.bus.dispatcher());
        if self.record_effects {
            self.config.bus.enable_recording();
        }
        Agent {
            tool_server_handle,
            config: self.config,
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
        let (dispatcher, mut driver) = Bus::channel_with(bus_config);
        let key = model_key(label.as_str());
        crate::agent::bus::register_generated(
            driver.register(key.clone(), CompletionAdapter::new(label, model)),
        );
        let bus = AgentBus::owned(dispatcher, driver);
        Self {
            config: AgentConfig::new(bus, key),
            tool_state: NoToolConfig,
            bus_config,
            record_effects: false,
            retrieval_indexes: 0,
        }
    }

    /// An agent over a host's bus: the model under `model` must be
    /// registered on `dispatcher`, and the host drives the bus.
    pub fn over_bus(dispatcher: Dispatcher, model: HandlerKey) -> Self {
        Self {
            config: AgentConfig::new(AgentBus::over(dispatcher), model),
            tool_state: NoToolConfig,
            bus_config: BusConfig::default(),
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
        self.build_agent(|_| ToolServer::new().run())
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
        self.build_agent(|state| state.handle)
    }
}

impl AgentBuilder<WithBuilderTools> {
    fn map_server(self, register: impl FnOnce(ToolServer) -> ToolServer) -> Self {
        let Self {
            config,
            tool_state,
            bus_config,
            record_effects,
            retrieval_indexes,
        } = self;
        Self {
            config,
            tool_state: WithBuilderTools(register(tool_state.0)),
            bus_config,
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
        self.build_agent(|state| state.0.run())
    }
}

#[cfg(test)]
mod tests;
