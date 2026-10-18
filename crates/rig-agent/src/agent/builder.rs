//! Agent construction over its own or a host's handler table, with the
//! builder's tools, a shared tool server, or both (the builder's tools join
//! the shared server at build).
//!
//! ```
//! use rig_core::{DynModel, operation::Completion};
//! use rig_agent::{Agent, AgentBuilder};
//! fn assistant(model: impl Into<DynModel<Completion>>) -> Agent {
//!     AgentBuilder::new(model).preamble("Be concise.").build()
//! }
//! ```

use std::sync::{Arc, OnceLock};

use crate::handlers::Handlers;
use rig_core::serve::adapters::{ModelAdapter, RetrieveAdapter};
use rig_core::serve::{ErasedHandler, Recorder};
use rig_core::{
    completion::{Document, ModelRef},
    effect::{HandlerKey, Key, family},
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

use super::{
    Agent, OutputMode,
    completion::{AgentConfig, default_owner, register_generated},
};

/// The `dynamic_context` hook: retrieves documents for the prompt through
/// the agent's handlers (an `IndexHandle` bound to the index registered at
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
                .find_map(|message| message.rag_text())
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
                results.into_iter().map(|result| {
                    Document {
                        text: serde_json::to_string_pretty(&result.document)
                            .unwrap_or_else(|_| result.document.to_string()),
                        id: result.id,
                        additional_props: Default::default(),
                    }
                }),
            )),
            Err(error) => {
                CompletionCallAction::stop(format!("failed to retrieve dynamic context: {error}"))
            }
        }
    }
}

/// The default model: a model the builder registers under a label, or the
/// key of one a host registers in its table.
enum DefaultModel {
    Labelled(ModelRef, ErasedHandler),
    Key(HandlerKey),
}

/// Builds an [`Agent`].
///
/// Every handler the builder registers (the default model, model routes,
/// dynamic-context indexes) is minted a key under the agent's owner label at
/// build: `<owner>/model:<label>` and `<owner>/retrieve:context#<n>`. Hosts sharing a table must choose distinct
/// owners to avoid replacing each other's bindings. The explicit owner takes
/// precedence over the agent name, then a generated `agent#<n>` label.
pub struct AgentBuilder {
    config: AgentConfig,
    /// The tools added through the builder.
    tools: ToolServer,
    /// The shared registry, if any; the builder's tools join it at build.
    shared: Option<ToolServerHandle>,
    owner: Option<String>,
    model: DefaultModel,
    /// Handlers to register at build, by key suffix.
    pending: Vec<(String, ErasedHandler)>,
    /// The dynamic-context hooks' key slots, by key suffix.
    dynamic_contexts: Vec<(String, Arc<OnceLock<Key<family::Retrieve>>>)>,
    recorder: Option<Arc<dyn Recorder + Send + Sync>>,
    retrieval_indexes: usize,
    /// The labels `model_route` registered, in order.
    routes: Vec<String>,
}

impl AgentBuilder {
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
        self.config.spec_mut().preamble = Some(preamble.into());
        self
    }

    /// Clear the system prompt.
    pub fn without_preamble(mut self) -> Self {
        self.config.spec_mut().preamble = None;
        self
    }

    /// Append a paragraph to the system prompt.
    pub fn append_preamble(mut self, doc: &str) -> Self {
        let preamble = &mut self.config.spec_mut().preamble;
        *preamble = Some(format!("{}\n{doc}", preamble.take().unwrap_or_default()));
        self
    }

    /// Add a static context document.
    pub fn context(mut self, doc: impl Into<String>) -> Self {
        let documents = &mut self.config.spec_mut().static_context;
        documents.push(Document {
            id: format!("static_doc_{}", documents.len()),
            text: doc.into(),
            additional_props: Default::default(),
        });
        self
    }

    /// Retrieve `samples` documents from `index` for every prompt and add
    /// them as context. The index is registered in the agent's table.
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
        self.config.spec_mut().tool_choice = Some(tool_choice);
        self
    }

    /// Set the default maximum number of turns.
    pub fn default_max_turns(mut self, default_max_turns: usize) -> Self {
        self.config.spec_mut().max_turns = Some(default_max_turns);
        self
    }

    /// Set the sampling temperature.
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.config.spec_mut().temperature = Some(temperature);
        self
    }

    /// Remove the sampling temperature, for test scenarios run against
    /// models that reject it.
    #[cfg(any(test, feature = "test-utils"))]
    pub(crate) fn clear_temperature(mut self) -> Self {
        self.config.spec_mut().temperature = None;
        self
    }

    /// Set the output-token cap.
    pub fn max_tokens(mut self, max_tokens: u64) -> Self {
        self.config.spec_mut().max_tokens = Some(max_tokens);
        self
    }

    /// Set provider passthrough parameters.
    pub fn additional_params(mut self, params: serde_json::Value) -> Self {
        self.config.spec_mut().additional_params = Some(params);
        self
    }

    /// Set the portable generation options every request of this agent
    /// carries. A run's [`AgentRunner::options`](crate::agent::AgentRunner::options)
    /// overlays them field by field. Calls apply in order: this replaces
    /// every field, so a shortcut such as [`Self::seed`] called before it is
    /// lost, and one called after it sets its one field on top.
    pub fn options(mut self, options: rig_core::completion::GenerationOptions) -> Self {
        self.config.spec_mut().options = options;
        self
    }

    generation_option_setters!("the agent's");

    /// Set the stop sequences in the agent's generation options, as
    /// [`GenerationOptions::stop`](rig_core::completion::GenerationOptions::stop) does,
    /// keeping its other fields. See
    /// [`Self::options`] for the order of calls.
    pub fn stop<S: Into<String>>(mut self, stop: impl IntoIterator<Item = S>) -> Self {
        self.config.spec_mut().options =
            std::mem::take(&mut self.config.spec_mut().options).stop(stop);
        self
    }

    /// Set the typed per-provider options every request of this agent
    /// carries. A run's
    /// [`AgentRunner::provider_options`](crate::agent::AgentRunner::provider_options)
    /// replaces the entry of each provider it names.
    ///
    /// Calls apply in order: this replaces every entry, so an entry set by
    /// an earlier [`Self::provider_option`] is lost, and a later
    /// [`Self::provider_option`] replaces its provider's entry on top.
    pub fn provider_options(mut self, options: rig_core::completion::ProviderOptions) -> Self {
        self.config.spec_mut().provider_options = options;
        self
    }

    /// Store `options` as the entry of their provider
    /// ([`ExtensionOptions::Ext`](rig_core::completion::ExtensionOptions::Ext))
    /// in the provider options every request of this agent carries,
    /// replacing that provider's entry and keeping every other
    /// ([`ProviderOptions::set`](rig_core::completion::ProviderOptions::set)).
    /// Options that do not serialize fail each request's encode. A
    /// third-party provider reusing a built-in options type needs
    /// [`ProviderOptions::with::<P>`](rig_core::completion::ProviderOptions::with)
    /// and [`Self::provider_options`]: this stores under the built-in key.
    pub fn provider_option<O: rig_core::completion::ExtensionOptions>(
        mut self,
        options: O,
    ) -> Self {
        self.config.spec_mut().provider_options =
            std::mem::take(&mut self.config.spec_mut().provider_options).set(options);
        self
    }

    /// Enable or disable sensitive message content on telemetry spans.
    /// Disabled by default; enabling may expose prompts, responses, and tool data.
    pub fn record_content_telemetry(mut self, enabled: bool) -> Self {
        self.config.record_telemetry_content = enabled;
        self
    }

    /// Accept finish reasons outside the normalized vocabulary
    /// ([`FinishReason::Other`](rig_core::completion::FinishReason::Other))
    /// as a normal stop. Off by default: such a turn fails the run,
    /// and history replay leaves it out. When on, the turn succeeds, its
    /// tool calls run, and it replays.
    ///
    /// `Other` also holds genuine failures, such as Gemini's
    /// `MALFORMED_FUNCTION_CALL` and `RECITATION` or Bedrock's
    /// `malformed_tool_use`, so tool calls from a malformed reply may run.
    /// Filtered content still fails the turn.
    pub fn accept_unknown_finish_reasons(mut self, accept: bool) -> Self {
        self.config.spec_mut().accept_unknown_finish_reasons = accept;
        self
    }

    /// Constrain the output to `T`'s JSON schema.
    pub fn output_schema<T>(mut self) -> Self
    where
        T: JsonSchema,
    {
        self.config.spec_mut().output_schema = Some(schema_for!(T));
        self
    }

    /// Constrain the output to a raw JSON schema.
    pub fn output_schema_raw(mut self, schema: Schema) -> Self {
        self.config.spec_mut().output_schema = Some(schema);
        self
    }

    /// Set the structured-output mode.
    pub fn output_mode(mut self, mode: OutputMode) -> Self {
        self.config.spec_mut().output_mode = mode;
        self
    }

    /// Register another model the run can select by label
    /// (`ModelSelectionAction::select(label)`, `using_model(label)`).
    pub fn model_route(
        mut self,
        label: impl Into<ModelRef>,
        model: impl Into<rig_core::DynModel<rig_core::operation::Completion>>,
    ) -> Self {
        let label = label.into();
        let model = model.into();
        let suffix = rig_core::effect::model_key(label.as_str()).to_string();
        self.routes.push(label.as_str().to_owned());
        self.pending
            .push((suffix, ErasedHandler::new(ModelAdapter::new(label, model))));
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

    /// Name the agent's keys: `<owner>/model:<label>`,
    /// `<owner>/retrieve:context#<n>`, and its own tools' `<owner>/tool:…`.
    /// Defaults to the agent's name, or a process-local `agent#<n>` counter.
    /// Use stable, distinct owners for replay and agents sharing a table.
    pub fn owner(mut self, label: impl Into<String>) -> Self {
        self.owner = Some(label.into());
        self
    }

    /// Keep each run's conversation in `store`: a run loads it before its
    /// first model call and saves its messages there once it finished, so
    /// the next run goes on from them. A run that fails or is cancelled
    /// saves nothing; a failed load or save fails the run with
    /// [`PromptError::Memory`](crate::completion::PromptError::Memory).
    /// See the [`conversation`](Self::conversation) it keeps.
    ///
    /// ```
    /// use rig_agent::AgentBuilder;
    /// use rig_core::test_utils::{MockCompletionModel, MockTurn};
    /// use rig_memory::{MemoryStore, load};
    ///
    /// # tokio_test::block_on(async {
    /// let store = MemoryStore::default();
    /// let model = MockCompletionModel::from_turns([MockTurn::text("Hi Alice!")]);
    /// let agent = AgentBuilder::new(model).memory(store.clone()).build();
    /// agent.prompt("I am Alice.").conversation("alice").await?;
    /// assert_eq!(load(&store, "alice")?.messages().len(), 2);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// # }).unwrap();
    /// ```
    pub fn memory(
        mut self,
        store: impl rig_memory::AsyncStore
        + rig_core::wasm_compat::WasmCompatSend
        + rig_core::wasm_compat::WasmCompatSync
        + 'static,
    ) -> Self {
        self.config.memory = Some(super::memory::Memory::new(store));
        self
    }

    /// The log of the [`memory`](Self::memory) store that runs load and
    /// save, one conversation per log: by default the agent's
    /// [`name`](Self::name), or `agent`. A log name is one part of a
    /// [`check_name`](rig_memory::check_name) name.
    pub fn conversation(mut self, log: impl Into<String>) -> Self {
        self.config.conversation = Some(log.into());
        self
    }

    /// Observe every dispatch with a caller-owned recorder, installed on
    /// the agent's handler table at build ([`Handlers::record_to`]). Over a
    /// host's table that is the host's: it replaces the host's recorder.
    /// Keep a clone of a shared recorder to inspect its observations.
    pub fn record_to(mut self, recorder: impl Recorder + Send + Sync) -> Self {
        self.recorder = Some(Arc::new(recorder));
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

    /// Build the agent.
    pub fn build(self) -> Agent {
        let Self {
            mut config,
            tools,
            shared,
            owner,
            model,
            mut pending,
            dynamic_contexts,
            recorder,
            retrieval_indexes: _,
            routes,
        } = self;
        // Named agents need stable keys across processes for replay.
        config.owner = Arc::from(
            owner
                .or_else(|| config.name.clone())
                .unwrap_or_else(default_owner),
        );
        config.model_key = match model {
            DefaultModel::Labelled(label, handler) => {
                let suffix = rig_core::effect::model_key(label.as_str()).to_string();
                pending.insert(0, (suffix, handler));
                config.model_key_for(label.as_str())
            }
            // A host may register its model after the agent is built; a
            // key of another family fails the run's first model call.
            DefaultModel::Key(key) => Key::new_unchecked(key),
        };
        for (suffix, handler) in pending {
            let key = config.raw_key(&suffix);
            register_generated(config.handlers.register_erased(key, handler));
        }
        for (suffix, slot) in dynamic_contexts {
            // The slot is this builder's own, filled exactly once.
            let key = config.key(&suffix);
            config.context_keys.push(key.clone());
            let _ = slot.set(key);
        }
        config.route_keys = routes
            .iter()
            .map(|label| config.model_key_for(label))
            .collect();
        if let Some(recorder) = recorder {
            config.handlers.record_with(recorder);
        }
        let tool_server_handle = match shared {
            Some(handle) => {
                handle.extend(tools);
                handle
            }
            // The builder's own registry takes the agent's owner, so a
            // named agent's tool keys are stable too.
            None => tools.owner(config.owner.to_string()).run(),
        };
        tool_server_handle.attach(&config.handlers);
        Agent {
            tool_server_handle,
            config,
        }
    }

    /// An agent over its own handler table, with `model` registered as the
    /// default model (label `default`).
    pub fn new(model: impl Into<rig_core::DynModel<rig_core::operation::Completion>>) -> Self {
        Self::named_model("default", model)
    }

    /// An agent over its own handler table, with `model` registered under
    /// `label`.
    pub fn named_model(
        label: impl Into<ModelRef>,
        model: impl Into<rig_core::DynModel<rig_core::operation::Completion>>,
    ) -> Self {
        let label = label.into();
        let handler = ErasedHandler::new(ModelAdapter::new(label.clone(), model.into()));
        Self::start(
            Handlers::default(),
            None,
            DefaultModel::Labelled(label, handler),
        )
    }

    /// An agent over a host's handler table, using the model under `model`
    /// verbatim and owner-qualified keys for the handlers it registers. The
    /// host registers the completion model before running the agent.
    pub fn over_handlers(handlers: Handlers, owner: impl Into<String>, model: HandlerKey) -> Self {
        Self::start(handlers, Some(owner.into()), DefaultModel::Key(model))
    }

    fn start(handlers: Handlers, owner: Option<String>, model: DefaultModel) -> Self {
        // The model key is a placeholder until build mints the real one
        // under the owner.
        let key = Key::new_unchecked(match &model {
            DefaultModel::Labelled(label, _) => HandlerKey::from(label.as_str()),
            DefaultModel::Key(key) => key.clone(),
        });
        Self {
            config: AgentConfig::new(handlers, key),
            tools: ToolServer::new(),
            shared: None,
            owner,
            model,
            pending: Vec::new(),
            dynamic_contexts: Vec::new(),
            recorder: None,
            retrieval_indexes: 0,
            routes: Vec::new(),
        }
    }

    /// Use a pre-existing shared registry; tools added through the builder
    /// join it at build.
    pub fn tool_server_handle(mut self, handle: ToolServerHandle) -> Self {
        self.shared = Some(handle);
        self
    }

    fn map_tools(mut self, add: impl FnOnce(ToolServer) -> ToolServer) -> Self {
        self.tools = add(std::mem::take(&mut self.tools));
        self
    }

    /// Add a typed tool.
    pub fn tool<T>(self, tool: T) -> Self
    where
        T: Tool + 'static,
    {
        self.map_tools(|tools| tools.tool(tool))
    }

    /// Add a runtime-defined tool.
    pub fn dynamic_tool(self, tool: DynamicTool) -> Self {
        self.map_tools(|tools| tools.dynamic_tool(tool))
    }

    /// Add runtime-defined tools.
    pub fn dynamic_tools(self, tools: Vec<DynamicTool>) -> Self {
        self.map_tools(|server| server.dynamic_tools(tools))
    }

    /// Add retrievable tools chosen per request by `index`.
    pub fn retrieved_tools<I, F>(self, sample: usize, index: I, toolset: ToolSet) -> Self
    where
        I: VectorStoreIndex<Filter = F> + 'static,
        F: DynamicSearchFilter + WasmCompatSend + WasmCompatSync + 'static,
    {
        self.map_tools(|tools| tools.retrieved_tools(sample, index, toolset))
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
        self.map_tools(|tools| tools.retrieved_tools_handler(sample, handler, toolset))
    }
}

#[cfg(test)]
mod tests;
