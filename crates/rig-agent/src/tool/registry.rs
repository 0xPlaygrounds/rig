//! The tool registry: a registration, an ordered set of them, and canonical
//! dispatch by name.
//!
//! A typed [`Tool`] erases once (`ErasedTool`) in rig-core; this module is
//! what a driver does with the erasure: stores it in an ordered [`ToolSet`],
//! executes it through one structured path ([`dispatch_tool`]), and pins a
//! per-turn [`ToolCatalog`]. The futures agent's live registry
//! ([`ToolServer`](super::server::ToolServer)) is layered over these types.

use std::collections::HashMap;

use indexmap::IndexMap;
use rig_core::{
    bus::{ErasedHandler, Serve, adapters::ToolAdapter, serve_inline},
    completion::{Document, ToolDefinition},
    effect::{
        EffectKind, FamilyDescriptor, Key, Outcome, ToolEmbeddingDescriptor, family, tool_key,
    },
    embeddings::{embed::EmbedError, tool::ToolSchema},
    error::{ErrorKind, ErrorReport},
    tool::{
        DynamicTool, LivenessFn, PortableDynamicTool, Tool, ToolContext, ToolEmbedding,
        ToolExecutionError, ToolResult, tool_definition,
    },
};

use super::catalog::ToolCatalog;

/// One registration: the tool's serde description, the key it is (or will
/// be) served under, and the erased handler a bus takes. Cloning shares the
/// handler.
#[derive(Clone)]
pub struct RegisteredTool {
    definition: ToolDefinition,
    embedding: Option<ToolEmbeddingDescriptor>,
    key: Key<family::Tool>,
    handler: ErasedHandler,
    liveness: Option<LivenessFn>,
}

impl std::fmt::Debug for RegisteredTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegisteredTool")
            .field("name", &self.definition.name)
            .field("key", &self.key)
            .field("retrievable", &self.embedding.is_some())
            .finish_non_exhaustive()
    }
}

impl RegisteredTool {
    /// Register a typed tool.
    pub fn from_tool<T>(tool: T) -> Self
    where
        T: Tool + 'static,
    {
        let definition = tool_definition(&tool);
        Self::from_parts(
            definition,
            None,
            ErasedHandler::new(ToolAdapter::new(tool)),
            None,
        )
    }

    /// Register a retrievable tool: its embedding context rides on the
    /// descriptor.
    pub fn from_retrievable<T>(tool: T) -> Result<Self, serde_json::Error>
    where
        T: ToolEmbedding + 'static,
    {
        let definition = tool_definition(&tool);
        let embedding = ToolEmbeddingDescriptor {
            context: serde_json::to_value(tool.context())?,
            embedding_docs: tool.embedding_docs(),
        };
        let adapter = ToolAdapter::retrievable(tool)?;
        Ok(Self::from_parts(
            definition,
            Some(embedding),
            ErasedHandler::new(adapter),
            None,
        ))
    }

    /// Register a runtime-defined tool.
    pub fn from_dynamic(tool: DynamicTool) -> Self {
        let (definition, handler, liveness) = tool.into_parts();
        Self::from_parts(definition, None, handler, liveness)
    }

    /// Register any tool-family handler under the key its descriptor names
    /// — a replayer answering a recorded tool from the effect log, a host's
    /// own handler. Fails when the handler is not of the tool family.
    pub fn from_handler(handler: impl Serve + 'static) -> Result<Self, ErrorReport> {
        let descriptor = handler.descriptor();
        let FamilyDescriptor::Tool {
            name,
            description,
            parameters,
            embedding,
        } = descriptor.family
        else {
            return Err(ErrorReport::new(
                ErrorKind::HandlerUnavailable,
                format!(
                    "handler `{}` serves the {} family, not tool_call",
                    descriptor.key,
                    descriptor.family.family()
                ),
            ));
        };
        Ok(Self {
            definition: ToolDefinition {
                name,
                description,
                parameters,
            },
            embedding,
            key: Key::new_unchecked(descriptor.key),
            handler: ErasedHandler::new(handler),
            liveness: None,
        })
    }

    /// Whether this registration is served under the default `tool:<name>`
    /// key (a registry that pins generations re-keys only those).
    pub fn has_default_key(&self) -> bool {
        *self.key.raw() == tool_key(&self.definition.name)
    }

    fn from_parts(
        definition: ToolDefinition,
        embedding: Option<ToolEmbeddingDescriptor>,
        handler: ErasedHandler,
        liveness: Option<LivenessFn>,
    ) -> Self {
        let key = Key::new_unchecked(tool_key(&definition.name));
        Self {
            definition,
            embedding,
            key,
            handler,
            liveness,
        }
    }

    /// Serve this registration under `key` instead of the default
    /// `tool:<name>` (a registry that pins generations does this).
    pub fn with_key(mut self, key: Key<family::Tool>) -> Self {
        self.key = key;
        self
    }

    /// The tool's name.
    pub fn name(&self) -> String {
        self.definition.name.clone()
    }

    /// The key the handler is served under: a tool key, proven at
    /// construction (the descriptor was checked to be tool-family).
    pub fn key(&self) -> &Key<family::Tool> {
        &self.key
    }

    /// The erased handler.
    pub fn handler(&self) -> &ErasedHandler {
        &self.handler
    }

    /// The definition, advertised under `name`.
    pub fn definition_with_name(&self, name: impl Into<String>) -> ToolDefinition {
        ToolDefinition {
            name: name.into(),
            description: self.definition.description.clone(),
            parameters: self.definition.parameters.clone(),
        }
    }

    /// The family-keyed descriptor of this registration.
    pub fn descriptor(&self) -> FamilyDescriptor {
        FamilyDescriptor::Tool {
            name: self.definition.name.clone(),
            description: self.definition.description.clone(),
            parameters: self.definition.parameters.clone(),
            embedding: self.embedding.clone(),
        }
    }

    /// The embedding context, for retrievable tools.
    pub fn embedding(&self) -> Option<&ToolEmbeddingDescriptor> {
        self.embedding.as_ref()
    }

    /// Whether the tool's owner still serves it (`true` without a probe).
    pub fn is_live(&self) -> bool {
        self.liveness.as_ref().is_none_or(|probe| probe())
    }

    /// Run the tool here, without a bus, publishing its result metadata
    /// into `context` — the inline path of a standalone tool set.
    pub async fn execute(&self, args: String, context: &mut ToolContext) -> ToolResult {
        let name = self.definition.name.clone();
        let outcome = serve_inline(
            &self.handler,
            EffectKind::ToolCall {
                name,
                args,
                context: std::mem::take(context),
            },
        )
        .await;
        match outcome {
            Ok(Outcome::ToolResult {
                result,
                context: published,
            }) => {
                *context = published;
                result
            }
            Ok(other) => ToolResult::failed(ToolExecutionError::other(format!(
                "tool handler answered with a {} outcome",
                other.family()
            ))),
            Err(report) => ToolResult::failed(ToolExecutionError::other(report.message)),
        }
    }
}

/// One entry of a [`ToolSet`]: the registration plus whether it is
/// advertised on every request or only when retrieved.
#[derive(Clone)]
pub(crate) struct ToolRegistration {
    tool: RegisteredTool,
    always_exposed: bool,
}

impl ToolRegistration {
    fn new(tool: RegisteredTool, always_exposed: bool) -> Self {
        Self {
            tool,
            always_exposed,
        }
    }
}

/// A tool call's result together with the dispatch-scoped context it
/// published into.
pub struct ToolDispatch {
    /// The result.
    pub result: ToolResult,
    /// The context after the tool ran.
    pub context: ToolContext,
}

impl ToolDispatch {
    /// Publish the dispatch's result metadata into `context` and return the
    /// result.
    pub fn publish_to(self, context: &mut ToolContext) -> ToolResult {
        context.accept_dispatch_result(self.context);
        self.result
    }
}

/// Run `tool` (or answer `not found`) on a dispatch-scoped copy of
/// `context`.
pub async fn dispatch_tool(
    name: &str,
    args: String,
    tool: Option<RegisteredTool>,
    context: &ToolContext,
) -> ToolDispatch {
    let mut dispatch_context = context.for_dispatch();
    let result = match tool {
        Some(tool) => {
            tracing::debug!(target: "rig", tool_name = name, "calling tool with args:\n{args}");
            tool.execute(args, &mut dispatch_context).await
        }
        None => ToolResult::failed(
            ToolExecutionError::not_found(format!("no tool named `{name}` is registered"))
                .with_model_feedback(format!("tool `{name}` not found")),
        ),
    };
    ToolDispatch {
        result,
        context: dispatch_context,
    }
}

/// A named set of tool registrations: the definition and advertisement
/// surface. Execution goes through the registrations' handlers — over a bus
/// once one has taken them, inline until then.
#[derive(Clone, Default)]
pub struct ToolSet {
    pub(crate) tools: IndexMap<String, ToolRegistration>,
}

impl ToolSet {
    /// A set from typed tools.
    pub fn from_tools<T>(tools: Vec<T>) -> Self
    where
        T: Tool + 'static,
    {
        let mut set = Self::default();
        for tool in tools {
            set.add_tool(tool);
        }
        set
    }

    /// A set from runtime-defined tools.
    pub fn from_dynamic_tools(tools: Vec<DynamicTool>) -> Self {
        let mut set = Self::default();
        for tool in tools {
            set.add_dynamic_tool(tool);
        }
        set
    }

    /// Whether a tool named `name` is registered.
    pub fn contains(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// Register a typed tool; returns its name.
    pub fn add_tool<T>(&mut self, tool: T) -> String
    where
        T: Tool + 'static,
    {
        self.insert(RegisteredTool::from_tool(tool))
    }

    /// Register a runtime-defined tool; returns its name.
    pub fn add_dynamic_tool(&mut self, tool: DynamicTool) -> String {
        self.insert(RegisteredTool::from_dynamic(tool))
    }

    /// Register a portable tool; returns its name.
    pub fn add_portable_dynamic_tool(&mut self, tool: PortableDynamicTool) -> String {
        self.add_dynamic_tool(DynamicTool::from_portable(tool))
    }

    /// Register a retrievable tool; returns its name. A context that does
    /// not serialize is the error, and nothing is registered.
    pub fn add_retrieved_tool<T>(&mut self, tool: T) -> Result<String, serde_json::Error>
    where
        T: ToolEmbedding + 'static,
    {
        RegisteredTool::from_retrievable(tool).map(|registered| self.insert(registered))
    }

    /// Register an already-built registration; returns its name.
    pub fn add_registered(&mut self, tool: RegisteredTool) -> String {
        self.insert(tool)
    }

    pub(crate) fn insert(&mut self, tool: RegisteredTool) -> String {
        let name = tool.name();
        self.insert_registration(name.clone(), ToolRegistration::new(tool, true));
        name
    }

    fn insert_registration(&mut self, name: String, mut registration: ToolRegistration) {
        if let Some(current) = self.tools.get_mut(&name) {
            registration.always_exposed |= current.always_exposed;
            *current = registration;
            tracing::warn!(tool_name = %name, "replacing an existing tool registration");
        } else {
            self.tools.insert(name, registration);
        }
    }

    /// Remove the tool named `name`.
    pub fn delete_tool(&mut self, name: &str) {
        self.tools.shift_remove(name);
    }

    /// Merge `set`'s registrations, keeping their exposure.
    pub fn add_tools(&mut self, set: ToolSet) {
        for (name, registration) in set.tools {
            self.insert_registration(name, registration);
        }
    }

    /// Merge `set`'s registrations as retrieval-only (not always exposed).
    pub fn add_retrievable_tools(&mut self, set: ToolSet) {
        for (name, mut registration) in set.tools {
            registration.always_exposed = false;
            self.insert_registration(name, registration);
        }
    }

    /// The registration named `name`.
    pub fn get(&self, name: &str) -> Option<&RegisteredTool> {
        self.tools.get(name).map(|registration| &registration.tool)
    }

    /// Every registration, in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &RegisteredTool)> {
        self.tools
            .iter()
            .map(|(name, registration)| (name.as_str(), &registration.tool))
    }

    /// Every name, in insertion order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tools.keys().map(String::as_str)
    }

    /// Number of registrations.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Whether the set is empty.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Names advertised on every request.
    pub fn always_exposed_names(&self) -> impl Iterator<Item = &str> {
        self.tools
            .iter()
            .filter_map(|(name, registration)| registration.always_exposed.then_some(name.as_str()))
    }

    /// Move `name` to the end of the insertion order.
    pub fn move_to_end(&mut self, name: &str) -> bool {
        self.tools
            .shift_remove_entry(name)
            .is_some_and(|(name, registration)| {
                self.tools.insert(name, registration);
                true
            })
    }

    /// The always-exposed registrations as a catalog.
    pub fn catalog(&self) -> ToolCatalog {
        ToolCatalog::from_registered(
            self.tools
                .iter()
                .filter(|(_, registration)| registration.always_exposed)
                .map(|(name, registration)| (name.clone(), registration.tool.clone()))
                .collect(),
        )
    }

    /// Every definition, in insertion order.
    pub fn tool_definitions(&self) -> Vec<ToolDefinition> {
        self.tools
            .iter()
            .map(|(name, registration)| registration.tool.definition_with_name(name.clone()))
            .collect()
    }

    /// Run the tool named `name` inline, publishing its result metadata
    /// into `context`.
    pub async fn execute(
        &self,
        name: &str,
        args: impl Into<String>,
        context: &mut ToolContext,
    ) -> ToolResult {
        context.clear_dispatch_result();
        let tool = self.get(name).cloned();
        let dispatch = dispatch_tool(name, args.into(), tool, context).await;
        dispatch.publish_to(context)
    }

    /// Every definition as a document, for embedding-based retrieval.
    pub fn documents(&self) -> Vec<Document> {
        self.tools
            .iter()
            .map(|(name, registration)| {
                let definition = registration.tool.definition_with_name(name.clone());
                let serialized =
                    serde_json::to_string_pretty(&definition).unwrap_or_else(|error| {
                        tracing::warn!(
                            tool_name = %name,
                            %error,
                            "tool definition could not be pretty-printed; using a plain representation"
                        );
                        format!(
                            "name: {}\ndescription: {}\nparameters: {}",
                            definition.name, definition.description, definition.parameters
                        )
                    });
                Document {
                    id: name.clone(),
                    text: format!("Tool: {name}\nDefinition: \n{serialized}"),
                    additional_props: HashMap::new(),
                }
            })
            .collect()
    }

    /// The embedding schemas of the retrievable registrations.
    pub fn schemas(&self) -> Result<Vec<ToolSchema>, EmbedError> {
        Ok(self
            .tools
            .iter()
            .filter_map(|(name, registration)| {
                registration.tool.embedding().map(|embedding| ToolSchema {
                    name: name.clone(),
                    context: embedding.context.clone(),
                    embedding_docs: embedding.embedding_docs.clone(),
                })
            })
            .collect())
    }
}

#[cfg(test)]
mod tests;
