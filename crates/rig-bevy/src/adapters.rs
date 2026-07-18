//! Adapters from portable Rig contracts to owned ECS effects.

use std::{any::Any, collections::BTreeMap, sync::Arc};

use rig_core::{
    completion::CompletionModel,
    memory::{ConversationMemory, MemoryError},
    message::Message,
    wasm_compat::{WasmBoxedFuture, WasmCompatSend, WasmCompatSync},
};

use crate::topology::AgentId;

type ErasedModel = Arc<dyn Any + Send + Sync>;

/// Concrete model implementations resolved to exact restored agent identities.
#[derive(Clone, Default)]
pub(crate) struct AgentModelBindings {
    models: BTreeMap<AgentId, ErasedModel>,
}

impl AgentModelBindings {
    pub(crate) fn bind<M>(&mut self, agent: AgentId, model: M) -> bool
    where
        M: CompletionModel + 'static,
    {
        if self.models.contains_key(&agent) {
            return false;
        }
        self.models.insert(agent, Arc::new(model));
        true
    }

    pub(crate) fn bind_erased(&mut self, agent: AgentId, model: ErasedModel) -> bool {
        if self.models.contains_key(&agent) {
            return false;
        }
        self.models.insert(agent, model);
        true
    }

    pub(crate) fn contains(&self, agent: AgentId) -> bool {
        self.models.contains_key(&agent)
    }

    pub(crate) fn get(&self, agent: AgentId) -> Option<ErasedModel> {
        self.models.get(&agent).cloned()
    }

    pub(crate) fn ids(&self) -> impl Iterator<Item = &AgentId> {
        self.models.keys()
    }

    pub(crate) fn resolve<M>(&self, agent: AgentId) -> Option<M>
    where
        M: CompletionModel + 'static,
    {
        self.models
            .get(&agent)
            .and_then(|model| model.downcast_ref::<M>())
            .cloned()
    }
}

trait ErasedMemory: WasmCompatSend + WasmCompatSync {
    fn load<'a>(
        &'a self,
        conversation: &'a str,
    ) -> WasmBoxedFuture<'a, Result<Vec<Message>, MemoryError>>;
    fn append<'a>(
        &'a self,
        conversation: &'a str,
        messages: Vec<Message>,
    ) -> WasmBoxedFuture<'a, Result<(), MemoryError>>;
}

impl<M> ErasedMemory for M
where
    M: ConversationMemory,
{
    fn load<'a>(
        &'a self,
        conversation: &'a str,
    ) -> WasmBoxedFuture<'a, Result<Vec<Message>, MemoryError>> {
        ConversationMemory::load(self, conversation)
    }

    fn append<'a>(
        &'a self,
        conversation: &'a str,
        messages: Vec<Message>,
    ) -> WasmBoxedFuture<'a, Result<(), MemoryError>> {
        ConversationMemory::append(self, conversation, messages)
    }
}

/// Explicit host implementation bindings. Only names are persisted; this map
/// is reconstructed after restart.
#[derive(Clone, Default)]
pub struct MemoryBindings {
    memories: BTreeMap<String, Arc<dyn ErasedMemory>>,
}

impl MemoryBindings {
    /// Bind a portable memory backend under a stable implementation name.
    pub fn bind<M>(&mut self, name: impl Into<String>, memory: M)
    where
        M: ConversationMemory + 'static,
    {
        self.memories.insert(name.into(), Arc::new(memory));
    }

    /// Whether a name has been explicitly rebound.
    pub fn contains(&self, name: &str) -> bool {
        self.memories.contains_key(name)
    }

    /// Load history through an owned backend reference. No world or runtime lock
    /// guard enters the returned future.
    pub async fn load(&self, name: &str, conversation: &str) -> Result<Vec<Message>, AdapterError> {
        let memory = self
            .memories
            .get(name)
            .cloned()
            .ok_or_else(|| AdapterError::MissingBinding(name.to_string()))?;
        memory
            .load(conversation)
            .await
            .map_err(AdapterError::Memory)
    }

    /// Append only messages selected by terminal commit logic.
    pub async fn append(
        &self,
        name: &str,
        conversation: &str,
        messages: Vec<Message>,
    ) -> Result<(), AdapterError> {
        let memory = self
            .memories
            .get(name)
            .cloned()
            .ok_or_else(|| AdapterError::MissingBinding(name.to_string()))?;
        memory
            .append(conversation, messages)
            .await
            .map_err(AdapterError::Memory)
    }
}

/// Typed adapter failure.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum AdapterError {
    /// An implementation was not rebound.
    #[error("missing implementation binding `{0}`")]
    MissingBinding(String),
    /// Portable memory backend failed.
    #[error(transparent)]
    Memory(#[from] MemoryError),
}
