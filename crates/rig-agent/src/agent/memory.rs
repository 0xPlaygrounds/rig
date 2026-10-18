//! A conversation kept across runs. An agent given a store
//! ([`AgentBuilder::memory`](super::AgentBuilder::memory)) loads the
//! conversation before a run's first model call and saves the run's
//! messages to it, with one append, once the run finished; a run that fails
//! or is cancelled saves nothing, and a failed load or save fails the run
//! ([`PromptError::Memory`](crate::completion::PromptError::Memory)).
//!
//! The store is any rig-memory [`AsyncStore`]: a local `MemoryStore` or
//! `DirStore`, whose calls are done when made, or one behind a network or a
//! JavaScript promise. A save refuses a conversation the log moved on from
//! since the run loaded it, so of two runs that loaded the same log the
//! second to finish fails and stores nothing.

use std::sync::Arc;

use rig_core::conversation::Conversation;
use rig_core::wasm_compat::{WasmBoxedFuture, WasmCompatSend, WasmCompatSync};
use rig_memory::{AsyncStore, SessionError, load_async, save_async};

/// An agent's store, behind the one box per load and save its `dyn`
/// boundary costs.
#[derive(Clone)]
pub(crate) struct Memory(Arc<dyn Persist>);

/// [`load_async`] and [`save_async`] of one store type, erased.
trait Persist: WasmCompatSend + WasmCompatSync {
    fn load<'a>(&'a self, log: &'a str) -> WasmBoxedFuture<'a, Result<Conversation, SessionError>>;

    fn save<'a>(
        &'a self,
        log: &'a str,
        conversation: &'a mut Conversation,
    ) -> WasmBoxedFuture<'a, Result<(), SessionError>>;
}

impl<S: AsyncStore + WasmCompatSend + WasmCompatSync> Persist for S {
    fn load<'a>(&'a self, log: &'a str) -> WasmBoxedFuture<'a, Result<Conversation, SessionError>> {
        Box::pin(load_async(self, log))
    }

    fn save<'a>(
        &'a self,
        log: &'a str,
        conversation: &'a mut Conversation,
    ) -> WasmBoxedFuture<'a, Result<(), SessionError>> {
        Box::pin(save_async(self, log, conversation))
    }
}

impl Memory {
    pub(crate) fn new(store: impl AsyncStore + WasmCompatSend + WasmCompatSync + 'static) -> Self {
        Self(Arc::new(store))
    }

    /// The conversation the log `log` holds; an empty one when there is none.
    pub(crate) async fn load(&self, log: &str) -> Result<Conversation, SessionError> {
        self.0.load(log).await
    }

    /// Appends the unsaved changes of `conversation` to the log `log`.
    pub(crate) async fn save(
        &self,
        log: &str,
        conversation: &mut Conversation,
    ) -> Result<(), SessionError> {
        self.0.save(log, conversation).await
    }
}

#[cfg(test)]
mod tests;
