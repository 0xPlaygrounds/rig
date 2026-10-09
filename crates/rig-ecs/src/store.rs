//! Where a session's journal is kept. The core reads and writes the
//! session only through a [`JournalStore`]: one append-only log per agent,
//! content-addressed blobs such as images, and the effect log. The app
//! inserts the [`SessionStore`] before the agent plugins are built; without
//! one nothing is kept. [`MemoryStore`] keeps a session in memory; with
//! feature `fs-journal`, [`JsonlDirStore`](super::fs_journal::JsonlDirStore)
//! keeps it as JSON-lines files in a directory.

use std::collections::BTreeMap;
use std::io;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use bevy_ecs::prelude::*;
use rig_cassette::effect_log::EffectLog;
use rig_core::effect::EffectId;

/// Storage for a session's journal. Every method takes `&self`, so one
/// store is shared by the agent logs and the effect log.
pub trait JournalStore: Send + Sync + 'static {
    /// The ids of the agents that have a log, in no particular order.
    fn agents(&self) -> io::Result<Vec<String>>;

    /// The whole log of `agent`.
    fn read(&self, agent: &str) -> io::Result<Vec<u8>>;

    /// Cuts the log of `agent` to its first `len` bytes, such as a torn
    /// last line a crash left.
    fn truncate(&self, agent: &str, len: u64) -> io::Result<()>;

    /// Appends `bytes` to the log of `agent`, which it starts when there is
    /// none.
    fn append(&self, agent: &str, bytes: &[u8]) -> io::Result<()>;

    /// Stores the blob `name`, unless it is stored already: a name is a
    /// hash of the content.
    fn put_blob(&self, name: &str, bytes: &[u8]) -> io::Result<()>;

    /// The blob `name`.
    fn blob(&self, name: &str) -> io::Result<Vec<u8>>;

    /// Appends resolved effects, with their header when it changed.
    fn append_effects(&self, log: &EffectLog) -> io::Result<()>;

    /// The highest effect id stored, so ids keep increasing across
    /// restarts.
    fn last_effect(&self) -> io::Result<Option<EffectId>>;
}

/// The session's [`JournalStore`].
#[derive(Resource, Clone)]
pub struct SessionStore(pub Arc<dyn JournalStore>);

impl SessionStore {
    /// The session kept in `store`.
    pub fn new(store: impl JournalStore) -> Self {
        Self(Arc::new(store))
    }
}

/// A session kept in memory, such as for tests or where there is no file
/// system. Cloning shares the contents.
#[derive(Clone, Default)]
pub struct MemoryStore(Arc<Mutex<Memory>>);

#[derive(Default)]
struct Memory {
    logs: BTreeMap<String, Vec<u8>>,
    blobs: BTreeMap<String, Vec<u8>>,
    effects: EffectLog,
}

impl MemoryStore {
    fn memory(&self) -> MutexGuard<'_, Memory> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Every effect appended, under the latest header.
    pub fn effects(&self) -> EffectLog {
        self.memory().effects.clone()
    }
}

fn missing(what: &str) -> io::Error {
    io::Error::new(io::ErrorKind::NotFound, format!("no {what}"))
}

impl JournalStore for MemoryStore {
    fn agents(&self) -> io::Result<Vec<String>> {
        Ok(self.memory().logs.keys().cloned().collect())
    }

    fn read(&self, agent: &str) -> io::Result<Vec<u8>> {
        self.memory()
            .logs
            .get(agent)
            .cloned()
            .ok_or_else(|| missing("such agent log"))
    }

    fn truncate(&self, agent: &str, len: u64) -> io::Result<()> {
        let len = usize::try_from(len).map_err(io::Error::other)?;
        if let Some(log) = self.memory().logs.get_mut(agent) {
            log.truncate(len);
        }
        Ok(())
    }

    fn append(&self, agent: &str, bytes: &[u8]) -> io::Result<()> {
        self.memory()
            .logs
            .entry(agent.to_owned())
            .or_default()
            .extend_from_slice(bytes);
        Ok(())
    }

    fn put_blob(&self, name: &str, bytes: &[u8]) -> io::Result<()> {
        self.memory()
            .blobs
            .entry(name.to_owned())
            .or_insert_with(|| bytes.to_vec());
        Ok(())
    }

    fn blob(&self, name: &str) -> io::Result<Vec<u8>> {
        self.memory()
            .blobs
            .get(name)
            .cloned()
            .ok_or_else(|| missing("such blob"))
    }

    fn append_effects(&self, log: &EffectLog) -> io::Result<()> {
        let mut memory = self.memory();
        memory.effects.header = log.header.clone();
        memory.effects.records.extend(log.records.iter().cloned());
        Ok(())
    }

    fn last_effect(&self) -> io::Result<Option<EffectId>> {
        Ok(self
            .memory()
            .effects
            .records
            .iter()
            .map(|record| record.id)
            .max())
    }
}
