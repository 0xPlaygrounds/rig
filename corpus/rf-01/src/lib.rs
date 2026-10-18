//! `shared-thread`: one support conversation answered by many stateless
//! clients, its history kept in a remote, versioned store.
//!
//! [`SharedThread::ask`] loads the thread from a [`ThreadStore`], sends the
//! window of it a [`Recent`] rule keeps (never starting at a tool result
//! whose call it cut), runs the rig agent with tools, and appends the run's
//! messages (the question, every tool call and result, the reply) with one
//! versioned append. A moved version is resolved by reloading and appending
//! after the other client's messages; every message carries a stable id, so
//! a retried save never writes one twice. Nothing is written before the run
//! finished, so a dropped `ask` leaves the store as it was.

use std::collections::HashSet;
use std::future::Future;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use rig_agent::Agent;
use rig_agent::completion::PromptError;
use rig_core::message::Message;
use rig_memory::recent::{LastMessages, Recent, view};
use serde::{Deserialize, Serialize};

/// One message of a thread, under the id it keeps in every store and retry.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredMessage {
    /// Stable, unique id.
    pub id: String,
    /// The message.
    pub message: Message,
}

impl StoredMessage {
    /// `message` under a fresh id.
    pub fn new(message: Message) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            message,
        }
    }
}

/// A thread as the store holds it now.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Snapshot {
    /// Every message, oldest first.
    pub messages: Vec<StoredMessage>,
    /// The version an append must name to go after these messages.
    pub version: u64,
}

/// A store call that failed.
#[derive(Clone, Debug, thiserror::Error)]
#[error("store failed: {0}")]
pub struct StoreError(pub String);

/// Why an append wrote nothing.
#[derive(Clone, Debug, thiserror::Error)]
pub enum AppendError {
    /// The thread is no longer at the expected version.
    #[error("the thread moved past the expected version")]
    Conflict,
    /// The store failed; the append may or may not have been applied.
    #[error(transparent)]
    Failed(#[from] StoreError),
}

/// Where threads are kept: a remote key-value or document store reached
/// asynchronously. An append is conditional on the version, checked and
/// applied in one step by the store.
pub trait ThreadStore: Send + Sync {
    /// The thread `thread`; an empty one at version 0 when there is none.
    fn get(&self, thread: &str) -> impl Future<Output = Result<Snapshot, StoreError>> + Send;

    /// Appends `messages` to `thread` if it is at `expected_version`,
    /// returning the new version.
    fn append(
        &self,
        thread: &str,
        expected_version: u64,
        messages: &[StoredMessage],
    ) -> impl Future<Output = Result<u64, AppendError>> + Send;
}

/// A shared store is a store.
impl<S: ThreadStore + ?Sized> ThreadStore for Arc<S> {
    fn get(&self, thread: &str) -> impl Future<Output = Result<Snapshot, StoreError>> + Send {
        (**self).get(thread)
    }

    fn append(
        &self,
        thread: &str,
        expected_version: u64,
        messages: &[StoredMessage],
    ) -> impl Future<Output = Result<u64, AppendError>> + Send {
        (**self).append(thread, expected_version, messages)
    }
}

/// A store in memory, each call taking `latency`: the stand-in for a
/// remote backend. The version is the thread's length. Cloning shares it.
#[derive(Clone, Default)]
pub struct InMemoryStore {
    threads: Arc<Mutex<std::collections::HashMap<String, Vec<StoredMessage>>>>,
    latency: Duration,
}

impl InMemoryStore {
    /// An empty store whose every call takes `latency`.
    pub fn with_latency(latency: Duration) -> Self {
        Self {
            latency,
            ..Self::default()
        }
    }

    /// The thread `thread` now, without latency.
    pub fn messages(&self, thread: &str) -> Vec<StoredMessage> {
        let threads = self.threads.lock().unwrap_or_else(PoisonError::into_inner);
        threads.get(thread).cloned().unwrap_or_default()
    }
}

impl ThreadStore for InMemoryStore {
    async fn get(&self, thread: &str) -> Result<Snapshot, StoreError> {
        tokio::time::sleep(self.latency).await;
        let messages = self.messages(thread);
        let version = messages.len() as u64;
        Ok(Snapshot { messages, version })
    }

    async fn append(
        &self,
        thread: &str,
        expected_version: u64,
        messages: &[StoredMessage],
    ) -> Result<u64, AppendError> {
        tokio::time::sleep(self.latency).await;
        let mut threads = self.threads.lock().unwrap_or_else(PoisonError::into_inner);
        let stored = threads.entry(thread.to_owned()).or_default();
        if stored.len() as u64 != expected_version {
            return Err(AppendError::Conflict);
        }
        stored.extend_from_slice(messages);
        Ok(stored.len() as u64)
    }
}

/// What [`SharedThread::ask`] answers.
#[derive(Clone, Debug)]
pub struct Reply {
    /// The model's reply text.
    pub text: String,
    /// The messages the exchange added to the thread, in order.
    pub messages: Vec<StoredMessage>,
    /// The thread's version after them.
    pub version: u64,
}

/// Why [`SharedThread::ask`] failed.
#[derive(Debug, thiserror::Error)]
pub enum AskError {
    /// Loading the thread failed; the model was not called.
    #[error("loading the thread failed")]
    Load(#[source] StoreError),
    /// The agent run failed; nothing was written.
    #[error("the agent run failed")]
    Model(#[source] Box<PromptError>),
    /// The model answered but saving the exchange failed after retries.
    /// [`SharedThread::save`] of `pending` retries it without duplicating.
    #[error("the reply could not be saved")]
    Save {
        /// The model's reply text.
        reply: String,
        /// The exchange's messages, under their stable ids.
        pending: Vec<StoredMessage>,
        /// The last store failure.
        source: StoreError,
    },
}

/// An agent whose conversation is a thread of a [`ThreadStore`], shared by
/// every client of that store.
pub struct SharedThread<S> {
    agent: Agent,
    store: S,
    window: Arc<dyn Recent>,
    save_attempts: usize,
}

impl<S: ThreadStore> SharedThread<S> {
    /// `agent` over `store`, sending the whole thread, saving with up to
    /// five attempts.
    pub fn new(agent: Agent, store: S) -> Self {
        Self {
            agent,
            store,
            window: Arc::new(LastMessages(usize::MAX)),
            save_attempts: 5,
        }
    }

    /// Sends only what `window` keeps of the thread; the store keeps all.
    #[must_use]
    pub fn window(mut self, window: impl Recent + 'static) -> Self {
        self.window = Arc::new(window);
        self
    }

    /// Tries a save at most `attempts` times (at least once).
    #[must_use]
    pub fn save_attempts(mut self, attempts: usize) -> Self {
        self.save_attempts = attempts.max(1);
        self
    }

    /// The store.
    pub fn store(&self) -> &S {
        &self.store
    }

    /// Asks `user_text` in `thread`: loads it, runs the agent on its window
    /// and appends the exchange.
    ///
    /// # Errors
    ///
    /// [`AskError::Load`] before any model call, [`AskError::Model`] when
    /// the run fails, [`AskError::Save`] (with the reply) when the save
    /// fails after retries.
    pub async fn ask(&self, thread: &str, user_text: &str) -> Result<Reply, AskError> {
        let snapshot = self.store.get(thread).await.map_err(AskError::Load)?;
        let history: Vec<Arc<Message>> = snapshot
            .messages
            .into_iter()
            .map(|stored| Arc::new(stored.message))
            .collect();
        let sent = view(&history, self.window.as_ref()).iter();
        let response = self
            .agent
            .prompt(user_text)
            .history(sent.map(|message| Message::clone(message)))
            .run()
            .await
            .map_err(|error| AskError::Model(Box::new(error)))?;
        let text = response.output();
        let messages: Vec<StoredMessage> = response
            .messages
            .into_iter()
            .map(StoredMessage::new)
            .collect();
        match self.append(thread, &messages, Some(snapshot.version)).await {
            Ok(version) => Ok(Reply {
                text,
                messages,
                version,
            }),
            Err(source) => Err(AskError::Save {
                reply: text,
                pending: messages,
                source,
            }),
        }
    }

    /// Appends `messages` to `thread` after whatever it holds now, skipping
    /// those already there by id, so retrying a failed save is safe.
    /// Returns the thread's version after them.
    ///
    /// # Errors
    ///
    /// The last store failure, after the configured attempts.
    pub async fn save(&self, thread: &str, messages: &[StoredMessage]) -> Result<u64, StoreError> {
        self.append(thread, messages, None).await
    }

    /// Appends at `version` when given, else (and after every refusal) at
    /// the reloaded version, without the messages the store already has.
    async fn append(
        &self,
        thread: &str,
        messages: &[StoredMessage],
        mut version: Option<u64>,
    ) -> Result<u64, StoreError> {
        let mut last = StoreError("the thread kept moving".to_owned());
        for _ in 0..self.save_attempts {
            let (at, todo) = match version.take() {
                Some(at) => (at, messages.to_vec()),
                None => match self.store.get(thread).await {
                    Ok(snapshot) => {
                        let have: HashSet<&str> =
                            snapshot.messages.iter().map(|m| m.id.as_str()).collect();
                        let todo = messages.iter().filter(|m| !have.contains(m.id.as_str()));
                        (snapshot.version, todo.cloned().collect())
                    }
                    Err(error) => {
                        last = error;
                        continue;
                    }
                },
            };
            if todo.is_empty() {
                return Ok(at);
            }
            match self.store.append(thread, at, &todo).await {
                Ok(version) => return Ok(version),
                Err(AppendError::Conflict) => {}
                Err(AppendError::Failed(error)) => last = error,
            }
        }
        Err(last)
    }
}

#[cfg(test)]
mod tests;
