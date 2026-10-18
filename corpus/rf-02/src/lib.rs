//! `app-chat`: rig agents over conversations kept in the app's own
//! repository. The repository is the only copy: every turn reads the
//! conversation from it as the agent's history
//! ([`AgentRunner::history`](rig_agent::AgentRunner::history)) and appends
//! the messages the run committed
//! ([`PromptResponse::messages`](rig_agent::agent::PromptResponse::messages))
//! back to it. rig keeps nothing between turns.
//!
//! The adapter is [`encode`] and [`decode`]: a rig [`Message`] is stored as
//! the JSON value its serde form is, so it comes back equal.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use rig_agent::Agent;
use rig_agent::completion::PromptError;
use rig_core::message::{AssistantContent, Message, Text, UserContent};
use serde_json::Value;

/// A message's id within its conversation, given by the repository.
pub type MessageId = u64;

/// The app's repository errors.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RepoError {
    /// No such conversation or message.
    #[error("not found")]
    NotFound,
    /// The conversation was soft-deleted.
    #[error("conversation deleted")]
    Deleted,
    /// The user may not use the conversation.
    #[error("access denied")]
    Forbidden,
    /// Someone appended to the conversation since it was read.
    #[error("conversation changed since it was read")]
    Conflict,
}

/// A message as the repository keeps it: its id and its stored body.
#[derive(Debug, Clone, PartialEq)]
pub struct StoredMessage {
    /// Its id.
    pub id: MessageId,
    /// The stored body ([`encode`]).
    pub body: Value,
}

/// The app's conversation repository, shared by every request handler, job
/// and agent as one `Arc<dyn Conversations>`.
pub trait Conversations: Send + Sync {
    /// Whether `user` may write to `conversation`.
    ///
    /// # Errors
    /// [`RepoError::NotFound`], [`RepoError::Deleted`] or [`RepoError::Forbidden`].
    fn authorize(&self, user: &str, conversation: &str) -> Result<(), RepoError>;

    /// The conversation's messages, oldest first.
    ///
    /// # Errors
    /// The repository's.
    fn messages(&self, conversation: &str) -> Result<Vec<StoredMessage>, RepoError>;

    /// Appends `bodies` if the conversation still holds `expected` messages.
    ///
    /// # Errors
    /// [`RepoError::Conflict`] when it does not, writing nothing.
    fn append(
        &self,
        conversation: &str,
        expected: usize,
        bodies: Vec<Value>,
    ) -> Result<Vec<MessageId>, RepoError>;

    /// Replaces the body of message `id`.
    ///
    /// # Errors
    /// The repository's.
    fn update(&self, conversation: &str, id: MessageId, body: Value) -> Result<(), RepoError>;
}

/// Why a [`ChatService`] call failed.
#[derive(Debug, thiserror::Error)]
pub enum ChatError {
    /// The repository refused, its error as it gave it.
    #[error(transparent)]
    Repo(#[from] RepoError),
    /// A stored body is not a message.
    #[error("stored message does not decode: {0}")]
    Decode(#[from] serde_json::Error),
    /// No agent of that name.
    #[error("no agent named {0:?}")]
    UnknownAgent(String),
    /// The edited message has no text to replace.
    #[error("message has no text")]
    NoText,
    /// The model run failed; the repository is unchanged.
    #[error(transparent)]
    Model(#[from] Box<PromptError>),
}

/// A message's stored body.
///
/// # Errors
/// When serialization fails.
pub fn encode(message: &Message) -> Result<Value, serde_json::Error> {
    serde_json::to_value(message)
}

/// The message a stored body holds.
///
/// # Errors
/// When `body` is not a message.
pub fn decode(body: Value) -> Result<Message, serde_json::Error> {
    serde_json::from_value(body)
}

/// Agents chatting over the app's conversations.
pub struct ChatService {
    repo: Arc<dyn Conversations>,
    agents: HashMap<String, Agent>,
}

impl ChatService {
    /// A service over `repo` with `agents` by name.
    pub fn new(
        repo: Arc<dyn Conversations>,
        agents: impl IntoIterator<Item = (impl Into<String>, Agent)>,
    ) -> Self {
        let agents = agents.into_iter().map(|(name, agent)| (name.into(), agent));
        Self {
            repo,
            agents: agents.collect(),
        }
    }

    /// Sends `text` from `user` to the agent `agent` in `conversation` and
    /// returns its reply. Every message the run added is in the repository
    /// when this returns.
    ///
    /// # Errors
    /// The repository's error before any model call when the user may not
    /// use the conversation; [`RepoError::Conflict`] when it was appended to
    /// during the turn; the run's error, writing nothing.
    pub async fn send(
        &self,
        user: &str,
        conversation: &str,
        text: &str,
        agent: &str,
    ) -> Result<String, ChatError> {
        let runner = self
            .agents
            .get(agent)
            .ok_or_else(|| ChatError::UnknownAgent(agent.to_owned()))?;
        self.repo.authorize(user, conversation)?;
        let history = self.history(conversation).await?;
        let expected = history.len();
        let response = runner
            .prompt(text)
            .history(history.into_iter().map(|(_, message)| message))
            .await
            .map_err(Box::new)?;
        let bodies = response.messages.iter().map(encode);
        let bodies = bodies.collect::<Result<Vec<_>, _>>()?;
        self.repo.append(conversation, expected, bodies)?;
        Ok(response.output())
    }

    /// The conversation as the agents see it, with each message's id.
    ///
    /// # Errors
    /// The repository's, or a body that does not decode.
    pub async fn history(
        &self,
        conversation: &str,
    ) -> Result<Vec<(MessageId, Message)>, ChatError> {
        let stored = self.repo.messages(conversation)?.into_iter();
        let decoded = stored.map(|row| Ok((row.id, decode(row.body)?)));
        decoded.collect()
    }

    /// Replaces the text of message `id` with `new_text`, keeping its other
    /// content.
    ///
    /// # Errors
    /// The repository's, or [`ChatError::NoText`].
    pub async fn edit(
        &self,
        conversation: &str,
        id: MessageId,
        new_text: &str,
    ) -> Result<(), ChatError> {
        let (_, message) = self
            .history(conversation)
            .await?
            .into_iter()
            .find(|(stored, _)| *stored == id)
            .ok_or(RepoError::NotFound)?;
        let edited = replace_text(message, new_text).ok_or(ChatError::NoText)?;
        Ok(self.repo.update(conversation, id, encode(&edited)?)?)
    }
}

/// `message` with its first text part `text` and its other text parts gone.
fn replace_text(message: Message, text: &str) -> Option<Message> {
    let mut replaced = false;
    let mut first = || !std::mem::replace(&mut replaced, true);
    let message = match message {
        Message::System { .. } => Message::system(text),
        Message::User { content } => Message::User {
            content: content
                .into_iter()
                .filter_map(|part| match part {
                    UserContent::Text(_) => first().then(|| UserContent::Text(Text::new(text))),
                    other => Some(other),
                })
                .collect(),
        },
        Message::Assistant(mut turn) => {
            let content = std::mem::take(&mut turn.content);
            let content = content.into_iter().filter_map(|part| match part {
                AssistantContent::Text(_) => {
                    first().then(|| AssistantContent::Text(Text::new(text)))
                }
                other => Some(other),
            });
            Message::Assistant(turn.with_content(content.collect()))
        }
    };
    (replaced || matches!(message, Message::System { .. })).then_some(message)
}

/// A conversation as the app keeps it: its own columns beside the messages.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ConversationRow {
    /// Its owner.
    pub owner: String,
    /// Its title.
    pub title: String,
    /// Its tags.
    pub tags: Vec<String>,
    /// Whether it was soft-deleted.
    pub deleted: bool,
    /// Its messages, oldest first.
    pub messages: Vec<StoredMessage>,
    next_id: MessageId,
}

/// The app's repository over an in-memory map behind a mutex, standing in
/// for its database.
#[derive(Default)]
pub struct InMemoryConversations(Mutex<BTreeMap<String, ConversationRow>>);

impl InMemoryConversations {
    fn rows(&self) -> MutexGuard<'_, BTreeMap<String, ConversationRow>> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn with<T>(
        &self,
        id: &str,
        f: impl FnOnce(&mut ConversationRow) -> Result<T, RepoError>,
    ) -> Result<T, RepoError> {
        self.rows()
            .get_mut(id)
            .ok_or(RepoError::NotFound)
            .and_then(f)
    }

    /// Starts the conversation `id` owned by `owner`.
    pub fn create(&self, id: &str, owner: &str, title: &str, tags: &[&str]) {
        let row = ConversationRow {
            owner: owner.to_owned(),
            title: title.to_owned(),
            tags: tags.iter().map(|tag| (*tag).to_owned()).collect(),
            ..ConversationRow::default()
        };
        self.rows().insert(id.to_owned(), row);
    }

    /// The conversation's row, as the app's admin listing shows it.
    ///
    /// # Errors
    /// [`RepoError::NotFound`].
    pub fn row(&self, id: &str) -> Result<ConversationRow, RepoError> {
        self.with(id, |row| Ok(row.clone()))
    }

    /// Soft-deletes the conversation.
    ///
    /// # Errors
    /// [`RepoError::NotFound`].
    pub fn soft_delete(&self, id: &str) -> Result<(), RepoError> {
        self.with(id, |row| {
            row.deleted = true;
            Ok(())
        })
    }

    /// Removes message `message`, as deleting an exchange does.
    ///
    /// # Errors
    /// [`RepoError::NotFound`].
    pub fn remove(&self, id: &str, message: MessageId) -> Result<(), RepoError> {
        self.with(id, |row| {
            row.messages.retain(|stored| stored.id != message);
            Ok(())
        })
    }
}

impl Conversations for InMemoryConversations {
    fn authorize(&self, user: &str, conversation: &str) -> Result<(), RepoError> {
        self.with(conversation, |row| match () {
            () if row.deleted => Err(RepoError::Deleted),
            () if row.owner != user => Err(RepoError::Forbidden),
            () => Ok(()),
        })
    }

    fn messages(&self, conversation: &str) -> Result<Vec<StoredMessage>, RepoError> {
        self.with(conversation, |row| Ok(row.messages.clone()))
    }

    fn append(
        &self,
        conversation: &str,
        expected: usize,
        bodies: Vec<Value>,
    ) -> Result<Vec<MessageId>, RepoError> {
        self.with(conversation, |row| {
            if row.messages.len() != expected {
                return Err(RepoError::Conflict);
            }
            let ids = bodies.into_iter().map(|body| {
                row.next_id += 1;
                row.messages.push(StoredMessage {
                    id: row.next_id,
                    body,
                });
                row.next_id
            });
            Ok(ids.collect())
        })
    }

    fn update(&self, conversation: &str, id: MessageId, body: Value) -> Result<(), RepoError> {
        self.with(conversation, |row| {
            let stored = row.messages.iter_mut().find(|stored| stored.id == id);
            stored.ok_or(RepoError::NotFound)?.body = body;
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests;
