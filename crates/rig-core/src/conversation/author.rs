//! Who wrote an entry or an aside of a conversation, or made a change to it.

use serde::{Deserialize, Serialize};

use super::short_id;

/// The id of a request to an agent, which the reply to it names, such as
/// the tool call that asked. Unique per sender, so it doubles as an
/// idempotency key.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RequestId(pub String);

/// Who wrote it.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthorKind {
    /// The user, typing.
    #[default]
    User,
    /// An agent, such as a subagent answering or an agent handing out a
    /// task.
    Agent,
    /// The plugin named.
    Plugin(String),
    /// rig itself: the stopped note, restart answers, the kernel's edits.
    Rig,
}

/// Who wrote an entry or an aside, or made a change: the kind of writer,
/// the id of the agent it came from, the request it answers or makes, and
/// what it is about. An entry without one is the user's own text, the
/// model's reply or a tool result.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Author {
    /// Who wrote it.
    #[serde(default)]
    pub kind: AuthorKind,
    /// The id of the agent it came from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub from: Option<String>,
    /// The request it answers or makes. Kept for correlation, such as a
    /// report to the call that asked; not shown in its header.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request: Option<RequestId>,
    /// What it is about, such as a subagent's task title, shown in its
    /// header.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
}

impl Author {
    /// The user, typing.
    pub fn user() -> Self {
        Self::default()
    }

    /// The agent whose id is `from`, answering or making `request`.
    pub fn agent(from: impl Into<String>, request: Option<RequestId>) -> Self {
        Self {
            kind: AuthorKind::Agent,
            from: Some(from.into()),
            request,
            title: None,
        }
    }

    /// The plugin `name`.
    pub fn plugin(name: impl Into<String>) -> Self {
        Self {
            kind: AuthorKind::Plugin(name.into()),
            ..Self::default()
        }
    }

    /// rig itself.
    pub fn rig() -> Self {
        Self {
            kind: AuthorKind::Rig,
            ..Self::default()
        }
    }

    /// This author, writing about `title`, such as a task's.
    #[must_use]
    pub fn titled(self, title: impl Into<String>) -> Self {
        Self {
            title: Some(title.into()),
            ..self
        }
    }

    /// Who wrote it, for views: such as `agent 1a2b3c4d "Fix the parser"`.
    /// The request id is left out; it stays in [`Author::request`].
    pub fn label(&self) -> String {
        let who = match &self.kind {
            AuthorKind::User => "you".to_owned(),
            AuthorKind::Agent => {
                format!("agent {}", self.from.as_deref().map_or("?", short_id))
            }
            AuthorKind::Plugin(name) => format!("plugin {name}"),
            AuthorKind::Rig => "rig".to_owned(),
        };
        match &self.title {
            Some(title) => format!("{who} \"{title}\""),
            None => who,
        }
    }

    /// The line that heads text by this author for the model, or `None`
    /// for the user's own: it names the writer and says the text is not
    /// the user's words.
    pub fn header(&self) -> Option<String> {
        match self.kind {
            AuthorKind::User => None,
            _ => Some(format!(
                "[Output of {}, not the user's words]",
                self.label()
            )),
        }
    }
}
