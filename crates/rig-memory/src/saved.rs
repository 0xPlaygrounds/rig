//! A saved agent's values read without the host that saved them: the keys
//! rig saves an agent's model and title and the session's current agent
//! under, typed reads of any saved value, and which agent of a session is
//! its main one.

use std::collections::BTreeMap;

use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::{SavedAgent, Session};

/// The key an agent's model is saved under: a string, `vendor/model`.
pub const MODEL: &str = "rig::model";

/// The key an agent's title is saved under, such as a subagent's task: a
/// string.
pub const TITLE: &str = "rig::title";

/// The key the session's current agent is saved under, the agent the user
/// talks to when none is named: a string, the agent's id. One of the
/// session's own values.
pub const CURRENT: &str = "rig::current_agent";

impl SavedAgent {
    /// Its saved value under `key`, such as a plugin's `goals.goal`, read
    /// as a `T`: `None` when it has none, an error when the value is not a
    /// `T`.
    pub fn value<T: DeserializeOwned>(&self, key: &str) -> Option<Result<T, serde_json::Error>> {
        value(&self.saved, key)
    }

    /// Its model, `vendor/model`, when one was saved ([`MODEL`]).
    pub fn model(&self) -> Option<String> {
        self.value(MODEL)?.ok()
    }

    /// Its title, when one was saved ([`TITLE`]).
    pub fn title(&self) -> Option<String> {
        self.value(TITLE)?.ok()
    }
}

/// The value of `key` in `state`, parsed.
fn value<T: DeserializeOwned>(
    state: &BTreeMap<String, Value>,
    key: &str,
) -> Option<Result<T, serde_json::Error>> {
    state.get(key).map(T::deserialize)
}

impl Session {
    /// The session's own saved value under `key` (the [`SESSION`](crate::SESSION)
    /// log's), read as a `T`: `None` when it has none, an error when the
    /// value is not a `T`.
    pub fn value<T: DeserializeOwned>(&self, key: &str) -> Option<Result<T, serde_json::Error>> {
        value(&self.saved, key)
    }

    /// The main agent: of the agents nothing spawned and not retired, the
    /// one [`CURRENT`] names, else the first by id.
    pub fn main_agent(&self) -> Option<&SavedAgent> {
        let top = (self.agents.iter()).filter(|agent| agent.parent.is_none() && !agent.retired);
        let current = self.value::<String>(CURRENT).and_then(Result::ok);
        let named = top
            .clone()
            .find(|agent| Some(&agent.id) == current.as_ref());
        named.or_else(|| top.min_by_key(|agent| &agent.id))
    }
}

#[cfg(test)]
mod tests;
