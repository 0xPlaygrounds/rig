//! A saved agent's values read without the host that saved them: the keys
//! rig saves an agent's model and title under, typed reads of any saved
//! value, and which agent of a session is its main one.

use std::collections::BTreeMap;

use serde::de::DeserializeOwned;

use crate::session::skipped;
use crate::{SavedAgent, Session, Upgrades, ValueError, Versioned};

/// The key an agent's model is saved under: a string, `vendor/model`.
pub const MODEL: &str = "rig::model";

/// The key an agent's title is saved under, such as a subagent's task: a
/// string.
pub const TITLE: &str = "rig::title";

impl SavedAgent {
    /// Its saved value under `key` as saved, not upgraded, read as a `T`:
    /// `None` when it has none, an error when the value is not a `T`. A
    /// program reads an older version of a plugin's value through
    /// [`Upgrades::upgrade`](crate::Upgrades::upgrade).
    pub fn value<T: DeserializeOwned>(&self, key: &str) -> Option<Result<T, serde_json::Error>> {
        self.saved
            .get(key)
            .map(|saved| T::deserialize(&saved.value))
    }

    /// Its saved value under `key`, such as a plugin's `goals.goal`, read
    /// at its current version through `upgrades` and parsed as a `T`:
    /// `None` when it has none.
    ///
    /// # Errors
    ///
    /// Inside the option: [`ValueError::Unread`] when no step reads its
    /// version, [`ValueError::Parse`] when the upgraded value is not a `T`.
    pub fn get<T: DeserializeOwned>(
        &self,
        key: &str,
        upgrades: &Upgrades,
    ) -> Option<Result<T, ValueError>> {
        get(&self.saved, key, upgrades)
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

/// The value of `key` in `state`, upgraded through `upgrades` and parsed.
fn get<T: DeserializeOwned>(
    state: &BTreeMap<String, Versioned>,
    key: &str,
    upgrades: &Upgrades,
) -> Option<Result<T, ValueError>> {
    let saved = state.get(key)?.clone();
    let value = upgrades.upgrade(key, saved).map_err(ValueError::from);
    Some(value.and_then(|value| serde_json::from_value(value).map_err(ValueError::from)))
}

/// Where an agent sorts among the agents of a session: those nothing
/// spawned first, each group by id. The first is the main agent, the one
/// the user talks to when none is named.
pub fn main_order(spawned: bool, id: &str) -> (bool, &str) {
    (spawned, id)
}

impl Session {
    /// The session's own saved value under `key` (the [`SESSION`](crate::SESSION)
    /// log's), read at its current version through `upgrades` and parsed as
    /// a `T`: `None` when it has none.
    ///
    /// # Errors
    ///
    /// Inside the option: as [`SavedAgent::get`].
    pub fn get<T: DeserializeOwned>(
        &self,
        key: &str,
        upgrades: &Upgrades,
    ) -> Option<Result<T, ValueError>> {
        get(&self.saved, key, upgrades)
    }

    /// The main agent: the first by [`main_order`] of the agents a restore
    /// brings back (not retired, nor spawned by one retired).
    pub fn main_agent(&self) -> Option<&SavedAgent> {
        let logs = self.agents.iter().map(|agent| {
            let parent = agent.parent.as_deref();
            (agent.id.as_str(), parent, agent.retired)
        });
        let skipped = skipped(logs);
        self.agents
            .iter()
            .filter(|agent| !skipped.contains(agent.id.as_str()))
            .min_by_key(|agent| main_order(agent.parent.is_some(), &agent.id))
    }
}

#[cfg(test)]
mod tests;
