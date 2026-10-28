//! The user's answers about the servers of the project's `.rig/mcp.json`:
//! each server approved or denied as it was written, kept for this user and
//! project in `mcp/approvals.json` of the project's plugin data, never in
//! the project. A server whose object changed since is unanswered again,
//! and so is every server of a file kept for another project, as two
//! projects' data can share a directory.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use bevy_ecs::prelude::*;
use rig_harness::plugin_data::PluginData;
use rig_harness::rig_memory::Files;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[cfg(test)]
mod tests;

/// The file in the project's plugin data.
const FILE: &str = "mcp/approvals.json";

/// What the file holds: the project it is about, and the answers.
#[derive(Serialize, Deserialize, Default)]
struct Kept {
    project: String,
    servers: BTreeMap<String, Answer>,
}

/// One answer: whether `server`, as written then, may start.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
struct Answer {
    approved: bool,
    server: Value,
}

/// The answers of this project, by server name.
#[derive(Resource, Default)]
pub struct Approvals {
    /// Where they are kept; in memory only without plugin data.
    files: Option<Arc<dyn Files>>,
    /// The project, as its canonical path.
    project: String,
    /// The lock a change holds while it reads the file again and writes it.
    lock: PathBuf,
    answers: BTreeMap<String, Answer>,
}

impl Approvals {
    /// The answers about `project` kept in `data`, if any, each change
    /// holding `lock`; a file that does not read, or is about another
    /// project, is no answers.
    pub fn read(data: Option<&PluginData>, project: String, lock: PathBuf) -> Self {
        let mut approvals = Self {
            files: data.map(|data| data.project.clone()),
            project,
            lock,
            answers: BTreeMap::new(),
        };
        approvals.answers = approvals.kept();
        approvals
    }

    /// The answers the file holds now.
    fn kept(&self) -> BTreeMap<String, Answer> {
        let kept = self
            .files
            .as_ref()
            .and_then(|files| files.read_file(FILE).ok())
            .and_then(|bytes| serde_json::from_slice::<Kept>(&bytes).ok())
            .filter(|kept| kept.project == self.project);
        kept.map(|kept| kept.servers).unwrap_or_default()
    }

    /// Whether the server `name`, written as `server`, was approved
    /// (`Some(true)`) or denied (`Some(false)`) as it is written now.
    pub fn answer(&self, name: &str, server: &Value) -> Option<bool> {
        let answer = self.answers.get(name)?;
        (answer.server == *server).then_some(answer.approved)
    }

    /// Keeps that the server `name`, written as `server`, is `approved` or
    /// denied, over the answers the file holds now, so another rig's
    /// answers since stay.
    ///
    /// # Errors
    ///
    /// When the answers cannot be written.
    pub fn set(&mut self, name: &str, server: Value, approved: bool) -> Result<(), String> {
        let answer = Answer { approved, server };
        let Some(files) = self.files.clone() else {
            self.answers.insert(name.to_owned(), answer);
            return Ok(());
        };
        let _lock = crate::config::lock(&self.lock)?;
        self.answers = self.kept();
        self.answers.insert(name.to_owned(), answer);
        let kept = Kept {
            project: self.project.clone(),
            servers: self.answers.clone(),
        };
        let bytes = serde_json::to_vec_pretty(&kept).map_err(|error| error.to_string())?;
        files
            .write_file(FILE, &bytes)
            .map_err(|error| format!("could not keep the approval: {error}"))
    }
}
