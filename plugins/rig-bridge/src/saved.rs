//! What is kept of a conversation on disk, so that a later process can go
//! on from it. Each session works in a directory of its own, where the
//! program saves the conversation its own way. Beside it the bridge keeps a
//! record of where Rig's history stood when the program last ended a turn:
//! a request whose history goes on from there starts the program on its
//! saved session, with everything it had, instead of on the conversation
//! written out as text.
//!
//! A kept directory is never removed by the bridge.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::session::process_alive;

#[cfg(test)]
mod tests;

/// The record's file in a session's directory.
const RECORD: &str = "rig.json";

/// Prefix of a session's directory, before the process that made it.
const PREFIX: &str = "rig-bridge-";

/// The directories the sessions of this process work in.
pub(crate) type Held = Arc<Mutex<HashSet<PathBuf>>>;

/// A session's directory: the program's working directory, with the files
/// it is pointed at and what it saves of the conversation. It is removed
/// with the session unless it was [kept](Self::keep).
pub(crate) struct SessionDir {
    path: PathBuf,
    held: Held,
    kept: bool,
}

impl SessionDir {
    /// A new, empty directory under `root`.
    pub(crate) fn create(root: &Path, held: &Held) -> std::io::Result<Self> {
        let mut builder = std::fs::DirBuilder::new();
        builder.recursive(true);
        #[cfg(unix)]
        std::os::unix::fs::DirBuilderExt::mode(&mut builder, 0o700);
        builder.create(root)?;
        let id = uuid::Uuid::new_v4().simple();
        let path = root.join(format!("{PREFIX}{}-{id}", std::process::id()));
        builder.recursive(false);
        builder.create(&path)?;
        lock(held).insert(path.clone());
        Ok(Self {
            path,
            held: held.clone(),
            kept: false,
        })
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    /// Writes `record` and keeps the directory from now on: whether it was
    /// written. The file is replaced whole, so a reader finds the record
    /// before or the one after.
    pub(crate) fn keep(&mut self, record: &Record) -> bool {
        let Ok(text) = serde_json::to_vec(record) else {
            return false;
        };
        let new = self.path.join(format!("{RECORD}.new"));
        let written = std::fs::write(&new, text)
            .and_then(|()| std::fs::rename(&new, self.path.join(RECORD)))
            .is_ok();
        self.kept |= written;
        written
    }

    /// Whether a record was written here.
    pub(crate) fn is_kept(&self) -> bool {
        self.kept
    }
}

impl Drop for SessionDir {
    fn drop(&mut self) {
        lock(&self.held).remove(&self.path);
        if !self.kept {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }
}

fn lock(held: &Held) -> std::sync::MutexGuard<'_, HashSet<PathBuf>> {
    held.lock().unwrap_or_else(PoisonError::into_inner)
}

/// What a kept directory holds, and where its conversation stands.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct Record {
    /// The Rig process whose session last worked here.
    pub pid: u32,
    pub vendor: String,
    pub model: String,
    /// The digest of the system prompt.
    pub system: String,
    /// The digest of the names of the tools the session was last given.
    pub tools: String,
    /// The program's own name for the session it saved.
    pub session: String,
    /// Where the conversation stands. `None` while the program answers:
    /// what it saved is then ahead of Rig's history, and nothing can go on
    /// from it.
    pub stands: Option<Stand>,
}

/// The point of Rig's history a saved session is at.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct Stand {
    /// How many messages of the history the program was given.
    pub seen: usize,
    /// The digest of those messages, the last of [`prefixes`] of them.
    pub key: String,
    /// The digest of what the program answered them with. `None` where the
    /// user stopped the answer: what Rig has of it, if anything, is then
    /// not what the program has.
    pub said: Option<String>,
}

/// The kept directory under `root` whose record `goes_on` accepts, as the
/// directory of a session of this process. One a live session works in,
/// here or in another Rig process, is passed over.
pub(crate) fn find(
    root: &Path,
    held: &Held,
    goes_on: impl Fn(&Record, &Stand) -> bool,
) -> Option<(SessionDir, Record)> {
    let me = std::process::id();
    for entry in std::fs::read_dir(root).ok()?.flatten() {
        let path = entry.path();
        let Some(record) = read(&path) else {
            continue;
        };
        let Some(stands) = &record.stands else {
            continue;
        };
        // Whether a process is alive costs more to ask than what it left.
        if !goes_on(&record, stands) || (record.pid != me && process_alive(record.pid)) {
            continue;
        }
        let mut held_now = lock(held);
        if held_now.insert(path.clone()) {
            drop(held_now);
            let dir = SessionDir {
                path,
                held: held.clone(),
                kept: true,
            };
            return Some((dir, record));
        }
    }
    None
}

fn read(dir: &Path) -> Option<Record> {
    serde_json::from_slice(&std::fs::read(dir.join(RECORD)).ok()?).ok()
}

/// Removes the directories under `root` that were never kept and whose
/// process is gone: a session that ended with its process, not by itself.
pub(crate) fn reap(root: &Path) {
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let pid = name
            .to_str()
            .and_then(|name| name.strip_prefix(PREFIX)?.split_once('-')?.0.parse().ok());
        let gone = pid.is_some_and(|pid: u32| pid != std::process::id() && !process_alive(pid));
        if gone && !path.join(RECORD).exists() {
            let _ = std::fs::remove_dir_all(&path);
        }
    }
}

/// The digest of `text`.
pub(crate) fn digest(text: &str) -> String {
    Sha256::digest(text.as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// The digest of each prefix of `texts`: the one at `n` is of the first
/// `n`, so it is the same for every sequence that starts with them.
pub(crate) fn prefixes(texts: impl IntoIterator<Item = String>) -> Vec<String> {
    let mut all = vec![digest("")];
    for text in texts {
        let last = all.last().map_or("", String::as_str);
        all.push(digest(&format!("{last}\n{text}")));
    }
    all
}
