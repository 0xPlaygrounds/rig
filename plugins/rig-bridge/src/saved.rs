//! What is kept of a conversation on disk, so that a later process can go
//! on from it. Each session works in a directory of its own, where the
//! program saves the conversation its own way. Beside it the bridge keeps a
//! record of where Rig's history stood when the program last ended a turn:
//! a request whose history goes on from there starts the program on its
//! saved session, with everything it had, instead of on the conversation
//! written out as text.
//!
//! A kept directory is never removed by the bridge.

use std::fs::File;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[cfg(test)]
mod tests;

/// The record's file in a session's directory.
const RECORD: &str = "rig.json";

/// The file a session holds locked in its directory while it lives.
const LOCK: &str = "rig.lock";

/// Prefix of a session's directory, before the process that made it.
const PREFIX: &str = "rig-bridge-";

/// A session's directory: the program's working directory, with the files
/// it is pointed at and what it saves of the conversation. One session
/// works in it at a time, of whichever Rig process: it holds a lock on a
/// file there for as long as it lives, which the system lets go of when
/// the process ends, however it ends. The directory is removed with the
/// session unless it was [kept](Self::keep).
pub(crate) struct SessionDir {
    path: PathBuf,
    kept: bool,
    _lock: File,
}

impl SessionDir {
    /// A new, empty directory under `root`.
    pub(crate) fn create(root: &Path) -> std::io::Result<Self> {
        let mut builder = std::fs::DirBuilder::new();
        builder.recursive(true);
        #[cfg(unix)]
        std::os::unix::fs::DirBuilderExt::mode(&mut builder, 0o700);
        builder.create(root)?;
        let id = uuid::Uuid::new_v4().simple();
        let path = root.join(format!("{PREFIX}{}-{id}", std::process::id()));
        builder.recursive(false);
        builder.create(&path)?;
        let lock = locked(&path).ok_or_else(|| std::io::Error::other("it is in use"))?;
        Ok(Self {
            path,
            kept: false,
            _lock: lock,
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
        if !self.kept {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }
}

/// The lock on `dir`, unless a session holds it.
fn locked(dir: &Path) -> Option<File> {
    let mut options = File::options();
    let lock = options.create(true).truncate(false).write(true);
    let lock = lock.open(dir.join(LOCK)).ok()?;
    lock.try_lock().ok()?;
    Some(lock)
}

/// What a kept directory holds, and where its conversation stands.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct Record {
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
    /// What the program answered them with.
    pub said: Reply,
}

/// What a program said in answer to the history it stands at.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Reply {
    /// A whole reply, by its digest.
    Whole(String),
    /// The text of a reply until the user stopped it. Rig may have less of
    /// it, never other text.
    Cut(String),
}

/// The kept directory under `root` whose record `goes_on` accepts, as the
/// directory of a session of this process. One a session works in, of this
/// Rig process or another, is passed over.
pub(crate) fn find(
    root: &Path,
    goes_on: impl Fn(&Record, &Stand) -> bool,
) -> Option<(SessionDir, Record)> {
    let stands = |path: &Path| {
        let record = read(path)?;
        let stands = record.stands.as_ref()?;
        goes_on(&record, stands).then_some(record)
    };
    for entry in std::fs::read_dir(root).ok()?.flatten() {
        let path = entry.path();
        if stands(&path).is_none() {
            continue;
        }
        let Some(lock) = locked(&path) else {
            continue;
        };
        // What was read before the lock was taken may be of a session that
        // has since gone on: only what is read under it holds.
        if let Some(record) = stands(&path) {
            let dir = SessionDir {
                path,
                kept: true,
                _lock: lock,
            };
            return Some((dir, record));
        }
    }
    None
}

fn read(dir: &Path) -> Option<Record> {
    serde_json::from_slice(&std::fs::read(dir.join(RECORD)).ok()?).ok()
}

/// Removes the directories under `root` that were never kept and that no
/// session works in: one whose session ended with its process, not by
/// itself.
pub(crate) fn reap(root: &Path) {
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let ours = entry.file_name().to_string_lossy().starts_with(PREFIX);
        if ours && !path.join(RECORD).exists() && locked(&path).is_some() {
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
