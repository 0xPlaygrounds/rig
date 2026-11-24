//! What is kept of a conversation on disk, so that a later process can go
//! on from it. Each session works in a directory of its own, where the
//! program saves the conversation its own way. Beside it the bridge keeps a
//! record of where Rig's history stood when the program last ended a turn:
//! a request whose history goes on from there starts the program on its
//! saved session, with everything it had, instead of on the conversation
//! written out as text.
//!
//! Kept directories are pruned on startup once unused for three days or
//! past a cap of 100 sessions (oldest first), and on demand via
//! `/bridge prune`.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::session::process_alive;

#[cfg(test)]
mod tests;

/// The record's file in a session's directory.
const RECORD: &str = "rig.json";

/// The file a session holds locked in its directory while it lives.
const LOCK: &str = "rig.lock";

/// Prefix of a session's directory, before the process that made it.
const PREFIX: &str = "rig-bridge-";

/// How long an unused saved session is kept before startup pruning removes it.
const MAX_AGE: Duration = Duration::from_secs(3 * 24 * 60 * 60);

/// How many saved sessions are kept at most on startup; older ones go first.
const MAX_KEPT: usize = 100;

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

/// Removes leftover directories from dead processes, unresumable records,
/// and saved sessions past the age or count limit.
pub(crate) fn reap(root: &Path) {
    let _ = prune_with(root, SystemTime::now(), Some(MAX_AGE), MAX_KEPT);
}

/// Removes every idle saved session and dead-process leftover under `root`,
/// returning how many directories were removed.
pub(crate) fn prune(root: &Path) -> usize {
    prune_with(root, SystemTime::now(), None, 0)
}

fn prune_with(root: &Path, now: SystemTime, max_age: Option<Duration>, max_kept: usize) -> usize {
    let Ok(entries) = std::fs::read_dir(root) else {
        return 0;
    };
    let mut removed = 0;
    let mut kept = Vec::new();

    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let Some(made_by) = name.to_str().and_then(|name| {
            name.strip_prefix(PREFIX)?
                .split_once('-')?
                .0
                .parse::<u32>()
                .ok()
        }) else {
            continue;
        };

        let record_path = path.join(RECORD);
        if !record_path.exists() {
            // A directory without a record whose creator lives may be one
            // it is still making.
            if made_by != std::process::id()
                && !process_alive(made_by)
                && let Some(_lock) = locked(&path)
                && !record_path.exists()
                && std::fs::remove_dir_all(&path).is_ok()
            {
                removed += 1;
            }
            continue;
        }

        let unresumable = |dir: &Path| read(dir).is_none_or(|record| record.stands.is_none());
        if unresumable(&path) {
            // A session that ended mid-turn, or whose record cannot be read,
            // can never be resumed.
            if let Some(_lock) = locked(&path)
                && record_path.exists()
                && unresumable(&path)
                && std::fs::remove_dir_all(&path).is_ok()
            {
                removed += 1;
            }
            continue;
        }

        let modified = std::fs::metadata(&record_path)
            .and_then(|meta| meta.modified())
            .ok();
        let expired = max_age.is_some_and(|limit| {
            modified
                .and_then(|time| now.duration_since(time).ok())
                .is_some_and(|age| age > limit)
        });
        if expired || max_kept == 0 {
            if let Some(_lock) = locked(&path) {
                let still_removable = if max_kept == 0 {
                    record_path.exists()
                } else {
                    std::fs::metadata(&record_path)
                        .and_then(|meta| meta.modified())
                        .ok()
                        == modified
                };
                if still_removable && std::fs::remove_dir_all(&path).is_ok() {
                    removed += 1;
                }
            }
            continue;
        }

        kept.push((modified.unwrap_or(SystemTime::UNIX_EPOCH), path));
    }

    if let Some(excess) = kept.len().checked_sub(max_kept).filter(|&n| n > 0) {
        // Newest first; everything past `max_kept` is removed.
        kept.sort_by_key(|entry| std::cmp::Reverse(entry.0));
        let start = kept.len() - excess;
        if let Some(to_remove) = kept.get(start..) {
            for (recorded_time, path) in to_remove {
                let record_path = path.join(RECORD);
                if let Some(_lock) = locked(path)
                    && std::fs::metadata(&record_path)
                        .and_then(|meta| meta.modified())
                        .ok()
                        .unwrap_or(SystemTime::UNIX_EPOCH)
                        == *recorded_time
                    && std::fs::remove_dir_all(path).is_ok()
                {
                    removed += 1;
                }
            }
        }
    }

    removed
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
