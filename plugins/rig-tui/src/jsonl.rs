//! Files of one JSON value per line that only grow, such as the prompt
//! history and the notices kept for the next build: values are appended,
//! and the newest are read back. A file is plain, or one of the session's
//! store.

use std::fmt;
use std::fs;
use std::io::{self, Write as _};
use std::path::PathBuf;
use std::sync::Arc;

use bevy_log::warn;
use rig_harness::prelude::JournalStore;
use serde::Serialize;
use serde::de::DeserializeOwned;

/// Where a file of values is kept.
#[derive(Clone)]
pub(crate) enum Kept {
    /// A plain file, its directory made as needed.
    Plain(PathBuf),
    /// A file of the session's store, by name.
    Stored(Arc<dyn JournalStore>, &'static str),
}

impl Kept {
    fn read(&self) -> io::Result<Vec<u8>> {
        match self {
            Self::Plain(path) => fs::read(path),
            Self::Stored(store, name) => store.read_file(name),
        }
    }

    fn append(&self, bytes: &[u8]) -> io::Result<()> {
        match self {
            Self::Plain(path) => {
                if let Some(parent) = path.parent() {
                    fs::create_dir_all(parent)?;
                }
                let mut out = fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)?;
                out.write_all(bytes)
            }
            Self::Stored(store, name) => store.append_file(name, bytes),
        }
    }

    /// Replaces the file with `bytes`, so a reader sees the old or the new.
    fn replace(&self, bytes: &[u8]) -> io::Result<()> {
        match self {
            Self::Plain(path) => {
                let temporary = path.with_extension("jsonl.new");
                fs::write(&temporary, bytes).and_then(|()| fs::rename(&temporary, path))
            }
            Self::Stored(store, name) => store.write_file(name, bytes),
        }
    }
}

impl fmt::Display for Kept {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Plain(path) => path.display().fmt(f),
            Self::Stored(_, name) => write!(f, "the session's {name}"),
        }
    }
}

/// `values`, one per line.
fn lines<'a, T: Serialize + 'a>(values: impl IntoIterator<Item = &'a T>) -> String {
    values
        .into_iter()
        .filter_map(|value| serde_json::to_string(value).ok())
        .map(|line| line + "\n")
        .collect()
}

/// Appends `values` to `file`, one per line.
pub(crate) fn append<'a, T: Serialize + 'a>(
    file: &Kept,
    values: impl IntoIterator<Item = &'a T>,
) -> io::Result<()> {
    let lines = lines(values);
    if lines.is_empty() {
        return Ok(());
    }
    file.append(lines.as_bytes())
}

/// The last `kept` values of `file`, oldest first, skipping lines that do
/// not read; the file is rewritten with them when it grew past twice that.
pub(crate) fn tail<T: Serialize + DeserializeOwned>(file: &Kept, kept: usize) -> Vec<T> {
    let Ok(bytes) = file.read() else {
        return Vec::new();
    };
    let text = String::from_utf8_lossy(&bytes);
    let lines: Vec<&str> = text.lines().collect();
    let total = lines.len();
    let values: Vec<T> = lines
        .into_iter()
        .skip(total.saturating_sub(kept))
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect();
    if total > 2 * kept
        && let Err(failure) = file.replace(self::lines(&values).as_bytes())
    {
        warn!("could not trim {file}: {failure}");
    }
    values
}
