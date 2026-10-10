//! A session kept as files in a directory: `<agent>.jsonl` per agent and
//! `blobs/<name>` for blobs, beside the effect log ([`EFFECT_LOG`]).

use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard, PoisonError};

use super::JournalStore;

/// The extension of an agent log.
const EXTENSION: &str = "jsonl";

/// The file of a session directory that holds its effect log, as
/// [`jsonl::Writer`](crate::effect_log::jsonl::Writer) writes it, and that
/// is no agent's log.
pub const EFFECT_LOG: &str = "effects.jsonl";

/// A [`JournalStore`] in the directory it was made for, created as files
/// are written. Agent logs stay open for appending once written.
pub struct JsonlDirStore {
    dir: PathBuf,
    open: Mutex<HashMap<String, File>>,
}

impl JsonlDirStore {
    /// The session in `dir`.
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self {
            dir: dir.into(),
            open: Mutex::new(HashMap::new()),
        }
    }

    fn agent_log(&self, agent: &str) -> PathBuf {
        self.dir.join(format!("{agent}.{EXTENSION}"))
    }

    fn blobs(&self) -> PathBuf {
        self.dir.join("blobs")
    }

    fn open(&self) -> MutexGuard<'_, HashMap<String, File>> {
        self.open.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl JournalStore for JsonlDirStore {
    fn agents(&self) -> io::Result<Vec<String>> {
        let entries = match fs::read_dir(&self.dir) {
            Err(failure) if failure.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
            entries => entries?,
        };
        Ok(entries
            .filter_map(Result::ok)
            .filter_map(|entry| {
                let path = entry.path();
                let stem = path.file_stem()?.to_str()?;
                let log = path.extension()? == EXTENSION && !stem.is_empty();
                let log = log && path.file_name()? != EFFECT_LOG;
                log.then(|| stem.to_owned())
            })
            .collect())
    }

    fn read(&self, agent: &str) -> io::Result<Vec<u8>> {
        fs::read(self.agent_log(agent))
    }

    fn truncate(&self, agent: &str, len: u64) -> io::Result<()> {
        self.open().remove(agent);
        OpenOptions::new()
            .write(true)
            .open(self.agent_log(agent))?
            .set_len(len)
    }

    fn append(&self, agent: &str, bytes: &[u8]) -> io::Result<()> {
        let mut open = self.open();
        let file = match open.remove(agent) {
            Some(file) => file,
            None => OpenOptions::new()
                .create(true)
                .append(true)
                .open(self.agent_log(agent))?,
        };
        open.entry(agent.to_owned())
            .or_insert(file)
            .write_all(bytes)
    }

    fn put_blob(&self, name: &str, bytes: &[u8]) -> io::Result<()> {
        let blobs = self.blobs();
        let path = blobs.join(name);
        if path.exists() {
            return Ok(());
        }
        fs::create_dir_all(&blobs)?;
        let temporary = path.with_extension("tmp");
        fs::write(&temporary, bytes)?;
        fs::rename(&temporary, &path)
    }

    fn blob(&self, name: &str) -> io::Result<Vec<u8>> {
        fs::read(self.blobs().join(name))
    }
}
