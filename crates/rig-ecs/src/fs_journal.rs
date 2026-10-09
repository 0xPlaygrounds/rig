//! A session kept as files in a directory: `<agent>.jsonl` per agent,
//! `blobs/<name>` for blobs, and `effects.jsonl` written by rig-cassette's
//! [`jsonl::Writer`], which [`jsonl::read`] reads back for its replayer.

use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, PoisonError};

use rig_cassette::effect_log::{EffectLog, jsonl};
use rig_core::effect::EffectId;

use super::store::JournalStore;

/// The extension of an agent log and of the effect log.
const EXTENSION: &str = "jsonl";

/// The effect log's file stem, which no agent id takes.
const EFFECTS: &str = "effects";

/// A [`JournalStore`] in the directory it was made for, created as files
/// are written. Agent logs stay open for appending once written.
pub struct JsonlDirStore {
    dir: PathBuf,
    open: Mutex<HashMap<String, File>>,
    effects: Mutex<jsonl::Writer>,
}

impl JsonlDirStore {
    /// The session in `dir`.
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        let dir = dir.into();
        let effects = jsonl::Writer::new(Self::effects_in(&dir));
        Self {
            dir,
            open: Mutex::new(HashMap::new()),
            effects: Mutex::new(effects),
        }
    }

    /// The log of `agent`.
    pub fn agent_log(&self, agent: &str) -> PathBuf {
        self.dir.join(format!("{agent}.{EXTENSION}"))
    }

    /// The effect log.
    pub fn effects(&self) -> PathBuf {
        Self::effects_in(&self.dir)
    }

    /// Where blobs are kept.
    pub fn blobs(&self) -> PathBuf {
        self.dir.join("blobs")
    }

    fn effects_in(dir: &Path) -> PathBuf {
        dir.join(format!("{EFFECTS}.{EXTENSION}"))
    }

    fn open(&self) -> MutexGuard<'_, HashMap<String, File>> {
        self.open.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn writer(&self) -> MutexGuard<'_, jsonl::Writer> {
        self.effects.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl JournalStore for JsonlDirStore {
    fn agents(&self) -> io::Result<Vec<String>> {
        let entries = match fs::read_dir(&self.dir) {
            Ok(entries) => entries,
            Err(failure) if failure.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(failure) => return Err(failure),
        };
        Ok(entries
            .filter_map(Result::ok)
            .filter_map(|entry| {
                let path = entry.path();
                if path.extension()? != EXTENSION {
                    return None;
                }
                let stem = path.file_stem()?.to_str()?;
                (!stem.is_empty() && stem != EFFECTS).then(|| stem.to_owned())
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

    fn append_effects(&self, log: &EffectLog) -> io::Result<()> {
        self.writer().append(log)
    }

    fn last_effect(&self) -> io::Result<Option<EffectId>> {
        self.writer().last_id()
    }
}
