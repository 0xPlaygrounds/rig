//! A session kept as files in a directory: `<agent>.jsonl` per agent,
//! `blobs/<name>` for blobs, and every other file at `<name>`, such as the
//! effect log ([`EFFECT_LOG`]). Every other top-level `*.jsonl` file is
//! read as an agent's log, so the store refuses such a name for a file,
//! as it does the plain files the directory keeps outside the store
//! ([`JsonlDirStore::with_plain`]).

use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, PoisonError};

use super::{JournalStore, check_file_name};

/// The extension of an agent log.
const EXTENSION: &str = "jsonl";

/// The directory of a session that holds its blobs.
const BLOBS: &str = "blobs";

/// The file of a session directory that holds its effect log, as
/// [`jsonl::Writer`](crate::effect_log::jsonl::Writer) writes it, and that
/// is no agent's log.
pub const EFFECT_LOG: &str = "effects.jsonl";

/// A [`JournalStore`] in the directory it was made for, created as files
/// are written. Agent logs stay open for appending once written.
pub struct JsonlDirStore {
    dir: PathBuf,
    plain: &'static [&'static str],
    open: Mutex<HashMap<String, File>>,
}

impl JsonlDirStore {
    /// The session in `dir`.
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self {
            dir: dir.into(),
            plain: &[],
            open: Mutex::new(HashMap::new()),
        }
    }

    /// The store, in a directory that also keeps the top-level files or
    /// directories `names` outside it, such as a rig session directory's
    /// markers (`rig::harness_protocol::SessionDir::PLAIN_FILES`): it
    /// neither lists nor takes them as its files.
    pub fn with_plain(mut self, names: &'static [&'static str]) -> Self {
        self.plain = names;
        self
    }

    fn agent_log(&self, agent: &str) -> PathBuf {
        self.dir.join(format!("{agent}.{EXTENSION}"))
    }

    fn blob_dir(&self) -> PathBuf {
        self.dir.join(BLOBS)
    }

    /// Where the file `name` is kept: refused when [`check_file_name`]
    /// refuses it, or when it would be read as an agent log or a blob, or
    /// is a plain file of the directory.
    fn file(&self, name: &str) -> io::Result<PathBuf> {
        check_file_name(name)?;
        let top = name.split('/').next().unwrap_or_default();
        let log = !name.contains('/')
            && name != EFFECT_LOG
            && Path::new(name)
                .extension()
                .is_some_and(|ext| ext == EXTENSION);
        if log || top == BLOBS || self.plain.contains(&top) {
            let why = format!(
                "{name:?} is where a session directory keeps its logs, blobs or plain files"
            );
            return Err(io::Error::new(io::ErrorKind::InvalidInput, why));
        }
        Ok(self.dir.join(name))
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
        let blobs = self.blob_dir();
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
        fs::read(self.blob_dir().join(name))
    }

    fn blobs(&self) -> io::Result<Vec<String>> {
        let entries = match fs::read_dir(self.blob_dir()) {
            Err(failure) if failure.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
            entries => entries?,
        };
        Ok(entries
            .filter_map(Result::ok)
            .filter_map(|entry| entry.file_name().into_string().ok())
            // A blob being written has a temporary name.
            .filter(|name| !name.ends_with(".tmp"))
            .collect())
    }

    fn remove(&self, agent: &str) -> io::Result<()> {
        self.open().remove(agent);
        match fs::remove_file(self.agent_log(agent)) {
            Err(failure) if failure.kind() == io::ErrorKind::NotFound => Ok(()),
            removed => removed,
        }
    }

    fn write_file(&self, name: &str, bytes: &[u8]) -> io::Result<()> {
        let path = self.file(name)?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut temporary = path.clone().into_os_string();
        temporary.push(".tmp");
        fs::write(&temporary, bytes)?;
        fs::rename(&temporary, &path)
    }

    fn append_file(&self, name: &str, bytes: &[u8]) -> io::Result<()> {
        let path = self.file(name)?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?
            .write_all(bytes)
    }

    fn read_file(&self, name: &str) -> io::Result<Vec<u8>> {
        fs::read(self.file(name)?)
    }

    fn remove_file(&self, name: &str) -> io::Result<()> {
        match fs::remove_file(self.file(name)?) {
            Err(failure) if failure.kind() == io::ErrorKind::NotFound => Ok(()),
            removed => removed,
        }
    }

    fn files(&self) -> io::Result<Vec<String>> {
        let mut files = Vec::new();
        let mut dirs = vec![(self.dir.clone(), String::new())];
        while let Some((dir, prefix)) = dirs.pop() {
            let entries = match fs::read_dir(&dir) {
                Err(failure) if failure.kind() == io::ErrorKind::NotFound => continue,
                entries => entries?,
            };
            for entry in entries {
                let entry = entry?;
                let Ok(name) = entry.file_name().into_string() else {
                    continue;
                };
                let name = format!("{prefix}{name}");
                // A file being written has a temporary name; the layout's
                // names are no files.
                if name.ends_with(".tmp") || self.file(&name).is_err() {
                    continue;
                }
                if entry.file_type()?.is_dir() {
                    dirs.push((entry.path(), format!("{name}/")));
                } else {
                    files.push(name);
                }
            }
        }
        Ok(files)
    }
}
