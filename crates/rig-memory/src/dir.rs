//! A session kept as files in a directory: `logs/<name>.jsonl` and
//! `files/<name…>`, its blobs under `files/blobs/`. The store reads only
//! these two folders, so whatever else the directory holds is left alone.

use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, PoisonError};

use crate::store::check_leaf;
use crate::{Files, SessionError, Store, check_name};

const LOGS: &str = "logs";
const FILES: &str = "files";

/// The extension of a log.
const LOG: &str = ".jsonl";

/// What a file being written is called until it is complete; a name ending
/// in it is refused.
const PART: &str = ".rig-part";

/// What a folder is renamed to while [`Store::destroy`] removes it.
const TRASH: &str = ".rig-trash";

/// A [`Store`] in the directory it was opened on, whose folders are made
/// as files are written. Logs stay open for appending once written.
pub struct DirStore {
    dir: PathBuf,
    files: DirFiles,
    open: Mutex<HashMap<String, File>>,
}

/// [`Files`] kept as plain files under a directory, each name its path
/// there, such as data kept outside any session. Folders are made as files
/// are written, and nothing is written before.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DirFiles(PathBuf);

impl DirFiles {
    /// The files under `dir`, which need not exist yet.
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self(dir.into())
    }

    /// The directory itself.
    pub fn path(&self) -> &Path {
        &self.0
    }

    fn file(&self, name: &str) -> io::Result<PathBuf> {
        check_name(name)?;
        refuse_part(name)?;
        Ok(self.0.join(name))
    }
}

impl DirStore {
    /// The session in `dir`, which need not exist yet. Nothing is written.
    ///
    /// # Errors
    ///
    /// [`SessionError::OldLayout`] when `dir` holds a top-level `*.jsonl`
    /// file, as a session saved by an older rig does; it is left as it is.
    /// [`SessionError::Io`] when `dir` cannot be listed.
    pub fn open(dir: impl Into<PathBuf>) -> Result<Self, SessionError> {
        let dir = dir.into();
        let old = names(&dir)?
            .into_iter()
            .any(|name| name.ends_with(LOG) && dir.join(&name).is_file());
        if old {
            return Err(SessionError::OldLayout);
        }
        Ok(Self {
            files: DirFiles::new(dir.join(FILES)),
            dir,
            open: Mutex::new(HashMap::new()),
        })
    }

    fn open_logs(&self) -> MutexGuard<'_, HashMap<String, File>> {
        self.open.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn log(&self, name: &str) -> io::Result<PathBuf> {
        check_leaf(name)?;
        Ok(self.dir.join(LOGS).join(format!("{name}{LOG}")))
    }
}

fn refuse_part(name: &str) -> io::Result<()> {
    if name.ends_with(PART) {
        let why = format!("{name:?} ends as a file being written does");
        return Err(io::Error::new(io::ErrorKind::InvalidInput, why));
    }
    Ok(())
}

/// The names in the directory `dir`; none when it does not exist.
fn names(dir: &Path) -> io::Result<Vec<String>> {
    let entries = match fs::read_dir(dir) {
        Err(failure) if failure.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        entries => entries?,
    };
    let names = entries.filter_map(|entry| entry.ok()?.file_name().into_string().ok());
    Ok(names.collect())
}

/// Removing what is not there succeeds.
fn absent_is_fine(removed: io::Result<()>) -> io::Result<()> {
    match removed {
        Err(failure) if failure.kind() == io::ErrorKind::NotFound => Ok(()),
        removed => removed,
    }
}

/// Writes `bytes` to `path` atomically: to a part file beside it, then
/// renamed over it.
fn write_atomic(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let part = write_part(path, bytes, false)?;
    fs::rename(&part, path)
}

/// As [`write_atomic`], and durably: the part file is synced before the
/// rename and the folder after it (best effort), so a power loss leaves
/// the old file or the new, never an empty one.
fn write_durable(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let part = write_part(path, bytes, true)?;
    fs::rename(&part, path)?;
    if let Some(parent) = path.parent() {
        let _ = File::open(parent).and_then(|folder| folder.sync_all());
    }
    Ok(())
}

/// Writes `bytes` to the part file beside `path`, whose folder it makes,
/// synced with `sync`, and returns the part file's path.
fn write_part(path: &Path, bytes: &[u8], sync: bool) -> io::Result<PathBuf> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut part = path.as_os_str().to_owned();
    part.push(PART);
    let part = PathBuf::from(part);
    let mut file = File::create(&part)?;
    file.write_all(bytes)?;
    if sync {
        file.sync_all()?;
    }
    Ok(part)
}

impl Files for DirFiles {
    fn write_file(&self, name: &str, bytes: &[u8]) -> io::Result<()> {
        write_atomic(&self.file(name)?, bytes)
    }

    fn append_file(&self, name: &str, bytes: &[u8]) -> io::Result<()> {
        let path = self.file(name)?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut file = OpenOptions::new().create(true).append(true).open(path)?;
        file.write_all(bytes)
    }

    fn read_file(&self, name: &str) -> io::Result<Vec<u8>> {
        fs::read(self.file(name)?)
    }

    fn remove_file(&self, name: &str) -> io::Result<()> {
        absent_is_fine(fs::remove_file(self.file(name)?))
    }

    fn files(&self) -> io::Result<Vec<String>> {
        let mut files = Vec::new();
        let mut dirs = vec![(self.0.clone(), String::new())];
        while let Some((dir, prefix)) = dirs.pop() {
            for name in names(&dir)? {
                if name.ends_with(PART) {
                    continue;
                }
                let path = dir.join(&name);
                let name = format!("{prefix}{name}");
                if path.is_dir() {
                    dirs.push((path, format!("{name}/")));
                } else {
                    files.push(name);
                }
            }
        }
        Ok(files)
    }
}

impl Files for DirStore {
    fn write_file(&self, name: &str, bytes: &[u8]) -> io::Result<()> {
        self.files.write_file(name, bytes)
    }

    fn append_file(&self, name: &str, bytes: &[u8]) -> io::Result<()> {
        self.files.append_file(name, bytes)
    }

    fn read_file(&self, name: &str) -> io::Result<Vec<u8>> {
        self.files.read_file(name)
    }

    fn remove_file(&self, name: &str) -> io::Result<()> {
        self.files.remove_file(name)
    }

    fn files(&self) -> io::Result<Vec<String>> {
        self.files.files()
    }
}

impl Store for DirStore {
    fn logs(&self) -> io::Result<Vec<String>> {
        let names = names(&self.dir.join(LOGS))?.into_iter();
        let logs = names.filter_map(|name| Some(name.strip_suffix(LOG)?.to_owned()));
        Ok(logs.filter(|name| !name.is_empty()).collect())
    }

    fn read_log(&self, name: &str) -> io::Result<Vec<u8>> {
        fs::read(self.log(name)?)
    }

    fn append_log(&self, name: &str, bytes: &[u8]) -> io::Result<()> {
        let mut open = self.open_logs();
        if let Some(file) = open.get_mut(name) {
            return file.write_all(bytes);
        }
        let path = self.log(name)?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        open.entry(name.to_owned()).or_insert(file).write_all(bytes)
    }

    /// Under the lock appends take, the log's open handle is dropped (it
    /// would append to the replaced file) and the new log is written
    /// beside, synced and renamed over it: the one synced write, as it is
    /// rare and replaces history.
    fn replace_log(&self, name: &str, bytes: &[u8]) -> io::Result<()> {
        let path = self.log(name)?;
        let mut open = self.open_logs();
        open.remove(name);
        write_durable(&path, bytes)
    }

    /// Each folder is renamed to a trash name inside the directory, so it
    /// is gone at once, then removed; a second call removes what a crash
    /// left of the first.
    fn destroy(&self) -> io::Result<()> {
        self.open_logs().clear();
        for folder in [LOGS, FILES] {
            let (path, trash) = (
                self.dir.join(folder),
                self.dir.join(format!("{folder}{TRASH}")),
            );
            if path.exists() {
                absent_is_fine(fs::remove_dir_all(&trash))?;
                fs::rename(&path, &trash)?;
            }
            absent_is_fine(fs::remove_dir_all(&trash))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
