//! Where a session is kept: [`Files`] by name, and a [`Store`] that also
//! holds each log of the session and the blobs its messages name.

use std::collections::BTreeMap;
use std::io;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

/// A place that keeps files by name, such as a session's plugin data or
/// effect log. A name is relative and `/`-separated ([`check_name`]).
/// Every method takes `&self`, so one value is shared.
pub trait Files: Send + Sync + 'static {
    /// Replaces the file `name` with `bytes`, atomically: a reader sees the
    /// old content or the new.
    fn write_file(&self, name: &str, bytes: &[u8]) -> io::Result<()>;

    /// Adds `bytes` at the end of the file `name`, which it creates when
    /// there is none.
    fn append_file(&self, name: &str, bytes: &[u8]) -> io::Result<()>;

    /// The file `name`; [`io::ErrorKind::NotFound`] when there is none.
    fn read_file(&self, name: &str) -> io::Result<Vec<u8>>;

    /// Removes the file `name`; removing one that is not there succeeds.
    fn remove_file(&self, name: &str) -> io::Result<()>;

    /// The names of the files, in no particular order.
    fn files(&self) -> io::Result<Vec<String>>;
}

/// Where a session is kept: its [`Files`], a log per agent plus the
/// session's own, appended to and read whole, and the content-addressed
/// blobs its messages name. Log and blob names are one part of a name
/// ([`check_name`] without `/`).
pub trait Store: Files {
    /// The names of the logs, in no particular order.
    fn logs(&self) -> io::Result<Vec<String>>;

    /// The whole log `name`; [`io::ErrorKind::NotFound`] when there is none.
    fn read_log(&self, name: &str) -> io::Result<Vec<u8>>;

    /// Appends `bytes` to the log `name`, which it starts when there is none.
    fn append_log(&self, name: &str, bytes: &[u8]) -> io::Result<()>;

    /// Cuts the log `name` to its first `len` bytes, such as a torn last
    /// line a crash left.
    fn truncate_log(&self, name: &str, len: u64) -> io::Result<()>;

    /// Removes the log `name`; removing one that is not there succeeds.
    fn remove_log(&self, name: &str) -> io::Result<()>;

    /// Stores the blob `name` unless it is stored already: a name is a hash
    /// of the content.
    fn put_blob(&self, name: &str, bytes: &[u8]) -> io::Result<()>;

    /// The blob `name`; [`io::ErrorKind::NotFound`] when there is none.
    fn blob(&self, name: &str) -> io::Result<Vec<u8>>;

    /// The names of the blobs, in no particular order.
    fn blobs(&self) -> io::Result<Vec<String>>;

    /// Removes everything the store holds: logs, blobs and files.
    /// Idempotent: calling it again after a crash midway finishes the job.
    fn destroy(&self) -> io::Result<()>;
}

/// Refuses `name` unless it is relative and `/`-separated, with no empty,
/// `.` or `..` part and no `\`, so it names a place inside the store
/// wherever the store keeps it.
///
/// ```
/// use rig_memory::check_name;
///
/// assert!(check_name("plugins/notes.json").is_ok());
/// assert!(check_name("../elsewhere").is_err());
/// assert!(check_name("/etc/passwd").is_err());
/// ```
///
/// # Errors
///
/// [`io::ErrorKind::InvalidInput`], naming it.
pub fn check_name(name: &str) -> io::Result<()> {
    let bad = |part: &str| matches!(part, "" | "." | "..") || part.contains('\\');
    if name.split('/').any(bad) {
        let why = format!("{name:?} is no name in a session");
        return Err(io::Error::new(io::ErrorKind::InvalidInput, why));
    }
    Ok(())
}

/// Refuses `name` as a log or blob name: [`check_name`], in one part.
pub(crate) fn check_leaf(name: &str) -> io::Result<()> {
    check_name(name)?;
    if name.contains('/') {
        let why = format!("{name:?} is no log or blob name");
        return Err(io::Error::new(io::ErrorKind::InvalidInput, why));
    }
    Ok(())
}

/// A session kept in memory, such as for tests or where there is no file
/// system. Cloning shares the contents.
#[derive(Clone, Default)]
pub struct MemoryStore(Arc<Mutex<Memory>>);

#[derive(Default)]
struct Memory {
    logs: BTreeMap<String, Vec<u8>>,
    blobs: BTreeMap<String, Vec<u8>>,
    files: BTreeMap<String, Vec<u8>>,
}

impl MemoryStore {
    fn memory(&self) -> MutexGuard<'_, Memory> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

fn not_found() -> io::Error {
    io::ErrorKind::NotFound.into()
}

impl Files for MemoryStore {
    fn write_file(&self, name: &str, bytes: &[u8]) -> io::Result<()> {
        check_name(name)?;
        self.memory().files.insert(name.to_owned(), bytes.to_vec());
        Ok(())
    }

    fn append_file(&self, name: &str, bytes: &[u8]) -> io::Result<()> {
        check_name(name)?;
        let mut memory = self.memory();
        let file = memory.files.entry(name.to_owned()).or_default();
        file.extend_from_slice(bytes);
        Ok(())
    }

    fn read_file(&self, name: &str) -> io::Result<Vec<u8>> {
        check_name(name)?;
        self.memory().files.get(name).cloned().ok_or_else(not_found)
    }

    fn remove_file(&self, name: &str) -> io::Result<()> {
        check_name(name)?;
        self.memory().files.remove(name);
        Ok(())
    }

    fn files(&self) -> io::Result<Vec<String>> {
        Ok(self.memory().files.keys().cloned().collect())
    }
}

impl Store for MemoryStore {
    fn logs(&self) -> io::Result<Vec<String>> {
        Ok(self.memory().logs.keys().cloned().collect())
    }

    fn read_log(&self, name: &str) -> io::Result<Vec<u8>> {
        self.memory().logs.get(name).cloned().ok_or_else(not_found)
    }

    fn append_log(&self, name: &str, bytes: &[u8]) -> io::Result<()> {
        check_leaf(name)?;
        let mut memory = self.memory();
        let log = memory.logs.entry(name.to_owned()).or_default();
        log.extend_from_slice(bytes);
        Ok(())
    }

    fn truncate_log(&self, name: &str, len: u64) -> io::Result<()> {
        let len = usize::try_from(len).map_err(io::Error::other)?;
        if let Some(log) = self.memory().logs.get_mut(name) {
            log.truncate(len);
        }
        Ok(())
    }

    fn remove_log(&self, name: &str) -> io::Result<()> {
        self.memory().logs.remove(name);
        Ok(())
    }

    fn put_blob(&self, name: &str, bytes: &[u8]) -> io::Result<()> {
        check_leaf(name)?;
        let mut memory = self.memory();
        memory
            .blobs
            .entry(name.to_owned())
            .or_insert_with(|| bytes.to_vec());
        Ok(())
    }

    fn blob(&self, name: &str) -> io::Result<Vec<u8>> {
        self.memory().blobs.get(name).cloned().ok_or_else(not_found)
    }

    fn blobs(&self) -> io::Result<Vec<String>> {
        Ok(self.memory().blobs.keys().cloned().collect())
    }

    fn destroy(&self) -> io::Result<()> {
        *self.memory() = Memory::default();
        Ok(())
    }
}
