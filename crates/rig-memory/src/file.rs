//! [`FileConversationMemory`], a [`ConversationMemory`] backend that keeps each
//! conversation in a JSON Lines file.

use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex as StdMutex, PoisonError};
use std::time::SystemTime;

use rig_core::completion::Message;
use rig_core::id::ConversationId;
use rig_core::memory::{ConversationMemory, MemoryError};
use rig_core::wasm_compat::WasmBoxedFuture;
use sha2::{Digest, Sha256};
use tokio::sync::{Mutex as AsyncMutex, OwnedMutexGuard};

const HISTORY_EXTENSION: &str = "jsonl";
const ID_EXTENSION: &str = "id";
const TEMP_EXTENSION: &str = "tmp";
/// Longest encoded stem used as a file name. Longer ones are hashed, keeping
/// every derived name, temporary files included, under the common 255-byte
/// file name limit.
const MAX_ENCODED_STEM: usize = 200;
/// Starts every hashed stem. Encoded stems never contain it.
const HASHED_PREFIX: char = '~';
/// Device names Windows reserves regardless of extension.
const RESERVED_STEMS: &[&str] = &[
    "con", "prn", "aux", "nul", "com0", "com1", "com2", "com3", "com4", "com5", "com6", "com7",
    "com8", "com9", "lpt0", "lpt1", "lpt2", "lpt3", "lpt4", "lpt5", "lpt6", "lpt7", "lpt8", "lpt9",
];

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Errors from [`FileConversationMemory`], returned as the source of
/// [`MemoryError::Backend`].
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum FileMemoryError {
    /// A file system operation on `path` failed.
    #[error("{}: {source}", path.display())]
    Io {
        /// The file or directory being accessed.
        path: PathBuf,
        /// The underlying error.
        #[source]
        source: io::Error,
    },

    /// A complete line of a conversation file is not a valid message.
    #[error("{}: line {line} is not a valid message: {source}", path.display())]
    MalformedLine {
        /// The conversation file.
        path: PathBuf,
        /// The 1-based line number.
        line: usize,
        /// The parse error.
        #[source]
        source: serde_json::Error,
    },

    /// A hashed file name's id file names a different conversation.
    #[error("{}: belongs to a different conversation id", path.display())]
    IdMismatch {
        /// The id file.
        path: PathBuf,
    },
}

/// A conversation found by [`FileConversationMemory::list`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct StoredConversation {
    /// The conversation's id.
    pub id: ConversationId,
    /// When its file was last written.
    pub modified: SystemTime,
}

/// Stores each conversation as a JSON Lines file in one directory, one
/// serialized [`Message`] per line.
///
/// File names are derived from conversation ids. ASCII lowercase letters,
/// digits, `-` and `_` are kept and every other byte is written as `%` plus two
/// lowercase hex digits, so names cannot traverse paths, contain no uppercase
/// letters to fold on case-insensitive file systems, and decode back to the id.
/// An id that is empty, a reserved Windows device name, or too long once
/// encoded is stored under a SHA-256 name instead, with the id kept in a
/// sibling `.id` file.
///
/// [`append`](ConversationMemory::append) writes whole lines and syncs them to
/// disk before returning. If a crash leaves an incomplete last line,
/// [`load`](ConversationMemory::load) skips it with a warning and the next
/// append truncates it. A malformed line anywhere else is an error.
/// [`replace`](Self::replace) swaps the whole history atomically.
///
/// File IO runs on Tokio's blocking pool when called inside a Tokio runtime and
/// on the calling thread otherwise. Operations on one conversation through one
/// value, or its clones, run one at a time, so appends never interleave.
/// Nothing locks across separately constructed values or processes. They can
/// share a directory as long as each conversation has one writer at a time.
/// Loads are always safe and skip a last line that is still being written.
/// Two writers appending to, replacing, or clearing the same conversation at
/// once may lose messages or corrupt the file.
///
/// ```no_run
/// # async fn run() -> Result<(), rig_memory::MemoryError> {
/// use rig_core::completion::Message;
/// use rig_memory::{ConversationMemory, FileConversationMemory};
///
/// let memory = FileConversationMemory::new("conversations");
/// let id = "thread-1".into();
/// memory.append(&id, vec![Message::user("Hello!")]).await?;
/// assert_eq!(memory.load(&id).await?.len(), 1);
///
/// let most_recent = memory.list().await?.into_iter().next();
/// # Ok(()) }
/// ```
#[derive(Clone)]
pub struct FileConversationMemory {
    dir: Arc<Path>,
    locks: Arc<LockMap>,
}

type LockMap = StdMutex<HashMap<String, Arc<AsyncMutex<()>>>>;

impl std::fmt::Debug for FileConversationMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileConversationMemory")
            .field("dir", &self.dir)
            .finish()
    }
}

impl FileConversationMemory {
    /// A store in `dir`. The directory is created on the first write.
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self {
            dir: Arc::from(dir.into()),
            locks: Arc::default(),
        }
    }

    /// The directory holding the conversation files.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The file holding `conversation_id`'s history, whether or not it exists.
    pub fn path(&self, conversation_id: &ConversationId) -> PathBuf {
        Stem::of(conversation_id.as_str()).history_path(&self.dir)
    }

    /// Replace the whole history of `conversation_id` with `messages`, for
    /// example after compaction.
    ///
    /// The new history is written to a temporary file in the same directory,
    /// synced, and renamed over the old one, so readers see either the old or
    /// the new history. An empty `messages` leaves an empty conversation.
    pub async fn replace(
        &self,
        conversation_id: &ConversationId,
        messages: Vec<Message>,
    ) -> Result<(), MemoryError> {
        let stem = Stem::of(conversation_id.as_str());
        let lock = self.lock(&stem).await;
        let dir = Arc::clone(&self.dir);
        let id = conversation_id.as_str().to_owned();
        run_blocking(move || {
            let _lock = lock;
            let lines = encode_lines(&messages)?;
            create_dir(&dir)?;
            if stem.hashed {
                ensure_id_file(&dir, &stem, &id)?;
            }
            write_atomically(&dir, &stem.history_path(&dir), &lines)
        })
        .await
    }

    /// Every stored conversation, most recently written first.
    ///
    /// Returns an empty list when the directory does not exist. Files that do
    /// not belong to the store are ignored.
    pub async fn list(&self) -> Result<Vec<StoredConversation>, MemoryError> {
        let dir = Arc::clone(&self.dir);
        run_blocking(move || list_dir(&dir)).await
    }

    async fn lock(&self, stem: &Stem) -> ConversationLock {
        let mutex = Arc::clone(
            self.locks
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .entry(stem.name.clone())
                .or_default(),
        );
        ConversationLock {
            guard: Some(mutex.lock_owned().await),
            locks: Arc::clone(&self.locks),
            name: stem.name.clone(),
        }
    }
}

impl ConversationMemory for FileConversationMemory {
    fn load<'a>(
        &'a self,
        conversation_id: &'a ConversationId,
    ) -> WasmBoxedFuture<'a, Result<Vec<Message>, MemoryError>> {
        Box::pin(async move {
            let stem = Stem::of(conversation_id.as_str());
            let lock = self.lock(&stem).await;
            let dir = Arc::clone(&self.dir);
            let id = conversation_id.as_str().to_owned();
            run_blocking(move || {
                let _lock = lock;
                if stem.hashed {
                    check_id_file(&dir, &stem, &id)?;
                }
                let path = stem.history_path(&dir);
                match fs::read(&path) {
                    Ok(bytes) => parse_lines(&path, &bytes),
                    Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(Vec::new()),
                    Err(error) => Err(io_error(&path, error)),
                }
            })
            .await
        })
    }

    fn append<'a>(
        &'a self,
        conversation_id: &'a ConversationId,
        messages: Vec<Message>,
    ) -> WasmBoxedFuture<'a, Result<(), MemoryError>> {
        Box::pin(async move {
            if messages.is_empty() {
                return Ok(());
            }
            let stem = Stem::of(conversation_id.as_str());
            let lock = self.lock(&stem).await;
            let dir = Arc::clone(&self.dir);
            let id = conversation_id.as_str().to_owned();
            run_blocking(move || {
                let _lock = lock;
                let lines = encode_lines(&messages)?;
                create_dir(&dir)?;
                if stem.hashed {
                    ensure_id_file(&dir, &stem, &id)?;
                }
                append_lines(&dir, &stem.history_path(&dir), &lines)
            })
            .await
        })
    }

    fn clear<'a>(
        &'a self,
        conversation_id: &'a ConversationId,
    ) -> WasmBoxedFuture<'a, Result<(), MemoryError>> {
        Box::pin(async move {
            let stem = Stem::of(conversation_id.as_str());
            let lock = self.lock(&stem).await;
            let dir = Arc::clone(&self.dir);
            run_blocking(move || {
                let _lock = lock;
                let mut removed = remove_if_present(&stem.history_path(&dir))?;
                if stem.hashed {
                    removed |= remove_if_present(&stem.id_path(&dir))?;
                }
                if removed {
                    sync_dir(&dir)?;
                }
                Ok(())
            })
            .await
        })
    }
}

/// Holds one conversation's lock and drops its map entry when nobody else is
/// waiting for it.
struct ConversationLock {
    guard: Option<OwnedMutexGuard<()>>,
    locks: Arc<LockMap>,
    name: String,
}

impl Drop for ConversationLock {
    fn drop(&mut self) {
        let mut locks = self.locks.lock().unwrap_or_else(PoisonError::into_inner);
        drop(self.guard.take());
        if locks
            .get(&self.name)
            .is_some_and(|mutex| Arc::strong_count(mutex) == 1)
        {
            locks.remove(&self.name);
        }
    }
}

/// The file name stem a conversation id maps to.
struct Stem {
    name: String,
    hashed: bool,
}

impl Stem {
    fn of(id: &str) -> Self {
        let name = encode_id(id);
        if name.is_empty()
            || name.len() > MAX_ENCODED_STEM
            || RESERVED_STEMS.contains(&name.as_str())
        {
            let digest = Sha256::digest(id.as_bytes());
            let mut name = String::from(HASHED_PREFIX);
            for byte in digest {
                name.push_str(&format!("{byte:02x}"));
            }
            Self { name, hashed: true }
        } else {
            Self {
                name,
                hashed: false,
            }
        }
    }

    fn history_path(&self, dir: &Path) -> PathBuf {
        dir.join(format!("{}.{HISTORY_EXTENSION}", self.name))
    }

    fn id_path(&self, dir: &Path) -> PathBuf {
        dir.join(format!("{}.{ID_EXTENSION}", self.name))
    }
}

fn encode_id(id: &str) -> String {
    let mut name = String::with_capacity(id.len());
    for byte in id.bytes() {
        if byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-' || byte == b'_' {
            name.push(char::from(byte));
        } else {
            name.push_str(&format!("%{byte:02x}"));
        }
    }
    name
}

/// Recovers the id from an encoded stem. Returns `None` for names this store
/// would not have produced.
fn decode_id(name: &str) -> Option<String> {
    let mut bytes = Vec::with_capacity(name.len());
    let mut rest = name.as_bytes();
    while let Some((&first, tail)) = rest.split_first() {
        if first == b'%' {
            let hex = std::str::from_utf8(tail.get(..2)?).ok()?;
            bytes.push(u8::from_str_radix(hex, 16).ok()?);
            rest = tail.get(2..)?;
        } else {
            bytes.push(first);
            rest = tail;
        }
    }
    let id = String::from_utf8(bytes).ok()?;
    let stem = Stem::of(&id);
    (!stem.hashed && stem.name == name).then_some(id)
}

/// Runs `operation` on Tokio's blocking pool when a runtime is available.
async fn run_blocking<T, F>(operation: F) -> Result<T, MemoryError>
where
    F: FnOnce() -> Result<T, MemoryError> + Send + 'static,
    T: Send + 'static,
{
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => handle
            .spawn_blocking(operation)
            .await
            .map_err(|error| MemoryError::Internal(error.to_string()))?,
        Err(_) => operation(),
    }
}

fn io_error(path: &Path, source: io::Error) -> MemoryError {
    MemoryError::backend(FileMemoryError::Io {
        path: path.to_owned(),
        source,
    })
}

fn encode_lines(messages: &[Message]) -> Result<Vec<u8>, MemoryError> {
    let mut lines = Vec::new();
    for message in messages {
        serde_json::to_writer(&mut lines, message).map_err(MemoryError::backend)?;
        lines.push(b'\n');
    }
    Ok(lines)
}

fn parse_lines(path: &Path, bytes: &[u8]) -> Result<Vec<Message>, MemoryError> {
    let mut messages = Vec::new();
    let mut segments = bytes.split(|&byte| byte == b'\n').enumerate().peekable();
    while let Some((index, segment)) = segments.next() {
        let line = segment.trim_ascii();
        if line.is_empty() {
            continue;
        }
        if segments.peek().is_none() {
            // Every line this store writes ends in a newline, so a last line
            // without one is a write that never completed.
            tracing::warn!(
                path = %path.display(),
                line = index + 1,
                "skipping incomplete last line of conversation file"
            );
            break;
        }
        let message = serde_json::from_slice(line).map_err(|source| {
            MemoryError::backend(FileMemoryError::MalformedLine {
                path: path.to_owned(),
                line: index + 1,
                source,
            })
        })?;
        messages.push(message);
    }
    Ok(messages)
}

fn append_lines(dir: &Path, path: &Path, lines: &[u8]) -> Result<(), MemoryError> {
    let (mut file, created) = match OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(path)
    {
        Ok(file) => (file, true),
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .open(path)
                .map_err(|error| io_error(path, error))?;
            (file, false)
        }
        Err(error) => return Err(io_error(path, error)),
    };
    if !created {
        truncate_incomplete_line(&mut file, path).map_err(|error| io_error(path, error))?;
    }
    file.seek(SeekFrom::End(0))
        .and_then(|_| file.write_all(lines))
        .and_then(|()| file.sync_data())
        .map_err(|error| io_error(path, error))?;
    if created {
        sync_dir(dir)?;
    }
    Ok(())
}

/// Cuts a last line left without its newline, so new lines do not join it.
fn truncate_incomplete_line(file: &mut File, path: &Path) -> io::Result<()> {
    let length = file.metadata()?.len();
    let mut end = length;
    let mut chunk = [0u8; 4096];
    while end > 0 {
        let start = end.saturating_sub(chunk.len() as u64);
        let window = chunk.get_mut(..(end - start) as usize).unwrap_or_default();
        file.seek(SeekFrom::Start(start))?;
        file.read_exact(window)?;
        if let Some(newline) = window.iter().rposition(|&byte| byte == b'\n') {
            end = start + newline as u64 + 1;
            break;
        }
        end = start;
    }
    if end < length {
        tracing::warn!(
            path = %path.display(),
            bytes = length - end,
            "truncating incomplete last line of conversation file before appending"
        );
        file.set_len(end)?;
    }
    Ok(())
}

fn write_atomically(dir: &Path, path: &Path, contents: &[u8]) -> Result<(), MemoryError> {
    let name = path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    let temp = dir.join(format!(
        ".{name}.{}-{}.{TEMP_EXTENSION}",
        std::process::id(),
        TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    let written = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp)
        .and_then(|mut file| {
            file.write_all(contents)?;
            file.sync_all()
        })
        .and_then(|()| fs::rename(&temp, path));
    if let Err(error) = written {
        let _ = fs::remove_file(&temp);
        return Err(io_error(path, error));
    }
    sync_dir(dir)
}

fn ensure_id_file(dir: &Path, stem: &Stem, id: &str) -> Result<(), MemoryError> {
    let path = stem.id_path(dir);
    match fs::read(&path) {
        Ok(stored) if stored == id.as_bytes() => Ok(()),
        Ok(_) => Err(MemoryError::backend(FileMemoryError::IdMismatch { path })),
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            write_atomically(dir, &path, id.as_bytes())
        }
        Err(error) => Err(io_error(&path, error)),
    }
}

fn check_id_file(dir: &Path, stem: &Stem, id: &str) -> Result<(), MemoryError> {
    let path = stem.id_path(dir);
    match fs::read(&path) {
        Ok(stored) if stored != id.as_bytes() => {
            Err(MemoryError::backend(FileMemoryError::IdMismatch { path }))
        }
        Ok(_) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(io_error(&path, error)),
    }
}

fn list_dir(dir: &Path) -> Result<Vec<StoredConversation>, MemoryError> {
    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(io_error(dir, error)),
    };
    let mut found = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|error| io_error(dir, error))?;
        let file_name = entry.file_name();
        let Some(name) = file_name
            .to_str()
            .and_then(|name| name.strip_suffix(HISTORY_EXTENSION))
            .and_then(|name| name.strip_suffix('.'))
        else {
            continue;
        };
        let id = if name.starts_with(HASHED_PREFIX) {
            match fs::read(dir.join(format!("{name}.{ID_EXTENSION}"))) {
                Ok(id) => String::from_utf8(id)
                    .ok()
                    .filter(|id| Stem::of(id).name == name),
                Err(error) if error.kind() == io::ErrorKind::NotFound => None,
                Err(error) => return Err(io_error(&entry.path(), error)),
            }
        } else {
            decode_id(name)
        };
        let Some(id) = id else {
            continue;
        };
        let modified = match entry.metadata().and_then(|metadata| metadata.modified()) {
            Ok(modified) => modified,
            // Cleared since the directory was read.
            Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
            Err(error) => return Err(io_error(&entry.path(), error)),
        };
        found.push(StoredConversation {
            id: ConversationId::new(id),
            modified,
        });
    }
    found.sort_by(|a, b| {
        b.modified
            .cmp(&a.modified)
            .then_with(|| a.id.as_str().cmp(b.id.as_str()))
    });
    Ok(found)
}

/// Creates `dir` if needed and makes its entry in the parent durable.
fn create_dir(dir: &Path) -> Result<(), MemoryError> {
    if dir.is_dir() {
        return Ok(());
    }
    fs::create_dir_all(dir).map_err(|error| io_error(dir, error))?;
    match dir.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => sync_dir(parent),
        _ => sync_dir(Path::new(".")),
    }
}

fn remove_if_present(path: &Path) -> Result<bool, MemoryError> {
    match fs::remove_file(path) {
        Ok(()) => Ok(true),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(io_error(path, error)),
    }
}

/// Makes renames, creations, and removals in `dir` durable. Only Unix can sync
/// a directory through the standard library.
fn sync_dir(dir: &Path) -> Result<(), MemoryError> {
    #[cfg(unix)]
    File::open(dir)
        .and_then(|dir| dir.sync_all())
        .map_err(|error| io_error(dir, error))?;
    #[cfg(not(unix))]
    let _ = dir;
    Ok(())
}

#[cfg(test)]
mod tests;
