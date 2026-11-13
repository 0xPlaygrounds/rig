//! Where sign-ins keep their credentials: the [`TokenStore`] trait, an
//! in-memory store, and the owner-only file store native targets default to.

use crate::error::BoxError;
use crate::wasm_compat::{WasmBoxedFuture, WasmCompatSend, WasmCompatSync};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};

/// Keeps the credentials a sign-in writes and refreshes, each a JSON text
/// under a key the provider names, such as `chatgpt` or `copilot`.
///
/// An authenticator reads its record before every request that needs one,
/// and saves it after a sign-in, a refresh, or an invalidated token. It never
/// touches storage except through its store. Clones of one authenticator
/// share a refresh lock, and a store shared with other processes holds the
/// record against them while a refresh or sign-in reads and saves it
/// ([`lock`](Self::lock)), so one refresh runs at a time and each request
/// reads what the last one saved. When a refresh token turns out to be spent, the
/// authenticator reads the store again and takes a fresh credential another
/// client or process saved there, before asking for a new sign-in.
///
/// ```
/// use rig_core::providers::chatgpt::auth::{AuthSource, Authenticator, DeviceCodeHandler, MemoryTokenStore};
/// use std::sync::Arc;
///
/// let store = Arc::new(MemoryTokenStore::default());
/// let auth = Authenticator::new(AuthSource::OAuth, store, DeviceCodeHandler::default(), true);
/// # let _ = auth;
/// ```
pub trait TokenStore: WasmCompatSend + WasmCompatSync {
    /// The record kept under `key`, or `None` when there is none. An error
    /// means the store could not answer, and fails the request.
    fn load<'a>(&'a self, key: &'a str) -> WasmBoxedFuture<'a, Result<Option<String>, BoxError>>;

    /// Keep `record` under `key`, replacing any record there.
    fn save<'a>(
        &'a self,
        key: &'a str,
        record: &'a str,
    ) -> WasmBoxedFuture<'a, Result<(), BoxError>>;

    /// Forget the record under `key`; forgetting a missing record succeeds.
    fn delete<'a>(&'a self, key: &'a str) -> WasmBoxedFuture<'a, Result<(), BoxError>>;

    /// Hold the record under `key` until the returned [`StoreLock`] is
    /// dropped, against every other holder of it, in this process or
    /// another (such as a refresh another rig runs at the same moment). The
    /// default holds nothing: a store no other process reads needs no
    /// lock beyond the authenticator's own.
    fn lock<'a>(&'a self, _key: &'a str) -> WasmBoxedFuture<'a, Result<StoreLock, BoxError>> {
        Box::pin(async { Ok(StoreLock::new(())) })
    }
}

/// A record held by [`TokenStore::lock`], let go when dropped: it owns the
/// store's guard and drops it then.
pub struct StoreLock {
    // Nothing reads the guard; the mutex only makes the lock `Sync` and
    // unwind safe whatever the guard is.
    _guard: Mutex<Box<dyn WasmCompatSend>>,
}

impl StoreLock {
    /// Holds `guard`, any value whose drop releases the store's lock, such
    /// as a locked file, an open database transaction or a lease on a
    /// distributed lock, until this lock is dropped. A store with nothing
    /// to release passes `()`.
    ///
    /// The guard's `Drop` runs synchronously, possibly inside an async task,
    /// so it must not block. A release that needs I/O should spawn it from
    /// `Drop` or let the lock's lease expire.
    pub fn new(guard: impl WasmCompatSend + 'static) -> Self {
        Self {
            _guard: Mutex::new(Box::new(guard)),
        }
    }
}

impl std::fmt::Debug for StoreLock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StoreLock").finish_non_exhaustive()
    }
}

impl<T: TokenStore + ?Sized> TokenStore for Arc<T> {
    fn load<'a>(&'a self, key: &'a str) -> WasmBoxedFuture<'a, Result<Option<String>, BoxError>> {
        (**self).load(key)
    }

    fn save<'a>(
        &'a self,
        key: &'a str,
        record: &'a str,
    ) -> WasmBoxedFuture<'a, Result<(), BoxError>> {
        (**self).save(key, record)
    }

    fn delete<'a>(&'a self, key: &'a str) -> WasmBoxedFuture<'a, Result<(), BoxError>> {
        (**self).delete(key)
    }

    fn lock<'a>(&'a self, key: &'a str) -> WasmBoxedFuture<'a, Result<StoreLock, BoxError>> {
        (**self).lock(key)
    }
}

/// A [`TokenStore`] in this process's memory, gone when it is dropped.
#[derive(Default)]
pub struct MemoryTokenStore(Mutex<HashMap<String, String>>);

impl std::fmt::Debug for MemoryTokenStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryTokenStore").finish_non_exhaustive()
    }
}

impl MemoryTokenStore {
    fn records(&self) -> std::sync::MutexGuard<'_, HashMap<String, String>> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl TokenStore for MemoryTokenStore {
    fn load<'a>(&'a self, key: &'a str) -> WasmBoxedFuture<'a, Result<Option<String>, BoxError>> {
        Box::pin(async move { Ok(self.records().get(key).cloned()) })
    }

    fn save<'a>(
        &'a self,
        key: &'a str,
        record: &'a str,
    ) -> WasmBoxedFuture<'a, Result<(), BoxError>> {
        Box::pin(async move {
            self.records().insert(key.to_owned(), record.to_owned());
            Ok(())
        })
    }

    fn delete<'a>(&'a self, key: &'a str) -> WasmBoxedFuture<'a, Result<(), BoxError>> {
        Box::pin(async move {
            self.records().remove(key);
            Ok(())
        })
    }
}

/// A [`TokenStore`] keeping each record in `<dir>/<key>.json`, written whole
/// or not at all (a temporary file renamed over it, and on Unix the
/// directory synced after), and held across processes with a file lock on
/// `<dir>/<key>.lock`. On Unix the files, and the directories it creates for
/// them, are readable by the owner alone. Native only.
///
/// [`load`](TokenStore::load) refuses a record that is a symlink or not a
/// regular file, or is over 64 KiB, with a [`FileStoreError`]. On Unix, when
/// group or others have any access to a record, it narrows the record to
/// owner-only before reading it, and refuses the record when that fails.
#[cfg(not(target_family = "wasm"))]
#[derive(Debug, Clone)]
pub struct FileTokenStore {
    dir: std::path::PathBuf,
    lock_wait: std::time::Duration,
}

#[cfg(not(target_family = "wasm"))]
impl FileTokenStore {
    /// A store over the files in `dir`, created when first written.
    pub fn new(dir: impl Into<std::path::PathBuf>) -> Self {
        Self {
            dir: dir.into(),
            lock_wait: LOCK_WAIT,
        }
    }

    /// This store, with [`lock`](TokenStore::lock) waiting at most `wait`
    /// for another holder instead of two minutes. A zero wait refuses a held
    /// lock at once.
    pub fn with_lock_wait(self, wait: std::time::Duration) -> Self {
        Self {
            lock_wait: wait,
            ..self
        }
    }

    /// The file the record under `key` is kept in.
    pub fn path(&self, key: &str) -> std::path::PathBuf {
        self.dir.join(format!("{key}.json"))
    }
}

#[cfg(not(target_family = "wasm"))]
impl TokenStore for FileTokenStore {
    fn load<'a>(&'a self, key: &'a str) -> WasmBoxedFuture<'a, Result<Option<String>, BoxError>> {
        Box::pin(async move { read_private(&self.path(key)) })
    }

    fn save<'a>(
        &'a self,
        key: &'a str,
        record: &'a str,
    ) -> WasmBoxedFuture<'a, Result<(), BoxError>> {
        Box::pin(async move { Ok(write_private(&self.path(key), record.as_bytes())?) })
    }

    fn delete<'a>(&'a self, key: &'a str) -> WasmBoxedFuture<'a, Result<(), BoxError>> {
        Box::pin(async move {
            let path = self.path(key);
            match std::fs::remove_file(&path) {
                Ok(()) => Ok(sync_parent(&path)?),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
                Err(error) => Err(error.into()),
            }
        })
    }

    /// Takes the file lock on `<dir>/<key>.lock`, waiting while another
    /// holder has it, for at most two minutes unless
    /// [`with_lock_wait`](FileTokenStore::with_lock_wait) set another wait,
    /// then refusing with [`FileStoreError::Busy`]. A process that dies lets
    /// go of it.
    fn lock<'a>(&'a self, key: &'a str) -> WasmBoxedFuture<'a, Result<StoreLock, BoxError>> {
        Box::pin(async move {
            let path = self.dir.join(format!("{key}.lock"));
            create_private_dir(&path)?;
            let mut options = std::fs::OpenOptions::new();
            options.read(true).write(true).create(true).truncate(false);
            #[cfg(unix)]
            std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
            let file = options.open(&path)?;
            let started = std::time::Instant::now();
            loop {
                match file.try_lock() {
                    Ok(()) => return Ok(StoreLock::new(file)),
                    Err(std::fs::TryLockError::WouldBlock)
                        if started.elapsed() < self.lock_wait =>
                    {
                        crate::wasm_compat::sleep(std::time::Duration::from_millis(50)).await;
                    }
                    Err(std::fs::TryLockError::WouldBlock) => {
                        return Err(FileStoreError::Busy {
                            path,
                            waited: self.lock_wait,
                        }
                        .into());
                    }
                    Err(std::fs::TryLockError::Error(error)) => return Err(error.into()),
                }
            }
        })
    }
}

/// Why a [`FileTokenStore`] refused to read a credential file or take its
/// lock. It reaches the caller inside the store's boxed error.
#[cfg(not(target_family = "wasm"))]
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum FileStoreError {
    /// The file is a symlink, a directory, or another kind of file than a
    /// regular one, or was replaced by one while it was opened.
    #[error("{} is a symlink or not a regular file; a credential is read only from a regular file", path.display())]
    NotAFile {
        /// The credential file.
        path: std::path::PathBuf,
    },
    /// The file holds more than `limit` bytes.
    #[error("{} is larger than {limit} bytes", path.display())]
    TooLarge {
        /// The credential file.
        path: std::path::PathBuf,
        /// The most a credential file may hold.
        limit: u64,
    },
    /// Group or others have access to the file and it could not be made
    /// owner-only, for example because another user owns it.
    #[error("{} is open to group or others and could not be made owner-only", path.display())]
    NotNarrowed {
        /// The credential file.
        path: std::path::PathBuf,
        /// Why the permissions could not be changed.
        #[source]
        source: std::io::Error,
    },
    /// Another holder, in this process or another, still held the lock when
    /// the wait ran out. Trying again later may succeed.
    #[error("another holder kept {} locked for {waited:?}", path.display())]
    Busy {
        /// The lock file.
        path: std::path::PathBuf,
        /// How long the lock was waited for.
        waited: std::time::Duration,
    },
}

/// The most a credential file may hold.
#[cfg(not(target_family = "wasm"))]
const RECORD_LIMIT: u64 = 64 * 1024;

/// Reads the credential file at `path`, `None` when there is none. The
/// checks look at the file that was opened, so it cannot be swapped for a
/// symlink between the check and the read: on Unix its identity is compared
/// with the unfollowed path's, and on Windows the open does not follow a
/// link.
#[cfg(not(target_family = "wasm"))]
fn read_private(path: &std::path::Path) -> Result<Option<String>, BoxError> {
    use std::io::Read;

    let not_a_file = || FileStoreError::NotAFile {
        path: path.to_owned(),
    };
    let linked = match std::fs::symlink_metadata(path) {
        Ok(linked) => linked,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    if !linked.file_type().is_file() {
        return Err(not_a_file().into());
    }
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    // FILE_FLAG_OPEN_REPARSE_POINT: open a symlink itself, which the handle
    // check below then refuses.
    #[cfg(windows)]
    std::os::windows::fs::OpenOptionsExt::custom_flags(&mut options, 0x0020_0000);
    let file = options.open(path)?;
    let opened = file.metadata()?;
    if !opened.is_file() {
        return Err(not_a_file().into());
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::{MetadataExt, PermissionsExt};
        if (opened.dev(), opened.ino()) != (linked.dev(), linked.ino()) {
            return Err(not_a_file().into());
        }
        if opened.mode() & 0o077 != 0 {
            file.set_permissions(std::fs::Permissions::from_mode(0o600))
                .map_err(|source| FileStoreError::NotNarrowed {
                    path: path.to_owned(),
                    source,
                })?;
        }
    }
    let mut record = Vec::new();
    file.take(RECORD_LIMIT + 1).read_to_end(&mut record)?;
    if record.len() as u64 > RECORD_LIMIT {
        return Err(FileStoreError::TooLarge {
            path: path.to_owned(),
            limit: RECORD_LIMIT,
        }
        .into());
    }
    Ok(Some(String::from_utf8(record)?))
}

/// How long [`FileTokenStore::lock`] waits for another holder by default:
/// far longer than a refresh's one request takes.
#[cfg(not(target_family = "wasm"))]
const LOCK_WAIT: std::time::Duration = std::time::Duration::from_secs(120);

/// Creates the directories above `path`, 0700 on Unix.
#[cfg(not(target_family = "wasm"))]
fn create_private_dir(path: &std::path::Path) -> Result<(), std::io::Error> {
    let Some(parent) = path.parent() else {
        return Ok(());
    };
    let mut builder = std::fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    std::os::unix::fs::DirBuilderExt::mode(&mut builder, 0o700);
    builder.create(parent)
}

/// Writes a credential file whole or not at all, creating the directories
/// above it: the bytes go to a fresh temporary file beside it, which is
/// synced and renamed over it, and on Unix the directory is synced so the
/// rename survives a crash, so a crash or a reader at the same moment never
/// sees half a credential. On Unix the directories it creates are
/// 0700, and the file is created 0600, or narrowed to 0600 before anything
/// is written to it.
#[cfg(not(target_family = "wasm"))]
fn write_private(path: &std::path::Path, bytes: &[u8]) -> Result<(), std::io::Error> {
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};

    static WRITES: AtomicU64 = AtomicU64::new(0);
    create_private_dir(path)?;
    let mut temp = path.as_os_str().to_owned();
    temp.push(format!(
        ".tmp-{}-{}",
        std::process::id(),
        WRITES.fetch_add(1, Ordering::Relaxed)
    ));
    let temp = std::path::PathBuf::from(temp);
    let write = || {
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
        let mut file = options.open(&temp)?;
        #[cfg(unix)]
        file.set_permissions(std::os::unix::fs::PermissionsExt::from_mode(0o600))?;
        file.write_all(bytes)?;
        file.sync_all()?;
        std::fs::rename(&temp, path)
    };
    let written = write();
    if written.is_err() {
        std::fs::remove_file(&temp).ok();
    }
    written?;
    sync_parent(path)
}

/// Syncs the directory holding `path`, so a rename or removal in it
/// survives a crash. Windows cannot open a directory to sync it, so there
/// this does nothing.
#[cfg(not(target_family = "wasm"))]
fn sync_parent(path: &std::path::Path) -> Result<(), std::io::Error> {
    #[cfg(unix)]
    if let Some(parent) = path.parent() {
        std::fs::File::open(parent)?.sync_all()?;
    }
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

/// The JSON record under `key`, or `T::default()` when there is none.
pub(crate) async fn load_record<T: Default + serde::de::DeserializeOwned>(
    store: &dyn TokenStore,
    key: &str,
) -> Result<T, super::AuthError> {
    match store.load(key).await.map_err(super::AuthError::Store)? {
        Some(record) => Ok(serde_json::from_str(&record)?),
        None => Ok(T::default()),
    }
}

/// Keep `record` under `key` as JSON.
pub(crate) async fn save_record<T: serde::Serialize>(
    store: &dyn TokenStore,
    key: &str,
    record: &T,
) -> Result<(), super::AuthError> {
    let text = serde_json::to_string_pretty(record)?;
    store
        .save(key, &text)
        .await
        .map_err(super::AuthError::Store)
}

#[cfg(test)]
mod tests;
