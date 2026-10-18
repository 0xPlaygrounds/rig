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
    _guard: Box<dyn WasmCompatSend>,
}

impl StoreLock {
    /// Holds `guard`, any value whose drop releases the store's lock, such
    /// as a locked file, an open database transaction or a lease on a
    /// distributed lock, until this lock is dropped. A store with nothing
    /// to release passes `()`.
    pub fn new(guard: impl WasmCompatSend + 'static) -> Self {
        Self {
            _guard: Box::new(guard),
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
/// or not at all (a temporary file renamed over it), and held across
/// processes with a file lock on `<dir>/<key>.lock`. On Unix the files, and
/// the directories it creates for them, are readable by the owner alone.
/// Native only.
#[cfg(not(target_family = "wasm"))]
#[derive(Debug, Clone)]
pub struct FileTokenStore {
    dir: std::path::PathBuf,
}

#[cfg(not(target_family = "wasm"))]
impl FileTokenStore {
    /// A store over the files in `dir`, created when first written.
    pub fn new(dir: impl Into<std::path::PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    /// The file the record under `key` is kept in.
    pub fn path(&self, key: &str) -> std::path::PathBuf {
        self.dir.join(format!("{key}.json"))
    }
}

#[cfg(not(target_family = "wasm"))]
impl TokenStore for FileTokenStore {
    fn load<'a>(&'a self, key: &'a str) -> WasmBoxedFuture<'a, Result<Option<String>, BoxError>> {
        Box::pin(async move {
            match std::fs::read_to_string(self.path(key)) {
                Ok(record) => Ok(Some(record)),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
                Err(error) => Err(error.into()),
            }
        })
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
            match std::fs::remove_file(self.path(key)) {
                Err(error) if error.kind() != std::io::ErrorKind::NotFound => Err(error.into()),
                _ => Ok(()),
            }
        })
    }

    /// Takes the file lock on `<dir>/<key>.lock`, waiting while another
    /// holder has it, for at most two minutes. A process that dies lets go
    /// of it.
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
                    Err(std::fs::TryLockError::WouldBlock) if started.elapsed() < LOCK_WAIT => {
                        crate::wasm_compat::sleep(std::time::Duration::from_millis(50)).await;
                    }
                    Err(std::fs::TryLockError::WouldBlock) => {
                        return Err(format!(
                            "another process has held {} for {} seconds",
                            path.display(),
                            LOCK_WAIT.as_secs()
                        )
                        .into());
                    }
                    Err(std::fs::TryLockError::Error(error)) => return Err(error.into()),
                }
            }
        })
    }
}

/// How long [`FileTokenStore::lock`] waits for another holder: far longer
/// than a refresh's one request takes.
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
/// synced and renamed over it, so a crash or a reader at the same moment
/// never sees half a credential. On Unix the directories it creates are
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
    written
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
