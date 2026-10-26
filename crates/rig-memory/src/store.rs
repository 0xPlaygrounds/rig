//! Where a session is kept: [`Files`] by name, and a [`Store`] that also
//! holds each log of the session. The blobs its messages name are its
//! files `blobs/<name>`, named by their content, which a directory store
//! keeps under `files/blobs/`. [`AsyncStore`] is the same storage reached
//! asynchronously, such as a database, an object store or IndexedDB; every
//! [`Store`] is one. A log write names the length the writer read, and the
//! store writes only if the log is still that long, checking and writing
//! in one step ([`LogWriteError::Changed`]).

use std::collections::BTreeMap;
use std::future::{Future, ready};
use std::io;
use std::pin::pin;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::task::{Context, Poll, Waker};

use rig_core::wasm_compat::WasmCompatSend;

use crate::SessionError;

/// Why a log write wrote nothing.
#[derive(Debug, thiserror::Error)]
pub enum LogWriteError {
    /// The log is not the `len` bytes the writer read: another writer
    /// changed it. Nothing was written.
    #[error("the log changed since it was read")]
    Changed,
    /// The store failed.
    #[error(transparent)]
    Io(#[from] io::Error),
}

impl From<LogWriteError> for io::Error {
    /// The store's own error back; [`LogWriteError::Changed`] as an
    /// [`io::ErrorKind::Other`] saying so.
    fn from(error: LogWriteError) -> Self {
        match error {
            LogWriteError::Changed => io::Error::other(LogWriteError::Changed.to_string()),
            LogWriteError::Io(error) => error,
        }
    }
}

/// A place that keeps files by name, such as a session's plugin data or
/// effect log. A name is relative and `/`-separated ([`check_name`]).
/// Every method takes `&self`, so one value is shared, between threads on
/// every target: a host keeps it as `Arc<dyn Files>` or `Arc<dyn Store>`
/// in state its systems share, such as a Bevy resource, which is `Send`
/// and `Sync` on wasm too; `Arc<dyn Store>` and `Box<dyn Store>` are
/// themselves a [`Store`], so an [`AsyncStore`] too. A store that is not
/// `Send` and `Sync`, such as one over a JavaScript promise, is an
/// [`AsyncStore`].
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

/// Where a session is kept: its [`Files`], with the content-addressed
/// blobs its messages name under `blobs/`, and a log per agent plus the
/// session's own, appended to and read whole. Log names are one part of a
/// name ([`check_name`] without `/`). The files under
/// [`migration::KEPT`](crate::migration::KEPT) are the copies a migration
/// kept of the session as it was.
pub trait Store: Files {
    /// The names of the logs, in no particular order.
    fn logs(&self) -> io::Result<Vec<String>>;

    /// The whole log `name`; [`io::ErrorKind::NotFound`] when there is none.
    fn read_log(&self, name: &str) -> io::Result<Vec<u8>>;

    /// Appends `bytes` to the log `name` if it is `len` bytes long now (0
    /// when there is none, which it then starts); else
    /// [`LogWriteError::Changed`], writing nothing. The check and the
    /// append are one step for every writer of the store, other processes
    /// included, so of two writers that read the log at the same length
    /// one appends and the other is refused. A log that shrank and grew
    /// back to exactly `len` bytes since it was read is not told apart;
    /// rig shrinks a log only by setting a torn last line aside, or by
    /// [`rewrite`](crate::rewrite) and [`Self::destroy`], which are for a
    /// stopped session.
    fn append_log(&self, name: &str, len: u64, bytes: &[u8]) -> Result<(), LogWriteError>;

    /// Replaces the log `name` with `bytes`, under the condition
    /// [`Self::append_log`] has, atomically and durably: a reader, and a
    /// crash or power loss, leave the old log or the new, never a part of
    /// either or an empty file; an append after it goes after the new
    /// bytes. Starts the log when there is none.
    fn replace_log(&self, name: &str, len: u64, bytes: &[u8]) -> Result<(), LogWriteError>;

    /// Removes everything the store holds: logs and files.
    /// Idempotent: calling it again after a crash midway finishes the job.
    fn destroy(&self) -> io::Result<()>;
}

/// A session's storage reached asynchronously: the part of [`Store`] that
/// [`load_async`](crate::load_async) and [`save_async`](crate::save_async)
/// use, for a store behind a network or a JavaScript promise. Every
/// [`Store`] is one, its calls done when made: a local store pays no thread
/// hop and no allocation for it. A type that implements [`Store`] gets this
/// from it and cannot have its own (the two impls would overlap): wrap it
/// in a type of your own instead.
///
/// Each method keeps [`Store`]'s contract. A name is checked by
/// [`check_name`] (a log name is one part) and may come from a log's
/// content, so a store refuses one that does not pass. The two log writes
/// check the log's length and write in one step, such as a conditional
/// `UPDATE … WHERE octet_length(bytes) = $len` or one IndexedDB
/// `readwrite` transaction, never a read followed by a write. An append
/// adds its bytes whole, or at worst leaves a cut last line, which the next
/// save sets aside: [`save_async`](crate::save_async) writes one turn with
/// one append, so a cancelled or failed save never stores part of one.
/// [`io::ErrorKind::NotFound`] means there is no such log or file.
///
/// Its futures are `Send` natively, and on browser wasm neither they nor
/// the store need be. A holder that shares the store, such as an agent,
/// asks for `WasmCompatSend + WasmCompatSync` itself. An async store that
/// is shared is `Clone` over an inner `Arc`: `Arc<S>` is an `AsyncStore`
/// only when `S` is a [`Store`].
pub trait AsyncStore {
    /// The whole log `name`.
    fn read_log_async(
        &self,
        name: &str,
    ) -> impl Future<Output = io::Result<Vec<u8>>> + WasmCompatSend;

    /// Appends `bytes` to the log `name` if it is `len` bytes long now
    /// ([`Store::append_log`]).
    fn append_log_async(
        &self,
        name: &str,
        len: u64,
        bytes: &[u8],
    ) -> impl Future<Output = Result<(), LogWriteError>> + WasmCompatSend;

    /// Replaces the log `name` with `bytes` if it is `len` bytes long now,
    /// atomically and durably ([`Store::replace_log`]).
    fn replace_log_async(
        &self,
        name: &str,
        len: u64,
        bytes: &[u8],
    ) -> impl Future<Output = Result<(), LogWriteError>> + WasmCompatSend;

    /// The file `name`.
    fn read_file_async(
        &self,
        name: &str,
    ) -> impl Future<Output = io::Result<Vec<u8>>> + WasmCompatSend;

    /// Replaces the file `name` with `bytes`, atomically.
    fn write_file_async(
        &self,
        name: &str,
        bytes: &[u8],
    ) -> impl Future<Output = io::Result<()>> + WasmCompatSend;
}

/// A local store's calls, done when made.
impl<S: Store + ?Sized> AsyncStore for S {
    fn read_log_async(
        &self,
        name: &str,
    ) -> impl Future<Output = io::Result<Vec<u8>>> + WasmCompatSend {
        ready(self.read_log(name))
    }

    fn append_log_async(
        &self,
        name: &str,
        len: u64,
        bytes: &[u8],
    ) -> impl Future<Output = Result<(), LogWriteError>> + WasmCompatSend {
        ready(self.append_log(name, len, bytes))
    }

    fn replace_log_async(
        &self,
        name: &str,
        len: u64,
        bytes: &[u8],
    ) -> impl Future<Output = Result<(), LogWriteError>> + WasmCompatSend {
        ready(self.replace_log(name, len, bytes))
    }

    fn read_file_async(
        &self,
        name: &str,
    ) -> impl Future<Output = io::Result<Vec<u8>>> + WasmCompatSend {
        ready(self.read_file(name))
    }

    fn write_file_async(
        &self,
        name: &str,
        bytes: &[u8],
    ) -> impl Future<Output = io::Result<()>> + WasmCompatSend {
        ready(self.write_file(name, bytes))
    }
}

/// The store a pointer points at: `Arc<dyn Store>` and `Box<dyn Store>`
/// are stores, so [`AsyncStore`]s too.
macro_rules! forward {
    ($($pointer:ident),*) => {$(
        impl<S: Files + ?Sized> Files for $pointer<S> {
            fn write_file(&self, name: &str, bytes: &[u8]) -> io::Result<()> {
                (**self).write_file(name, bytes)
            }

            fn append_file(&self, name: &str, bytes: &[u8]) -> io::Result<()> {
                (**self).append_file(name, bytes)
            }

            fn read_file(&self, name: &str) -> io::Result<Vec<u8>> {
                (**self).read_file(name)
            }

            fn remove_file(&self, name: &str) -> io::Result<()> {
                (**self).remove_file(name)
            }

            fn files(&self) -> io::Result<Vec<String>> {
                (**self).files()
            }
        }

        impl<S: Store + ?Sized> Store for $pointer<S> {
            fn logs(&self) -> io::Result<Vec<String>> {
                (**self).logs()
            }

            fn read_log(&self, name: &str) -> io::Result<Vec<u8>> {
                (**self).read_log(name)
            }

            fn append_log(&self, name: &str, len: u64, bytes: &[u8]) -> Result<(), LogWriteError> {
                (**self).append_log(name, len, bytes)
            }

            fn replace_log(&self, name: &str, len: u64, bytes: &[u8]) -> Result<(), LogWriteError> {
                (**self).replace_log(name, len, bytes)
            }

            fn destroy(&self) -> io::Result<()> {
                (**self).destroy()
            }
        }
    )*};
}

forward!(Arc, Box);

/// The output of `work`, polled once. A [`Store`]'s calls through
/// [`AsyncStore`] are done when made, so its saves finish on the first
/// poll: the synchronous API runs the one asynchronous implementation.
pub(crate) fn now<T>(
    work: impl Future<Output = Result<T, SessionError>>,
) -> Result<T, SessionError> {
    match pin!(work).poll(&mut Context::from_waker(Waker::noop())) {
        Poll::Ready(output) => output,
        Poll::Pending => Err(io::Error::other("a local store's call did not finish").into()),
    }
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

/// Refuses `name` as a log name: [`check_name`], in one part.
///
/// ```
/// use rig_memory::check_log_name;
///
/// assert!(check_log_name("main").is_ok());
/// assert!(check_log_name("agents/main").is_err());
/// ```
///
/// # Errors
///
/// [`io::ErrorKind::InvalidInput`], naming it.
pub fn check_log_name(name: &str) -> io::Result<()> {
    check_name(name)?;
    if name.contains('/') {
        let why = format!("{name:?} is no log name");
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
        check_log_name(name)?;
        self.memory().logs.get(name).cloned().ok_or_else(not_found)
    }

    fn append_log(&self, name: &str, len: u64, bytes: &[u8]) -> Result<(), LogWriteError> {
        check_log_name(name)?;
        let mut memory = self.memory();
        if memory.logs.get(name).map_or(0, Vec::len) as u64 != len {
            return Err(LogWriteError::Changed);
        }
        let log = memory.logs.entry(name.to_owned()).or_default();
        log.extend_from_slice(bytes);
        Ok(())
    }

    fn replace_log(&self, name: &str, len: u64, bytes: &[u8]) -> Result<(), LogWriteError> {
        check_log_name(name)?;
        let mut memory = self.memory();
        if memory.logs.get(name).map_or(0, Vec::len) as u64 != len {
            return Err(LogWriteError::Changed);
        }
        memory.logs.insert(name.to_owned(), bytes.to_vec());
        Ok(())
    }

    fn destroy(&self) -> io::Result<()> {
        *self.memory() = Memory::default();
        Ok(())
    }
}

/// Where plugins' files are in a session's store.
pub const PLUGINS: &str = "plugins/";

/// The files of a session's store under [`PLUGINS`], by their names there:
/// what a plugin keeps for the session.
pub struct PluginFiles(pub Arc<dyn Files>);

impl Files for PluginFiles {
    fn write_file(&self, name: &str, bytes: &[u8]) -> io::Result<()> {
        self.0.write_file(&format!("{PLUGINS}{name}"), bytes)
    }

    fn append_file(&self, name: &str, bytes: &[u8]) -> io::Result<()> {
        self.0.append_file(&format!("{PLUGINS}{name}"), bytes)
    }

    fn read_file(&self, name: &str) -> io::Result<Vec<u8>> {
        self.0.read_file(&format!("{PLUGINS}{name}"))
    }

    fn remove_file(&self, name: &str) -> io::Result<()> {
        self.0.remove_file(&format!("{PLUGINS}{name}"))
    }

    fn files(&self) -> io::Result<Vec<String>> {
        let files = self.0.files()?.into_iter();
        let within = files.filter_map(|name| Some(name.strip_prefix(PLUGINS)?.to_owned()));
        Ok(within.collect())
    }
}
