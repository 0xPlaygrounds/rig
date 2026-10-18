//! Where a session is kept: [`Files`] by name, and a [`Store`] that also
//! holds each log of the session. The blobs its messages name are files
//! under `blobs/`, named by their content. [`AsyncStore`] is the same
//! storage reached asynchronously, such as a database, an object store or
//! IndexedDB; every [`Store`] is one.

use std::collections::BTreeMap;
use std::future::{Future, ready};
use std::io;
use std::pin::pin;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::task::{Context, Poll, Waker};

use rig_core::wasm_compat::WasmCompatSend;

use crate::SessionError;

/// A place that keeps files by name, such as a session's plugin data or
/// effect log. A name is relative and `/`-separated ([`check_name`]).
/// Every method takes `&self`, so one value is shared, between threads on
/// every target: a host keeps it as `Arc<dyn Files>` or `Arc<dyn Store>`
/// in state its systems share, such as a Bevy resource, which is `Send`
/// and `Sync` on wasm too. A store that is not, such as one over a
/// JavaScript promise, is an [`AsyncStore`].
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
/// name ([`check_name`] without `/`).
pub trait Store: Files {
    /// The names of the logs, in no particular order.
    fn logs(&self) -> io::Result<Vec<String>>;

    /// The whole log `name`; [`io::ErrorKind::NotFound`] when there is none.
    fn read_log(&self, name: &str) -> io::Result<Vec<u8>>;

    /// Appends `bytes` to the log `name`, which it starts when there is none.
    fn append_log(&self, name: &str, bytes: &[u8]) -> io::Result<()>;

    /// Replaces the log `name` with `bytes`, atomically and durably: a
    /// reader, and a crash or power loss, leave the old log or the new,
    /// never a part of either or an empty file; an append after it goes
    /// after the new bytes. Starts the log when there is none.
    fn replace_log(&self, name: &str, bytes: &[u8]) -> io::Result<()>;

    /// Removes everything the store holds: logs and files.
    /// Idempotent: calling it again after a crash midway finishes the job.
    fn destroy(&self) -> io::Result<()>;
}

/// A session's storage reached asynchronously: the part of [`Store`] that
/// [`load_async`](crate::load_async) and [`save_async`](crate::save_async)
/// use, for a store behind a network or a JavaScript promise. Every
/// [`Store`] is one, its calls done when made: a local store pays no thread
/// hop and no allocation for it.
///
/// Each method keeps [`Store`]'s contract. A name is checked by
/// [`check_name`] (a log name is one part) and may come from a log's
/// content, so a store refuses one that does not pass. An append adds its
/// bytes whole, or at worst leaves a cut last line, which the next save
/// sets aside: [`save_async`](crate::save_async) writes one turn with one
/// append, so a cancelled or failed save never stores part of one.
/// [`io::ErrorKind::NotFound`] means there is no such log or file.
///
/// Its futures are `Send` natively, and on browser wasm neither they nor
/// the store need be. A holder that shares the store, such as an agent,
/// asks for `WasmCompatSend + WasmCompatSync` itself.
pub trait AsyncStore {
    /// The names of the logs, in no particular order.
    fn logs(&self) -> impl Future<Output = io::Result<Vec<String>>> + WasmCompatSend;

    /// The whole log `name`.
    fn read_log(&self, name: &str) -> impl Future<Output = io::Result<Vec<u8>>> + WasmCompatSend;

    /// Appends `bytes` to the log `name`, which it starts when there is none.
    fn append_log(
        &self,
        name: &str,
        bytes: &[u8],
    ) -> impl Future<Output = io::Result<()>> + WasmCompatSend;

    /// Replaces the log `name` with `bytes`, atomically and durably
    /// ([`Store::replace_log`]).
    fn replace_log(
        &self,
        name: &str,
        bytes: &[u8],
    ) -> impl Future<Output = io::Result<()>> + WasmCompatSend;

    /// The names of the files, in no particular order.
    fn files(&self) -> impl Future<Output = io::Result<Vec<String>>> + WasmCompatSend;

    /// The file `name`.
    fn read_file(&self, name: &str) -> impl Future<Output = io::Result<Vec<u8>>> + WasmCompatSend;

    /// Replaces the file `name` with `bytes`, atomically.
    fn write_file(
        &self,
        name: &str,
        bytes: &[u8],
    ) -> impl Future<Output = io::Result<()>> + WasmCompatSend;
}

/// A local store's calls, done when made.
impl<S: Store + ?Sized> AsyncStore for S {
    fn logs(&self) -> impl Future<Output = io::Result<Vec<String>>> + WasmCompatSend {
        ready(Store::logs(self))
    }

    fn read_log(&self, name: &str) -> impl Future<Output = io::Result<Vec<u8>>> + WasmCompatSend {
        ready(Store::read_log(self, name))
    }

    fn append_log(
        &self,
        name: &str,
        bytes: &[u8],
    ) -> impl Future<Output = io::Result<()>> + WasmCompatSend {
        ready(Store::append_log(self, name, bytes))
    }

    fn replace_log(
        &self,
        name: &str,
        bytes: &[u8],
    ) -> impl Future<Output = io::Result<()>> + WasmCompatSend {
        ready(Store::replace_log(self, name, bytes))
    }

    fn files(&self) -> impl Future<Output = io::Result<Vec<String>>> + WasmCompatSend {
        ready(Files::files(self))
    }

    fn read_file(&self, name: &str) -> impl Future<Output = io::Result<Vec<u8>>> + WasmCompatSend {
        ready(Files::read_file(self, name))
    }

    fn write_file(
        &self,
        name: &str,
        bytes: &[u8],
    ) -> impl Future<Output = io::Result<()>> + WasmCompatSend {
        ready(Files::write_file(self, name, bytes))
    }
}

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
pub(crate) fn check_leaf(name: &str) -> io::Result<()> {
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
        self.memory().logs.get(name).cloned().ok_or_else(not_found)
    }

    fn append_log(&self, name: &str, bytes: &[u8]) -> io::Result<()> {
        check_leaf(name)?;
        let mut memory = self.memory();
        let log = memory.logs.entry(name.to_owned()).or_default();
        log.extend_from_slice(bytes);
        Ok(())
    }

    fn replace_log(&self, name: &str, bytes: &[u8]) -> io::Result<()> {
        check_leaf(name)?;
        self.memory().logs.insert(name.to_owned(), bytes.to_vec());
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
