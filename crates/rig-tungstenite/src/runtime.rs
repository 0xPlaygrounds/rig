//! A lazy single-worker Tokio runtime for websocket I/O from other executors.
//!
//! Sockets remain on this runtime for their lifetime; callers communicate over
//! channels without needing a reactor. Each [`OwnedTask`] aborts on drop so
//! cancelled connections release their transport resources.

use rig_core::{http_client::Error, wasm_compat::WasmCompatSend};
use std::future::Future;
use std::sync::LazyLock;
use tokio::runtime::{Handle, Runtime};
use tokio::task::JoinHandle;

/// The shared fallback runtime, caching initialization failure for subsequent
/// connections.
static RUNTIME: LazyLock<Result<Runtime, RuntimeUnavailable>> = LazyLock::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .thread_name("rig-tungstenite")
        .enable_all()
        .build()
        .map_err(|err| RuntimeUnavailable(err.to_string()))
});

/// The fallback tokio runtime could not be started.
#[derive(Debug, Clone, thiserror::Error)]
#[error("rig-tungstenite: failed to start the fallback tokio runtime: {0}")]
struct RuntimeUnavailable(String);

fn runtime() -> Result<&'static Runtime, Error> {
    RUNTIME.as_ref().map_err(|err| Error::instance(err.clone()))
}

/// Return whether the current task has a Tokio runtime handle.
///
/// Hosts must enable I/O and timers on their runtime; this check cannot detect
/// missing drivers, which can cause socket I/O to panic.
pub(crate) fn in_tokio() -> bool {
    Handle::try_current().is_ok()
}

/// A fallback-runtime task whose handle aborts it on drop.
pub(crate) struct OwnedTask<T> {
    handle: JoinHandle<T>,
}

impl<T> OwnedTask<T> {
    /// Await the task's output, returning a join error if it fails.
    /// Dropping the returned future aborts the task.
    pub(crate) async fn join(mut self) -> Result<T, Error> {
        // Borrow the handle so cancellation still drops the abort guard.
        (&mut self.handle).await.map_err(Error::instance)
    }
}

impl<T> Drop for OwnedTask<T> {
    fn drop(&mut self) {
        self.handle.abort();
    }
}

/// Run `future` to completion on the fallback runtime, awaiting its result from
/// whatever executor the caller is on. Only call this when [`in_tokio`] is
/// false. Returns an error if startup or the spawned task fails.
pub(crate) async fn run_off_runtime<F>(future: F) -> Result<F::Output, Error>
where
    F: Future + WasmCompatSend + 'static,
    F::Output: WasmCompatSend + 'static,
{
    spawn_off_runtime(future)?.join().await
}

/// Spawn `future` on the fallback runtime and return its owning handle, or an
/// error if the runtime cannot start.
pub(crate) fn spawn_off_runtime<F>(future: F) -> Result<OwnedTask<F::Output>, Error>
where
    F: Future + WasmCompatSend + 'static,
    F::Output: WasmCompatSend + 'static,
{
    Ok(OwnedTask {
        handle: runtime()?.spawn(future),
    })
}
