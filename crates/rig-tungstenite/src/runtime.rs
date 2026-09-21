//! Running websocket I/O when the caller has no tokio runtime.
//!
//! `tokio-tungstenite` needs a tokio reactor. Inside a tokio runtime this
//! backend drives the socket directly. Outside one — Bevy task pools, smol,
//! `futures::executor::block_on` — it moves the socket onto a lazily started,
//! single-worker fallback runtime and talks to it over `futures` channels, so
//! the caller only ever polls runtime-agnostic futures and no thread parks.
//!
//! `rig-reqwest` has a fallback runtime for the same reason, but not the same
//! shape: a unary request is *context-bound* there — the caller keeps owning
//! the future and only the reactor context is borrowed for each poll, so
//! Rig spawns no per-request forwarding task. A socket outlives any single call, so here it is
//! moved onto the runtime instead. Anything moved that way is still owned by
//! the caller's handle: every spawn below comes back as an [`OwnedTask`] whose
//! drop aborts it, so abandoning a connect or dropping a connection releases
//! the transport rather than leaving a task running with no way to reach it.

use rig_core::{http_client::Error, wasm_compat::WasmCompatSend};
use std::future::Future;
use std::sync::LazyLock;
use tokio::runtime::{Handle, Runtime};
use tokio::task::JoinHandle;

/// The fallback runtime, or the reason it could not start. A `LazyLock`
/// initializer cannot return an error, so the failure is stored and surfaced as
/// a transport error on every connection that needs the runtime.
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

/// Whether the current task already runs inside a tokio runtime.
///
/// A caveat this shares with `rig-reqwest`: a `current_thread` runtime built
/// without `enable_io()`/`enable_time()` answers `true` here, and tungstenite
/// then panics with "there is no reactor running". `Handle::try_current()`
/// cannot distinguish a runtime with the I/O driver from one without it, so a
/// host that builds its own runtime must enable I/O.
pub(crate) fn in_tokio() -> bool {
    Handle::try_current().is_ok()
}

/// A task on the fallback runtime, owned by whoever holds this handle: its
/// drop aborts the task.
///
/// Work moved off the caller's executor still belongs to the caller. Without
/// this, dropping the owner leaves a task the caller can no longer reach still
/// holding a socket, and only the process teardown ends it.
pub(crate) struct OwnedTask<T> {
    handle: JoinHandle<T>,
}

impl<T> OwnedTask<T> {
    /// Await the task's output. Dropping *this future* aborts the task, which
    /// is the point: an abandoned connect must not go on connecting.
    pub(crate) async fn join(mut self) -> Result<T, Error> {
        // Borrowed, not consumed, so the guard is still armed while parked
        // here and disarms only by running to completion.
        (&mut self.handle).await.map_err(Error::instance)
    }
}

impl<T> Drop for OwnedTask<T> {
    fn drop(&mut self) {
        // A no-op once the task has finished.
        self.handle.abort();
    }
}

/// Run `future` to completion on the fallback runtime, awaiting its result from
/// whatever executor the caller is on. Only call this when [`in_tokio`] is
/// false; inside a runtime, just `.await` the future.
pub(crate) async fn run_off_runtime<F>(future: F) -> Result<F::Output, Error>
where
    F: Future + WasmCompatSend + 'static,
    F::Output: WasmCompatSend + 'static,
{
    spawn_off_runtime(future)?.join().await
}

/// Move `future` onto the fallback runtime, handing back the owning handle.
pub(crate) fn spawn_off_runtime<F>(future: F) -> Result<OwnedTask<F::Output>, Error>
where
    F: Future + WasmCompatSend + 'static,
    F::Output: WasmCompatSend + 'static,
{
    Ok(OwnedTask {
        handle: runtime()?.spawn(future),
    })
}
