//! Running reqwest futures when the caller has no tokio runtime.
//!
//! Async `reqwest` needs a tokio reactor on native targets. Inside a tokio
//! runtime the transport awaits reqwest futures directly. Outside one — Bevy
//! task pools, smol, `futures::executor::block_on` — it spawns them onto a
//! lazily started, single-worker fallback runtime and awaits the resulting
//! [`JoinHandle`](tokio::task::JoinHandle) from the caller's executor; a
//! `JoinHandle` is a plain runtime-agnostic future, so no `block_on` or
//! thread parking is involved. Everything that touches the reqwest response —
//! reading a body, polling a byte stream — must likewise run on the tokio
//! side, which is why the off-runtime paths read bodies eagerly or forward
//! streams through a channel instead of handing reqwest futures back.

use rig_core::http_client::Error;
use std::future::Future;
use std::sync::LazyLock;
use tokio::runtime::{Handle, Runtime};

/// The fallback runtime, or the reason it could not start. A `LazyLock`
/// initializer cannot return an error, so the failure is stored and surfaced
/// as a transport error on every request that needs the runtime.
static RUNTIME: LazyLock<Result<Runtime, RuntimeUnavailable>> = LazyLock::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .thread_name("rig-reqwest")
        .enable_all()
        .build()
        .map_err(|err| RuntimeUnavailable(err.to_string()))
});

/// The fallback tokio runtime could not be started.
///
/// Private: this module is private, so a `pub` here was unnameable by callers
/// anyway. The failure reaches them as the `Error::Instance` this is boxed
/// into, whose `Display` carries the message. `rig-tungstenite` keeps the same
/// type private for the same reason.
#[derive(Debug, Clone, thiserror::Error)]
#[error("rig-reqwest: failed to start the fallback tokio runtime: {0}")]
struct RuntimeUnavailable(String);

fn runtime() -> Result<&'static Runtime, Error> {
    RUNTIME.as_ref().map_err(|err| Error::instance(err.clone()))
}

/// Whether the current task already runs inside a tokio runtime.
///
/// A caveat: a `current_thread` runtime built without
/// `enable_io()`/`enable_time()` answers `true` here, and reqwest then panics
/// with "there is no reactor running". `Handle::try_current()` cannot
/// distinguish a runtime with the I/O driver from one without it, so a host
/// that builds its own runtime must enable I/O. `rig-tungstenite`'s backend
/// shares the caveat and documents it in the same place.
pub(crate) fn in_tokio() -> bool {
    Handle::try_current().is_ok()
}

/// Run `future` to completion on the fallback runtime, awaiting its result
/// from whatever executor the caller is on. Only call this when
/// [`in_tokio`] is false; inside a runtime, just `.await` the future.
pub(crate) async fn run_off_runtime<F>(future: F) -> Result<F::Output, Error>
where
    F: Future + Send + 'static,
    F::Output: Send + 'static,
{
    runtime()?.spawn(future).await.map_err(Error::instance)
}

/// Spawn a detached task on the fallback runtime (used to drive a body
/// stream into a channel while the caller polls the receiver elsewhere).
pub(crate) fn spawn_off_runtime<F>(future: F) -> Result<(), Error>
where
    F: Future<Output = ()> + Send + 'static,
{
    runtime()?.spawn(future);
    Ok(())
}
