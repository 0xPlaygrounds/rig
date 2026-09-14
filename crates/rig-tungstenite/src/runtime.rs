//! Running websocket I/O when the caller has no tokio runtime.
//!
//! `tokio-tungstenite` needs a tokio reactor. Inside a tokio runtime this
//! backend drives the socket directly. Outside one — Bevy task pools, smol,
//! `futures::executor::block_on` — it moves the socket onto a lazily started,
//! single-worker fallback runtime and talks to it over `futures` channels, so
//! the caller only ever polls runtime-agnostic futures and no thread parks.
//!
//! This mirrors `rig-reqwest`'s own fallback runtime, for the same reason and
//! with the same shape; a websocket differs only in living longer, which is why
//! the socket is moved rather than each request driven individually.

use rig_core::http_client::Error;
use std::future::Future;
use std::sync::LazyLock;
use tokio::runtime::{Handle, Runtime};

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

/// Run `future` to completion on the fallback runtime, awaiting its result from
/// whatever executor the caller is on. Only call this when [`in_tokio`] is
/// false; inside a runtime, just `.await` the future.
pub(crate) async fn run_off_runtime<F>(future: F) -> Result<F::Output, Error>
where
    F: Future + Send + 'static,
    F::Output: Send + 'static,
{
    runtime()?.spawn(future).await.map_err(Error::instance)
}

/// Spawn a detached task on the fallback runtime — the connection actor, which
/// owns the socket for as long as the caller holds the connection.
pub(crate) fn spawn_off_runtime<F>(future: F) -> Result<(), Error>
where
    F: Future<Output = ()> + Send + 'static,
{
    runtime()?.spawn(future);
    Ok(())
}
