//! The tokio runtime the bundled transports poll their I/O in.
//!
//! [`bind`] and [`bind_stream`] enter a runtime handle on every poll, so a
//! caller on another executor (Bevy task pools, smol, `futures::executor`)
//! owns and polls the future or stream itself: no spawned task, no channel.
//! The handle is the current runtime's, else a process-wide fallback runtime
//! started on first use (one worker thread, I/O and timers enabled).
//!
//! ```no_run
//! let handle = rig_http::runtime::handle()?;
//! assert_eq!(handle.block_on(async { 6 * 7 }), 42);
//! # Ok::<(), rig_http::http_client::Error>(())
//! ```

use crate::http_client::Error;
use futures::{Stream, stream};
use std::future::{Future, poll_fn};
use std::pin::pin;
use std::sync::LazyLock;
use tokio::runtime::{Handle, Runtime};

static RUNTIME: LazyLock<Result<Runtime, RuntimeUnavailable>> = LazyLock::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .thread_name("rig-http")
        .enable_all()
        .build()
        .map_err(|err| RuntimeUnavailable(err.to_string()))
});

#[derive(Debug, Clone, thiserror::Error)]
#[error("rig-http: failed to start the fallback tokio runtime: {0}")]
struct RuntimeUnavailable(String);

/// The current runtime when called inside one, else the fallback runtime,
/// which runs for the rest of the process. A host with no tokio runtime of
/// its own hands this to tokio-based libraries instead of starting another.
/// A supplied current runtime must have I/O and timers enabled and stay
/// driven until its operations finish.
///
/// # Errors
///
/// When there is no current runtime and the fallback could not start.
pub fn handle() -> Result<Handle, Error> {
    match Handle::try_current() {
        Ok(handle) => Ok(handle),
        Err(_) => RUNTIME
            .as_ref()
            .map(|runtime| runtime.handle().clone())
            .map_err(|error| Error::permanent(error.clone())),
    }
}

/// `future`, polled with [`handle`] entered.
///
/// # Errors
///
/// As [`handle`].
pub fn bind<F: Future>(future: F) -> Result<impl Future<Output = F::Output>, Error> {
    let handle = handle()?;
    Ok(async move {
        let mut future = pin!(future);
        poll_fn(|cx| {
            let _entered = handle.enter();
            future.as_mut().poll(cx)
        })
        .await
    })
}

/// `stream`, polled with [`handle`] entered.
///
/// # Errors
///
/// As [`handle`].
pub fn bind_stream<S: Stream>(stream: S) -> Result<impl Stream<Item = S::Item>, Error> {
    let handle = handle()?;
    let mut stream = Box::pin(stream);
    Ok(stream::poll_fn(move |cx| {
        let _entered = handle.enter();
        stream.as_mut().poll_next(cx)
    }))
}

#[cfg(test)]
mod tests;
