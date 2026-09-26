//! Poll reqwest in its selected reactor context without spawning an operation.
//!
//! The caller owns the future/stream throughout its lifetime. Entering a handle
//! for each poll lets non-Tokio executors use the fallback reactor without a
//! detached request or a forwarding task. Bodies capture the same context as
//! the request, even if the host moves them to a different executor.

use crate::RuntimePolicy;
use futures::{Stream, stream};
use rig_http::http_client::Error;
use std::{
    future::{Future, poll_fn},
    pin::pin,
    sync::LazyLock,
};
use tokio::runtime::{Handle, Runtime};

static RUNTIME: LazyLock<Result<Runtime, RuntimeUnavailable>> = LazyLock::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .thread_name("rig-reqwest")
        .enable_all()
        .build()
        .map_err(|err| RuntimeUnavailable(err.to_string()))
});

#[derive(Debug, Clone, thiserror::Error)]
#[error("rig-reqwest: failed to start the fallback tokio runtime: {0}")]
struct RuntimeUnavailable(String);

/// A supplied current runtime must have enabled I/O and timers and remain
/// driven until its operations finish. A Handle does not keep a Runtime alive
/// or prove its drivers were enabled; that remains the host's responsibility.
fn context(policy: RuntimePolicy) -> Result<Handle, Error> {
    if matches!(policy, RuntimePolicy::Caller)
        && let Ok(handle) = Handle::try_current()
    {
        return Ok(handle);
    }
    RUNTIME
        .as_ref()
        .map(|runtime| runtime.handle().clone())
        .map_err(|error| Error::instance(error.clone()))
}

pub(crate) fn bind<F: Future>(future: F) -> Result<impl Future<Output = F::Output>, Error> {
    bind_on(future, RuntimePolicy::Caller)
}

pub(crate) fn bind_on<F: Future>(
    future: F,
    policy: RuntimePolicy,
) -> Result<impl Future<Output = F::Output>, Error> {
    let handle = context(policy)?;
    Ok(async move {
        let mut future = pin!(future);
        poll_fn(|cx| {
            let _entered = handle.enter();
            future.as_mut().poll(cx)
        })
        .await
    })
}

pub(crate) fn bind_stream<S: Stream>(stream: S) -> Result<impl Stream<Item = S::Item>, Error> {
    let handle = context(RuntimePolicy::Caller)?;
    let mut stream = Box::pin(stream);
    Ok(stream::poll_fn(move |cx| {
        let _entered = handle.enter();
        stream.as_mut().poll_next(cx)
    }))
}
