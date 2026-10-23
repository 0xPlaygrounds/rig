//! The contract every [`AsyncStore`] keeps, checked against a store: what
//! a missing log or file reads as, the length check of the two log writes,
//! names refused, and racing writers, of which exactly one wins. A store's
//! own tests run [`check_store`] on two handles to one empty storage, such
//! as two connections to one database, so it also checks that what one
//! handle writes the other reads.
//!
//! ```
//! use std::future::Future;
//! use std::pin::pin;
//! use std::task::{Context, Poll, Waker};
//!
//! use rig_memory::MemoryStore;
//! use rig_memory::conformance::check_store;
//!
//! let store = MemoryStore::default();
//! let other = store.clone();
//! let check = pin!(check_store(&store, &other));
//! // A local store's calls are done when made, so one poll finishes it.
//! let Poll::Ready(checked) = check.poll(&mut Context::from_waker(Waker::noop())) else {
//!     return Err("pending".into());
//! };
//! checked?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use std::future::{Future, poll_fn};
use std::io;
use std::pin::Pin;
use std::task::Poll;

use rig_core::message::Message;

use crate::{AsyncStore, LogWriteError, SessionError, load_async, save_async};

/// A part of the contract a store broke.
#[derive(Debug, thiserror::Error)]
#[error("{check}: {detail}")]
pub struct Violation {
    /// The part of the contract.
    pub check: &'static str,
    /// What the store did instead.
    pub detail: String,
}

/// How many writers race for one log.
const RACERS: usize = 8;

/// Checks that `a` and `b`, two handles to one storage that holds nothing
/// yet, keep [`AsyncStore`]'s contract together.
///
/// # Errors
///
/// The first [`Violation`] found.
pub async fn check_store<S: AsyncStore + ?Sized>(a: &S, b: &S) -> Result<(), Violation> {
    missing(a).await?;
    appends(a, b).await?;
    replaces(a, b).await?;
    files(a, b).await?;
    names(a).await?;
    races(a, b).await?;
    sessions(a, b).await
}

fn ensure(ok: bool, check: &'static str, detail: impl FnOnce() -> String) -> Result<(), Violation> {
    if ok {
        return Ok(());
    }
    Err(Violation {
        check,
        detail: detail(),
    })
}

fn is_kind<T>(result: &io::Result<T>, kind: io::ErrorKind) -> bool {
    matches!(result, Err(error) if error.kind() == kind)
}

fn is_changed(result: &Result<(), LogWriteError>) -> bool {
    matches!(result, Err(LogWriteError::Changed))
}

/// `read`, which must be `want`.
fn reads(check: &'static str, read: io::Result<Vec<u8>>, want: &[u8]) -> Result<(), Violation> {
    match read {
        Ok(bytes) => ensure(bytes == want, check, || {
            format!("read {bytes:?}, not {want:?}")
        }),
        Err(error) => ensure(false, check, || format!("read failed: {error}")),
    }
}

/// `write`, which must succeed.
fn wrote<E: std::fmt::Display>(check: &'static str, write: Result<(), E>) -> Result<(), Violation> {
    match write {
        Ok(()) => Ok(()),
        Err(error) => ensure(false, check, || format!("write failed: {error}")),
    }
}

/// A log or file that was never written is not found, and a log write
/// that names a length other than 0 neither writes nor starts it.
async fn missing<S: AsyncStore + ?Sized>(a: &S) -> Result<(), Violation> {
    let check = "a missing log or file is not found";
    let read = a.read_log_async("missing").await;
    ensure(is_kind(&read, io::ErrorKind::NotFound), check, || {
        format!("read_log_async gave {read:?}")
    })?;
    let read = a.read_file_async("missing/file").await;
    ensure(is_kind(&read, io::ErrorKind::NotFound), check, || {
        format!("read_file_async gave {read:?}")
    })?;
    let check = "a log write to a missing log at a length other than 0 is refused";
    let append = a.append_log_async("missing", 3, b"abc").await;
    ensure(is_changed(&append), check, || {
        format!("append gave {append:?}")
    })?;
    let replace = a.replace_log_async("missing", 1, b"abc").await;
    ensure(is_changed(&replace), check, || {
        format!("replace gave {replace:?}")
    })?;
    let read = a.read_log_async("missing").await;
    ensure(is_kind(&read, io::ErrorKind::NotFound), check, || {
        format!("the refused writes left {read:?}")
    })
}

/// An append at the log's length adds its bytes, binary ones included;
/// one at another length writes nothing; an empty append at 0 starts an
/// empty log.
async fn appends<S: AsyncStore + ?Sized>(a: &S, b: &S) -> Result<(), Violation> {
    let check = "an append at the log's length adds its bytes";
    let first = b"one\n\0\xff\n";
    wrote(check, a.append_log_async("appends", 0, first).await)?;
    reads(check, b.read_log_async("appends").await, first)?;
    let second = b"two\n";
    let len = first.len() as u64;
    wrote(check, b.append_log_async("appends", len, second).await)?;
    reads(
        check,
        a.read_log_async("appends").await,
        b"one\n\0\xff\ntwo\n",
    )?;

    let check = "an append at another length is refused and writes nothing";
    for stale in [0, len, len + 100, i64::MAX as u64, u64::MAX] {
        let append = a.append_log_async("appends", stale, b"lost\n").await;
        ensure(is_changed(&append), check, || {
            format!("an append at {stale} gave {append:?}")
        })?;
    }
    reads(
        check,
        b.read_log_async("appends").await,
        b"one\n\0\xff\ntwo\n",
    )?;

    let check = "an empty append at 0 starts an empty log";
    wrote(check, a.append_log_async("empty", 0, b"").await)?;
    reads(check, b.read_log_async("empty").await, b"")?;
    wrote(check, b.append_log_async("empty", 0, b"after\n").await)?;
    wrote(check, a.append_log_async("empty", 6, b"").await)?;
    reads(check, a.read_log_async("empty").await, b"after\n")
}

/// A replace at the log's length replaces it, and starts a missing one at
/// 0; at another length it writes nothing; an append after it goes after
/// the new bytes.
async fn replaces<S: AsyncStore + ?Sized>(a: &S, b: &S) -> Result<(), Violation> {
    let check = "a replace at the log's length replaces it";
    wrote(
        check,
        a.append_log_async("replaces", 0, b"old\nlog\n").await,
    )?;
    wrote(check, b.replace_log_async("replaces", 8, b"new\n").await)?;
    reads(check, a.read_log_async("replaces").await, b"new\n")?;
    wrote(check, a.append_log_async("replaces", 4, b"more\n").await)?;
    reads(check, b.read_log_async("replaces").await, b"new\nmore\n")?;
    wrote(check, b.replace_log_async("replaces", 9, b"").await)?;
    reads(check, a.read_log_async("replaces").await, b"")?;

    let check = "a replace at another length is refused and writes nothing";
    let replace = a.replace_log_async("replaces", 9, b"lost\n").await;
    ensure(is_changed(&replace), check, || {
        format!("replace gave {replace:?}")
    })?;
    reads(check, b.read_log_async("replaces").await, b"")?;

    let check = "a replace at 0 starts a missing log";
    wrote(check, a.replace_log_async("started", 0, b"fresh\n").await)?;
    reads(check, b.read_log_async("started").await, b"fresh\n")
}

/// A file write replaces the whole file, under a nested name too.
async fn files<S: AsyncStore + ?Sized>(a: &S, b: &S) -> Result<(), Violation> {
    let check = "a file write replaces the whole file";
    let name = "blobs/nested/file";
    wrote(check, a.write_file_async(name, b"long content\0\xff").await)?;
    reads(check, b.read_file_async(name).await, b"long content\0\xff")?;
    wrote(check, b.write_file_async(name, b"short").await)?;
    reads(check, a.read_file_async(name).await, b"short")?;
    wrote(check, a.write_file_async("top", b"").await)?;
    reads(check, b.read_file_async("top").await, b"")?;
    let check = "a file and a log of one name are apart";
    let read = a.read_file_async("appends").await;
    ensure(is_kind(&read, io::ErrorKind::NotFound), check, || {
        format!("the file `appends` read {read:?}")
    })?;
    let read = b.read_log_async("top").await;
    ensure(is_kind(&read, io::ErrorKind::NotFound), check, || {
        format!("the log `top` read {read:?}")
    })
}

/// Names that [`check_name`](crate::check_name) refuses, and log names in
/// more than one part, are refused with [`io::ErrorKind::InvalidInput`].
async fn names<S: AsyncStore + ?Sized>(a: &S) -> Result<(), Violation> {
    let check = "a name that is no name in a session is refused";
    let invalid = |result: &Result<(), LogWriteError>| matches!(result, Err(LogWriteError::Io(error)) if error.kind() == io::ErrorKind::InvalidInput);
    for name in ["", ".", "..", "../up", "/root", "a//b", "a\\b", "a/./b"] {
        let read = a.read_file_async(name).await;
        ensure(is_kind(&read, io::ErrorKind::InvalidInput), check, || {
            format!("read_file_async({name:?}) gave {read:?}")
        })?;
        let write = a.write_file_async(name, b"x").await;
        ensure(is_kind(&write, io::ErrorKind::InvalidInput), check, || {
            format!("write_file_async({name:?}) gave {write:?}")
        })?;
    }
    for name in ["", "..", "../up", "/root", "a\\b", "nested/log"] {
        let read = a.read_log_async(name).await;
        ensure(is_kind(&read, io::ErrorKind::InvalidInput), check, || {
            format!("read_log_async({name:?}) gave {read:?}")
        })?;
        let append = a.append_log_async(name, 0, b"x").await;
        ensure(invalid(&append), check, || {
            format!("append_log_async({name:?}) gave {append:?}")
        })?;
        let replace = a.replace_log_async(name, 0, b"x").await;
        ensure(invalid(&replace), check, || {
            format!("replace_log_async({name:?}) gave {replace:?}")
        })?;
    }
    Ok(())
}

/// Every future in `futures`, polled together until all are done: writes
/// in flight at once, on whichever connections the handles hold.
async fn together<'f, T>(futures: Vec<Pin<Box<dyn Future<Output = T> + 'f>>>) -> Vec<T> {
    let mut futures: Vec<_> = futures.into_iter().map(Some).collect();
    let mut outputs: Vec<Option<T>> = futures.iter().map(|_| None).collect();
    poll_fn(|cx| {
        for (slot, output) in futures.iter_mut().zip(outputs.iter_mut()) {
            if let Some(future) = slot
                && let Poll::Ready(done) = future.as_mut().poll(cx)
            {
                *output = Some(done);
                *slot = None;
            }
        }
        if futures.iter().all(Option::is_none) {
            Poll::Ready(())
        } else {
            Poll::Pending
        }
    })
    .await;
    outputs.into_iter().flatten().collect()
}

/// Of writers that read a log at one length and write to it at once,
/// through both handles, exactly one writes and the others are refused:
/// when they start the log, when they append to it, and when a replace
/// races the appends. Writers of one file at once all write, and it holds
/// one of their bytes whole, as two saves storing one blob do.
async fn races<S: AsyncStore + ?Sized>(a: &S, b: &S) -> Result<(), Violation> {
    let lines: Vec<Vec<u8>> = (0..RACERS)
        .map(|racer| format!("writer {racer}\n").into_bytes())
        .collect();
    let mut log = Vec::new();
    for (check, replacing) in [
        ("of writers starting a log at once, one writes", false),
        (
            "of writers appending at one length at once, one writes",
            false,
        ),
        (
            "of a replace and appends at one length at once, one writes",
            true,
        ),
    ] {
        let len = log.len() as u64;
        let writes = lines.iter().enumerate().map(|(racer, line)| {
            let store = if racer % 2 == 0 { a } else { b };
            let write: Pin<Box<dyn Future<Output = _>>> = if replacing && racer == 0 {
                Box::pin(store.replace_log_async("raced", len, line))
            } else {
                Box::pin(store.append_log_async("raced", len, line))
            };
            write
        });
        let results = together(writes.collect()).await;
        let mut winners = Vec::new();
        for (racer, result) in results.iter().enumerate() {
            match result {
                Ok(()) => winners.push(racer),
                Err(LogWriteError::Changed) => {}
                Err(error) => {
                    return ensure(false, check, || format!("writer {racer} failed: {error}"));
                }
            }
        }
        let &[winner] = winners.as_slice() else {
            return ensure(false, check, || format!("writers {winners:?} all wrote"));
        };
        let line = lines.get(winner).map_or(&[][..], Vec::as_slice);
        if replacing && winner == 0 {
            log = line.to_vec();
        } else {
            log.extend_from_slice(line);
        }
        reads(check, a.read_log_async("raced").await, &log)?;
    }

    let check = "of writers of one file at once, all write and one's bytes are kept";
    let writes = lines.iter().enumerate().map(|(racer, line)| {
        let store = if racer % 2 == 0 { a } else { b };
        let write: Pin<Box<dyn Future<Output = _>>> =
            Box::pin(store.write_file_async("blobs/raced", line));
        write
    });
    for (racer, result) in together(writes.collect()).await.into_iter().enumerate() {
        if let Err(error) = result {
            return ensure(false, check, || format!("writer {racer} failed: {error}"));
        }
    }
    match b.read_file_async("blobs/raced").await {
        Ok(file) => ensure(lines.contains(&file), check, || {
            format!("the file holds {file:?}")
        }),
        Err(error) => ensure(false, check, || format!("read failed: {error}")),
    }
}

/// Through [`save_async`] and [`load_async`]: a conversation saved through
/// one handle loads through the other, and of two saves that loaded the
/// same conversation the second is refused with [`SessionError::Changed`],
/// until it loads again.
async fn sessions<S: AsyncStore + ?Sized>(a: &S, b: &S) -> Result<(), Violation> {
    let check = "a saved conversation loads through the other handle";
    let failed = |error: SessionError| Violation {
        check,
        detail: error.to_string(),
    };
    let mut first = load_async(a, "chat").await.map_err(failed)?;
    first.extend([
        Message::user("My name is Alice."),
        Message::assistant("Hi Alice!"),
    ]);
    save_async(a, "chat", &mut first).await.map_err(failed)?;
    let mut second = load_async(b, "chat").await.map_err(failed)?;
    ensure(second == first, check, || {
        format!("loaded {second:?}, not {first:?}")
    })?;

    let check = "of two saves that loaded one conversation, the second is refused";
    first.push(Message::user("What is my name?"));
    save_async(a, "chat", &mut first)
        .await
        .map_err(|error| Violation {
            check,
            detail: error.to_string(),
        })?;
    second.push(Message::user("Forget my name."));
    let refused = save_async(b, "chat", &mut second).await;
    ensure(
        matches!(refused, Err(SessionError::Changed { ref log }) if log == "chat"),
        check,
        || format!("the second save gave {refused:?}"),
    )?;
    let loaded = load_async(b, "chat").await.map_err(|error| Violation {
        check,
        detail: error.to_string(),
    })?;
    ensure(loaded == first, check, || {
        format!("loaded {loaded:?}, not the first save's {first:?}")
    })
}

#[cfg(test)]
mod tests;
