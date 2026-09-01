//! Capture what a code path logs, without the capture quietly going dead.
//!
//! Asserting on `tracing` output from a test binary has a hazard that is easy
//! to get wrong and hard to notice: interest is cached **per callsite,
//! globally**, computed by whichever thread reaches the callsite first, while
//! `set_default` installs a subscriber for the current thread only. A sibling
//! test that reaches the callsite without a subscriber can cache
//! `Interest::never` and silence the capture mid-test.
//!
//! The failure is asymmetric. A silenced capture makes `assert!(contains(..))`
//! fail loudly, but makes `assert!(!contains(..))` pass **vacuously** — the
//! test still reports green while covering nothing.
//!
//! [`captured_logs`] packages the defence described on
//! `rig_core::test_utils::scoped_tracing_subscriber_guard`, so the safe
//! sequence is the one you get by default rather than one each caller
//! reassembles.

use std::future::Future;
use std::io::Write;
use std::sync::{Arc, Mutex, PoisonError};

/// Shared buffer behind the subscriber's writer.
#[derive(Clone)]
struct SharedWriter(Arc<Mutex<Vec<u8>>>);

impl Write for SharedWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn drain(buffer: &Arc<Mutex<Vec<u8>>>) -> String {
    let mut guard = buffer.lock().unwrap_or_else(PoisonError::into_inner);
    let bytes = std::mem::take(&mut *guard);
    String::from_utf8_lossy(&bytes).into_owned()
}

/// Run `scenario` under a scoped subscriber and return everything it logged at
/// `level` or above.
///
/// `anchor` is a *different* run of the same code path, chosen so that it
/// always logs — every string in `anchor_markers` must appear in its output.
/// It does two jobs:
///
/// 1. **Warms** the callsites, so they first-register against this subscriber
///    rather than against whichever thread happened to reach them first.
/// 2. **Proves the capture is live** after the interest-cache rebuild, so a
///    subsequent absence assertion over `scenario`'s output means the path
///    stayed quiet — not that nothing was listening.
///
/// That second job is why this takes an anchor at all: without it, the only
/// thing keeping an absence assertion honest is the test's own isolation, and
/// nothing checks that isolation still holds.
///
/// Both closures are invoked more than once, so they must be repeatable — take
/// a factory (`|| run(pages.clone())`), not a one-shot future.
///
/// # Panics
///
/// If the anchor's output is missing any of `anchor_markers`, which means the
/// capture is not live and no assertion on the returned logs would be sound.
pub async fn captured_logs<A, AFut, S, SFut>(
    level: tracing::Level,
    mut anchor: A,
    anchor_markers: &[&str],
    mut scenario: S,
) -> String
where
    A: FnMut() -> AFut,
    AFut: Future<Output = ()>,
    S: FnMut() -> SFut,
    SFut: Future<Output = ()>,
{
    assert!(
        !anchor_markers.is_empty(),
        "captured_logs needs at least one anchor marker; an anchor nothing is \
         asserted about cannot prove the capture is live"
    );

    // Scoped-subscriber tests must not run concurrently.
    let _isolation = rig_core::test_utils::scoped_tracing_subscriber_guard().await;

    let buffer = Arc::new(Mutex::new(Vec::new()));
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(level)
        .with_ansi(false)
        .without_time()
        .with_writer({
            let buffer = buffer.clone();
            move || SharedWriter(buffer.clone())
        })
        .finish();

    let _default = tracing::subscriber::set_default(subscriber);

    // (1) Warm: first-register this path's callsites against THIS subscriber.
    anchor().await;
    // (2) Rebuild: heal callsites a foreign thread already poisoned.
    tracing::callsite::rebuild_interest_cache();
    drain(&buffer);

    // (3) Prove the capture survived the rebuild, at the callsites that matter.
    let anchor_logs = {
        anchor().await;
        drain(&buffer)
    };
    for marker in anchor_markers {
        assert!(
            anchor_logs.contains(marker),
            "tracing capture is not live: the anchor run did not log {marker:?}, so \
             nothing asserted about the captured output below would be meaningful \
             (an absence assertion would pass vacuously). Captured:\n{anchor_logs}"
        );
    }

    // (4) The run under assertion.
    scenario().await;
    drain(&buffer)
}
