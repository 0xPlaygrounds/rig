#![allow(clippy::unwrap_used)]
use super::*;

/// A finished task raises the wake; a runner waiting on it returns at once,
/// and an untouched wake times out as not raised.
#[test]
fn the_wake_is_raised_by_signal_and_taken_by_wait() {
    let wake = Wake::default();
    assert!(!wake.wait(Duration::from_millis(1)));
    let raiser = wake.clone();
    std::thread::spawn(move || raiser.signal());
    assert!(wake.wait(Duration::from_secs(5)));
    assert!(!wake.take(), "wait took the signal");
}
