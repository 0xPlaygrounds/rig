use super::{Elapsed, timeout};
use std::time::Duration;

#[tokio::test]
async fn timeout_returns_ok_for_a_future_that_completes_in_time() {
    let result = timeout(Duration::from_secs(5), async { 42 }).await;
    assert_eq!(result, Ok(42));
}

#[tokio::test]
async fn timeout_returns_elapsed_for_a_future_that_never_completes() {
    let result = timeout(Duration::from_millis(20), std::future::pending::<()>()).await;
    assert_eq!(result, Err(Elapsed));
}

#[tokio::test]
async fn timeout_zero_duration_still_polls_a_ready_future_once() {
    // Documented contract: a zero/already-elapsed duration still polls the
    // future once before electing `Elapsed`, so a ready future wins.
    let result = timeout(Duration::ZERO, async { 7 }).await;
    assert_eq!(result, Ok(7));
}
