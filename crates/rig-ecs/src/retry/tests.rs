use std::time::{Duration, SystemTime};

use rig_core::error::{ErrorKind, ErrorReport};

use super::{RetryPolicy, Verdict};

fn report(message: &str) -> ErrorReport {
    ErrorReport::new(ErrorKind::Provider, message)
}

#[test]
fn retries_back_off_and_give_up() {
    let policy = RetryPolicy::DEFAULT;
    let busy = report("overloaded").with_retryable(true);
    let now = SystemTime::now();
    let verdict = policy.verdict(&busy, 1, now);
    assert!(
        matches!(verdict, Verdict::Retry(wait) if wait <= Duration::from_secs(4) && wait >= Duration::from_secs(3)),
        "{verdict:?}"
    );
    assert!(matches!(
        policy.verdict(&busy, policy.max_retries, now),
        Verdict::GaveUp(_)
    ));
    assert_eq!(
        policy.verdict(&report("bad request"), 0, now),
        Verdict::Final
    );
    let asked = report("try again in 120s").with_retryable(true);
    assert!(matches!(policy.verdict(&asked, 0, now), Verdict::GaveUp(_)));
    assert!(policy.backoff(10) <= policy.max_backoff);
}
