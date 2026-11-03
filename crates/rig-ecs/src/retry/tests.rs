use std::time::{Duration, SystemTime};

use rig_core::error::{ErrorKind, ErrorReport};

use super::{RetryPolicy, Verdict};

fn report(message: &str) -> ErrorReport {
    ErrorReport::new(ErrorKind::Provider, message)
}

#[test]
fn retries_back_off_and_give_up() {
    let policy = RetryPolicy {
        max_retries: Some(4),
        max_asked_wait: Some(Duration::from_secs(60)),
        ..RetryPolicy::DEFAULT
    };
    let busy = report("overloaded").with_retryable(true);
    let now = SystemTime::now();
    let verdict = policy.verdict(&busy, 1, now);
    assert!(
        matches!(verdict, Verdict::Retry(wait) if wait <= Duration::from_secs(4) && wait >= Duration::from_secs(3)),
        "{verdict:?}"
    );
    assert!(matches!(
        policy.verdict(&busy, 4, now),
        Verdict::GaveUp(why) if why.contains("`[retry] max_retries` is 4")
    ));
    assert_eq!(
        policy.verdict(&report("bad request"), 0, now),
        Verdict::Final
    );
    let asked = report("try again in 120s").with_retryable(true);
    assert!(matches!(
        policy.verdict(&asked, 0, now),
        Verdict::GaveUp(why) if why.contains("`[retry] max_wait_secs` is 60")
    ));
    assert!(policy.backoff(10) <= policy.max_backoff);
}

/// With no limit set, as by default, a transient failure is retried however
/// often it failed and however long the provider asks to wait.
#[test]
fn by_default_a_transient_failure_is_always_retried() {
    let policy = RetryPolicy::default();
    let now = SystemTime::now();
    let busy = report("overloaded").with_retryable(true);
    assert!(matches!(policy.verdict(&busy, 50, now), Verdict::Retry(_)));
    let asked = report("try again in 7200s").with_retryable(true);
    assert_eq!(
        policy.verdict(&asked, 0, now),
        Verdict::Retry(Duration::from_secs(7200))
    );
}
