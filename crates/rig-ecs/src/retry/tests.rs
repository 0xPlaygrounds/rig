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

/// The default gives up after four retries and past a minute's asked
/// wait; a policy set higher retries on.
#[test]
fn the_default_is_four_retries_and_a_minute_and_a_set_policy_goes_further() {
    let now = SystemTime::now();
    let busy = report("overloaded").with_retryable(true);
    let asked = report("try again in 7200s").with_retryable(true);
    let default = RetryPolicy::default();
    assert!(matches!(default.verdict(&busy, 4, now), Verdict::GaveUp(_)));
    assert!(matches!(
        default.verdict(&asked, 0, now),
        Verdict::GaveUp(_)
    ));
    let more = RetryPolicy {
        max_retries: 50,
        max_asked_wait: Duration::from_secs(7200),
        ..RetryPolicy::DEFAULT
    };
    assert!(matches!(more.verdict(&busy, 49, now), Verdict::Retry(_)));
    assert_eq!(
        more.verdict(&asked, 0, now),
        Verdict::Retry(Duration::from_secs(7200))
    );
}
