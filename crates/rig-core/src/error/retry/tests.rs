use std::time::{Duration, SystemTime, UNIX_EPOCH};

use super::{RetryPolicy, Verdict, http_date};
use crate::error::{ErrorKind, ErrorReport};

fn report(message: &str) -> ErrorReport {
    ErrorReport::new(ErrorKind::Provider, message)
}

#[test]
fn providers_words_for_a_full_context_are_an_overflow() {
    for message in [
        "prompt is too long: 210000 tokens > 200000 maximum",
        "This model's maximum context length is 128000 tokens.",
        "Input token count 1048577 exceeds the maximum number of tokens allowed",
        "Your input exceeds the context window of this model",
    ] {
        assert!(report(message).is_context_overflow(), "{message}");
    }
    assert!(
        report("anything")
            .with_code("context_length_exceeded")
            .is_context_overflow()
    );
    assert!(
        report("payload")
            .with_http_status(413)
            .is_context_overflow()
    );
}

#[test]
fn rate_limits_and_retryable_failures_are_not_an_overflow() {
    assert!(!report("Rate limit reached: too many tokens per minute").is_context_overflow());
    assert!(
        !report("too many tokens")
            .with_retryable(true)
            .is_context_overflow()
    );
}

#[test]
fn a_body_hint_names_the_wait() {
    let now = SystemTime::now();
    let gemini = report(r#"{"error": {"details": [{"retryDelay": "7s"}]}}"#);
    assert_eq!(gemini.retry_after(now), Some(Duration::from_secs(7)));
    let openai = report("Rate limit reached. Please try again in 1.5s.");
    assert_eq!(openai.retry_after(now), Some(Duration::from_millis(1500)));
    let millis = report("Please try again in 20ms.");
    assert_eq!(millis.retry_after(now), Some(Duration::from_millis(20)));
    assert_eq!(report("try again later").retry_after(now), None);
}

#[test]
fn an_http_date_is_read() {
    assert_eq!(
        http_date("Sun, 06 Nov 1994 08:49:37 GMT"),
        Some(UNIX_EPOCH + Duration::from_secs(784_111_777))
    );
    assert_eq!(http_date("Sunday, 06-Nov-94 08:49:37 GMT"), None);
}

#[test]
fn retries_back_off_and_give_up() {
    let policy = RetryPolicy::DEFAULT;
    let busy = report("overloaded").with_retryable(true);
    let now = SystemTime::now();
    match policy.verdict(&busy, 1, now) {
        Verdict::Retry(wait) => {
            assert!(wait <= Duration::from_secs(4) && wait >= Duration::from_secs(3));
        }
        other => panic!("expected a retry, got {other:?}"),
    }
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
