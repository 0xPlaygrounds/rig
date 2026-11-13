use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::error::{ErrorKind, ErrorReport};
use crate::provider_response::ProviderResponseError;

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
fn a_body_hint_in_minutes_or_hours_names_the_wait() {
    let now = SystemTime::now();
    for (message, wait) in [
        ("Please try again in 6m0s.", Duration::from_secs(360)),
        (
            "Please try again in 1h2m3.5s.",
            Duration::from_millis(3_723_500),
        ),
        (
            "Please try again in 2m59.56s.",
            Duration::from_millis(179_560),
        ),
        ("Please try again in 1.5 s.", Duration::from_millis(1500)),
        ("Please try again in .5s.", Duration::from_millis(500)),
        ("Please try again in 1h1us.", Duration::new(3600, 1_000)),
        (
            "Please try again in 250\u{b5}s.",
            Duration::from_micros(250),
        ),
        ("Please try again in 40ns.", Duration::from_nanos(40)),
        (
            "Please try again in 0.0000000001h.",
            Duration::from_nanos(360),
        ),
        (
            "Please try again in 0.0000000006s.",
            Duration::from_nanos(1),
        ),
        (
            "Please try again in 3s 2 tokens left.",
            Duration::from_secs(3),
        ),
    ] {
        assert_eq!(report(message).retry_after(now), Some(wait), "{message}");
    }
    for message in [
        "try again in 2 minutes",
        "try again in 6m0x",
        "try again in 5mins",
    ] {
        assert_eq!(report(message).retry_after(now), None, "{message}");
    }
}

#[test]
fn a_retry_after_date_names_the_wait() {
    let mut headers = http::HeaderMap::new();
    headers.insert(
        "retry-after",
        http::HeaderValue::from_static("Sun, 06 Nov 1994 08:49:37 GMT"),
    );
    let mut dated = report("busy");
    dated.provider_response = Some(
        ProviderResponseError::new(http::StatusCode::TOO_MANY_REQUESTS, "busy")
            .with_headers(Some(headers)),
    );
    let now = UNIX_EPOCH + Duration::from_secs(784_111_770);
    assert_eq!(dated.retry_after(now), Some(Duration::from_secs(7)));
}
