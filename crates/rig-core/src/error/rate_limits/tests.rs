use std::time::{Duration, SystemTime, UNIX_EPOCH};

use http::{HeaderMap, HeaderName, HeaderValue};

use super::{Allowance, RateLimits};

/// 2026-01-01T00:00:00Z.
const NEW_YEAR: u64 = 1_767_225_600;

fn at(secs: u64) -> SystemTime {
    UNIX_EPOCH + Duration::from_secs(secs)
}

fn headers(pairs: &[(&'static str, &str)]) -> HeaderMap {
    pairs
        .iter()
        .filter_map(|(name, value)| {
            Some((
                HeaderName::from_static(name),
                HeaderValue::from_str(value).ok()?,
            ))
        })
        .collect()
}

fn allowance(limit: u64, remaining: u64, resets_in: Option<u64>) -> Option<Allowance> {
    Some(Allowance {
        limit,
        remaining,
        resets_in: resets_in.map(Duration::from_secs),
    })
}

#[test]
fn each_dialect_gives_its_allowances() {
    let now = at(NEW_YEAR - 34);
    for (dialect, pairs, requests, tokens) in [
        (
            "openai",
            vec![
                ("x-ratelimit-limit-requests", "50"),
                ("x-ratelimit-remaining-requests", "42"),
                ("x-ratelimit-reset-requests", "34s"),
                ("x-ratelimit-limit-tokens", "40000"),
                ("x-ratelimit-remaining-tokens", "18000"),
                ("x-ratelimit-reset-tokens", "6m0s"),
            ],
            allowance(50, 42, Some(34)),
            allowance(40_000, 18_000, Some(360)),
        ),
        (
            "anthropic",
            vec![
                ("anthropic-ratelimit-requests-limit", "50"),
                ("anthropic-ratelimit-requests-remaining", "49"),
                ("anthropic-ratelimit-requests-reset", "2026-01-01T00:00:00Z"),
                ("anthropic-ratelimit-tokens-limit", "40000"),
                ("anthropic-ratelimit-tokens-remaining", "39000"),
                ("anthropic-ratelimit-tokens-reset", "2026-01-01T00:01:00Z"),
            ],
            allowance(50, 49, Some(34)),
            allowance(40_000, 39_000, Some(94)),
        ),
        (
            "anthropic input tokens",
            vec![
                ("anthropic-ratelimit-input-tokens-limit", "30000"),
                ("anthropic-ratelimit-input-tokens-remaining", "29000"),
            ],
            None,
            allowance(30_000, 29_000, None),
        ),
        (
            "openrouter",
            vec![
                ("x-ratelimit-limit", "20"),
                ("x-ratelimit-remaining", "19"),
                ("x-ratelimit-reset", "1767225600000"),
            ],
            allowance(20, 19, Some(34)),
            None,
        ),
        (
            "ietf draft",
            vec![
                ("ratelimit-limit", "100;w=60"),
                ("ratelimit-remaining", "99"),
                ("ratelimit-reset", "34"),
            ],
            allowance(100, 99, Some(34)),
            None,
        ),
    ] {
        let read = RateLimits::from_headers(&headers(&pairs), now);
        assert_eq!(read, RateLimits { requests, tokens }, "{dialect}");
    }
}

#[test]
fn each_reset_form_is_a_wait_from_the_reply() {
    let now = at(NEW_YEAR);
    let half = Duration::from_millis(500);
    for (reset, wait) in [
        ("34", Some(Duration::from_secs(34))),
        ("1.5", Some(Duration::from_millis(1500))),
        ("1767225634", Some(Duration::from_secs(34))),
        ("1767225634500", Some(Duration::from_secs(34) + half)),
        ("20ms", Some(Duration::from_millis(20))),
        ("1h2m3.5s", Some(Duration::from_secs(3723) + half)),
        ("2026-01-01T00:00:34Z", Some(Duration::from_secs(34))),
        (
            "2026-01-01T00:00:34.5Z",
            Some(Duration::from_secs(34) + half),
        ),
        ("2026-01-01T01:00:34+01:00", Some(Duration::from_secs(34))),
        ("2025-12-31T23:01:34-01:00", Some(Duration::from_secs(94))),
        (
            "Thu, 01 Jan 2026 00:00:34 GMT",
            Some(Duration::from_secs(34)),
        ),
        ("2025-12-31T23:59:00Z", Some(Duration::ZERO)),
        ("1767225000", Some(Duration::ZERO)),
        ("2026-02-30T00:00:00Z", None),
        ("2026-01-01T24:00:00Z", None),
        ("2026-01-01T00:00:00", None),
        ("soon", None),
        // A fraction is plain digits, cut at the nanosecond.
        ("2026-01-01T00:00:00.18446744073709549568e20Z", None),
        ("2026-01-01T00:00:34.Z", None),
        (
            "2026-01-01T00:00:34.1234567899Z",
            Some(Duration::new(34, 123_456_789)),
        ),
        ("1969-12-31T23:59:59Z", Some(Duration::ZERO)),
        ("2026-01-01T00:00:34+99:99", None),
        ("2026-01-01T00:00:34+24:00", None),
        ("1h1us", Some(Duration::new(3600, 1_000))),
        ("1500us", Some(Duration::from_micros(1500))),
        ("3ns", Some(Duration::from_nanos(3))),
        ("34s, roughly", Some(Duration::from_secs(34))),
        ("99999999999999999999999999h", None),
        ("1e3", None),
        ("-3", None),
    ] {
        let pairs = [
            ("ratelimit-limit", "10"),
            ("ratelimit-remaining", "5"),
            ("ratelimit-reset", reset),
        ];
        let read = RateLimits::from_headers(&headers(&pairs), now).requests;
        assert_eq!(read.map(|read| read.resets_in), Some(wait), "{reset}");
    }
}

#[test]
fn a_unix_time_reset_is_exact_to_the_nanosecond() {
    let now = at(NEW_YEAR) + Duration::from_nanos(1);
    let pairs = [
        ("ratelimit-limit", "10"),
        ("ratelimit-remaining", "5"),
        ("ratelimit-reset", "1767225634"),
    ];
    let read = RateLimits::from_headers(&headers(&pairs), now).requests;
    let wait = Duration::from_secs(34) - Duration::from_nanos(1);
    assert_eq!(read.map(|read| read.resets_in), Some(Some(wait)));
}

#[test]
fn an_allowance_needs_a_limit_and_a_remainder() {
    let now = at(NEW_YEAR);
    let only_limit = headers(&[("x-ratelimit-limit-requests", "50")]);
    assert_eq!(
        RateLimits::from_headers(&only_limit, now),
        RateLimits::default()
    );
    let not_a_number = headers(&[
        ("x-ratelimit-limit-requests", "many"),
        ("x-ratelimit-remaining-requests", "42"),
        ("ratelimit-limit", "10"),
        ("ratelimit-remaining", "5"),
    ]);
    let read = RateLimits::from_headers(&not_a_number, now);
    assert_eq!(read.requests, allowance(10, 5, None));
}
