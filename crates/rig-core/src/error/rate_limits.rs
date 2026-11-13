//! The request and token allowances a provider's reply states in its
//! rate-limit headers: OpenAI's and Groq's `x-ratelimit-*-requests|tokens`,
//! Anthropic's `anthropic-ratelimit-*`, OpenRouter's `x-ratelimit-*` and the
//! IETF draft's `ratelimit-*`. Reading them never touches the clock: the
//! caller passes the time the reply came, so resets given as a date or a Unix
//! time become a wait.
//!
//! ```
//! use std::time::{Duration, SystemTime};
//! use rig_core::error::RateLimits;
//!
//! let mut headers = http::HeaderMap::new();
//! headers.insert("x-ratelimit-limit-requests", "50".parse()?);
//! headers.insert("x-ratelimit-remaining-requests", "42".parse()?);
//! headers.insert("x-ratelimit-reset-requests", "34s".parse()?);
//! let limits = RateLimits::from_headers(&headers, SystemTime::now());
//! let requests = limits.requests.ok_or("no request allowance")?;
//! assert_eq!((requests.remaining, requests.limit), (42, 50));
//! assert_eq!(requests.resets_in, Some(Duration::from_secs(34)));
//! assert_eq!(limits.tokens, None);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use super::retry::{NANOS, decimal_nanos, hinted_wait, nanoseconds};

/// The `(limit, remaining, reset)` headers of a request allowance; the
/// first pair present wins.
const REQUESTS: [[&str; 3]; 4] = [
    [
        "x-ratelimit-limit-requests",
        "x-ratelimit-remaining-requests",
        "x-ratelimit-reset-requests",
    ],
    [
        "anthropic-ratelimit-requests-limit",
        "anthropic-ratelimit-requests-remaining",
        "anthropic-ratelimit-requests-reset",
    ],
    [
        "x-ratelimit-limit",
        "x-ratelimit-remaining",
        "x-ratelimit-reset",
    ],
    ["ratelimit-limit", "ratelimit-remaining", "ratelimit-reset"],
];

/// The `(limit, remaining, reset)` headers of a token allowance.
const TOKENS: [[&str; 3]; 3] = [
    [
        "x-ratelimit-limit-tokens",
        "x-ratelimit-remaining-tokens",
        "x-ratelimit-reset-tokens",
    ],
    [
        "anthropic-ratelimit-tokens-limit",
        "anthropic-ratelimit-tokens-remaining",
        "anthropic-ratelimit-tokens-reset",
    ],
    [
        "anthropic-ratelimit-input-tokens-limit",
        "anthropic-ratelimit-input-tokens-remaining",
        "anthropic-ratelimit-input-tokens-reset",
    ],
];

/// One allowance as a reply states it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Allowance {
    /// Requests or tokens per window.
    pub limit: u64,
    /// Requests or tokens left in this window.
    pub remaining: u64,
    /// How long after the reply the window resets, when the reply said so
    /// in a form that could be read.
    pub resets_in: Option<Duration>,
}

/// The allowances a reply's rate-limit headers state.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RateLimits {
    /// The request allowance, when the reply gave its limit and remainder.
    pub requests: Option<Allowance>,
    /// The token allowance, when the reply gave its limit and remainder.
    pub tokens: Option<Allowance>,
}

impl RateLimits {
    /// Reads the allowances in `headers`, a reply that came at `now`.
    ///
    /// An allowance needs both a limit and a remainder whose values start
    /// with a whole number (`100` of `100;w=60`). Its reset may be seconds
    /// (`34`, `1.5`), a Unix time in seconds or milliseconds, a value that
    /// starts with a Go duration (`6m0s`, `20ms`, `1h1us`), an RFC 3339 date
    /// (`2026-01-01T00:00:00Z`) or an HTTP date
    /// (`Thu, 01 Jan 2026 00:00:34 GMT`). A reset already past is zero; any
    /// other form is none.
    pub fn from_headers(headers: &http::HeaderMap, now: SystemTime) -> Self {
        Self {
            requests: allowance(headers, &REQUESTS, now),
            tokens: allowance(headers, &TOKENS, now),
        }
    }
}

/// The first allowance of `names` the headers give both a limit and a
/// remainder for.
fn allowance(headers: &http::HeaderMap, names: &[[&str; 3]], now: SystemTime) -> Option<Allowance> {
    let text = |name: &str| headers.get(name)?.to_str().ok().map(str::trim);
    names.iter().find_map(|[limit, remaining, reset]| {
        Some(Allowance {
            limit: leading_number(text(limit)?)?,
            remaining: leading_number(text(remaining)?)?,
            resets_in: text(reset).and_then(|reset| reset_in(reset, now)),
        })
    })
}

/// The whole number `text` starts with.
fn leading_number(text: &str) -> Option<u64> {
    let end = text
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(text.len());
    text.get(..end)?.parse().ok()
}

/// How long after `now` the reset `text` names comes.
fn reset_in(text: &str, now: SystemTime) -> Option<Duration> {
    if let Some(number) = decimal_nanos(text, NANOS) {
        // Numbers this large are Unix times, not waits: no window lasts 30 years.
        let at = if number > 1_000_000_000_000 * NANOS {
            decimal_nanos(text, 1_000_000)?
        } else if number > 1_000_000_000 * NANOS {
            number
        } else {
            return nanoseconds(number);
        };
        return until(at, now);
    }
    if let Some(wait) = hinted_wait(text) {
        return Some(wait);
    }
    let at = match rfc3339(text) {
        Some(at) => at,
        None => since_epoch(httpdate::parse_http_date(text).ok()?)?,
    };
    until(at, now)
}

/// The wait from `now` until `at`, in nanoseconds since the Unix epoch;
/// zero once it has passed.
fn until(at: i128, now: SystemTime) -> Option<Duration> {
    nanoseconds(at.checked_sub(since_epoch(now)?)?)
}

/// `time` in nanoseconds since the Unix epoch, negative before it.
fn since_epoch(time: SystemTime) -> Option<i128> {
    match time.duration_since(UNIX_EPOCH) {
        Ok(after) => i128::try_from(after.as_nanos()).ok(),
        Err(before) => i128::try_from(before.duration().as_nanos())
            .ok()
            .map(|n| -n),
    }
}

/// The instant an RFC 3339 date names, in nanoseconds since the Unix epoch:
/// `2026-01-01T00:00:00Z`, with an optional fraction of a second (digits
/// past the ninth dropped, not rounded) and a `Z` or `+hh:mm` offset.
fn rfc3339(text: &str) -> Option<i128> {
    let (date, time) = text.split_once(['T', 't', ' '])?;
    let mut date = date.split('-');
    let year = digits(date.next()?, 4)?;
    let month = digits(date.next()?, 2)?;
    let day = digits(date.next()?, 2)?;
    let (clock, offset) = match time.strip_suffix(['Z', 'z']) {
        Some(clock) => (clock, 0),
        None => {
            let (clock, offset) = time.split_at_checked(time.rfind(['+', '-'])?)?;
            let sign = if offset.starts_with('-') { -1 } else { 1 };
            let (hours, minutes) = offset.get(1..)?.split_once(':')?;
            let (hours, minutes) = (digits(hours, 2)?, digits(minutes, 2)?);
            if hours > 23 || minutes > 59 {
                return None;
            }
            (clock, sign * (hours * 3600 + minutes * 60))
        }
    };
    let (clock, fraction) = clock.split_once('.').unwrap_or((clock, "0"));
    if fraction.is_empty() || !fraction.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    // Digits past the nanosecond are dropped, as chrono does.
    let fraction: i128 = format!("{:0<9}", fraction.get(..9).unwrap_or(fraction))
        .parse()
        .ok()?;
    let mut clock = clock.split(':');
    let hour = digits(clock.next()?, 2)?;
    let minute = digits(clock.next()?, 2)?;
    let second = digits(clock.next()?, 2)?;
    let days = days_since_epoch(year, month, day)?;
    let valid = date.next().is_none() && clock.next().is_none();
    if !valid || hour > 23 || minute > 59 || second > 60 {
        return None;
    }
    let seconds = days * 86_400 + hour * 3600 + minute * 60 + second - offset;
    i128::from(seconds)
        .checked_mul(NANOS)?
        .checked_add(fraction)
}

/// `text` as a number of exactly `width` ASCII digits.
fn digits(text: &str, width: usize) -> Option<i64> {
    let ascii = text.len() == width && text.bytes().all(|b| b.is_ascii_digit());
    ascii.then(|| text.parse().ok())?
}

/// Days from 1970-01-01 to a date of the proleptic Gregorian calendar, or
/// `None` for a date that does not exist (Howard Hinnant's `days_from_civil`).
fn days_since_epoch(year: i64, month: i64, day: i64) -> Option<i64> {
    let days = |year: i64, month: i64, day: i64| {
        let year = if month <= 2 { year - 1 } else { year };
        let era = year.div_euclid(400);
        let of_era = year - era * 400;
        let of_year = (153 * ((month + 9) % 12) + 2) / 5 + day - 1;
        era * 146_097 + of_era * 365 + of_era / 4 - of_era / 100 + of_year - 719_468
    };
    let (next_year, next_month) = if month == 12 {
        (year + 1, 1)
    } else {
        (year, month + 1)
    };
    let length = days(next_year, next_month, 1) - days(year, month, 1);
    ((1..=12).contains(&month) && (1..=length).contains(&day)).then(|| days(year, month, day))
}

#[cfg(test)]
mod tests;
