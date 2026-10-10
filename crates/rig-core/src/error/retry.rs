//! What to do about a failed model call: wait and send it again after a
//! transient failure, honouring the wait the provider asked for; recover
//! from a request that outgrew the model's context window; or give up.
//!
//! ```
//! use std::time::{Duration, SystemTime};
//! use rig_core::error::retry::{RetryPolicy, Verdict};
//! use rig_core::error::{ErrorKind, ErrorReport};
//!
//! let overloaded = ErrorReport::new(ErrorKind::Provider, "overloaded").with_retryable(true);
//! let policy = RetryPolicy::DEFAULT;
//! assert!(matches!(policy.verdict(&overloaded, 0, SystemTime::now()), Verdict::Retry(_)));
//!
//! let full = ErrorReport::new(ErrorKind::Provider, "prompt is too long: 210000 tokens");
//! assert!(full.is_context_overflow());
//! ```
//!
//! Nothing here reads the clock: a caller passes `now`, which a browser
//! target gets from its own clock.

use std::time::{Duration, SystemTime};

use super::{ErrorKind, ErrorReport};

/// How often and how long a failed call is retried.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RetryPolicy {
    /// Retries of failed calls in a row before giving up.
    pub max_retries: u32,
    /// The first wait; each retry doubles it, up to `max_backoff`.
    pub first_backoff: Duration,
    /// The longest wait the policy picks itself.
    pub max_backoff: Duration,
    /// The longest wait a provider may ask for; past it the policy gives
    /// up, as waiting longer in front of a user is worse than saying so.
    pub max_asked_wait: Duration,
}

impl RetryPolicy {
    /// Four retries, waiting 2 s doubling to 30 s, or what the provider asks
    /// for up to a minute.
    pub const DEFAULT: Self = Self {
        max_retries: 4,
        first_backoff: Duration::from_secs(2),
        max_backoff: Duration::from_secs(30),
        max_asked_wait: Duration::from_secs(60),
    };

    /// What to do about `report`, the failure of a call already retried
    /// `retries` times in a row, at `now`.
    pub fn verdict(&self, report: &ErrorReport, retries: u32, now: SystemTime) -> Verdict {
        if report.is_context_overflow() {
            return Verdict::Overflow;
        }
        if !report.is_retryable() || report.kind == ErrorKind::Cancelled {
            return Verdict::Final;
        }
        if retries >= self.max_retries {
            return Verdict::GaveUp(format!("it failed {} times in a row", retries + 1));
        }
        match report.retry_after(now) {
            Some(asked) if asked > self.max_asked_wait => Verdict::GaveUp(format!(
                "the provider asks to wait {}s before the next call",
                asked.as_secs()
            )),
            Some(asked) => Verdict::Retry(asked),
            None => Verdict::Retry(self.backoff(retries)),
        }
    }

    /// The wait before retry `retries + 1`: doubling from `first_backoff`,
    /// capped at `max_backoff`, less up to a quarter at random so that
    /// callers failing together do not retry together.
    pub fn backoff(&self, retries: u32) -> Duration {
        let doubled = self.first_backoff.saturating_mul(1 << retries.min(16));
        let wait = doubled.min(self.max_backoff);
        wait.saturating_sub(wait * fastrand::u32(0..250) / 1000)
    }
}

/// What to do about a failed call.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// The request does not fit the model's context window: make it
    /// smaller, then send it again.
    Overflow,
    /// Wait this long, then send the same request again.
    Retry(Duration),
    /// Transient, but retrying is not worth it, for this reason: the
    /// retries ran out, or the provider asks for too long a wait. Nothing
    /// is wrong with the request itself.
    GaveUp(String),
    /// The same request fails again.
    Final,
}

impl ErrorReport {
    /// The wait the provider asked for, as of `now`: the `retry-after-ms`
    /// header (OpenAI), else `retry-after` in seconds or as an HTTP date,
    /// else a hint in the body: Gemini's `"retryDelay": "2s"` or OpenAI's
    /// "try again in 1.5s".
    pub fn retry_after(&self, now: SystemTime) -> Option<Duration> {
        let header = |name: &str| {
            self.provider_response_headers()
                .and_then(|headers| headers.get(name))
                .and_then(|value| value.to_str().ok())
                .map(str::trim)
        };
        if let Some(millis) = header("retry-after-ms").and_then(|value| value.parse::<f64>().ok()) {
            return seconds(millis / 1000.0);
        }
        if let Some(value) = header("retry-after") {
            return match value.parse::<f64>() {
                Ok(secs) => seconds(secs),
                Err(_) => httpdate::parse_http_date(value)
                    .ok()
                    .map(|at| at.duration_since(now).unwrap_or_default()),
            };
        }
        let body = self
            .provider_response_body()
            .unwrap_or(&self.message)
            .to_ascii_lowercase();
        ["\"retrydelay\"", "try again in "]
            .iter()
            .find_map(|marker| {
                let after = body.get(body.find(marker)? + marker.len()..)?;
                hinted_wait(after.trim_start_matches([' ', ':', '"']))
            })
    }

    /// Whether this report says the request does not fit the model's
    /// context window: the `context_length_exceeded` code, a 413, or the
    /// provider saying so in its own words, less rate limits that happen to
    /// mention tokens. A retryable failure is never an overflow, which fails
    /// the same way every time (Bedrock's "too many tokens" throttling is
    /// retryable).
    pub fn is_context_overflow(&self) -> bool {
        if self.code.as_deref() == Some("context_length_exceeded") {
            return true;
        }
        if self.is_retryable() {
            return false;
        }
        if self.http_status == Some(413) {
            return true;
        }
        [Some(self.message.as_str()), self.provider_response_body()]
            .into_iter()
            .flatten()
            .map(str::to_ascii_lowercase)
            .any(|text| says_overflow(&text) && !says_rate_limit(&text))
    }
}

/// The ways providers say the context is full, lower case, in the forms pi
/// knows (`packages/ai/src/utils/overflow.ts`).
const OVERFLOW: &[&str] = &[
    "prompt is too long",
    "prompt too long",
    "prompt exceeds max length",
    "request_too_large",
    "input is too long for requested model",
    "exceeds the context window",
    "maximum context length",
    "maximum prompt length is",
    "reduce the length of the messages",
    "maximum allowed input length",
    "than the model's context length",
    "than the models context length",
    "greater than the context length",
    "exceeds the limit of ",
    "exceeds the available context size",
    "context window exceeds limit",
    "exceeded model token limit",
    "but the configured context size is",
    "model_context_window_exceeded",
    "exceeded context length",
    "exceeded max context length",
    "range of input length should be",
    "context_length_exceeded",
    "context length exceeded",
    "too many tokens",
    "token limit exceeded",
];

/// Whether lower-case `text` says the context is full.
fn says_overflow(text: &str) -> bool {
    OVERFLOW.iter().any(|phrase| text.contains(phrase))
        || (text.contains("input token count") && text.contains("exceeds the maximum"))
}

/// Whether lower-case `text` is a rate limit or an outage whose words also
/// match [`OVERFLOW`].
fn says_rate_limit(text: &str) -> bool {
    text.starts_with("throttling error:")
        || text.starts_with("service unavailable:")
        || text.contains("rate limit")
        || text.contains("too many requests")
}

/// `secs` as a duration, when it is a sane one.
fn seconds(secs: f64) -> Option<Duration> {
    Duration::try_from_secs_f64(secs.max(0.0)).ok()
}

/// The wait `text` starts with: a number and `ms` or `s`.
fn hinted_wait(text: &str) -> Option<Duration> {
    let end = text
        .find(|c: char| !(c.is_ascii_digit() || c == '.'))
        .unwrap_or(text.len());
    let amount: f64 = text.get(..end)?.parse().ok()?;
    let unit = text.get(end..)?.trim_start();
    let unit_end = unit
        .find(|c: char| !c.is_ascii_alphanumeric())
        .unwrap_or(unit.len());
    match unit.get(..unit_end)? {
        "ms" => seconds(amount / 1000.0),
        "s" => seconds(amount),
        _ => None,
    }
}

#[cfg(test)]
mod tests;
