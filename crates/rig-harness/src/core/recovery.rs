//! What a turn does when a model call fails: wait and call again after a
//! transient failure, honouring the provider's `Retry-After`; clear old
//! tool outputs and call again when the conversation outgrew the model's
//! context window. Each turn counts its attempts in its [`Recovery`]; a
//! wait is a call entity of the turn with a [`Backoff`], so interrupting
//! the turn cancels it like any other call.

use std::hash::{BuildHasher, RandomState};
use std::sync::LazyLock;
use std::time::{Duration, Instant};

use bevy_ecs::prelude::*;
use regex::Regex;
use rig_core::completion::Message;
use rig_core::error::{ErrorKind, ErrorReport};
use rig_core::message::{ToolResultContent, UserContent};

use super::compaction::estimate_content;

/// Retries of failed model calls in a row, per turn. A reply resets it.
pub const MAX_RETRIES: u32 = 4;
/// The first wait; each retry doubles it, up to [`MAX_BACKOFF`].
const FIRST_BACKOFF: Duration = Duration::from_secs(2);
/// The longest wait rig-harness picks itself.
const MAX_BACKOFF: Duration = Duration::from_secs(30);
/// The longest wait a provider may ask for; past it the turn fails, as
/// waiting longer in front of the user is worse than saying so (pi's
/// `DEFAULT_MAX_RETRY_DELAY_MS`).
const MAX_ASKED_WAIT: Duration = Duration::from_secs(60);
/// Times one turn clears tool outputs to fit the context window.
pub const MAX_CLEARINGS: u32 = 2;
/// Tokens of the newest tool outputs a first clearing keeps (opencode's
/// `PRUNE_PROTECT`).
pub const KEEP_RECENT_OUTPUTS: u64 = 40_000;
/// What a cleared tool output says instead.
pub const CLEARED: &str =
    "[output cleared to fit the context window; run the tool again if needed]";

/// A turn's recovery so far: the failed calls retried since its last
/// reply, how often it cleared tool outputs and how often it compacted
/// the conversation.
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct Recovery {
    /// Retries since the last reply.
    pub retries: u32,
    /// Clearings in this turn.
    pub clearings: u32,
    /// Compactions in this turn.
    pub compactions: u32,
}

/// A wait before the turn's next model call, on a call entity of the turn;
/// its task is a [`Running<RetryDue>`](super::calls::Running).
#[derive(Component, Clone, Debug)]
pub struct Backoff {
    /// Which retry this wait is for, from 1 to [`MAX_RETRIES`].
    pub attempt: u32,
    /// When the call is sent again.
    pub until: Instant,
    /// Why the last call failed.
    pub why: String,
}

impl Backoff {
    /// Whole seconds left to wait, rounded up.
    pub fn seconds_left(&self) -> u64 {
        let left = self.until.saturating_duration_since(Instant::now());
        left.as_secs() + u64::from(left.subsec_nanos() > 0)
    }
}

/// What a [`Backoff`]'s task returns once its wait is over.
pub struct RetryDue;

/// Waits `delay`, on futures-timer's timer thread: no pool thread sleeps,
/// and dropping the task stops the wait.
pub(crate) async fn wait(delay: Duration) -> RetryDue {
    futures_timer::Delay::new(delay).await;
    RetryDue
}

/// What to do about a failed model call.
#[derive(Debug, PartialEq, Eq)]
pub enum Verdict {
    /// The conversation does not fit the model's window: clear or compact,
    /// and retry.
    Overflow,
    /// Wait this long, then send the same request again.
    Retry(Duration),
    /// Transient, but retrying is not worth it: the retries ran out, or the
    /// provider asks for a wait longer than a minute. Nothing is
    /// wrong with the request, so the user's message is kept.
    GaveUp(String),
    /// The same request fails again: the turn fails.
    Final,
}

/// Decides what to do about `report`, the failure of a turn whose
/// [`Recovery`] counted `retries` so far.
pub fn verdict(report: &ErrorReport, retries: u32) -> Verdict {
    if is_overflow(report) {
        return Verdict::Overflow;
    }
    if !report.is_retryable() || report.kind == ErrorKind::Cancelled {
        return Verdict::Final;
    }
    if retries >= MAX_RETRIES {
        return Verdict::GaveUp(format!("it failed {} times in a row", retries + 1));
    }
    match asked_wait(report) {
        Some(asked) if asked > MAX_ASKED_WAIT => Verdict::GaveUp(format!(
            "the provider asks to wait {}s before the next call",
            asked.as_secs()
        )),
        Some(asked) => Verdict::Retry(asked),
        None => Verdict::Retry(backoff(retries)),
    }
}

/// The wait before retry `retries + 1`: doubling from [`FIRST_BACKOFF`],
/// capped, less up to a quarter at random so that agents failing together
/// do not retry together.
fn backoff(retries: u32) -> Duration {
    let doubled = FIRST_BACKOFF.saturating_mul(1 << retries.min(16));
    let wait = doubled.min(MAX_BACKOFF);
    let jitter = RandomState::new().hash_one(retries) % 250;
    wait.saturating_sub(wait * u32::try_from(jitter).unwrap_or(0) / 1000)
}

/// The wait the provider asked for: the `retry-after-ms` header (OpenAI),
/// else `retry-after` in seconds or as an HTTP date, else a hint in the
/// body: Gemini's `retryDelay` or OpenAI's "try again in 1.5s".
fn asked_wait(report: &ErrorReport) -> Option<Duration> {
    let header = |name: &str| {
        report
            .provider_response_headers()
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
            Err(_) => chrono::DateTime::parse_from_rfc2822(value).ok().map(|at| {
                (at.to_utc() - chrono::Utc::now())
                    .to_std()
                    .unwrap_or_default()
            }),
        };
    }
    let body = report.provider_response_body().unwrap_or(&report.message);
    let captures = RETRY_HINT.as_ref()?.captures(body)?;
    let amount: f64 = captures.name("amount")?.as_str().parse().ok()?;
    match captures.name("unit").map(|unit| unit.as_str()) {
        Some("ms") => seconds(amount / 1000.0),
        _ => seconds(amount),
    }
}

/// `secs` as a duration, when it is a sane one.
fn seconds(secs: f64) -> Option<Duration> {
    Duration::try_from_secs_f64(secs.max(0.0)).ok()
}

/// A wait the body of a failure names.
static RETRY_HINT: LazyLock<Option<Regex>> = LazyLock::new(|| {
    Regex::new(
        r#"(?i)(?:"retryDelay"\s*:\s*"|try again in )(?P<amount>\d+(?:\.\d+)?)\s*(?P<unit>ms|s)\b"#,
    )
    .ok()
});

/// Whether `report` says the request does not fit the model's context
/// window: a 413, or the provider saying so in its own words, in the
/// forms pi knows (`references/pi/packages/ai/src/utils/overflow.ts:37-63`),
/// less rate limits that happen to mention tokens.
pub fn is_overflow(report: &ErrorReport) -> bool {
    if report.code.as_deref() == Some("context_length_exceeded") {
        return true;
    }
    // A full context fails the same way every time; a failure worth
    // retrying (Bedrock's "too many tokens" throttling) is not one.
    if report.is_retryable() {
        return false;
    }
    let texts = [
        Some(report.message.as_str()),
        report.provider_response_body(),
    ];
    let matches = |pattern: &LazyLock<Option<Regex>>| {
        pattern
            .as_ref()
            .is_some_and(|pattern| texts.iter().flatten().any(|text| pattern.is_match(text)))
    };
    report.http_status == Some(413) || (matches(&OVERFLOW) && !matches(&NOT_OVERFLOW))
}

/// The ways providers say the context is full.
static OVERFLOW: LazyLock<Option<Regex>> = LazyLock::new(|| {
    Regex::new(concat!(
        r"(?i)prompt (?:is )?too long|prompt exceeds max length|request_too_large",
        r"|input is too long for requested model|exceeds the context window",
        r"|exceeds (?:the )?(?:model'?s )?maximum context length",
        r"|input token count.*exceeds the maximum|maximum prompt length is \d+",
        r"|reduce the length of the messages|maximum context length is \d+ tokens",
        r"|exceeds (?:the )?maximum allowed input length|is longer than the model'?s context length",
        r"|exceeds the limit of \d+|exceeds the available context size",
        r"|greater than the context length|context window exceeds limit",
        r"|exceeded model token limit|too large for model with \d+ maximum context length",
        r"|but the configured context size is|model_context_window_exceeded",
        r"|exceeded (?:max )?context length|range of input length should be",
        r"|context[_ ]length[_ ]exceeded|too many tokens|token limit exceeded",
    ))
    .ok()
});

/// Rate limits and outages whose words also match [`OVERFLOW`].
static NOT_OVERFLOW: LazyLock<Option<Regex>> = LazyLock::new(|| {
    Regex::new(r"(?i)^(?:throttling error|service unavailable):|rate limit|too many requests").ok()
});

/// What a clearing took out.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Cleared {
    /// Tool results cleared.
    pub results: usize,
    /// Their estimated tokens.
    pub tokens: u64,
}

/// Clears the outputs of older tool calls in `messages` to free context,
/// newest kept first: walking back from the end, the outputs within the
/// first `keep` estimated tokens stay, and every older one is replaced by
/// [`CLEARED`]. The last message is never touched: it is what the model
/// must answer. Error results stay, being short and telling the model what
/// went wrong. The calls themselves stay, so the conversation keeps its
/// shape (opencode's `prune`,
/// `references/opencode/packages/opencode/src/session/compaction.ts:271-316`).
pub fn clear_tool_outputs(messages: &mut [Message], keep: u64) -> Cleared {
    let mut cleared = Cleared::default();
    let mut kept = 0u64;
    let Some((_, earlier)) = messages.split_last_mut() else {
        return cleared;
    };
    for message in earlier.iter_mut().rev() {
        let Message::User { content } = message else {
            continue;
        };
        for item in content.iter_mut().rev() {
            let tokens = estimate_content(item);
            let UserContent::ToolResult(result) = item else {
                continue;
            };
            if result.is_error || is_cleared(&result.content) {
                continue;
            }
            kept += tokens;
            if kept <= keep {
                continue;
            }
            result.content = vec![ToolResultContent::text(CLEARED)];
            cleared.results += 1;
            cleared.tokens += tokens;
        }
    }
    cleared
}

/// How much a recovering turn keeps on its `clearing`th clearing: the
/// newest outputs at first, nothing after.
pub fn keep_for(clearing: u32) -> u64 {
    if clearing == 0 {
        KEEP_RECENT_OUTPUTS
    } else {
        0
    }
}

fn is_cleared(content: &[ToolResultContent]) -> bool {
    matches!(content, [only] if only.as_text() == Some(CLEARED))
}
