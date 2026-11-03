//! What to do about a failed model call: wait and send it again after a
//! transient failure, honouring the wait the provider asked for; recover
//! from a request that outgrew the model's context window; or give up.
//! The kernel reads its policy from the [`RetryPolicy`] resource, which
//! rig's `[retry]` table sets.

use std::time::{Duration, SystemTime};

use bevy_ecs::prelude::Resource;
use rig_core::error::{ErrorKind, ErrorReport};

/// How often and how long a failed call is retried: a resource the kernel
/// reads each time a call fails.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq)]
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

impl Default for RetryPolicy {
    fn default() -> Self {
        Self::DEFAULT
    }
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
            let most = self.max_retries;
            let times = match retries + 1 {
                1 => "once".to_owned(),
                times => format!("{times} times in a row"),
            };
            return Verdict::GaveUp(format!(
                "it failed {times} (`[retry] max_retries` is {most})"
            ));
        }
        match report.retry_after(now) {
            Some(asked) if asked > self.max_asked_wait => Verdict::GaveUp(format!(
                "the provider asks to wait {}s before the next call (`[retry] max_wait_secs` \
                 is {})",
                asked.as_secs(),
                self.max_asked_wait.as_secs()
            )),
            Some(asked) if web_time::Instant::now().checked_add(asked).is_none() => {
                Verdict::GaveUp(format!(
                    "the provider asks to wait {}s before the next call, longer than this \
                     machine's clock counts",
                    asked.as_secs()
                ))
            }
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

#[cfg(test)]
mod tests;
