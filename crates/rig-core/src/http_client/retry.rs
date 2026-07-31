//! Helpers to handle connection delays when receiving errors

use super::Error;
use std::time::Duration;

/// Exponential-backoff reconnection settings for an SSE event source.
///
/// This is plain configuration the event source owns and consults; there is no
/// policy trait. A server-supplied `retry:` field updates it in place through
/// [`ExponentialBackoff::set_reconnection_time`].
#[derive(Debug, Clone)]
pub struct ExponentialBackoff {
    /// The start of the backoff
    pub start: Duration,
    /// The factor of which to backoff by
    pub factor: f64,
    /// The maximum duration to delay
    pub max_duration: Option<Duration>,
    /// The maximum number of retries before giving up
    pub max_retries: Option<usize>,
}

impl ExponentialBackoff {
    /// Create a new exponential backoff retry policy
    pub const fn new(
        start: Duration,
        factor: f64,
        max_duration: Option<Duration>,
        max_retries: Option<usize>,
    ) -> Self {
        Self {
            start,
            factor,
            max_duration,
            max_retries,
        }
    }

    /// The delay before the next reconnection attempt, or `None` to give up.
    ///
    /// `last_retry` carries the previous attempt's number and delay; `None`
    /// means this is the first attempt of a fresh retry cycle, which uses
    /// [`Self::start`] unchanged.
    pub fn retry(&self, _error: &Error, last_retry: Option<(usize, Duration)>) -> Option<Duration> {
        if let Some((retry_num, last_duration)) = last_retry {
            if self
                .max_retries
                .is_none_or(|max_retries| retry_num < max_retries)
            {
                let duration = last_duration.mul_f64(self.factor);
                if let Some(max_duration) = self.max_duration {
                    Some(duration.min(max_duration))
                } else {
                    Some(duration)
                }
            } else {
                None
            }
        } else {
            Some(self.start)
        }
    }
    /// Adopt a server-supplied reconnection time from an SSE `retry:` field.
    ///
    /// The clamp is widened when the server asks for a longer delay than the
    /// configured maximum, so an explicit server instruction is never shortened.
    pub fn set_reconnection_time(&mut self, duration: Duration) {
        self.start = duration;
        if let Some(max_duration) = self.max_duration {
            self.max_duration = Some(max_duration.max(duration))
        }
    }
}

/// The default backoff settings when initializing an event source
pub const DEFAULT_RETRY: ExponentialBackoff = ExponentialBackoff::new(
    Duration::from_millis(300),
    2.,
    Some(Duration::from_secs(5)),
    None,
);

#[cfg(test)]
mod tests {
    use super::*;

    /// The delay computation ignores the error, so any variant will do.
    fn error() -> Error {
        Error::StreamEnded
    }

    fn backoff(max_duration: Option<Duration>, max_retries: Option<usize>) -> ExponentialBackoff {
        ExponentialBackoff::new(Duration::from_millis(100), 2., max_duration, max_retries)
    }

    #[test]
    fn first_attempt_of_a_cycle_uses_start_unscaled() {
        let policy = backoff(None, None);
        assert_eq!(
            policy.retry(&error(), None),
            Some(Duration::from_millis(100))
        );
    }

    #[test]
    fn subsequent_attempts_scale_by_the_factor() {
        let policy = backoff(None, None);
        assert_eq!(
            policy.retry(&error(), Some((1, Duration::from_millis(100)))),
            Some(Duration::from_millis(200))
        );
        assert_eq!(
            policy.retry(&error(), Some((2, Duration::from_millis(200)))),
            Some(Duration::from_millis(400))
        );
    }

    #[test]
    fn max_duration_clamps_the_scaled_delay() {
        let policy = backoff(Some(Duration::from_millis(250)), None);
        assert_eq!(
            policy.retry(&error(), Some((1, Duration::from_millis(200)))),
            Some(Duration::from_millis(250))
        );
    }

    #[test]
    fn max_retries_ends_the_cycle() {
        let policy = backoff(None, Some(2));
        // Attempts below the limit keep going.
        assert!(
            policy
                .retry(&error(), Some((1, Duration::from_millis(100))))
                .is_some()
        );
        // Reaching the limit gives up rather than delaying again.
        assert_eq!(
            policy.retry(&error(), Some((2, Duration::from_millis(200)))),
            None
        );
        assert_eq!(
            policy.retry(&error(), Some((3, Duration::from_millis(400)))),
            None
        );
        // A fresh cycle (no history) still reconnects.
        assert!(policy.retry(&error(), None).is_some());
    }

    #[test]
    fn server_reconnection_time_replaces_start_and_widens_the_clamp() {
        let mut policy = backoff(Some(Duration::from_secs(1)), None);
        // A server asking for longer than the clamp widens the clamp, so the
        // instruction is never shortened.
        policy.set_reconnection_time(Duration::from_secs(4));
        assert_eq!(policy.start, Duration::from_secs(4));
        assert_eq!(policy.max_duration, Some(Duration::from_secs(4)));
        assert_eq!(policy.retry(&error(), None), Some(Duration::from_secs(4)));

        // A shorter instruction leaves the existing clamp alone.
        policy.set_reconnection_time(Duration::from_millis(500));
        assert_eq!(policy.start, Duration::from_millis(500));
        assert_eq!(policy.max_duration, Some(Duration::from_secs(4)));
    }

    #[test]
    fn default_retry_matches_its_documented_settings() {
        assert_eq!(DEFAULT_RETRY.start, Duration::from_millis(300));
        assert_eq!(DEFAULT_RETRY.factor, 2.);
        assert_eq!(DEFAULT_RETRY.max_duration, Some(Duration::from_secs(5)));
        assert_eq!(DEFAULT_RETRY.max_retries, None);
    }
}
