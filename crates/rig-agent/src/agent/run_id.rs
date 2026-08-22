//! Identity of one agent run.

use std::fmt;
use std::num::NonZeroU64;
use std::str::FromStr;
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};

/// Identifier of one agent run within this process.
///
/// A `Copy`, hashable, ordered, serializable handle id minted from a global
/// counter — the same shape as `std::thread::ThreadId`, `tokio::task::Id` or
/// `tracing::span::Id`: unique for the life of the process, increasing in
/// minting order, and making no claim beyond that. A host that correlates
/// runs across processes or machines composes it with its own scope (a host
/// id, a session key) rather than relying on the id alone. It is not secret.
///
/// Every hook event of a run sees the same id through
/// [`HookContext::run_id`](crate::agent::HookContext::run_id), and every
/// [`RunEvent`](crate::agent::RunEvent) on a [`run_channel`](crate::agent::AgentRunner::run_channel)
/// feed carries it, so a host driving many runs at once can route each event
/// to the run (entity, job, session) it belongs to without a side table.
///
/// A host that already has an identity for the work passes its own with
/// [`AgentRunner::with_run_id`](crate::agent::AgentRunner::with_run_id)
/// before the run starts; otherwise one is minted when the run starts.
///
/// Serializes as a plain non-zero integer (`#[serde(transparent)]` over
/// `NonZeroU64`); [`Display`](fmt::Display) renders the decimal number, which
/// [`FromStr`] accepts back. `Option<RunId>` is the size of `RunId`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RunId(NonZeroU64);

static NEXT: AtomicU64 = AtomicU64::new(1);

impl RunId {
    /// Mint a fresh id: unique within this process, greater than every id
    /// minted before it.
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        // The counter starts at 1 and would take ~585 years at one id per
        // nanosecond to wrap; if it ever does, zero is skipped rather than
        // handed out as an id.
        loop {
            if let Some(raw) = NonZeroU64::new(NEXT.fetch_add(1, Ordering::Relaxed)) {
                return Self(raw);
            }
        }
    }

    /// Build an id from its raw value. `None` for zero, which is never a
    /// valid id.
    pub const fn from_raw(raw: u64) -> Option<Self> {
        match NonZeroU64::new(raw) {
            Some(raw) => Some(Self(raw)),
            None => None,
        }
    }

    /// The id's raw value (never zero).
    pub const fn to_raw(self) -> u64 {
        self.0.get()
    }
}

impl From<NonZeroU64> for RunId {
    fn from(raw: NonZeroU64) -> Self {
        Self(raw)
    }
}

impl From<RunId> for NonZeroU64 {
    fn from(id: RunId) -> Self {
        id.0
    }
}

impl fmt::Display for RunId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

impl fmt::Debug for RunId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RunId({})", self.0)
    }
}

/// Error from parsing a [`RunId`]: the text was not a non-zero decimal `u64`.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("invalid run id: expected a non-zero integer, got {0:?}")]
pub struct ParseRunIdError(String);

impl FromStr for RunId {
    type Err = ParseRunIdError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        s.parse::<NonZeroU64>()
            .map(Self)
            .map_err(|_| ParseRunIdError(s.to_owned()))
    }
}

const _: () = assert!(std::mem::size_of::<Option<RunId>>() == std::mem::size_of::<u64>());

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_ids_are_unique_and_increasing() {
        let a = RunId::new();
        let b = RunId::new();
        assert_ne!(a, b);
        assert!(b > a);
    }

    #[test]
    fn raw_round_trip_and_zero_is_rejected() {
        let id = RunId::new();
        assert_eq!(RunId::from_raw(id.to_raw()), Some(id));
        assert_eq!(RunId::from_raw(0), None);
        assert_eq!(RunId::from_raw(7).map(RunId::to_raw), Some(7));
    }

    #[test]
    fn display_and_parse_round_trip() {
        let id = RunId::from_raw(42).expect("non-zero");
        assert_eq!(id.to_string(), "42");
        assert_eq!("42".parse::<RunId>(), Ok(id));
        assert!("0".parse::<RunId>().is_err());
        assert!("abc".parse::<RunId>().is_err());
        assert_eq!(format!("{id:?}"), "RunId(42)");
    }

    #[test]
    fn serde_round_trip_is_transparent() {
        let id = RunId::from_raw(42).expect("non-zero");
        let json = serde_json::to_string(&id).expect("serialize");
        assert_eq!(json, "42");
        assert_eq!(
            serde_json::from_str::<RunId>(&json).expect("deserialize"),
            id
        );
        assert!(serde_json::from_str::<RunId>("0").is_err());
    }
}
