//! Identity of one agent run.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

/// Identifier of one agent run: 128 random bits, `Copy`, hashable,
/// serializable.
///
/// Every hook event of a run sees the same id through
/// [`HookContext::run_id`](crate::agent::HookContext::run_id), and every
/// [`RunEvent`](crate::agent::RunEvent) on a [`run_channel`](crate::agent::AgentRunner::run_channel)
/// feed carries it, so a host driving many runs at once can route each event
/// to the run (entity, job, session) it belongs to without a side table.
///
/// The id is process-minted and not secret. A host that wants to choose the
/// id itself — to match an entity or job it already created — passes one with
/// [`AgentRunner::with_run_id`](crate::agent::AgentRunner::with_run_id)
/// before the run starts; otherwise one is minted when the run starts.
///
/// Serializes as a plain integer (`#[serde(transparent)]` over `u128`);
/// [`Display`](fmt::Display) renders 32 lowercase hex digits, which
/// [`FromStr`] accepts back.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RunId(u128);

impl RunId {
    /// Mint a fresh, random id.
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self(rig_core::id::random_u128())
    }

    /// Build an id from its 128-bit representation.
    pub const fn from_bits(bits: u128) -> Self {
        Self(bits)
    }

    /// The id's 128-bit representation.
    pub const fn to_bits(self) -> u128 {
        self.0
    }
}

impl fmt::Display for RunId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:032x}", self.0)
    }
}

impl fmt::Debug for RunId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RunId({self})")
    }
}

/// Error from parsing a [`RunId`]: the text was not 32 hex digits.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("invalid run id: expected 32 hex digits, got {0:?}")]
pub struct ParseRunIdError(String);

impl FromStr for RunId {
    type Err = ParseRunIdError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.len() != 32 {
            return Err(ParseRunIdError(s.to_owned()));
        }
        u128::from_str_radix(s, 16)
            .map(Self)
            .map_err(|_| ParseRunIdError(s.to_owned()))
    }
}

const _: () = assert!(std::mem::size_of::<RunId>() == 16);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bits_round_trip() {
        let id = RunId::from_bits(0x0123_4567_89ab_cdef_fedc_ba98_7654_3210);
        assert_eq!(RunId::from_bits(id.to_bits()), id);
    }

    #[test]
    fn display_and_parse_round_trip() {
        let id = RunId::new();
        let text = id.to_string();
        assert_eq!(text.len(), 32);
        assert_eq!(text.parse::<RunId>(), Ok(id));
        assert_eq!(RunId::from_bits(1).to_string(), format!("{:032x}", 1));
    }

    #[test]
    fn parse_rejects_bad_text() {
        assert!("abc".parse::<RunId>().is_err());
        assert!("zz".repeat(16).parse::<RunId>().is_err());
    }

    #[test]
    fn serde_round_trip_is_transparent() {
        let id = RunId::from_bits(42);
        let json = serde_json::to_string(&id).expect("serialize");
        assert_eq!(json, "42");
        assert_eq!(
            serde_json::from_str::<RunId>(&json).expect("deserialize"),
            id
        );
    }

    #[test]
    fn new_ids_differ() {
        assert_ne!(RunId::new(), RunId::new());
    }
}
