//! Which saved log a conversation continues.

use serde::{Deserialize, Serialize};

/// The saved log a [`Conversation`](super::Conversation) continues: a token
/// a store chose when it started that log, carried by every value loaded
/// from it or saved to it. A store appends a value's unsaved changes to a
/// log only when the value holds what that log holds
/// ([`Conversation::continues`](super::Conversation::continues)): its
/// lineage, and as many of its changes. The count alone cannot tell two
/// conversations apart.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Lineage(u64);

impl Lineage {
    /// A conversation begun here from nothing and never saved: it continues
    /// any empty log.
    pub const EMPTY: Self = Self(0);

    /// A conversation read back from its serde form, which does not carry
    /// its lineage: it continues no log, so saving it writes it whole.
    pub const DETACHED: Self = Self(u64::MAX);

    /// The lineage of a log just started: neither [`EMPTY`](Self::EMPTY)
    /// nor [`DETACHED`](Self::DETACHED), and unlike any other log's but by
    /// a one in 2^64 chance.
    pub fn fresh() -> Self {
        let now = web_time::SystemTime::now()
            .duration_since(web_time::UNIX_EPOCH)
            .unwrap_or_default();
        let clock = now.as_secs().rotate_left(32) ^ u64::from(now.subsec_nanos());
        Self((fastrand::u64(..) ^ clock).clamp(1, u64::MAX - 1))
    }

    /// Whether this is a log's lineage: neither [`EMPTY`](Self::EMPTY) nor
    /// [`DETACHED`](Self::DETACHED).
    pub fn is_log(self) -> bool {
        self != Self::EMPTY && self != Self::DETACHED
    }
}
