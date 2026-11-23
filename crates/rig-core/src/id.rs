//! Non-cryptographic random identifiers and process-local run counters. Do
//! not use generated IDs as secrets.
//!
//! ```
//! let id = rig_core::id::generate();
//! assert_eq!(id.len(), 21);
//! ```

use serde::{Deserialize, Serialize};
use std::{
    fmt,
    num::NonZeroU64,
    sync::atomic::{AtomicU64, Ordering},
};

/// URL-safe ASCII alphabet for random identifiers.
const ALPHABET: &[u8; 64] = b"_-0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";

/// Number of characters returned by [`generate`].
const DEFAULT_LEN: usize = 21;

/// Generate a 21-character URL-safe random identifier. Not cryptographically
/// secure; collisions are possible.
pub fn generate() -> String {
    std::iter::repeat_with(|| {
        let idx = fastrand::usize(..ALPHABET.len());
        ALPHABET.get(idx).copied().unwrap_or(b'_') as char
    })
    .take(DEFAULT_LEN)
    .collect()
}

/// Process-local run identifier shared by that run's hook events, a nonzero
/// counter serialized as its number. Not secret; add a host or session scope
/// for cross-process correlation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RunId(NonZeroU64);

const _: () = assert!(std::mem::size_of::<Option<RunId>>() == std::mem::size_of::<u64>());

impl RunId {
    /// Mint the next run id. Values increase until wraparound and may then
    /// repeat or collide with deserialized ids.
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        // Zero is skipped, also after wraparound.
        loop {
            if let Some(raw) = NonZeroU64::new(NEXT.fetch_add(1, Ordering::Relaxed)) {
                return Self(raw);
            }
        }
    }
}

impl fmt::Display for RunId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

#[cfg(test)]
mod tests;
