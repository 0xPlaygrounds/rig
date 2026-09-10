//! Lightweight generation of short, unique, URL-safe identifiers.
//!
//! These IDs are used purely to disambiguate things like in-flight tool calls
//! and request headers — they are *not* security-sensitive and do not need a
//! cryptographic source of randomness. We therefore generate them with
//! [`fastrand`] (already a hard dependency of this crate) rather than pulling in
//! `nanoid` → `rand` → `getrandom`, which would add a cryptographic RNG (and a
//! `getrandom/js` shim on wasm) for no benefit here.

/// The URL-safe alphabet used by `nanoid` (`A-Za-z0-9_-`), preserved so the
/// shape of generated IDs is unchanged from the previous implementation.
const ALPHABET: &[u8; 64] = b"_-0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";

/// Default ID length, matching `nanoid`'s default.
const DEFAULT_LEN: usize = 21;

/// Generate a 21-character, URL-safe, non-cryptographic identifier.
///
/// This is a drop-in replacement for the previous `nanoid!()` usage for
/// internal, non-security-sensitive use.
pub fn generate() -> String {
    generate_with_len(DEFAULT_LEN)
}

/// Define a process-unique counter identifier: a `NonZeroU64` minted from a
/// process-global `AtomicU64`, the shape of `std::thread::ThreadId`,
/// `tokio::task::Id` and `tracing::span::Id`. Unique and increasing within the
/// process, `Copy + Hash + Ord`, transparent serde as the integer, decimal
/// `Display`/`FromStr`, `Option<Id>` the size of `u64`. No randomness, no claim
/// beyond the process — a host that correlates across processes composes its
/// own scope. Zero is never issued.
macro_rules! counter_id {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(
            Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
        )]
        #[serde(transparent)]
        pub struct $name(std::num::NonZeroU64);

        impl $name {
            fn counter() -> &'static std::sync::atomic::AtomicU64 {
                static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
                &NEXT
            }

            /// Mint a fresh id: unique within this process, greater than every
            /// id minted before it.
            #[allow(clippy::new_without_default)]
            pub fn new() -> Self {
                // The counter starts at 1 and would take ~585 years at one id
                // per nanosecond to wrap; if it ever does, zero is skipped.
                loop {
                    if let Some(raw) = std::num::NonZeroU64::new(
                        Self::counter().fetch_add(1, std::sync::atomic::Ordering::Relaxed),
                    ) {
                        return Self(raw);
                    }
                }
            }

            /// Advance this process's counter past `raw`, so ids minted after
            /// the call are strictly greater than it. Call it when loading ids
            /// persisted by an earlier process (a resumed run) so fresh mints
            /// cannot collide with ids consumers already saw.
            pub fn advance_past(raw: u64) {
                Self::counter()
                    .fetch_max(raw.saturating_add(1), std::sync::atomic::Ordering::Relaxed);
            }

            /// Build an id from its raw value. `None` for zero, which is never
            /// a valid id.
            pub const fn from_raw(raw: u64) -> Option<Self> {
                match std::num::NonZeroU64::new(raw) {
                    Some(raw) => Some(Self(raw)),
                    None => None,
                }
            }

            /// The id's raw value (never zero).
            pub const fn to_raw(self) -> u64 {
                self.0.get()
            }
        }

        impl From<std::num::NonZeroU64> for $name {
            fn from(raw: std::num::NonZeroU64) -> Self {
                Self(raw)
            }
        }

        impl From<$name> for std::num::NonZeroU64 {
            fn from(id: $name) -> Self {
                id.0
            }
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                std::fmt::Display::fmt(&self.0, f)
            }
        }

        impl std::fmt::Debug for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}({})", stringify!($name), self.0)
            }
        }

        impl std::str::FromStr for $name {
            type Err = ParseIdError;

            fn from_str(s: &str) -> Result<Self, Self::Err> {
                s.parse::<std::num::NonZeroU64>()
                    .map(Self)
                    .map_err(|_| ParseIdError(s.to_owned()))
            }
        }

        const _: () = assert!(
            std::mem::size_of::<Option<$name>>() == std::mem::size_of::<u64>()
        );
    };
}

/// Error from parsing a counter id: the text was not a non-zero decimal `u64`.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("invalid id: expected a non-zero integer, got {0:?}")]
pub struct ParseIdError(String);

counter_id! {
    /// Identifier of one agent run within this process.
    ///
    /// Every hook event of a run sees the same id, and a host driving many
    /// runs at once routes by it. Minted when a run starts; unique and
    /// increasing within the process, not secret, and meaningful only in this
    /// process — compose it with your own scope (a host id, a session key) to
    /// correlate across processes.
    RunId
}

counter_id! {
    /// Rig-generated correlator for one tool call across its stream items:
    /// the id that ties a call's argument deltas to its completed call and
    /// its execution/result items. Minted by the streaming assembler when a
    /// call opens (never provider-supplied); unique and increasing within
    /// the process. Persisted with run state so a resumed run keeps emitting
    /// the ids consumers already saw — a loader of persisted ids must call
    /// [`InternalCallId::advance_past`] so fresh mints cannot collide.
    InternalCallId
}

/// Identifier of a conversation, scoping [`ConversationMemory`] persistence
/// and threading provider-side conversation state across runs.
///
/// String-backed because it is user- and provider-facing (callers bring their
/// own session keys); a newtype so it cannot be confused with the other id
/// strings in flight, and so map keys hash a dedicated type.
///
/// [`ConversationMemory`]: crate::memory::ConversationMemory
#[derive(
    Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
#[serde(transparent)]
pub struct ConversationId(String);

impl ConversationId {
    /// Wrap a caller-supplied conversation key.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// The id as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Unwrap into the underlying string.
    pub fn into_string(self) -> String {
        self.0
    }
}

impl std::fmt::Display for ConversationId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for ConversationId {
    fn from(id: String) -> Self {
        Self(id)
    }
}

impl From<&str> for ConversationId {
    fn from(id: &str) -> Self {
        Self(id.to_owned())
    }
}

impl AsRef<str> for ConversationId {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

/// Generate a `len`-character, URL-safe, non-cryptographic identifier.
pub fn generate_with_len(len: usize) -> String {
    std::iter::repeat_with(|| {
        let idx = fastrand::usize(..ALPHABET.len());
        // `idx` is always in bounds, but use `get` to avoid the `indexing_slicing` lint.
        ALPHABET.get(idx).copied().unwrap_or(b'_') as char
    })
    .take(len)
    .collect()
}

#[cfg(test)]
mod tests;
