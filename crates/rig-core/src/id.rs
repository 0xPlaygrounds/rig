//! Non-cryptographic random identifiers, process-local run counters, and
//! caller-supplied conversation keys. Do not use generated IDs as secrets.
//!
//! ```
//! let id = rig_core::id::generate();
//! assert_eq!(id.len(), 21);
//! ```

/// URL-safe ASCII alphabet for random identifiers.
const ALPHABET: &[u8; 64] = b"_-0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";

/// Number of characters returned by [`generate`].
const DEFAULT_LEN: usize = 21;

/// Generate a 21-character URL-safe random identifier. Not cryptographically
/// secure; collisions are possible.
pub fn generate() -> String {
    generate_with_len(DEFAULT_LEN)
}

/// Define a nonzero process-local counter with transparent integer serde and
/// decimal parsing/display. Values increase until wraparound; raw construction
/// and deserialization do not reserve counter values.
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

            /// Mint the next nonzero process-local counter value. Values may repeat
            /// after wraparound or collide with raw or deserialized IDs.
            #[allow(clippy::new_without_default)]
            pub fn new() -> Self {
                // Zero is reserved even after counter wraparound.
                loop {
                    if let Some(raw) = std::num::NonZeroU64::new(
                        Self::counter().fetch_add(1, std::sync::atomic::Ordering::Relaxed),
                    ) {
                        return Self(raw);
                    }
                }
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
    /// Process-local run identifier shared by that run's hook events.
    /// Not secret; add a host or session scope for cross-process correlation.
    RunId
}

/// Caller-supplied key scoping [`ConversationMemory`] and provider conversation
/// state across runs. Wraps the string without validation.
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
pub(crate) fn generate_with_len(len: usize) -> String {
    std::iter::repeat_with(|| {
        let idx = fastrand::usize(..ALPHABET.len());
        ALPHABET.get(idx).copied().unwrap_or(b'_') as char
    })
    .take(len)
    .collect()
}

#[cfg(test)]
mod tests;
