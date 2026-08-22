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
            /// Mint a fresh id: unique within this process, greater than every
            /// id minted before it.
            #[allow(clippy::new_without_default)]
            pub fn new() -> Self {
                static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
                // The counter starts at 1 and would take ~585 years at one id
                // per nanosecond to wrap; if it ever does, zero is skipped.
                loop {
                    if let Some(raw) = std::num::NonZeroU64::new(
                        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
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
    /// Identifier of one agent run within this process.
    ///
    /// Every hook event of a run sees the same id, and a host driving many
    /// runs at once routes by it. Minted when a run starts; unique and
    /// increasing within the process, not secret, and meaningful only in this
    /// process — compose it with your own scope (a host id, a session key) to
    /// correlate across processes.
    RunId
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
mod tests {
    use super::*;

    #[test]
    fn default_length_and_alphabet() {
        let id = generate();
        assert_eq!(id.len(), DEFAULT_LEN);
        assert!(id.bytes().all(|b| ALPHABET.contains(&b)));
    }

    #[test]
    fn ids_are_unique() {
        let a = generate();
        let b = generate();
        assert_ne!(a, b);
    }

    #[test]
    fn custom_length() {
        assert_eq!(generate_with_len(8).len(), 8);
    }

    #[test]
    fn run_ids_are_unique_increasing_and_round_trip() {
        let a = RunId::new();
        let b = RunId::new();
        assert_ne!(a, b);
        assert!(b > a);
        assert_eq!(RunId::from_raw(a.to_raw()), Some(a));
        assert_eq!(RunId::from_raw(0), None);
        let id = RunId::from_raw(42).expect("non-zero");
        assert_eq!(id.to_string(), "42");
        assert_eq!("42".parse::<RunId>(), Ok(id));
        assert!("0".parse::<RunId>().is_err());
        assert_eq!(format!("{id:?}"), "RunId(42)");
        assert_eq!(serde_json::to_string(&id).expect("serialize"), "42");
        assert_eq!(
            serde_json::from_str::<RunId>("42").expect("deserialize"),
            id
        );
    }
}
