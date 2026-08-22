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

/// A process-unique 128-bit identifier: `process_nonce << 64 | sequence`.
///
/// The high word is drawn once per process (from the same non-cryptographic
/// source as [`generate`]); the low word is a monotonically increasing
/// counter. Two calls in one process therefore never collide, ids from one
/// process are ordered by creation, and ids from different processes differ
/// in their high word with the same odds as a 64-bit random value — which is
/// what a routing key for in-flight work (a run, a job) needs; it is not a
/// secret and makes no cryptographic claim.
pub fn next_u128() -> u128 {
    use std::sync::LazyLock;
    use std::sync::atomic::{AtomicU64, Ordering};

    static PROCESS_NONCE: LazyLock<u64> = LazyLock::new(|| fastrand::u64(..));
    static SEQUENCE: AtomicU64 = AtomicU64::new(0);

    let sequence = SEQUENCE.fetch_add(1, Ordering::Relaxed);
    (u128::from(*PROCESS_NONCE) << 64) | u128::from(sequence)
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
    fn next_u128_is_unique_ordered_and_process_stamped() {
        let a = next_u128();
        let b = next_u128();
        assert_ne!(a, b);
        assert!(b > a, "ids are ordered by creation within a process");
        assert_eq!(a >> 64, b >> 64, "same process, same nonce");
        assert_eq!((b & u128::from(u64::MAX)) - (a & u128::from(u64::MAX)), 1);
    }
}
