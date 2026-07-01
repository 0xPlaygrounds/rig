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
}
