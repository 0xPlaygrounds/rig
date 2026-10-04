//! The client-side validation surface, exhaustively.
//!
//! Every cell here is free: it never opens a socket, so the full Cartesian
//! product is affordable where a recorded matrix would have to sample.

use super::*;

/// `models/x` from every spelling a caller might reach for.
#[test]
fn model_qualification_is_total_and_idempotent() {
    for (input, expected) in [
        ("gemini-2.5-flash", "models/gemini-2.5-flash"),
        ("models/gemini-2.5-flash", "models/gemini-2.5-flash"),
        ("", "models/"),
        ("models/", "models/"),
        ("tunedModels/x", "models/tunedModels/x"),
    ] {
        assert_eq!(qualify_model(input), expected, "input {input:?}");
    }
}

/// A cache with nothing in it would bill for storage and cache nothing.
#[test]
fn emptiness_is_rejected_but_either_payload_alone_suffices() {
    assert!(
        NewCachedContent::new("gemini-2.5-flash")
            .validate()
            .is_err(),
        "an empty cached content should be refused"
    );
    assert!(
        NewCachedContent::new("gemini-2.5-flash")
            .content("corpus")
            .validate()
            .is_ok()
    );
    assert!(
        NewCachedContent::new("gemini-2.5-flash")
            .system_instruction("be brief")
            .validate()
            .is_ok()
    );
    // Display name and expiry are not payload.
    assert!(
        NewCachedContent::new("gemini-2.5-flash")
            .display_name("x")
            .expiry(CacheExpiry::ttl(Duration::from_secs(60)))
            .validate()
            .is_err()
    );
}
