//! The empty-string-is-absent rule on the response metadata types is
//! structural, not advisory (#2336).
//!
//! `StreamFinal` and `completion::CompletionResponse` normalize an empty
//! identifier to `None` in their `with_*_id` setters. The fields are public,
//! so before the identifier types became `Option<WireId>` a caller could
//! assign `Some(String::new())` directly and reintroduce the sentinel the
//! setters exist to remove. Now the assignment does not compile, and
//! `WireId::new` — whose only path rejects the empty string — is the sole way
//! in.

fn main() {
    let mut response = rig_core::completion::CompletionResponse::new(
        Vec::new(),
        rig_core::completion::Usage::new(),
        "provider",
    );

    // The sentinel the invariant forbids: not merely discouraged, unrepresentable.
    response.message_id = Some(String::new());
}
