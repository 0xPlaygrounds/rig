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

    // The sentinel the invariant forbids: not merely discouraged,
    // unrepresentable. Bound first so the diagnostic is the assignment's own
    // type error, with no note pointing into the standard library (whose
    // source is not always available to render).
    let empty_sentinel: Option<String> = Some(String::new());
    response.message_id = empty_sentinel.clone();
    response.response_id = empty_sentinel.clone();
    response.provider_request_id = empty_sentinel.clone();

    // The same holds for the streamed terminal record and the identity
    // carrier the two convert through.
    let mut identity = rig_core::completion::ResponseIdentity::default();
    identity.message_id = empty_sentinel.clone();

    let mut terminal = rig_core::streaming::StreamFinal::new(
        "provider",
        rig_core::completion::Usage::new(),
    );
    terminal.message_id = empty_sentinel;
}
