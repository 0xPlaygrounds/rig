//! `Reasoning::id` is a durable handle, so the empty-string sentinel is
//! unrepresentable there too (#2336).
//!
//! `Reasoning` is persisted in history and replayed upstream, and two provider
//! paths (`xai::api` and `openai::responses_api`) gate on `Some` and send the
//! value straight into a request body. A `Some("")` walked through both and
//! put `{"type":"reasoning","id":""}` on the wire; the field is now an
//! `Option<WireId>`, whose only constructor rejects the empty string, so the
//! assignment below does not compile.
//!
//! The sentinel is bound to a local first so the diagnostic is this
//! assignment's own type error, with no note pointing into the standard
//! library — whose source CI does not have.

fn main() {
    let mut reasoning = rig_core::message::Reasoning::new("thinking");

    let empty_sentinel: Option<String> = Some(String::new());
    reasoning.id = empty_sentinel;
}
