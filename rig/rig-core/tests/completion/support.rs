//! Shared fixtures and assertions for completion smoke tests.

pub(crate) const PREAMBLE: &str = "You are a concise assistant. Answer directly.";
pub(crate) const PROMPT: &str = "In one or two sentences, explain what Rust programming language is and why memory safety matters.";

pub(crate) fn assert_nontrivial_response(response: &str) {
    let trimmed = response.trim();

    assert!(
        !trimmed.is_empty(),
        "Completion returned an empty or whitespace-only response."
    );
    assert!(
        trimmed.len() > 20,
        "Completion returned a suspiciously short response ({} chars): {:?}",
        trimmed.len(),
        trimmed
    );
}
