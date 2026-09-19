use super::*;

fn scrub(text: &str) -> String {
    CassetteScrubber::new(CassettePolicy::default()).scrub_text(text)
}

/// `token_end` accepts offset 0 unconditionally, so a non-token character
/// straight after the collection prefix used to come back as the "id" —
/// swallowing a closing quote and writing invalid JSON into a fixture.
#[test]
fn a_handle_with_no_id_does_not_eat_the_next_character() {
    assert_eq!(
        scrub(r#"{"cachedContent":"cachedContents/"}"#),
        r#"{"cachedContent":"cachedContents/"}"#
    );
}

/// The same shape in prose — a provider error quoting the handle template.
#[test]
fn a_handle_placeholder_in_prose_is_left_alone() {
    assert_eq!(
        scrub("expected cachedContents/<id>"),
        "expected cachedContents/<id>"
    );
}
