use super::*;

fn scrub(text: &str) -> String {
    CassetteScrubber::new(CassettePolicy::default()).scrub_text(text)
}

/// The handle is account-scoped and server-generated, and it rides in
/// request bodies, request paths and response bodies alike.
#[test]
fn a_cached_content_handle_is_placeholdered() {
    let scrubbed = scrub(r#"{"cachedContent":"cachedContents/n3v1qk0nqz9k"}"#);
    assert!(!scrubbed.contains("n3v1qk0nqz9k"), "{scrubbed}");
    assert!(scrubbed.contains("cachedContents/"), "{scrubbed}");
}

/// Equal originals must map to equal placeholders, or a request body and the
/// path it was sent to stop agreeing and replay cannot match.
#[test]
fn the_same_handle_scrubs_to_the_same_placeholder() {
    let scrubbed = scrub(
        "/v1beta/cachedContents/abc123def and {\"cachedContent\":\"cachedContents/abc123def\"}",
    );
    let placeholders: Vec<&str> = scrubbed
        .match_indices("cachedContents/")
        .map(|(i, _)| {
            let rest = &scrubbed[i + "cachedContents/".len()..];
            let end = rest
                .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_' || c == '-'))
                .unwrap_or(rest.len());
            &rest[..end]
        })
        .collect();
    assert_eq!(placeholders.len(), 2, "{scrubbed}");
    assert_eq!(placeholders[0], placeholders[1], "{scrubbed}");
    assert!(!placeholders[0].contains("abc123def"), "{scrubbed}");
}

/// The collection endpoint has no id; scrubbing it would break the recorded
/// request path.
#[test]
fn the_bare_collection_path_is_untouched() {
    assert_eq!(
        scrub("/v1beta/cachedContents?pageSize=1000"),
        "/v1beta/cachedContents?pageSize=1000"
    );
}
