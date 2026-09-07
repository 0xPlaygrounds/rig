use super::*;

/// Gemini's `cachedContents` cursors are base64 protobuf carrying the real
/// resource ids of the entries either side of the page boundary — so a
/// recorded cursor re-exposed ids that were placeholdered everywhere else in
/// the same fixture.
#[test]
fn a_next_page_token_is_placeholdered_in_the_body() {
    let scrubbed = CassetteScrubber::new(CassettePolicy::default())
        .scrub_body(r#"{"nextPageToken":"cjwKDoIBCwifnJDUBhDo0c8VCipCKHZi"}"#);
    assert!(
        !scrubbed.contains("cjwKDoIBCwifnJDUBhDo0c8VCipCKHZi"),
        "{scrubbed}"
    );
    assert!(scrubbed.contains("nextPageToken"), "{scrubbed}");
}

/// Distinct cursors must stay distinct: a loop that follows them compares
/// each against the last to spot a server that stops advancing, so
/// collapsing them to one value would end that loop early on replay.
#[test]
fn distinct_cursors_get_distinct_placeholders() {
    let mut scrubber = CassetteScrubber::new(CassettePolicy::default());
    let first = scrubber.placeholder("cursor-one-original", "cursor-");
    let second = scrubber.placeholder("cursor-two-original", "cursor-");
    let first_again = scrubber.placeholder("cursor-one-original", "cursor-");

    assert_ne!(first, second, "different cursors must not collapse");
    assert_eq!(
        first, first_again,
        "the same cursor must map to the same placeholder, or the request that \
         replays it stops matching the response that issued it"
    );
}
