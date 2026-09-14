//! Absolute home-directory paths must not reach a committed fixture.
//!
//! A locally-hosted provider echoes the path it was launched with:
//! `llama-server` puts the full GGUF path in `model` on every chat
//! response. Nothing else in the scrubber reaches it, and no
//! `FORBIDDEN_CASSETTE_PATTERNS` entry trips on it, so without this rule the
//! safety scan passes a fixture carrying the operator's username.

use super::scrub_local_filesystem_paths as scrub;

#[test]
fn a_home_path_keeps_only_its_basename() {
    let scrubbed = scrub(
        "/Users/someone/.cache/huggingface/hub/models--org--repo/snapshots/abc123/Model-Q8_0.gguf",
    );
    assert_eq!(scrubbed, "/REDACTED_PATH/Model-Q8_0.gguf");
    assert!(
        !scrubbed.contains("someone"),
        "the username must not survive"
    );
    assert!(
        !scrubbed.contains("abc123"),
        "the snapshot hash must not survive"
    );
}

#[test]
fn linux_and_root_homes_are_covered_too() {
    assert_eq!(scrub("/home/dev/models/m.gguf"), "/REDACTED_PATH/m.gguf");
    assert_eq!(scrub("/root/m.gguf"), "/REDACTED_PATH/m.gguf");
}

/// The path sits inside a JSON string in practice, so the rule has to stop
/// at the closing quote rather than swallowing the rest of the body.
#[test]
fn a_path_inside_json_stops_at_the_quote() {
    let scrubbed = scrub(r#"{"model":"/Users/a/b/m.gguf","object":"chat.completion"}"#);
    assert_eq!(
        scrubbed,
        r#"{"model":"/REDACTED_PATH/m.gguf","object":"chat.completion"}"#
    );
}

/// Several paths in one body are all replaced.
#[test]
fn every_occurrence_is_replaced() {
    let scrubbed = scrub(r#"{"a":"/Users/x/one.gguf","b":"/Users/y/two.gguf"}"#);
    assert!(!scrubbed.contains("/Users/"), "{scrubbed}");
    assert!(
        scrubbed.contains("one.gguf") && scrubbed.contains("two.gguf"),
        "{scrubbed}"
    );
}

/// A home directory containing a space is the case that makes whitespace a
/// forbidden terminator.
///
/// macOS creates `/Users/John Smith` from a full account name. Ending the
/// path at the space kept `John` as the basename and copied `Smith/...`
/// through untouched — the operator's name in the fixture, in output the
/// rule then considers already-scrubbed, so the safety scan reports nothing.
#[test]
fn a_home_directory_containing_a_space_is_fully_replaced() {
    let scrubbed = scrub(r#"{"model":"/Users/John Smith/models/m.gguf"}"#);
    assert_eq!(scrubbed, r#"{"model":"/REDACTED_PATH/m.gguf"}"#);
    assert!(!scrubbed.contains("John"), "{scrubbed}");
    assert!(!scrubbed.contains("Smith"), "{scrubbed}");
}

/// Re-scrubbing scrubbed output must not change it — the cassette safety
/// check re-scrubs its own output, so a non-idempotent rule fails there.
#[test]
fn the_rule_is_idempotent() {
    let once = scrub("/Users/a/.cache/m.gguf");
    assert_eq!(scrub(&once), once);
}

/// A URL that merely *contains* one of these segments is not a local path
/// and must survive verbatim.
///
/// This is not hypothetical: anthropic's recorded web-search fixtures cite
/// `https://math.ucr.edu/home/baez/physics/...`, and an unanchored rule
/// rewrote it — corrupting three committed fixtures and failing the safety
/// scan with "not in scrubbed cassette form".
#[test]
fn a_url_containing_a_home_segment_is_not_a_local_path() {
    let body = r#"{"url":"https://math.ucr.edu/home/baez/physics/General/BlueSky/blue_sky.html"}"#;
    assert_eq!(scrub(body), body);
    let body = r#"{"url":"https://example.com/Users/profile.html"}"#;
    assert_eq!(scrub(body), body);
}

/// The anchored rule still catches the real case sitting next to a URL.
#[test]
fn a_local_path_beside_a_url_is_still_scrubbed() {
    let scrubbed =
        scrub(r#"{"url":"https://math.ucr.edu/home/baez/x.html","model":"/Users/me/m.gguf"}"#);
    assert!(
        scrubbed.contains("math.ucr.edu/home/baez/x.html"),
        "{scrubbed}"
    );
    assert!(
        scrubbed.contains(r#""model":"/REDACTED_PATH/m.gguf""#),
        "{scrubbed}"
    );
}

/// Text carrying no path is untouched, so the rule cannot churn unrelated
/// fixtures.
#[test]
fn unrelated_text_is_untouched() {
    let body = r#"{"model":"gpt-4o-mini","choices":[]}"#;
    assert_eq!(scrub(body), body);
}
