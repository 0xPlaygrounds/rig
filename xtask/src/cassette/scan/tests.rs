use super::*;

fn kinds(text: &str) -> Vec<&'static str> {
    scan(text, &[]).into_iter().map(|hit| hit.kind).collect()
}

// Synthetic shapes built at runtime so this file holds no key-shaped literal.
fn repeat(ch: char, count: usize) -> String {
    std::iter::repeat_n(ch, count).collect()
}

#[test]
fn every_credential_shape_is_found_as_a_whole_token() {
    let cases = [
        (format!("key AIza{} end", repeat('B', 35)), "google_key"),
        (format!("\"sk-proj-{}\"", repeat('a', 40)), "openai_key"),
        (format!("sk-ant-api03-{}", repeat('Z', 48)), "anthropic_key"),
        (format!("sk-or-v1-{}", repeat('f', 64)), "openrouter_key"),
        (format!("AKIA{}", repeat('Q', 16)), "aws_key"),
        (format!("xai-{}", repeat('k', 48)), "xai_key"),
        (format!("gsk_{}", repeat('g', 48)), "groq_key"),
        (
            format!("organization `org_{}`", repeat('0', 26)),
            "groq_org",
        ),
        (format!("Bearer {}", repeat('t', 30)), "bearer"),
        ("  - name: set-cookie\n".to_owned(), "cookie_header"),
        ("\"user_id\": \"user_2x\"".to_owned(), "user_id"),
        (
            "team 0a1b2c3d-0000-0000-0000-000000000000".to_owned(),
            "xai_team",
        ),
        ("mail alice@example.com here".to_owned(), "email"),
        ("/Users/bob/project".to_owned(), "home_path"),
    ];
    for (text, kind) in cases {
        assert!(kinds(&text).contains(&kind), "{kind} not found in {text}");
    }
}

#[test]
fn a_shape_inside_a_longer_token_is_not_a_hit() {
    // Base64 image data contains every letter sequence eventually.
    let embedded = format!("xAIza{}", repeat('B', 35));
    assert!(!kinds(&embedded).contains(&"google_key"));
    let too_long = format!("AIza{}", repeat('B', 36));
    assert!(!kinds(&too_long).contains(&"google_key"));
    assert!(kinds("sk-short").is_empty());
    assert!(kinds("the user_id field").is_empty());
}

#[test]
fn an_exported_secret_is_found_but_never_echoed() {
    let secret = format!("zz-{}", repeat('s', 20));
    let hits = scan(&format!("body: '{secret}'"), std::slice::from_ref(&secret));
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].kind, "exported_secret");
    assert!(!hits[0].excerpt.contains(&secret));
}

#[test]
fn multi_byte_text_before_a_uuid_does_not_panic() {
    // The 16-byte lookback starts inside a two-byte character here.
    let text = "ééééééééééé team: 0a1b2c3d-0000-0000-0000-000000000000";
    assert!(kinds(text).contains(&"xai_team"));
    assert!(kinds("ééééééééééé 0a1b2c3d-0000-0000-0000-000000000000").is_empty());
}

#[test]
fn a_git_failure_is_an_error_not_a_clean_scan() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let error = changed_fixtures(&root, "no-such-ref-xtask-scan").expect_err("bad ref");
    assert!(error.contains("no-such-ref-xtask-scan"), "{error}");
}
