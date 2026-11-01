//! What the rules find: each kind by its shape, where a token starts, and
//! nothing that only looks like the start of a key.

use serde_json::json;

use super::*;

/// One key of every kind, each found whole, and redacted once.
#[test]
fn every_kind_is_found_by_its_shape_and_redacted() {
    let keys = [
        (
            "sk-proj-abcdefghijklmnopqrstuvwxyz1234567890",
            Kind::OpenAiApiKey,
        ),
        ("sk-svcacct-abcdefghijklmnop1234", Kind::OpenAiApiKey),
        (
            "sk-abcdefghijklmnopqrstuvwxyz1234567890",
            Kind::OpenAiApiKey,
        ),
        (
            "sk-ant-api03-abcdefghijklmnopqrstuvwxyz1234567890",
            Kind::AnthropicApiKey,
        ),
        ("AIzaSyExampleSecretToken1234567890", Kind::GoogleApiKey),
        ("AKIA1234567890ABCDEF", Kind::AwsAccessKey),
        ("ASIA0123456789ABCDEF", Kind::AwsAccessKey),
        (
            "ghp_abcdefghijklmnopqrstuvwxyz0123456789",
            Kind::GitHubToken,
        ),
        (
            "github_pat_11ABCDEFG0123456789_abcdefghij",
            Kind::GitHubToken,
        ),
        ("glpat-abcdefghij0123456789", Kind::GitLabToken),
        ("xoxb-1234567890-abcdefghij", Kind::SlackToken),
        ("sk_live_abcdefghij0123456789", Kind::StripeKey),
        (
            "hf_abcdefghijklmnopqrstuvwxyz0123456789",
            Kind::HuggingFaceToken,
        ),
        (
            "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiIxMjM0In0.c2lnbmF0dXJlLWJ5dGVz",
            Kind::JsonWebToken,
        ),
        (
            "eyJhbGciOiJub25lIn0.eyJzdWIiOiIxMjM0In0.",
            Kind::JsonWebToken,
        ),
        (
            "-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXk=\n-----END OPENSSH PRIVATE KEY-----",
            Kind::PrivateKey,
        ),
    ];
    for (key, kind) in keys {
        let line = format!("before \"{key}\", after");
        let found: Vec<SecretMatch> = find(&line).collect();
        let start = "before \"".len();
        assert_eq!(
            found,
            [SecretMatch {
                kind,
                range: start..start + key.len()
            }],
            "{key}"
        );
        let redacted = text(&line);
        assert_eq!(redacted, format!("before \"{REDACTED}\", after"));
        assert!(!has_secret(&redacted));
    }
}

/// Key prefixes inside a word or a base64 blob, and prefixes with a short
/// or letterless tail, are not keys; text without one is not copied.
#[test]
fn only_a_whole_token_of_a_key_shape_is_a_key() {
    let harmless = [
        "the sk-etch marker is harmless",
        "sk-12345678901234567890123456789012345",
        "rig-file-id-page-two-verifier-8c27 file_011Cb1W1wnAxQP1a6AuVcPx5",
        "ErUBCkYIBRgCIkAIzaSyExampleSecretToken1234567890ABCDEFGHIJAKIA0123456789ABCDEFsk-proj-notakeyhereatall0123456789abcdef",
        "Eq/AIzaSyExampleSecretToken1234567890ABCDEFGHIJ+AKIA0123456789ABCDEF==",
        "AKIA0123456789ABCDEFG is one letter too long",
        "-----BEGIN PUBLIC KEY-----\nMFkw\n-----END PUBLIC KEY-----",
        "eyJhbGciOiJSUzI1NiJ9 alone, or eyJhbGciOiJSUzI1NiJ9.eyJz.c2ln with a short payload",
        "theyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiIxMjM0In0.c2ln inside a word",
    ];
    for line in harmless {
        assert!(matches!(text(line), Cow::Borrowed(_)), "{line}");
    }
}

/// Every string of a JSON value is redacted, keys and other values are
/// kept, and a private key cut off mid-block is redacted to the end.
#[test]
fn json_redacts_every_string_and_a_cut_off_key_to_the_end() {
    let mut value = json!({
        "sk-abcdefghijklmnopqrstuvwxyz1234567890": 1,
        "headers": ["Bearer ghp_abcdefghijklmnopqrstuvwxyz0123456789", "plain"],
        "chunk": "key: -----BEGIN RSA PRIVATE KEY-----\nMIIEow",
        "count": 7,
    });
    json(&mut value);
    assert_eq!(
        value,
        json!({
            "sk-abcdefghijklmnopqrstuvwxyz1234567890": 1,
            "headers": [format!("Bearer {REDACTED}"), "plain"],
            "chunk": format!("key: {REDACTED}"),
            "count": 7,
        })
    );
}

/// A value assigned to a secret-named variable is redacted with the name
/// kept; a count, a reference, a short value and a name that only holds a
/// secret word are not secrets, and a known key keeps its own kind.
#[test]
fn a_value_assigned_to_a_secret_name_is_redacted_and_counts_are_not() {
    let line = "AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY region=us-east-1";
    assert_eq!(
        text(line),
        "AWS_SECRET_ACCESS_KEY=[REDACTED] region=us-east-1"
    );
    assert_eq!(
        text(r#"{"DB_PASSWORD": "correct horse battery", "GITHUB_TOKEN": 'abcdefgh'}"#),
        r#"{"DB_PASSWORD": "[REDACTED]", "GITHUB_TOKEN": '[REDACTED]'}"#
    );
    assert_eq!(
        text("?API_KEY=abcdefghij&page=2"),
        "?API_KEY=[REDACTED]&page=2"
    );
    for kept in [
        "MAX_TOKENS=4096",
        "TOKEN_COUNT=12345678",
        "MAX_TOKENS_PER_CALL=unlimited",
        "TOKEN_BUDGET=generous-enough",
        "GH_TOKEN=$GITHUB_TOKEN",
        "API_KEY=short",
        "my_token=abcdefghijk",
        "API_KEY=[REDACTED]",
        "API_KEY=‹SECRET_1›",
    ] {
        assert_eq!(text(kept), kept);
    }
    let shaped: Vec<Kind> = find("GITHUB_TOKEN=ghp_abcdefghijklmnopqrstuvwxyz0123456789")
        .map(|secret| secret.kind)
        .collect();
    assert_eq!(shaped, [Kind::GitHubToken]);
}

/// Serialized JSON is redacted as its value is: escaped strings are
/// decoded first, so a key after a newline or inside quotes is found; keys
/// and clean text are left as they are, and the result still parses.
#[test]
fn json_text_redacts_serialized_json_as_its_value_is_redacted() {
    let value = json!({
        "OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz1234567890": "a key as a name stays",
        "output": "PATH=/bin\nOPENAI_API_KEY=sk-test-probe-1234567890abcdef\nHOME=/home",
        "quoted": "export GITHUB_TOKEN=\"abcdefgh12345678\"",
        "token": "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiIxMjM0In0.c2lnbmF0dXJlLWJ5dGVz",
        "plain": ["no secret", "AKIA0123456789ABCDEF"],
        "n": 3,
    });
    let line = serde_json::to_string(&value).unwrap_or_default();
    let redacted = json_text(&line);
    let mut expected = value.clone();
    json(&mut expected);
    let parsed: serde_json::Value = serde_json::from_str(&redacted).unwrap_or_default();
    assert_eq!(parsed, expected);
    assert!(!redacted.contains("sk-test-probe"), "{redacted}");
    assert!(!redacted.contains("abcdefgh12345678"), "{redacted}");

    let clean = serde_json::to_string(&json!({"a": "line\none", "b": [1, "two"]}));
    let clean = clean.unwrap_or_default();
    assert!(matches!(json_text(&clean), Cow::Borrowed(_)));
    // A letter written as an escape is decoded before the search.
    let escaped = r#"{"a":"id \u0041KIA0123456789ABCDEF\n"}"#;
    assert_eq!(json_text(escaped), r#"{"a":"id [REDACTED]\n"}"#);
    let unclosed = r#"{"a":"AKIA0123456789ABCDEF"#;
    assert_eq!(json_text(unclosed), unclosed);
}

/// A text redacted in pieces, cut anywhere, joins to the text redacted
/// whole: no secret the whole holds is shown, however it streams.
#[test]
fn a_streamed_text_redacts_as_the_whole_text_wherever_it_is_cut() {
    let texts = [
        "Set OPENAI_API_KEY=sk-proj-abcdefghijklmnop1234 first, then run it.",
        "export GITHUB_TOKEN=ghp_abcdefghijklmnopqrstuvwxyz0123456789\nnext",
        "PASSWORD: hunter2hunter2 and DB_PASSWORD = \"two words here\" end",
        "id AKIA0123456789ABCDEF and eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0.c2ln done",
        "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA\nabcd\n-----END RSA PRIVATE KEY-----\nafter",
        "-----BEGIN PRIVATE KEY-----\nMIIEowIBAAKCAQEA never ends",
        "Prose with é and 日本語の文 words, no key, MAX_TOKENS=4096 and TOKEN_COUNT: 12345678.",
        "https://example.com/a+b/sk-proj-abcdefghijklmnop1234,(ghp_abcdefghijklmnopqrstuvwxyz0123456789)",
        "\"API_KEY\": \"abcdefgh12345678\", \"name\": \"x\"\n\tSECRET=$HOME",
    ];
    for text_ in texts {
        let whole = text(text_).into_owned();
        let cuts: Vec<usize> = (0..=text_.len())
            .filter(|&at| text_.is_char_boundary(at))
            .collect();
        // Every cut into two pieces, and the text one character at a time.
        let splits = cuts.iter().map(|&at| vec![&text_[..at], &text_[at..]]);
        let chars = cuts
            .windows(2)
            .map(|pair| &text_[pair[0]..pair[1]])
            .collect::<Vec<_>>();
        for pieces in splits.chain(std::iter::once(chars)) {
            let mut streamed = Streamed::default();
            let mut shown: String = pieces.iter().map(|piece| streamed.push(piece)).collect();
            shown += &streamed.finish();
            assert_eq!(shown, whole, "{pieces:?}");
        }
    }
}

/// A run of key characters with no end, such as a base64 blob, is held
/// back no further than `HELD_MOST`, so it streams and is not searched
/// again with every piece.
#[test]
fn a_streamed_run_with_no_end_is_held_back_no_further_than_its_bound() {
    let mut streamed = Streamed::default();
    let piece = "a".repeat(100);
    let held_back = (0..400).map(|_| streamed.push(&piece).len()).sum::<usize>();
    assert!(400 * 100 - held_back <= HELD_MOST + 100, "{held_back}");
    let rest = streamed.finish();
    assert_eq!(held_back + rest.len(), 400 * 100);
}
