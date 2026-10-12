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
            "-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXk=\n-----END OPENSSH PRIVATE KEY-----",
            Kind::PrivateKey,
        ),
    ];
    for (key, kind) in keys {
        let line = format!("before \"{key}\", after");
        let found: Vec<Secret> = find(&line).collect();
        let start = "before \"".len();
        assert_eq!(
            found,
            [Secret {
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
