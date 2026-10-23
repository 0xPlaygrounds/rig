use super::*;

/// The observed native recipe is checked offline with independently calculated controls.
#[test]
fn current_request_checksum_controls() {
    for (body, expected) in [
        (
            r#"{"model":"claude-opus-5-5","max_tokens":128000,"messages":[{"role":"user","content":"hello"}],"system":[{"type":"text","text":"cch=00000;"}],"stream":true}"#,
            "fa6ee",
        ),
        (
            r#"{"max_tokens":42,"fallbacks":[{"model":"other","data":["bracket ] and quote \\\"",[1,2]]}],"messages":[],"fallback_credit_token":"synthetic","model":"claude-opus-5-5"}"#,
            "63005",
        ),
        (
            r#"{"messages":[{"role":"user","content":"literal \\\"max_tokens\\\":42"}],"thinking":{"type":"adaptive"}}"#,
            "0a455",
        ),
    ] {
        assert_eq!(compute(body.as_bytes()), expected);
    }
}

/// Synthetic requests isolate fields omitted by the native checksum contract.
#[test]
fn volatile_fields_do_not_change_attestation() {
    let base = br#"{"model":"first","messages":[]}"#;
    for body in [
        br#"{"model":"second","messages":[]}"#.as_slice(),
        br#"{"max_tokens":128000,"model":"first","messages":[]}"#,
        br#"{"model":"first","max_tokens":16,"messages":[]}"#,
        br#"{"model":"first","messages":[],"max_tokens":42}"#,
        br#"{"fallbacks":[{"model":"fallback","array":[[1],"]"]}],"model":"first","messages":[]}"#,
        br#"{"model":"first","messages":[],"fallback_credit_token":"synthetic"}"#,
        br#"{"model":"first","max_tokens":2,"fallback_credit_token":"synthetic","messages":[],"fallbacks":[]}"#,
    ] {
        assert_eq!(compute(body), compute(base));
    }
    assert_ne!(
        compute(br#"{"model":"first","messages":[{"role":"user","content":"hello"}]}"#),
        compute(base)
    );
}

/// Escaped field names inside text remain part of the signed request bytes.
#[test]
fn literal_data_is_hashed_without_reserializing() {
    assert_ne!(
        compute(br#"{"text":"literal \"max_tokens\":42"}"#),
        compute(br#"{"text":"literal \"max_tokens\":43"}"#)
    );
    assert_ne!(
        compute(br#"{"max_tokens":-1}"#),
        compute(br#"{"max_tokens":-2}"#)
    );
    assert_ne!(
        compute(r#"{"text":"é"}"#.as_bytes()),
        compute(br#"{"text":"\u00e9"}"#)
    );
}
