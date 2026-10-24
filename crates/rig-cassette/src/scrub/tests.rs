use super::*;
use serde_json::json;

#[test]
fn credentials_are_learned_from_each_wire_representation() {
    let mut scrubber = Scrubber::new(Purpose::Traffic);
    scrubber.learn_header("authorization", "Bearer opaque-authorization");
    scrubber.learn_header("cookie", "a=opaque-cookie-one; b=opaque-cookie-two");
    scrubber.learn_header("set-cookie", "sid=opaque-session; Path=/; Secure");
    scrubber.learn_header("x-service-key", "opaque-header-key");
    scrubber.learn_url("https://username:opaque%2Fpassword@example.test/?%61pi_key=opaque%2Fquery");
    scrubber.learn_body("refresh_token=opaque%2Frefresh&client_secret=opaque-form-secret");
    scrubber.learn_body(
        r#"{"client_secret":{"value":"opaque-json-secret"},"codeVerifier":1234567890}"#,
    );
    for secret in [
        "opaque-authorization",
        "opaque-cookie-one",
        "opaque-cookie-two",
        "opaque-session",
        "opaque-header-key",
        "opaque%2Fpassword",
        "opaque/password",
        "opaque%2Fquery",
        "opaque/query",
        "opaque%2Frefresh",
        "opaque/refresh",
        "opaque-form-secret",
        "opaque-json-secret",
        "1234567890",
    ] {
        assert_eq!(
            scrubber.scrub_text(&format!("echo {secret}")),
            "echo [REDACTED]"
        );
    }
    scrubber.learn_header("x-key", REDACTED);
    scrubber.learn_header("x-key", "id_REDACTED_1");
    scrubber.learn_header("authorization", "Bearer id_REDACTED_1");
    scrubber.learn_url("https://example.test/?key=%5BREDACTED%5D");
    assert!(
        !scrubber
            .learned
            .iter()
            .any(|value| value.contains("REDACTED"))
    );
    assert!(!is_secret_header("www-authenticate"));
    assert!(is_secret_header("x-custom-auth"));
}

#[test]
fn encoded_urls_forms_and_incomplete_json_are_masked() {
    let mut scrubber = Scrubber::new(Purpose::Traffic);
    assert_eq!(
        scrubber.scrub_url("https://user:pw@example.test/?%74oken=abc&alt=sse"),
        "https://[REDACTED]@example.test/?%74oken=[REDACTED]&alt=sse"
    );
    assert_eq!(
        scrubber.scrub_text("see https://u:p@example.test/?%61pi_key=secret"),
        "see https://[REDACTED]@example.test/?%61pi_key=[REDACTED]"
    );
    assert_eq!(
        scrubber.scrub_body("grant_type=refresh_token&client%5Fsecret=abc%2Fdef"),
        "grant_type=refresh_token&client%5Fsecret=[REDACTED]"
    );
    assert_eq!(
        scrubber.scrub_body(r#"prefix {"client_secret":{"value":"cut"#),
        "prefix {\"client_secret\":\"[REDACTED]\""
    );
}

#[test]
fn json_fields_nested_strings_and_application_tokens_follow_the_purpose() {
    let original = json!({
        "password": 1234,
        "arguments": r#"{"clientSecret":"opaque-client-secret","query":"say \"hi"}"#,
        "token": "checkpoint-value",
        "logprobs": {"token": "ordinary-word", "logprob": -0.1},
        "properties": {
            "token": {"type": "string", "description": "ordinary description"},
            "authorization_token": {"type": ["string", "null"], "default": "schema-default-value"}
        },
        "created_at": 42,
    });
    for purpose in [Purpose::Traffic, Purpose::Cassette] {
        let mut scrubber = Scrubber::new(purpose);
        scrubber.learn_json(&original);
        scrubber.learn_body(
            "data: {\"token\":\"ordinary-word\",\"logprob\":0}\r\n\r\ndata: [DONE]\r\n\r\n",
        );
        let mut value = original.clone();
        scrubber.scrub_json_value(None, &mut value);
        assert_eq!(value.get("password"), Some(&json!(REDACTED)));
        let nested = value
            .get("arguments")
            .and_then(Value::as_str)
            .and_then(|text| serde_json::from_str::<Value>(text).ok());
        assert_eq!(
            nested.as_ref().and_then(|value| value.get("clientSecret")),
            Some(&json!(REDACTED))
        );
        assert_eq!(
            value.pointer("/logprobs/token"),
            Some(&json!("ordinary-word"))
        );
        assert_eq!(value.get("properties"), original.get("properties"));
        assert_eq!(
            value.get("token"),
            Some(&json!(if purpose == Purpose::Traffic {
                REDACTED
            } else {
                "checkpoint-value"
            }))
        );
        assert_eq!(
            value.get("created_at"),
            Some(&json!(if purpose == Purpose::Traffic { 42 } else { 0 }))
        );
        let again = value.clone();
        scrubber.scrub_json_value(None, &mut value);
        assert_eq!(again, value);
    }
}

#[cfg(feature = "http")]
#[test]
fn cassettes_learn_both_sides_before_scrubbing_echoes() {
    let cassette = crate::http::scrub_cassette_contents(
        r#"when:
  path: /token
  method: POST
  header:
  - name: authorization
    value: Bearer opaque-header-secret
  body: '{"password":"opaque-request-secret","echo":"opaque-response-secret"}'
then:
  status: 401
  body: '{"client_secret":"opaque-response-secret","error":"opaque-request-secret opaque-header-secret"}'
"#,
    );
    for secret in [
        "opaque-header-secret",
        "opaque-request-secret",
        "opaque-response-secret",
    ] {
        assert!(!cassette.contains(secret), "{cassette}");
    }
    assert_eq!(crate::http::scrub_cassette_contents(&cassette), cassette);
}

#[test]
fn short_credentials_and_numeric_echoes_are_masked_without_matching_longer_words() {
    for purpose in [Purpose::Traffic, Purpose::Cassette] {
        let mut scrubber = Scrubber::new(purpose);
        scrubber.learn_header("authorization", "Bearer abc123");
        scrubber.learn_header("cookie", "sid=c12");
        scrubber.learn_header("set-cookie", "sid=s34; Path=/");
        scrubber.learn_url("https://user:p56@example.test/?key=q78");
        scrubber.learn_body("client_secret=f90");
        let mut request = json!({"client_secret": {"value": "s3cr3t!"}, "password": 4321});
        scrubber.scrub_json_value(None, &mut request);
        assert_eq!(
            request.pointer("/client_secret/value"),
            Some(&json!(REDACTED))
        );
        let mut reply = json!({
            "error": "invalid abc123 s3cr3t! c12 s34 p56 q78 f90",
            "scalar": "s3cr3t!",
            "echo": 4321,
            "nested": r#"{"echo":4321}"#,
            "longer": "xabc123y xs3cr3t!x xc12x xs34x xp56x xq78x xf90x",
            "longer_number": 14321,
        });
        scrubber.scrub_json_value(None, &mut reply);
        assert_eq!(
            reply.get("error"),
            Some(&json!(
                "invalid [REDACTED] [REDACTED] [REDACTED] [REDACTED] [REDACTED] [REDACTED] [REDACTED]"
            ))
        );
        assert_eq!(reply.get("echo"), Some(&json!(REDACTED)));
        assert_eq!(reply.get("scalar"), Some(&json!(REDACTED)));
        assert_eq!(
            reply.get("nested"),
            Some(&json!(r#"{"echo":"[REDACTED]"}"#))
        );
        assert_eq!(
            reply.get("longer"),
            Some(&json!("xabc123y xs3cr3t!x xc12x xs34x xp56x xq78x xf90x"))
        );
        assert_eq!(reply.get("longer_number"), Some(&json!(14321)));
    }
    let mut scrubber = Scrubber::new(Purpose::Traffic);
    scrubber.learn_url("https://user@example.test");
    scrubber.learn_url("https://%61%62%63@example.test");
    assert_eq!(scrubber.scrub_text("user abc"), "user abc");
}

#[test]
fn traffic_keeps_paths_and_account_ids_while_cassettes_scrub_them() {
    let path = "/home/alice/code/src/main.rs";
    let arn = "arn:aws:bedrock:us-east-1:123456789012:model/example";
    let mut traffic = Scrubber::new(Purpose::Traffic);
    assert_eq!(traffic.scrub_text(path), path);
    assert_eq!(traffic.scrub_text(arn), arn);
    let mut cassette = Scrubber::default();
    assert_eq!(cassette.scrub_text(path), "/REDACTED_PATH/main.rs");
    assert_eq!(
        cassette.scrub_text(arn),
        "arn:aws:bedrock:us-east-1:account_REDACTED_1:model/example"
    );
}
