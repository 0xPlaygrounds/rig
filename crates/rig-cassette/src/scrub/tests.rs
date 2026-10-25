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
            .any(|secret| secret.value.contains("REDACTED"))
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
        "tools": [{"name": "sign_in", "parameters": {"type": "object", "properties": {
            "token": {"type": "string", "description": "ordinary description"},
            "authorization_token": {"type": ["string", "null"], "default": "schema-default-value"}
        }}}],
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
        assert_eq!(value.get("tools"), original.get("tools"));
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
            "error": "invalid [REDACTED] [REDACTED] c12 s34 p56 [REDACTED] [REDACTED]",
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
                "invalid [REDACTED] [REDACTED] c12 s34 p56 [REDACTED] [REDACTED]"
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

#[test]
fn described_credentials_keep_their_secret_context_and_schemas_keep_their_definitions() {
    let secrets = [
        "opaque-credential-12345",
        "opaque-test-credential-012345",
        "abc12345",
    ];
    for purpose in [Purpose::Cassette, Purpose::Traffic] {
        for secret in secrets {
            let request = json!({
                "client_secret": {"value": secret, "description": "temporary credential"},
                "echo": secret,
                "tools": [{"name": "sign_in", "parameters": {
                    "type": "object", "properties": {
                        "password": {"description": "The user's password", "type": "string"},
                        "token": {"type": ["string", "null"]},
                        "client_secret": {"description": "A credential object", "properties": {
                            "value": {"type": "string"}
                        }}
                    }
                }}]
            });
            let mut scrubber = Scrubber::new(purpose);
            scrubber.learn_body(&request.to_string());
            let scrubbed = scrubber.scrub_body(&request.to_string());
            assert!(!scrubbed.contains(secret), "{scrubbed}");
            let parsed: Value = serde_json::from_str(&scrubbed).unwrap_or_default();
            assert_eq!(
                parsed.pointer("/client_secret/value"),
                Some(&json!(REDACTED))
            );
            assert_eq!(parsed.get("tools"), request.get("tools"));
            let reply = scrubber.scrub_body(&json!({"error": secret}).to_string());
            assert!(!reply.contains(secret), "{reply}");
            let mut scrubber = Scrubber::new(purpose);
            let ordinary_parameters = json!({"name": "call", "parameters": {"properties": {
                "client_secret": {"value": secret, "description": "temporary credential"}
            }}, "echo": secret});
            let scrubbed = scrubber.scrub_body(&ordinary_parameters.to_string());
            assert!(!scrubbed.contains(secret), "{scrubbed}");
        }
    }
}

#[test]
fn prose_does_not_promote_values_to_numbers_but_authoritative_credentials_do() {
    for purpose in [Purpose::Cassette, Purpose::Traffic] {
        let original = json!({
            "type": "message",
            "content_type": "text",
            "count": 200,
            "large_count": 12345678,
            "content": "Read https://example.test/?key=type and https://example.test/?code=200 and https://example.test/?code=12345678",
            "quoted": "example: {\"password\":\"type\",\"device_code\":200}",
        });
        let mut scrubber = Scrubber::new(purpose);
        let mut value = original.clone();
        scrubber.scrub_json_value(None, &mut value);
        assert_eq!(value.get("type"), original.get("type"));
        assert_eq!(value.get("content_type"), original.get("content_type"));
        assert_eq!(value.get("count"), original.get("count"));
        assert_eq!(value.get("large_count"), original.get("large_count"));
        assert!(
            !value
                .as_object()
                .is_some_and(|fields| fields.contains_key(REDACTED))
        );
        assert!(
            !scrubber
                .learned
                .iter()
                .any(|secret| secret.value == "type" || secret.value == "200")
        );
        let mut scrubber = Scrubber::new(purpose);
        scrubber.learn_header("x-api-key", "200");
        let output: Value =
            serde_json::from_str(&scrubber.scrub_body(r#"{"value":200,"text":"200"}"#))
                .expect("JSON");
        assert_eq!(output, json!({"value":REDACTED,"text":REDACTED}));
        scrubber.learn_body(r#"{"password":1234567890}"#);
        assert_eq!(
            scrubber.scrub_body(r#"{"value":1234567890}"#),
            r#"{"value":"[REDACTED]"}"#
        );
        scrubber.learn_header("x-api-key", "type");
        let output: Value =
            serde_json::from_str(&scrubber.scrub_body(r#"{"type":"type","content_type":"text"}"#))
                .expect("JSON");
        assert_eq!(output.get(REDACTED), Some(&json!(REDACTED)));
        assert_eq!(output.get("content_[REDACTED]"), Some(&json!("text")));
    }
}

#[test]
fn query_credentials_are_explicit_and_cursors_remain_distinct() {
    for name in [
        "page_token",
        "next_token",
        "continuation_token",
        "last_key",
        "content_signature",
        "prompt_cache_key",
        "eos_token",
    ] {
        assert!(!is_secret_param(name), "{name}");
    }
    for name in [
        "page-token",
        "sort-key",
        "x-api-key",
        "access_token",
        "refresh-token",
        "idToken",
        "session_token",
        "authToken",
        "api_key",
        "client_secret",
        "x-amz-credential",
        "x-amz-signature",
        "x-amz-security-token",
    ] {
        assert!(is_secret_param(name), "{name}");
    }
    for purpose in [Purpose::Cassette, Purpose::Traffic] {
        let mut scrubber = Scrubber::new(purpose);
        assert_eq!(
            scrubber.scrub_url("https://example.test/?page_token=p2&access_token=abc123"),
            "https://example.test/?page_token=p2&access_token=[REDACTED]"
        );
        scrubber.learn_url("https://example.test/?page_token=p2");
        assert_eq!(
            scrubber.scrub_text("p2 and page-token-p2"),
            "p2 and page-token-p2"
        );
    }
}

#[test]
fn one_sse_parser_handles_crlf_multiline_fields_offsets_and_partial_events() {
    let first = ": comment\r\nevent: auth\r\nid: 7\r\ndata: {\"client_secret\":\r\ndata: {\"value\":\"abc12345\",\"description\":\"temporary\"}}\r\n\r\n";
    let tail = "event: delta\ndata: {\"echo\":\"abc12345\"}";
    let body = format!("{first}{tail}");
    let events = parse_sse(&body, false).unwrap_or_default();
    assert_eq!(events.len(), 2);
    assert_eq!(events.first().map(|event| event.end), Some(first.len()));
    assert_eq!(events.last().map(|event| event.end), Some(body.len()));
    assert_eq!(
        events.first().and_then(|event| event.event.as_deref()),
        Some("auth")
    );
    assert_eq!(
        events.first().and_then(|event| event.id.as_deref()),
        Some("7")
    );
    assert_eq!(
        events.first().and_then(|event| event.data.as_deref()),
        Some("{\"client_secret\":\n{\"value\":\"abc12345\",\"description\":\"temporary\"}}")
    );
    assert!(parse_sse("data", false).is_none());
    assert!(parse_sse("data", true).is_some());
    for purpose in [Purpose::Cassette, Purpose::Traffic] {
        let mut scrubber = Scrubber::new(purpose);
        let scrubbed = scrubber.scrub_body(&body);
        assert!(!scrubbed.contains("abc12345"), "{scrubbed}");
        assert_eq!(scrubber.scrub_body(&scrubbed), scrubbed);
    }
}

#[cfg(feature = "http")]
#[tokio::test]
async fn traffic_recognizes_sse_without_a_content_type_hint() {
    use futures::StreamExt;
    use rig_core::http_client::{Bytes, HttpClientExt, Request};
    let secret = "opaque-described-secret-12345";
    let upstream = httpmock::MockServer::start_async().await;
    upstream.mock_async(|when, then| {
        when.method("POST").path("/events");
        then.status(200).header("content-type", "text/plain").body(format!(
            "event: delta\r\nid: 7\r\ndata: {{\"type\":\"delta\",\"count\":200,\r\ndata: \"echo\":\"{secret}\",\"text\":\"https://example.test/?key=type&code=200\"}}\r\n\r\n"
        ));
    }).await;
    let directory = assert_fs::TempDir::new().expect("log directory");
    let path = directory.path().join("traffic.jsonl");
    let log = crate::traffic::TrafficLog::new(std::fs::File::create(&path).expect("log file"))
        .expect("writer");
    let client = rig_reqwest::ReqwestClient::from(
        rig_reqwest::reqwest::Client::builder()
            .no_proxy()
            .build()
            .expect("client"),
    );
    let client = log.wrap_http(client);
    let request = Request::post(upstream.url("/events"))
        .body(Bytes::from(
            json!({"client_secret": {"value": secret, "description": "temporary credential"}})
                .to_string(),
        ))
        .expect("request");
    let response = client.send_streaming(request).await.expect("reply");
    let _: Vec<_> = response.into_body().collect().await;
    log.flush();
    let text = std::fs::read_to_string(path).expect("written line");
    assert!(!text.contains(secret), "{text}");
    let line: Value = serde_json::from_str(&text).expect("JSON line");
    assert_eq!(
        line.pointer("/response/events/0/event"),
        Some(&json!("delta"))
    );
    assert_eq!(line.pointer("/response/events/0/id"), Some(&json!("7")));
    assert_eq!(
        line.pointer("/response/events/0/data/type"),
        Some(&json!("delta"))
    );
    assert_eq!(
        line.pointer("/response/events/0/data/count"),
        Some(&json!(200))
    );
    assert_eq!(
        line.pointer("/response/events/0/data/echo"),
        Some(&json!(REDACTED))
    );
}

#[cfg(feature = "http")]
#[tokio::test]
async fn replay_preserves_raw_pagination_cursors_and_rejects_the_wrong_cursor() {
    use crate::http::{CassetteError, CassetteMode, CassetteSpec, ProviderCassette, RecordVia};
    use rig_reqwest::reqwest::{Client, StatusCode};
    let raw = r#"when:
  path: /v1/items
  method: GET
  query_param:
  - name: page_token
    value: p2
then:
  status: 200
  body: '{"page":2}'
"#;
    let recorded = crate::http::scrub_cassette_contents(raw);
    assert!(recorded.contains("value: p2"));
    for (contents, cursor, expected) in [
        (raw, "p2", StatusCode::OK),
        (recorded.as_str(), "p2", StatusCode::OK),
        (recorded.as_str(), "p1", StatusCode::NOT_FOUND),
    ] {
        let directory = assert_fs::TempDir::new().expect("fixture directory");
        let path = directory.path().join("pagination.yaml");
        std::fs::write(&path, contents).expect("fixture");
        let cassette = ProviderCassette::try_start_at(
            RecordVia::Proxy,
            "example",
            CassetteSpec::new("pagination"),
            "https://example.invalid/v1",
            CassetteMode::Replay,
            path,
        )
        .await
        .expect("replay");
        let response = Client::builder()
            .no_proxy()
            .build()
            .expect("client")
            .get(format!("{}/items?page_token={cursor}", cassette.base_url()))
            .send()
            .await
            .expect("replayed request");
        assert_eq!(response.status(), expected);
        response.bytes().await.expect("response body");
        if expected == StatusCode::OK {
            cassette.try_finish().await.expect("consumed recording");
        } else {
            assert!(matches!(cassette.try_finish().await,
                Err(CassetteError::ReplayMismatch { unexpected_requests, .. }) if !unexpected_requests.is_empty()));
        }
    }
}

#[test]
fn wire_credentials_mask_object_keys_and_collisions_keep_values() {
    for purpose in [Purpose::Traffic, Purpose::Cassette] {
        let mut scrubber = Scrubber::new(purpose);
        scrubber.learn_header("authorization", "Bearer opaque-header-credential");
        scrubber.learn_header("x-api-key", "abc");
        scrubber.learn_body("example: https://example.test/?api_key=inferred-credential");
        let input = json!({
            "opaque-header-credential": 1,
            REDACTED: 2,
            "abc": 3,
            "inferred-credential": 4,
            "nested": "{\"opaque-header-credential\":5}",
        });
        let output: Value =
            serde_json::from_str(&scrubber.scrub_body(&input.to_string())).expect("JSON");
        let object = output.as_object().expect("object");
        assert!(!object.contains_key("opaque-header-credential"));
        assert_eq!(object.len(), 5);
        assert!(object.values().any(|value| value == &json!(1)));
        assert!(object.values().any(|value| value == &json!(2)));
        assert!(!object.contains_key("abc"));
        assert!(object.values().any(|value| value == &json!(3)));
        assert_eq!(object.get("inferred-credential"), Some(&json!(4)));
        assert!(
            !output
                .get("nested")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .contains("opaque-header-credential")
        );
        let sse = scrubber.scrub_body("\u{feff}id: 7\rdata: {\"opaque-header-credential\":1}\r\rdata: {\"echo\":\"opaque-header-credential\"}");
        assert!(sse.starts_with('\u{feff}'));
        assert!(!sse.contains("opaque-header-credential"), "{sse}");
        let parsed = parse_sse(&sse, true).expect("SSE");
        assert_eq!(
            parsed.last().and_then(|event| event.id.as_deref()),
            Some("7")
        );
    }
}

#[test]
fn bare_token_and_cookie_values_do_not_erase_ordinary_text() {
    for purpose in [Purpose::Traffic, Purpose::Cassette] {
        let mut scrubber = Scrubber::new(purpose);
        scrubber.learn_header("cookie", "variant=control; sid=abc");
        scrubber.learn_header("set-cookie", "sid=def; Path=/");
        let request = r#"{"token":"main","password":"abcd","echo":"git checkout main; abcd control abc def"}"#;
        let output: Value = serde_json::from_str(&scrubber.scrub_body(request)).expect("JSON");
        assert_eq!(
            output.get("echo"),
            Some(&json!("git checkout main; [REDACTED] control abc def"))
        );
        assert_eq!(output.get("password"), Some(&json!(REDACTED)));
        assert_eq!(
            output.get("token"),
            Some(&json!(if purpose == Purpose::Traffic {
                REDACTED
            } else {
                "main"
            }))
        );
        assert_eq!(scrubber.scrub_header_value("cookie", "sid=abc"), REDACTED);
    }
}

#[cfg(feature = "http")]
#[tokio::test]
async fn a_utf8_cut_keeps_whole_events_and_learns_the_valid_prefix() {
    use futures::StreamExt;
    use rig_core::http_client::{Bytes, HttpClientExt, Request};
    let secret = "opaque-prefix-credential";
    let mut bytes = format!("\u{feff}id: 7\rdata: {{\"echo\":\"{secret}\"}}\r\rdata: {{\"client_secret\":\"{secret}\"}}\r\r").into_bytes();
    bytes.extend_from_slice(&[0xe2, 0x82]);
    let upstream = httpmock::MockServer::start_async().await;
    upstream
        .mock_async(|when, then| {
            when.method("POST").path("/events");
            then.status(200)
                .header("content-type", "text/event-stream")
                .body(bytes);
        })
        .await;
    let directory = assert_fs::TempDir::new().expect("directory");
    let path = directory.path().join("traffic.jsonl");
    let log = crate::traffic::TrafficLog::new(std::fs::File::create(&path).expect("file"))
        .expect("writer");
    let client = log.wrap_http(rig_reqwest::ReqwestClient::from(
        rig_reqwest::reqwest::Client::builder()
            .no_proxy()
            .build()
            .expect("client"),
    ));
    let response = client
        .send_streaming(
            Request::post(upstream.url("/events"))
                .body(Bytes::new())
                .expect("request"),
        )
        .await
        .expect("reply");
    let _: Vec<_> = response.into_body().collect().await;
    log.flush();
    let text = std::fs::read_to_string(path).expect("line");
    assert!(!text.contains(secret), "{text}");
    let line: Value = serde_json::from_str(&text).expect("JSON");
    assert_eq!(
        line.pointer("/response/events/0/data/echo"),
        Some(&json!(REDACTED))
    );
    assert_eq!(line.pointer("/response/events/1/id"), Some(&json!("7")));
    assert_eq!(
        line.pointer("/response/events/2/data"),
        Some(&json!({"bytes":2}))
    );
    assert_eq!(
        line.pointer("/response/events")
            .and_then(Value::as_array)
            .map(Vec::len),
        Some(3)
    );
}

#[cfg(feature = "http")]
fn gzip(body: &[u8]) -> Vec<u8> {
    use std::io::Write;
    let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    encoder.write_all(body).expect("compress");
    encoder.finish().expect("gzip bytes")
}

#[cfg(feature = "http")]
#[tokio::test]
async fn compressed_and_raw_multipart_credentials_are_private_in_both_sinks() {
    use crate::http::{CassetteMode, CassetteSpec, ProviderCassette, RecordVia};
    use base64::{Engine, prelude::BASE64_STANDARD};
    let secret = "opaque-shared-credential-97531";
    let multipart = format!(
        "--sample\r\nContent-Disposition: form-data; name=\"client_secret\"\r\n\r\n{secret}\r\n--sample\r\nContent-Disposition: form-data; name=\"file\"; filename=\"data.bin\"\r\n\r\n"
    );
    let mut multipart = multipart.into_bytes();
    multipart.extend_from_slice(b"\xff\x00\r\n--sample--\r\n");
    let json_body = json!({"client_secret":secret,"message":"keep this"}).to_string();
    for (label, content_type, encoding, body) in [
        (
            "gzip",
            "application/json",
            Some("gzip"),
            gzip(json_body.as_bytes()),
        ),
        (
            "multipart",
            "multipart/form-data; boundary=sample",
            None,
            multipart,
        ),
    ] {
        let upstream = httpmock::MockServer::start_async().await;
        upstream
            .mock_async(|when, then| {
                when.method("POST").path("/submit");
                then.status(200)
                    .header("content-type", "application/json")
                    .body(json!({"echo":secret}).to_string());
            })
            .await;
        let directory = assert_fs::TempDir::new().expect("directory");
        let path = directory.path().join("fixture.yaml");
        let session = ProviderCassette::try_start_at(
            RecordVia::Proxy,
            "example",
            CassetteSpec::new("prepared"),
            &upstream.base_url(),
            CassetteMode::Record,
            path.clone(),
        )
        .await
        .expect("record session");
        let log_path = directory.path().join("traffic.jsonl");
        let log = crate::traffic::TrafficLog::new(std::fs::File::create(&log_path).expect("log"))
            .expect("writer");
        let raw = rig_reqwest::reqwest::Client::builder()
            .no_proxy()
            .build()
            .expect("client");
        let client = log.wrap_http(rig_reqwest::ReqwestClient::from(raw.clone()));
        use rig_core::http_client::{Bytes, HttpClientExt, Request};
        let mut request = Request::post(format!("{}/submit", session.base_url()))
            .header("content-type", content_type);
        if let Some(encoding) = encoding {
            request = request.header("content-encoding", encoding);
        }
        let response = client
            .send::<_, Bytes>(
                request
                    .body(Bytes::copy_from_slice(&body))
                    .expect("request"),
            )
            .await
            .expect("reply");
        let reply = response.into_body().await.expect("body");
        assert!(String::from_utf8_lossy(&reply).contains(secret));
        log.flush();
        session.try_finish().await.expect("safe recording");
        let text = std::fs::read_to_string(&log_path).expect("traffic");
        assert!(!text.contains(secret), "{label}: {text}");
        let yaml = std::fs::read_to_string(&path).expect("fixture");
        assert!(!yaml.contains(secret), "{label}: {yaml}");
        let fixture: serde_yaml::Value = serde_yaml::from_str(&yaml).expect("YAML");
        for side in ["when", "then"] {
            let side = fixture.get(side).expect("side");
            let text = side
                .get("body")
                .and_then(serde_yaml::Value::as_str)
                .expect("stored body");
            let bytes = if side
                .get("body_encoding")
                .and_then(serde_yaml::Value::as_str)
                == Some("base64")
            {
                BASE64_STANDARD.decode(text).expect("base64")
            } else {
                text.as_bytes().to_vec()
            };
            if label == "multipart" && side.get("status").is_none() {
                assert!(bytes.windows(2).any(|window| window == b"\xff\x00"));
            }
            assert!(
                !bytes
                    .windows(secret.len())
                    .any(|window| window == secret.as_bytes()),
                "{label}: decoded {side:?}"
            );
        }
        let replay = ProviderCassette::try_start_at(
            RecordVia::Proxy,
            "example",
            CassetteSpec::new("prepared"),
            &upstream.base_url(),
            CassetteMode::Replay,
            path.clone(),
        )
        .await
        .expect("replay");
        let mut request = raw
            .post(format!("{}/submit", replay.base_url()))
            .header("content-type", content_type)
            .body(body);
        if let Some(encoding) = encoding {
            request = request.header("content-encoding", encoding);
        }
        let response = request.send().await.expect("replayed response");
        assert_eq!(response.status().as_u16(), 200, "{label}");
        let text = response.text().await.expect("replayed body");
        assert!(text.contains(REDACTED), "{text}");
        replay.try_finish().await.expect("consumed");
    }
}

#[test]
fn a_boundary_parameter_without_multipart_parts_does_not_bypass_body_scrubbing() {
    for content_type in [
        "application/json; boundary=absent",
        "multipart/form-data; boundary=absent",
    ] {
        let mut scrubber = Scrubber::new(Purpose::Traffic);
        let prepared = scrubber.prepare_exchange(
            "https://example.test/",
            [("content-type", content_type)],
            RequestBodyRef::Bytes(br#"{"client_secret":"opaque-boundary-credential","echo":"opaque-boundary-credential"}"#),
            std::iter::empty(),
            None,
        );
        let bytes = scrubber.scrub_prepared(prepared.request_body.expect("prepared body"));
        let text = std::str::from_utf8(&bytes).expect("text");
        assert!(!text.contains("opaque-boundary-credential"), "{text}");
        assert!(text.contains(REDACTED));
    }
}

#[test]
fn a_home_path_keeps_only_its_basename() {
    let scrubbed = scrub_local_filesystem_paths(
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
    assert_eq!(
        scrub_local_filesystem_paths("/home/dev/models/m.gguf"),
        "/REDACTED_PATH/m.gguf"
    );
    assert_eq!(
        scrub_local_filesystem_paths("/root/m.gguf"),
        "/REDACTED_PATH/m.gguf"
    );
}

/// A JSON string's closing quote terminates the path.
#[test]
fn a_path_inside_json_stops_at_the_quote() {
    let scrubbed =
        scrub_local_filesystem_paths(r#"{"model":"/Users/a/b/m.gguf","object":"chat.completion"}"#);
    assert_eq!(
        scrubbed,
        r#"{"model":"/REDACTED_PATH/m.gguf","object":"chat.completion"}"#
    );
}

/// Several paths in one body are all replaced.
#[test]
fn every_occurrence_is_replaced() {
    let scrubbed =
        scrub_local_filesystem_paths(r#"{"a":"/Users/x/one.gguf","b":"/Users/y/two.gguf"}"#);
    assert!(!scrubbed.contains("/Users/"), "{scrubbed}");
    assert!(
        scrubbed.contains("one.gguf") && scrubbed.contains("two.gguf"),
        "{scrubbed}"
    );
}

/// Spaces remain part of a home-directory account name.
#[test]
fn a_home_directory_containing_a_space_is_fully_replaced() {
    let scrubbed = scrub_local_filesystem_paths(r#"{"model":"/Users/John Smith/models/m.gguf"}"#);
    assert_eq!(scrubbed, r#"{"model":"/REDACTED_PATH/m.gguf"}"#);
    assert!(!scrubbed.contains("John"), "{scrubbed}");
    assert!(!scrubbed.contains("Smith"), "{scrubbed}");
}

/// Scrubbing is idempotent.
#[test]
fn the_rule_is_idempotent() {
    let once = scrub_local_filesystem_paths("/Users/a/.cache/m.gguf");
    assert_eq!(scrub_local_filesystem_paths(&once), once);
}

/// Home-directory segments inside public URLs stay unchanged.
#[test]
fn a_url_containing_a_home_segment_is_not_a_local_path() {
    let body = r#"{"url":"https://math.ucr.edu/home/baez/physics/General/BlueSky/blue_sky.html"}"#;
    assert_eq!(scrub_local_filesystem_paths(body), body);
    let body = r#"{"url":"https://example.com/Users/profile.html"}"#;
    assert_eq!(scrub_local_filesystem_paths(body), body);
}

/// The anchored rule still catches the real case sitting next to a URL.
#[test]
fn a_local_path_beside_a_url_is_still_scrubbed() {
    let scrubbed = scrub_local_filesystem_paths(
        r#"{"url":"https://math.ucr.edu/home/baez/x.html","model":"/Users/me/m.gguf"}"#,
    );
    assert!(
        scrubbed.contains("math.ucr.edu/home/baez/x.html"),
        "{scrubbed}"
    );
    assert!(
        scrubbed.contains(r#""model":"/REDACTED_PATH/m.gguf""#),
        "{scrubbed}"
    );
}

/// Text without a home path stays unchanged.
#[test]
fn unrelated_text_is_untouched() {
    let body = r#"{"model":"gpt-4o-mini","choices":[]}"#;
    assert_eq!(scrub_local_filesystem_paths(body), body);
}

#[test]
fn urls_in_keys_learn_credentials_before_either_side_is_rendered() {
    let response = json!({
        "https://user:opaque-key-password-97531@example.test/file": {"size":12},
        "https://example.test/file?signature=opaque-key-signature-97531": 2,
        "nested": "{\"https://example.test/cb#access_token=opaque-fragment-97531\":3}"
    })
    .to_string();
    let request = json!({"echo":"opaque-key-password-97531 opaque-key-signature-97531 opaque-fragment-97531"}).to_string();
    for purpose in [Purpose::Traffic, Purpose::Cassette] {
        let mut scrubber = Scrubber::new(purpose);
        let prepared = scrubber.prepare_exchange(
            "https://example.test/",
            [],
            RequestBodyRef::Bytes(request.as_bytes()),
            [],
            Some(response.as_bytes()),
        );
        let request = scrubber.scrub_prepared(prepared.request_body.expect("request"));
        let response = scrubber.scrub_prepared(prepared.response_body.expect("response"));
        for bytes in [&request, &response] {
            let text = std::str::from_utf8(bytes).expect("UTF-8");
            for secret in [
                "opaque-key-password-97531",
                "opaque-key-signature-97531",
                "opaque-fragment-97531",
            ] {
                assert!(!text.contains(secret), "{text}");
            }
        }
    }
}

#[test]
fn short_authoritative_credentials_are_masked_in_place_and_as_whole_word_echoes() {
    for purpose in [Purpose::Traffic, Purpose::Cassette] {
        let mut scrubber = Scrubber::new(purpose);
        scrubber.learn_header("authorization", "Bearer local");
        scrubber.learn_header("x-api-key", "abc123");
        scrubber.learn_url("https://example.test/?api_key=ollama");
        scrubber.learn_body("client_secret=EMPTY");
        assert_eq!(
            scrubber.scrub_text("local abc123 ollama EMPTY none"),
            "[REDACTED] [REDACTED] [REDACTED] [REDACTED] none"
        );
        assert_eq!(
            scrubber.scrub_header_value("authorization", "Bearer local"),
            REDACTED
        );
        assert_eq!(scrubber.scrub_param_value("api_key", "ollama"), REDACTED);
        let output: Value =
            serde_json::from_str(&scrubber.scrub_body(r#"{"password":"none","echo":"none"}"#))
                .expect("JSON");
        assert_eq!(output, json!({"password":REDACTED,"echo":REDACTED}));
    }
}

#[test]
fn json_header_fields_fragments_and_escaped_cut_text_are_scrubbed_and_learned() {
    for purpose in [Purpose::Traffic, Purpose::Cassette] {
        let mut scrubber = Scrubber::new(purpose);
        let input = json!({"authorization":"opaque-authorization-97531", "headers":{"Authorization":"Bearer opaque-bearer-97531", "x-service-key":"opaque-service-97531"}}).to_string();
        scrubber.learn_body(&input);
        let output = scrubber.scrub_body(&input);
        for secret in [
            "opaque-authorization-97531",
            "opaque-bearer-97531",
            "opaque-service-97531",
        ] {
            assert!(!output.contains(secret), "{output}");
            assert_eq!(scrubber.scrub_text(secret), REDACTED);
        }
        let cut = r#"data: {"arguments":"{\"password\":\"hunter2-opaque\"}"#;
        scrubber.learn_body(cut);
        let output = scrubber.scrub_body(cut);
        assert!(!output.contains("hunter2-opaque"), "{output}");
        assert_eq!(scrubber.scrub_text("hunter2-opaque"), REDACTED);
        let url =
            "https://example.test/cb?state=kept#access_token=opaque-fragment-secret&state=kept";
        scrubber.learn_header("location", url);
        assert!(
            !scrubber
                .scrub_header_value("location", url)
                .contains("opaque-fragment-secret")
        );
        assert_eq!(scrubber.scrub_text("opaque-fragment-secret"), REDACTED);
    }
}

#[test]
fn malformed_multipart_is_redacted_instead_of_treated_as_ordinary_text() {
    let raw = b"--sample\nContent-Disposition: form-data; name=\"api_key\"\n\nopaque-malformed-secret\n--sample--\n";
    for purpose in [Purpose::Traffic, Purpose::Cassette] {
        let mut scrubber = Scrubber::new(purpose);
        let prepared = scrubber.prepare_exchange(
            "",
            [("content-type", "multipart/form-data; boundary=sample")],
            RequestBodyRef::Bytes(raw),
            [],
            None,
        );
        assert_eq!(
            scrubber.scrub_prepared(prepared.request_body.expect("body")),
            Bytes::from_static(REDACTED.as_bytes())
        );
    }
}

#[cfg(feature = "http")]
#[tokio::test]
async fn url_keys_and_json_authorization_are_private_in_live_and_cassette_output() {
    use crate::http::{CassetteMode, CassetteSpec, ProviderCassette, RecordVia};
    use rig_core::http_client::{HttpClientExt, Request};
    let key_password = "opaque-only-key-password-97531";
    let signed = "opaque-only-key-signature-97531";
    let nested = "opaque-only-nested-fragment-97531";
    let auth = "opaque-mcp-authorization-97531";
    let header_auth = "opaque-mcp-header-authorization-97531";
    let response = json!({
        format!("https://user:{key_password}@example.test/file"):1,
        format!("https://example.test/file?signature={signed}"):2,
        "nested":json!({format!("https://example.test/cb#access_token={nested}"):3}).to_string(),
        "echo":format!("{auth} {header_auth}"),
        "fingerprint":"local"
    })
    .to_string();
    let upstream = httpmock::MockServer::start_async().await;
    upstream
        .mock_async(|when, then| {
            when.method("POST").path("/submit");
            then.status(200)
                .header("content-type", "application/json")
                .body(response);
        })
        .await;
    let directory = assert_fs::TempDir::new().expect("directory");
    let fixture = directory.path().join("fixture.yaml");
    let session = ProviderCassette::try_start_at(
        RecordVia::Proxy,
        "example",
        CassetteSpec::new("key-privacy"),
        &upstream.base_url(),
        CassetteMode::Record,
        fixture.clone(),
    )
    .await
    .expect("record");
    let log_path = directory.path().join("traffic.jsonl");
    let log = crate::traffic::TrafficLog::new(std::fs::File::create(&log_path).expect("log"))
        .expect("writer");
    let raw = rig_reqwest::reqwest::Client::builder()
        .no_proxy()
        .build()
        .expect("client");
    let client = log.wrap_http(rig_reqwest::ReqwestClient::from(raw));
    let request = json!({"authorization":auth,"headers":{"Authorization":format!("Bearer {header_auth}")},"echo":format!("{key_password} {signed} {nested}"),"fingerprint":"local"});
    let response = client
        .send::<_, Bytes>(
            Request::post(format!("{}/submit", session.base_url()))
                .header("authorization", "Bearer local")
                .body(Bytes::from(request.to_string()))
                .expect("request"),
        )
        .await
        .expect("reply");
    response.into_body().await.expect("body");
    log.flush();
    session.try_finish().await.expect("safe recording");
    for output in [
        std::fs::read_to_string(&log_path).expect("traffic"),
        std::fs::read_to_string(&fixture).expect("cassette"),
    ] {
        for secret in [key_password, signed, nested, auth, header_auth] {
            assert!(!output.contains(secret), "{output}");
        }
        assert!(!output.contains("local"), "{output}");
    }
}
#[test]
fn author_and_lookup_keys_stay_data_while_headers_keep_their_credential_context() {
    for purpose in [Purpose::Traffic, Purpose::Cassette] {
        let mut scrubber = Scrubber::new(purpose);
        let input = json!({
            "author":{"name":"Ursula K. Le Guin"},
            "authority":"example.test",
            "key":"harbor",
            "arguments":"{\"author\":{\"name\":\"Ursula K. Le Guin\"},\"key\":\"harbor\"}",
            "headers":{"key":"opaque-header-key-97531","Authorization":"Bearer opaque-auth-97531","X-Custom-Auth":"opaque-custom-auth-97531"},
            "echo":"opaque-header-key-97531 opaque-auth-97531 opaque-custom-auth-97531"
        });
        let output: Value =
            serde_json::from_str(&scrubber.scrub_body(&input.to_string())).expect("JSON");
        assert_eq!(output.get("author"), input.get("author"));
        assert_eq!(output.get("authority"), input.get("authority"));
        assert_eq!(output.get("key"), input.get("key"));
        assert_eq!(output.get("arguments"), input.get("arguments"));
        assert_eq!(output.pointer("/headers/key"), Some(&json!(REDACTED)));
        assert_eq!(
            output.pointer("/headers/Authorization"),
            Some(&json!(REDACTED))
        );
        assert_eq!(
            output.pointer("/headers/X-Custom-Auth"),
            Some(&json!(REDACTED))
        );
        assert!(!output.to_string().contains("opaque-"));
    }
}

#[test]
fn auth_headers_match_whole_components() {
    for name in [
        "auth",
        "authorization",
        "authentication",
        "x-custom-auth",
        "auth-token",
        "X_Authentication_Custom",
        "x-api-key",
        "client_secret",
        "refresh_token",
        "api_key",
    ] {
        assert!(is_secret_header(name), "{name}");
    }
    for name in ["author", "authority", "authorship", "www-authenticate"] {
        assert!(!is_secret_header(name), "{name}");
    }
}

#[test]
fn parsed_argument_credentials_scrub_echoed_keys_but_prose_credentials_do_not() {
    let secret = "opaque-nested-password-97531";
    let header_secret = "opaque-nested-header-key-97531";
    let request = json!({"arguments":json!({"password":secret}).to_string(), "headers":json!({"key":header_secret}).to_string()}).to_string();
    let response = json!({secret:"invalid credential",header_secret:"invalid header"}).to_string();
    for purpose in [Purpose::Traffic, Purpose::Cassette] {
        let mut scrubber = Scrubber::new(purpose);
        let prepared = scrubber.prepare_exchange(
            "",
            [],
            RequestBodyRef::Bytes(request.as_bytes()),
            [],
            Some(response.as_bytes()),
        );
        let request = scrubber.scrub_prepared(prepared.request_body.expect("request"));
        let response = scrubber.scrub_prepared(prepared.response_body.expect("response"));
        let output = format!(
            "{} {}",
            String::from_utf8_lossy(&request),
            String::from_utf8_lossy(&response)
        );
        assert!(!output.contains(secret), "{output}");
        assert!(!output.contains(header_secret), "{output}");
        assert!(output.contains("invalid credential"));
        let prose = format!(
            "Example: {}",
            serde_json::to_string(&json!({"password":secret}).to_string()).expect("quoted JSON")
        );
        let mut scrubber = Scrubber::new(purpose);
        scrubber.learn_body(&prose);
        let output: Value =
            serde_json::from_str(&scrubber.scrub_body(&json!({secret:"ordinary key"}).to_string()))
                .expect("JSON");
        assert_eq!(output.get(secret), Some(&json!("ordinary key")));
    }
}

#[test]
fn query_values_keep_their_wire_spelling_while_multipart_text_remains_a_body() {
    for purpose in [Purpose::Traffic, Purpose::Cassette] {
        let mut scrubber = Scrubber::new(purpose);
        for value in [
            "1.50",
            "1e3",
            r#"{ "b": 2, "a": 1 }"#,
            r#""quoted""#,
            r#""{\"b\": 2, \"a\": 1}""#,
        ] {
            assert_eq!(scrubber.scrub_param_value("filter", value), value);
            assert_eq!(scrubber.scrub_param_value("api_key", value), REDACTED);
        }
        assert_eq!(scrubber.scrub_part_value("message", "1.50"), "1.5");
    }
}

#[cfg(feature = "http")]
#[test]
fn cassette_query_values_preserve_numeric_and_json_spelling() {
    let pairs = json!([
        {"name":"temperature","value":"1.50"},
        {"name":"n","value":"1e3"},
        {"name":"filter","value":"{ \"b\": 2, \"a\": 1 }"},
        {"name":"encoded","value":r#""{\"b\": 2, \"a\": 1}""#}
    ]);
    let fixture =
        json!({"when":{"path":"/search","method":"GET","query_param":pairs},"then":{"status":200}});
    let output =
        crate::http::scrub_cassette_contents(&serde_yaml::to_string(&fixture).expect("YAML"));
    let output: Value = serde_yaml::from_str(&output).expect("fixture");
    assert_eq!(output.pointer("/when/query_param"), Some(&pairs));
}

#[cfg(feature = "http")]
#[tokio::test]
async fn multipart_surroundings_are_private_in_both_sinks_and_decoded_fixture_bodies() {
    use crate::http::{CassetteMode, CassetteSpec, ProviderCassette, RecordVia};
    use base64::{Engine, prelude::BASE64_STANDARD};
    use rig_core::http_client::{HttpClientExt, Request};
    let secret = "opaque-review-credential-97531";
    for payload in [b"hello".as_slice(), b"\xff\x00".as_slice()] {
        let mut raw = format!(
            "{secret}\r\n--sample\r\nContent-Disposition: form-data; name=\"message\"\r\n\r\n"
        )
        .into_bytes();
        raw.extend_from_slice(payload);
        raw.extend_from_slice(format!("\r\n--sample--\r\n{secret}").as_bytes());
        let upstream = httpmock::MockServer::start_async().await;
        upstream
            .mock_async(|when, then| {
                when.method("POST").path("/submit");
                then.status(200)
                    .header("content-type", "multipart/form-data; boundary=sample")
                    .body(raw.clone());
            })
            .await;
        let directory = assert_fs::TempDir::new().expect("directory");
        let path = directory.path().join("fixture.yaml");
        let session = ProviderCassette::try_start_at(
            RecordVia::Proxy,
            "example",
            CassetteSpec::new("multipart-surroundings"),
            &upstream.base_url(),
            CassetteMode::Record,
            path.clone(),
        )
        .await
        .expect("record");
        let log_path = directory.path().join("traffic.jsonl");
        let log = crate::traffic::TrafficLog::new(std::fs::File::create(&log_path).expect("file"))
            .expect("writer");
        let client = log.wrap_http(rig_reqwest::ReqwestClient::from(
            rig_reqwest::reqwest::Client::builder()
                .no_proxy()
                .build()
                .expect("client"),
        ));
        let response = client
            .send::<_, Bytes>(
                Request::post(format!("{}/submit", session.base_url()))
                    .header("content-type", "multipart/form-data; boundary=sample")
                    .header("authorization", format!("Bearer {secret}"))
                    .body(Bytes::from(raw.clone()))
                    .expect("request"),
            )
            .await
            .expect("reply");
        assert_eq!(
            response.into_body().await.expect("body").as_ref(),
            raw.as_slice()
        );
        log.flush();
        session.try_finish().await.expect("safe recording");
        let traffic = std::fs::read_to_string(&log_path).expect("traffic");
        assert!(!traffic.contains(secret), "{traffic}");
        if payload == b"hello" {
            let traffic: Value = serde_json::from_str(&traffic).expect("JSON");
            for side in ["request", "response"] {
                let body = traffic
                    .get(side)
                    .and_then(|side| side.get("body"))
                    .and_then(Value::as_str)
                    .expect("text multipart");
                assert!(body.starts_with("[REDACTED]\r\n--sample\r\n"), "{body}");
                assert!(body.ends_with("--sample--\r\n[REDACTED]"), "{body}");
            }
        }
        let fixture: serde_yaml::Value =
            serde_yaml::from_str(&std::fs::read_to_string(&path).expect("fixture")).expect("YAML");
        for side in ["when", "then"] {
            let side = fixture.get(side).expect("side");
            let body = side
                .get("body")
                .and_then(serde_yaml::Value::as_str)
                .expect("body");
            let bytes = if side
                .get("body_encoding")
                .and_then(serde_yaml::Value::as_str)
                == Some("base64")
            {
                BASE64_STANDARD.decode(body).expect("base64")
            } else {
                body.as_bytes().to_vec()
            };
            assert!(
                !bytes
                    .windows(secret.len())
                    .any(|window| window == secret.as_bytes())
            );
            assert!(bytes.windows(payload.len()).any(|window| window == payload));
            assert!(bytes.starts_with(b"[REDACTED]\r\n--sample\r\n"));
            assert!(bytes.ends_with(b"--sample--\r\n[REDACTED]"));
        }
    }
}

#[test]
fn short_authoritative_credentials_scrub_unary_and_sse_echoes_for_both_purposes() {
    let request =
        json!({"password":"hunter2", "arguments":r#"{"client_secret":"n5Q9x4"}"#, "token":"main"})
            .to_string();
    for purpose in [Purpose::Traffic, Purpose::Cassette] {
        for response in [
            json!({"error":"invalid q7Z2p9 hunter2 n5Q9x4", "longer":"xq7Z2p9y hunter2ish éq7Z2p9é", "echo":"main"}).to_string(),
            "data: {\"error\":\"invalid q7Z2p9 hunter2 n5Q9x4\",\"longer\":\"xq7Z2p9y hunter2ish éq7Z2p9é\",\"echo\":\"main\"}\n\n".to_owned(),
        ] {
            let mut scrubber = Scrubber::new(purpose);
            let prepared = scrubber.prepare_exchange("https://example.test/", [("x-api-key","q7Z2p9")], RequestBodyRef::Bytes(request.as_bytes()), [], Some(response.as_bytes()));
            let request = scrubber.scrub_prepared(prepared.request_body.expect("request"));
            assert!(!String::from_utf8_lossy(&request).contains("hunter2"));
            let response = scrubber.scrub_prepared(prepared.response_body.expect("response"));
            let response = std::str::from_utf8(&response).expect("text");
            let payload = if response.starts_with("data:") {
                rig_core::http_client::framing::Framing::Sse.split(response.as_bytes()).first().map(|frame| frame.as_str().into_owned()).expect("event")
            } else { response.to_owned() };
            let payload: Value = serde_json::from_str(&payload).expect("JSON");
            assert_eq!(payload.get("error"),Some(&json!("invalid [REDACTED] [REDACTED] [REDACTED]")));
            assert_eq!(payload.get("longer"),Some(&json!("xq7Z2p9y hunter2ish éq7Z2p9é")));
            assert_eq!(payload.get("echo"),Some(&json!("main")));
        }
    }
}

#[test]
fn short_credentials_match_word_boundaries_in_opaque_bytes_too() {
    let mut scrubber = Scrubber::new(Purpose::Cassette);
    scrubber.learn_header("x-api-key", "q7Z2p9");
    assert_eq!(
        scrubber.scrub_bytes(b"\xff invalid q7Z2p9\x00"),
        Bytes::from_static(REDACTED.as_bytes())
    );
    for bytes in [
        b"\xff xq7Z2p9y\x00".as_slice(),
        "\u{fffd} éq7Z2p9é".as_bytes(),
    ] {
        assert_eq!(scrubber.scrub_bytes(bytes).as_ref(), bytes);
    }
    let mut unicode = vec![0xff];
    unicode.extend_from_slice("éq7Z2p9é".as_bytes());
    assert_eq!(scrubber.scrub_bytes(&unicode).as_ref(), unicode.as_slice());
}

#[test]
fn secret_names_and_url_authorities_are_shared_across_surfaces() {
    let secret = "opaque-shared-name-97531";
    for purpose in [Purpose::Traffic, Purpose::Cassette] {
        let mut scrubber = Scrubber::new(purpose);
        for prefix in ["postgres", "custom+scheme", "FTP"] {
            let url = format!("{prefix}://user:{secret}@example.test/db");
            scrubber.learn_body(&format!("connect to {url}"));
            let output = scrubber.scrub_text(&format!("connect to {url}"));
            assert!(!output.contains(secret), "{output}");
            assert!(output.contains("://[REDACTED]@example.test/db"), "{output}");
        }
        assert_eq!(scrubber.scrub_text(secret), REDACTED);
        let mut query = Scrubber::new(purpose);
        query.learn_url("https://example.test/?x-api-key=q7Z2p9");
        assert_eq!(query.scrub_text("invalid q7Z2p9"), "invalid [REDACTED]");
        assert_eq!(
            Scrubber::new(purpose).scrub_url("https://example.test/?x-api-key=q7Z2p9"),
            "https://example.test/?x-api-key=[REDACTED]"
        );
        for text in [
            format!("x-api-key={secret}"),
            format!("failed with ?%78-api-key={secret}"),
            format!("auth={secret}"),
        ] {
            let output = Scrubber::new(purpose).scrub_text(&text);
            assert!(!output.contains(secret), "{output}");
        }
        for name in ["prompt_cache_key", "eos_token", "page_token"] {
            assert!(!is_secret_param(name));
            assert_eq!(
                scrubber.scrub_text(&format!("{name}=ordinary")),
                format!("{name}=ordinary")
            );
        }
    }
}

#[cfg(feature = "http")]
#[tokio::test]
async fn short_password_and_header_echoes_are_private_in_both_output_files() {
    use crate::http::{AccountFailure, CassetteMode, CassetteSpec, ProviderCassette, RecordVia};
    use futures::StreamExt;
    use rig_core::http_client::{HttpClientExt, Request, framing::Framing};
    let reply = json!({"error":{"message":"invalid q7Z2p9 hunter2"},"longer":"xq7Z2p9y hunter2ish","ordinary":"main"});
    for streaming in [false, true] {
        let upstream = httpmock::MockServer::start_async().await;
        let body = if streaming {
            format!("data: {reply}\n\ndata: [DONE]\n\n")
        } else {
            reply.to_string()
        };
        upstream
            .mock_async(|when, then| {
                when.method("POST").path("/submit");
                then.status(if streaming { 200 } else { 401 })
                    .header(
                        "content-type",
                        if streaming {
                            "text/event-stream"
                        } else {
                            "application/json"
                        },
                    )
                    .body(body);
            })
            .await;
        let directory = assert_fs::TempDir::new().expect("directory");
        let path = directory.path().join("fixture.yaml");
        let session = ProviderCassette::try_start_at(
            RecordVia::Proxy,
            "example",
            CassetteSpec::new("short-echoes").expects_account_failure(AccountFailure::Auth),
            &upstream.base_url(),
            CassetteMode::Record,
            path.clone(),
        )
        .await
        .expect("record");
        let log_path = directory.path().join("traffic.jsonl");
        let log = crate::traffic::TrafficLog::new(std::fs::File::create(&log_path).expect("log"))
            .expect("writer");
        let client = log.wrap_http(rig_reqwest::ReqwestClient::from(
            rig_reqwest::reqwest::Client::builder()
                .no_proxy()
                .build()
                .expect("client"),
        ));
        let request = Request::post(format!("{}/submit", session.base_url()))
            .header("x-api-key", "q7Z2p9")
            .body(Bytes::from_static(
                br#"{"password":"hunter2","token":"main"}"#,
            ))
            .expect("request");
        if streaming {
            let response = client.send_streaming(request).await.expect("reply");
            let _: Vec<_> = response.into_body().collect().await;
        } else {
            assert!(client.send::<_, Bytes>(request).await.is_err());
        }
        log.flush();
        session.try_finish().await.expect("safe recording");
        let traffic: Value =
            serde_json::from_str(&std::fs::read_to_string(&log_path).expect("traffic"))
                .expect("JSON");
        assert_eq!(
            traffic.pointer("/request/body/password"),
            Some(&json!(REDACTED))
        );
        let traffic_reply = traffic
            .pointer(if streaming {
                "/response/events/0/data"
            } else {
                "/response/body"
            })
            .expect("reply data");
        let fixture: Value =
            serde_yaml::from_str(&std::fs::read_to_string(&path).expect("fixture")).expect("YAML");
        let body = fixture
            .pointer("/then/body")
            .and_then(Value::as_str)
            .expect("body");
        let payload = if streaming {
            Framing::Sse
                .split(body.as_bytes())
                .first()
                .map(|frame| frame.as_str().into_owned())
                .expect("event")
        } else {
            body.to_owned()
        };
        let cassette_reply: Value = serde_json::from_str(&payload).expect("reply JSON");
        for payload in [traffic_reply, &cassette_reply] {
            assert_eq!(
                payload.pointer("/error/message"),
                Some(&json!("invalid [REDACTED] [REDACTED]"))
            );
            assert_eq!(payload.get("longer"), Some(&json!("xq7Z2p9y hunter2ish")));
            assert_eq!(payload.get("ordinary"), Some(&json!("main")));
        }
        let request: Value = serde_json::from_str(
            fixture
                .pointer("/when/body")
                .and_then(Value::as_str)
                .expect("request body"),
        )
        .expect("request JSON");
        assert_eq!(request.get("password"), Some(&json!(REDACTED)));
    }
}

#[test]
fn loose_parameter_scrubbing_preserves_cassette_application_tokens_and_quoted_markers() {
    let mut cassette = Scrubber::new(Purpose::Cassette);
    for text in [
        "Call checkpoint_step with token=start; then return next_token.",
        "export ANTHROPIC_AUTH_TOKEN=\"[REDACTED]\"\nclaude",
        "llm = LLM(\napi_key=\"YOUR_HOLYSHEEP_API_KEY\",\n)",
    ] {
        assert_eq!(cassette.scrub_text(text), text);
    }
    for purpose in [Purpose::Traffic, Purpose::Cassette] {
        let mut scrubber = Scrubber::new(purpose);
        for text in [
            "failed x-api-key=\"opaque-query-97531\"",
            "failed x-api-key='opaque-query-97531'",
            "failed ?token=opaque-query-97531",
        ] {
            let output = scrubber.scrub_text(text);
            assert!(!output.contains("opaque-query-97531"), "{output}");
            assert_eq!(scrubber.scrub_text(&output), output);
        }
    }
}

#[test]
fn retained_sse_comments_and_unknown_fields_teach_both_renderers_about_echoes() {
    use rig_core::http_client::framing::Framing;
    let comment = "opaque-comment-credential-1234";
    let extension = "opaque-extension-credential-5678";
    let source = format!(
        ": {{\"client_secret\":\"{comment}\"}}\n\nx-extra: {{\"api_key\":\"{extension}\"}}\n\ndata: {{\"echo\":\"{comment} {extension}\"}}\n\n"
    );
    for purpose in [Purpose::Traffic, Purpose::Cassette] {
        let mut scrubber = Scrubber::new(purpose);
        let prepared = scrubber.prepare_exchange(
            "https://example.test/",
            [],
            RequestBodyRef::Bytes(b""),
            [("content-type", "text/event-stream")],
            Some(source.as_bytes()),
        );
        let response = prepared.response_body.expect("response");
        let (serialized, payload) = if purpose == Purpose::Traffic {
            let events = scrubber.scrub_stream(response, [source.len()]);
            let data: Vec<_> = events.into_iter().map(|event| event.data).collect();
            (
                serde_json::to_string(&data).expect("events"),
                data.first().cloned().expect("data event"),
            )
        } else {
            let bytes = scrubber.scrub_prepared(response);
            let frames = Framing::Sse.split(&bytes);
            let payload = serde_json::from_slice(frames.first().expect("data event").as_bytes())
                .expect("JSON");
            (
                String::from_utf8(bytes.to_vec()).expect("SSE text"),
                payload,
            )
        };
        assert!(!serialized.contains(comment), "{serialized}");
        assert!(!serialized.contains(extension), "{serialized}");
        assert_eq!(payload.get("echo"), Some(&json!("[REDACTED] [REDACTED]")));
    }
}

#[cfg(feature = "http")]
#[test]
fn content_coding_lists_are_bounded_and_report_uninspectable_bodies() {
    use std::io::Write;
    let plain = br#"{"password":"opaque-coding-credential-97531"}"#;
    let inner = gzip(plain);
    let twice = gzip(&inner);
    assert_eq!(
        decoded(&twice, Some("gzip, x-gzip, identity"))
            .expect("decode")
            .bytes
            .as_ref(),
        plain
    );
    let mut zlib = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
    zlib.write_all(&inner).expect("deflate");
    let stacked = zlib.finish().expect("bytes");
    let result = decoded(&stacked, Some(" GZIP , deflate ")).expect("decode mixed stack");
    assert_eq!(result.status, DecodeStatus::Decoded);
    assert_eq!(result.bytes.as_ref(), plain);
    assert_eq!(
        decoded(plain, Some("identity, identity"))
            .expect("identity")
            .status,
        DecodeStatus::NotEncoded
    );
    for coding in ["br", "zstd", "gzip, br"] {
        assert_eq!(
            decoded(&twice, Some(coding)).err(),
            Some(DecodeFailure::UnsupportedCoding)
        );
    }
    assert_eq!(
        decoded(plain, Some("gzip")).err(),
        Some(DecodeFailure::InvalidData)
    );
    assert_eq!(
        decoded_with_limit(&gzip(b"12345678"), Some("gzip"), 8)
            .expect("exact limit")
            .bytes
            .as_ref(),
        b"12345678"
    );
    assert_eq!(
        decoded_with_limit(&gzip(b"123456789"), Some("gzip"), 8).err(),
        Some(DecodeFailure::OverLimit)
    );
}

#[test]
fn every_auth_scheme_uses_the_same_credential_learning_without_splitting_descriptions() {
    for purpose in [Purpose::Traffic, Purpose::Cassette] {
        for json_source in [false, true] {
            let mut scrubber = Scrubber::new(purpose);
            if json_source {
                scrubber
                    .learn_body(r#"{"authorization":"Custom-Scheme opaque-custom-scheme-97531"}"#);
            } else {
                scrubber.learn_header("authorization", "Custom-Scheme opaque-custom-scheme-97531");
            }
            assert_eq!(scrubber.scrub_text("opaque-custom-scheme-97531"), REDACTED);
            scrubber.learn_body(r#"{"client_secret":{"value":"opaque-explicit-secret-97531","description":"temporary credential"}}"#);
            assert_eq!(
                scrubber.scrub_text("A credential object"),
                "A credential object"
            );
        }
    }
}
