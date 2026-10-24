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
                "invalid [REDACTED] s3cr3t! c12 s34 p56 [REDACTED] [REDACTED]"
            ))
        );
        assert_eq!(reply.get("echo"), Some(&json!(4321)));
        assert_eq!(reply.get("scalar"), Some(&json!("s3cr3t!")));
        assert_eq!(reply.get("nested"), Some(&json!(r#"{"echo":4321}"#)));
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
fn free_text_does_not_promote_short_values_or_string_credentials_to_numeric_secrets() {
    for purpose in [Purpose::Cassette, Purpose::Traffic] {
        let original = json!({
            "type": "message",
            "content_type": "text",
            "count": 200,
            "large_count": 12345678,
            "content": "Read https://example.test/?key=type and https://example.test/?code=200 and https://example.test/?code=12345678",
            "quoted": "example: {\"password\":\"type\",\"device_code\":200}",
            "json_in_string": "{\"password\":\"type\",\"device_code\":200}"
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
        assert_eq!(
            scrubber.scrub_body(r#"{"value":200,"text":"200"}"#),
            if purpose == Purpose::Traffic {
                r#"{"value":200,"text":"[REDACTED]"}"#
            } else {
                r#"{"text":"[REDACTED]","value":200}"#
            }
        );
        scrubber.learn_body(r#"{"password":1234567890}"#);
        assert_eq!(
            scrubber.scrub_body(r#"{"value":1234567890}"#),
            r#"{"value":"[REDACTED]"}"#
        );
        scrubber.learn_header("x-api-key", "type");
        assert_eq!(
            scrubber.scrub_body(r#"{"type":"type","content_type":"text"}"#),
            if purpose == Purpose::Traffic {
                r#"{"type":"[REDACTED]","content_type":"text"}"#
            } else {
                r#"{"content_type":"text","type":"[REDACTED]"}"#
            }
        );
    }
}

#[test]
fn query_credentials_are_explicit_and_cursors_remain_distinct() {
    for name in [
        "page_token",
        "page-token",
        "next_token",
        "continuation_token",
        "last_key",
        "sort-key",
        "content_signature",
    ] {
        assert!(!is_secret_param(name), "{name}");
    }
    for name in [
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
fn only_long_wire_credentials_mask_object_keys_and_collisions_keep_values() {
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
        assert_eq!(object.get("abc"), Some(&json!(3)));
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
fn short_json_and_cookie_values_do_not_erase_ordinary_text() {
    for purpose in [Purpose::Traffic, Purpose::Cassette] {
        let mut scrubber = Scrubber::new(purpose);
        scrubber.learn_header("cookie", "variant=control; sid=abc");
        scrubber.learn_header("set-cookie", "sid=def; Path=/");
        let request = r#"{"token":"main","password":"abcd","echo":"git checkout main; abcd control abc def"}"#;
        let output: Value = serde_json::from_str(&scrubber.scrub_body(request)).expect("JSON");
        assert_eq!(
            output.get("echo"),
            Some(&json!("git checkout main; abcd control abc def"))
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
