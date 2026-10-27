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
fn cassettes_scrub_openai_user_ids_and_keep_other_user_strings() {
    let echoed = r#"{"safety_identifier":"user-AbCdEfGhIjKlMnOpQrStUvWx","user":"user-1"}"#;
    let mut traffic = Scrubber::new(Purpose::Traffic);
    assert_eq!(traffic.scrub_text(echoed), echoed);
    let mut cassette = Scrubber::default();
    assert_eq!(
        cassette.scrub_text(echoed),
        r#"{"safety_identifier":"user-REDACTED_1","user":"user-1"}"#
    );
    assert_eq!(
        cassette.scrub_text("user-REDACTED_1"),
        "user-REDACTED_1",
        "a placeholder stays as it is"
    );
    assert_eq!(
        cassette.scrub_text("é user-AbCdEfGhIjKlMnOpQrStUvWx é"),
        "é user-REDACTED_1 é"
    );
    for kept in [
        "superuser-AbCdEfGhIjKlMnOpQrStUvWx",
        "user-AbCdEfGhIjKlMnOpQrStUvWxY",
    ] {
        assert_eq!(cassette.scrub_text(kept), kept);
    }
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
async fn a_utf8_cut_withholds_the_entire_stream() {
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
                .body(bytes.clone());
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
        line.pointer("/response/events/0/data"),
        Some(&json!({"bytes":bytes.len(), "content_type":"text/event-stream"}))
    );
    assert_eq!(
        line.pointer("/response/events")
            .and_then(Value::as_array)
            .map(Vec::len),
        Some(1)
    );
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
        assert!(text.contains(REDACTED) || text.contains("bytes"));
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

/// A local path beside a URL is scrubbed.
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
            serde_json::from_slice::<Value>(
                &scrubber.scrub_prepared(prepared.request_body.expect("body"))
            )
            .expect("metadata"),
            json!({"bytes":raw.len(),"content_type":"multipart/form-data; boundary=sample"})
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
    let client = log.wrap_http(rig_reqwest::ReqwestClient::from(raw.clone()));
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
async fn raw_multipart_capture_keeps_only_metadata_and_omits_request_bytes() {
    use crate::http::{CassetteMode, CassetteSpec, ProviderCassette, RecordVia};
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
        let traffic: Value = serde_json::from_str(&traffic).expect("JSON");
        for side in ["request", "response"] {
            assert_eq!(
                traffic.get(side).and_then(|side| side.get("body")),
                Some(
                    &json!({"bytes":raw.len(),"content_type": if side == "request" { "multipart/form-data; boundary=sample" } else { "multipart/form-data" }})
                )
            );
        }
        let fixture: Value =
            serde_yaml::from_str(&std::fs::read_to_string(&path).expect("fixture")).expect("YAML");
        assert!(fixture.pointer("/when/body").is_none_or(Value::is_null));
        assert!(fixture.pointer("/then/body").is_none_or(Value::is_null));
        assert_eq!(
            fixture.pointer("/then/withheld/bytes"),
            Some(&json!(raw.len()))
        );
        assert!(!fixture.to_string().contains(secret));
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

#[cfg(feature = "http")]
struct CapturedScrubCase {
    traffic: Value,
    fixture: Value,
    path: std::path::PathBuf,
    _directory: assert_fs::TempDir,
}

#[cfg(feature = "http")]
async fn capture_scrub_case(
    query: &str,
    request_headers: &[(&str, &str)],
    request: &[u8],
    response: &[u8],
    streaming: bool,
    response_encoding: Option<&str>,
) -> CapturedScrubCase {
    capture_scrub_case_with_headers(
        query,
        request_headers,
        request,
        response,
        streaming,
        response_encoding,
        &[],
    )
    .await
}

#[cfg(feature = "http")]
async fn capture_scrub_case_with_headers(
    query: &str,
    request_headers: &[(&str, &str)],
    request: &[u8],
    response: &[u8],
    streaming: bool,
    response_encoding: Option<&str>,
    response_headers: &[(&str, &str)],
) -> CapturedScrubCase {
    let mut headers = HeaderMap::new();
    for (name, value) in request_headers {
        headers.append(
            name.parse::<rig_reqwest::reqwest::header::HeaderName>()
                .expect("name"),
            value.parse().expect("value"),
        );
    }
    if !headers.contains_key("content-type") {
        headers.insert("content-type", "application/json".parse().expect("type"));
    }
    capture_scrub_case_with_header_map(
        query,
        &headers,
        request,
        response,
        streaming,
        response_encoding,
        response_headers,
    )
    .await
}

#[cfg(feature = "http")]
async fn capture_scrub_case_with_header_map(
    query: &str,
    request_headers: &HeaderMap,
    request: &[u8],
    response: &[u8],
    streaming: bool,
    response_encoding: Option<&str>,
    response_headers: &[(&str, &str)],
) -> CapturedScrubCase {
    capture_scrub_request_with_header_map(
        query,
        request_headers,
        RequestBodyRef::Bytes(request),
        response,
        streaming,
        response_encoding,
        response_headers,
    )
    .await
}

#[cfg(feature = "http")]
async fn capture_scrub_request_with_header_map(
    query: &str,
    request_headers: &HeaderMap,
    request: RequestBodyRef<'_>,
    response: &[u8],
    streaming: bool,
    response_encoding: Option<&str>,
    response_headers: &[(&str, &str)],
) -> CapturedScrubCase {
    use crate::http::{CassetteMode, CassetteSpec, ProviderCassette, RecordVia};
    use futures::StreamExt;
    use rig_core::http_client::{HttpClientExt, Request};
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("listener");
    let origin = format!("http://{}", listener.local_addr().expect("address"));
    let mut reply_headers = HeaderMap::new();
    if !response_headers
        .iter()
        .any(|(name, _)| name.eq_ignore_ascii_case("content-type"))
    {
        reply_headers.insert(
            "content-type",
            if streaming {
                "text/event-stream"
            } else {
                "application/json"
            }
            .parse()
            .expect("type"),
        );
    }
    for (name, value) in response_headers {
        reply_headers.append(
            name.parse::<rig_reqwest::reqwest::header::HeaderName>()
                .expect("name"),
            value.parse().expect("value"),
        );
    }
    if let Some(coding) = response_encoding {
        reply_headers.insert("content-encoding", coding.parse().expect("coding"));
    }
    let response = Bytes::copy_from_slice(response);
    let app = axum::Router::new().fallback(axum::routing::any(move || {
        let body = response.clone();
        let headers = reply_headers.clone();
        async move {
            let mut response = axum::response::Response::new(axum::body::Body::from(body));
            *response.headers_mut() = headers;
            response
        }
    }));
    let (stop, stopped) = tokio::sync::oneshot::channel();
    let server = tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(async {
                let _ = stopped.await;
            })
            .await
            .expect("server");
    });
    let directory = assert_fs::TempDir::new().expect("directory");
    let fixture = directory.path().join("fixture.yaml");
    let session = ProviderCassette::try_start_at(
        RecordVia::Proxy,
        "example",
        CassetteSpec::new("scrub-exchange"),
        &origin,
        CassetteMode::Record,
        fixture.clone(),
    )
    .await
    .expect("record");
    let path = directory.path().join("traffic.jsonl");
    let log = crate::traffic::TrafficLog::new(std::fs::File::create(&path).expect("file"))
        .expect("writer");
    let raw = rig_reqwest::reqwest::Client::builder()
        .no_proxy()
        .build()
        .expect("client");
    let client = log.wrap_http(rig_reqwest::ReqwestClient::from(raw));
    let mut builder = Request::post(format!("{}/submit{query}", session.base_url()));
    for (name, value) in request_headers {
        builder = builder.header(name.clone(), value.clone());
    }
    match request {
        RequestBodyRef::Bytes(bytes) => {
            let request = builder
                .body(Bytes::copy_from_slice(bytes))
                .expect("request");
            if streaming {
                let reply = client.send_streaming(request).await.expect("reply");
                let _: Vec<_> = reply.into_body().collect().await;
            } else {
                client
                    .send::<_, Bytes>(request)
                    .await
                    .expect("reply")
                    .into_body()
                    .await
                    .expect("body");
            }
        }
        RequestBodyRef::Multipart(form) => {
            assert!(!streaming, "typed multipart uses unary replies");
            client
                .send_multipart::<Bytes>(builder.body(form.clone()).expect("request"))
                .await
                .expect("reply")
                .into_body()
                .await
                .expect("body");
        }
    }
    log.flush();
    session.try_finish().await.expect("safe recording");
    let _ = stop.send(());
    server.await.expect("server joined");
    CapturedScrubCase {
        traffic: serde_json::from_str(&std::fs::read_to_string(path).expect("traffic"))
            .expect("JSON"),
        fixture: serde_yaml::from_str(&std::fs::read_to_string(&fixture).expect("fixture"))
            .expect("YAML"),
        path: fixture,
        _directory: directory,
    }
}

#[cfg(feature = "http")]
#[tokio::test]
async fn numeric_header_query_and_string_password_echoes_are_private_in_both_files() {
    use rig_core::http_client::framing::Framing;
    let payload = json!({"error":{"header":1234567890u64,"query":2345678901u64,"field":3456789012u64},"ordinary":17});
    for streaming in [false, true] {
        let response = if streaming {
            format!("data: {payload}\n\ndata: [DONE]\n\n")
        } else {
            payload.to_string()
        };
        let captured = capture_scrub_case(
            "?api_key=2345678901",
            &[("x-api-key", "1234567890")],
            br#"{"password":"3456789012"}"#,
            response.as_bytes(),
            streaming,
            None,
        )
        .await;
        let (traffic, fixture) = (&captured.traffic, &captured.fixture);
        for output in [&traffic, &fixture] {
            let text = output.to_string();
            for secret in ["1234567890", "2345678901", "3456789012"] {
                assert!(!text.contains(secret), "{text}");
            }
        }
        let traffic_payload = traffic
            .pointer(if streaming {
                "/response/events/0/data"
            } else {
                "/response/body"
            })
            .expect("reply");
        let body = fixture
            .pointer("/then/body")
            .and_then(Value::as_str)
            .expect("body");
        let body = if streaming {
            Framing::Sse
                .split(body.as_bytes())
                .first()
                .map(|frame| frame.as_str().into_owned())
                .expect("event")
        } else {
            body.to_owned()
        };
        let cassette_payload: Value = serde_json::from_str(&body).expect("reply JSON");
        for payload in [traffic_payload, &cassette_payload] {
            assert_eq!(
                payload.get("error"),
                Some(&json!({"header":REDACTED,"query":REDACTED,"field":REDACTED}))
            );
            assert_eq!(payload.get("ordinary"), Some(&json!(17)));
        }
    }
}

#[cfg(feature = "http")]
#[tokio::test]
async fn newly_recorded_header_and_query_body_redactions_replay_the_same_request() {
    use crate::http::{CassetteMode, CassetteSpec, ProviderCassette, RecordVia};
    let query = "?api_key=opaque-query-replay-97531";
    let request =
        br#"{"fingerprint":"local","query_echo":"opaque-query-replay-97531","message":"right"}"#;
    let captured = capture_scrub_case(
        query,
        &[("authorization", "Bearer local")],
        request,
        b"{}",
        false,
        None,
    )
    .await;
    let recorded: Value = serde_json::from_str(
        captured
            .fixture
            .pointer("/when/body")
            .and_then(Value::as_str)
            .expect("body"),
    )
    .expect("JSON");
    assert_eq!(recorded.get("fingerprint"), Some(&json!(REDACTED)));
    assert_eq!(recorded.get("query_echo"), Some(&json!(REDACTED)));
    let replay = ProviderCassette::try_start_at(
        RecordVia::Proxy,
        "example",
        CassetteSpec::new("scrub-exchange"),
        "https://example.test",
        CassetteMode::Replay,
        captured.path.clone(),
    )
    .await
    .expect("replay");
    let response = rig_reqwest::reqwest::Client::builder()
        .no_proxy()
        .build()
        .expect("client")
        .post(format!("{}/submit{query}", replay.base_url()))
        .header("authorization", "Bearer local")
        .header("content-type", "application/json")
        .body(request.to_vec())
        .send()
        .await
        .expect("reply");
    assert_eq!(response.status().as_u16(), 200);
    response.bytes().await.expect("body");
    replay
        .try_finish()
        .await
        .expect("same request passes matching and snapshots");
}

#[test]
fn uninspectable_captures_retain_only_metadata_for_both_purposes() {
    let secret = "uninspectable-arbitrary-material";
    let malformed = format!("{{\"message\":\"{secret}\"");
    let partial_sse = format!("data: {secret}\n");
    let multipart = format!(
        "--sample\r\nContent-Disposition: form-data; name=\"file\"\r\nAuthorization Bearer {secret}\r\n\r\nhello\r\n--sample--\r\n"
    );
    for purpose in [Purpose::Traffic, Purpose::Cassette] {
        for (content_type, encoding, source) in [
            ("application/json", Some("gzip"), secret.as_bytes()),
            ("application/json", Some("identity, identity"), b"{}"),
            ("application/json", Some(""), b"{}"),
            ("application/json", None, malformed.as_bytes()),
            ("application/json", None, b"{}\xe2\x82"),
            (
                "multipart/mixed; boundary=sample",
                None,
                b"\xff\x00private-container",
            ),
            ("text/event-stream", None, partial_sse.as_bytes()),
            (
                "application/jsonl",
                None,
                b"{\"api_key\":\"abc\"}\n{\"echo\":",
            ),
            (
                "multipart/form-data; boundary=sample",
                None,
                multipart.as_bytes(),
            ),
        ] {
            let mut scrubber = Scrubber::new(purpose);
            let prepared = scrubber.prepare_exchange(
                "",
                [],
                RequestBodyRef::Bytes(b""),
                [("content-type", content_type)]
                    .into_iter()
                    .chain(encoding.map(|value| ("content-encoding", value))),
                Some(source),
            );
            let body = prepared.response_body.expect("body");
            let expected = json!({"bytes":source.len(), "content_type":content_type});
            assert_eq!(body.opaque().map(OpaqueBody::value), Some(expected.clone()));
            assert_eq!(
                serde_json::from_slice::<Value>(&scrubber.scrub_prepared(body)).expect("metadata"),
                expected
            );
            let prepared = scrubber.prepare_exchange(
                "",
                [],
                RequestBodyRef::Bytes(b""),
                [("content-type", content_type)]
                    .into_iter()
                    .chain(encoding.map(|value| ("content-encoding", value))),
                Some(source),
            );
            let events =
                scrubber.scrub_stream(prepared.response_body.expect("body"), [source.len()]);
            assert_eq!(events.len(), 1);
            assert_eq!(events.first().map(|event| &event.data), Some(&expected));
            assert!(!expected.to_string().contains(secret));
        }
    }
}

#[test]
fn declared_forms_keep_authoritative_learning_with_whitespace() {
    for purpose in [Purpose::Traffic, Purpose::Cassette] {
        let mut scrubber = Scrubber::new(purpose);
        let prepared = scrubber.prepare_exchange(
            "",
            [("content-type", "application/x-www-form-urlencoded")],
            RequestBodyRef::Bytes(b"api_key=abc&echo=abc value"),
            [("content-type", "application/json")],
            Some(br#"{"echo":"abc","longer":"abcd"}"#),
        );
        let request = prepared.request_body.expect("request");
        assert!(request.opaque().is_none());
        assert_eq!(
            scrubber.scrub_prepared(request),
            Bytes::from_static(b"api_key=[REDACTED]&echo=[REDACTED] value")
        );
        let response: Value = serde_json::from_slice(
            &scrubber.scrub_prepared(prepared.response_body.expect("response")),
        )
        .expect("reply");
        assert_eq!(response, json!({"echo":REDACTED,"longer":"abcd"}));
    }
}

#[test]
fn recording_rejects_conflicting_and_non_utf8_body_headers() {
    let mut scrubber = Scrubber::default();
    for headers in [
        vec![
            ("content-encoding", "identity"),
            ("content-encoding", "gzip"),
        ],
        vec![
            ("content-type", "application/json"),
            ("content-type", "application/octet-stream"),
        ],
    ] {
        let prepared =
            scrubber.prepare_exchange("", headers, RequestBodyRef::Bytes(b"{}"), [], None);
        assert!(prepared.request_body.expect("request").opaque().is_some());
    }
    for name in ["content-encoding", "content-type"] {
        let mut headers = HeaderMap::new();
        headers.insert(
            name,
            rig_core::http_client::HeaderValue::from_bytes(b"\xff").expect("header"),
        );
        let prepared = scrubber.prepare_exchange_with_headers(
            "",
            header_pairs(&headers),
            RequestBodyRef::Bytes(b"secret"),
            [],
            None,
        );
        assert!(prepared.request_body.expect("request").opaque().is_some());
    }
}

#[test]
fn binary_media_keep_exact_cassette_bytes_and_traffic_metadata() {
    let bytes = b"\xff\x00secret-raw-binary-material\xe2\x82";
    for content_type in [
        "audio/mpeg",
        "image/png",
        "video/mp4",
        "application/octet-stream",
        "application/pdf",
        "application/vnd.custom-binary",
    ] {
        for purpose in [Purpose::Traffic, Purpose::Cassette] {
            let mut scrubber = Scrubber::new(purpose);
            let prepared = scrubber.prepare_exchange(
                "",
                [],
                RequestBodyRef::Bytes(b""),
                [("content-type", content_type)],
                Some(bytes),
            );
            let body = prepared.response_body.expect("body");
            assert!(body.opaque().is_none());
            assert!(body.is_binary());
            assert_eq!(scrubber.scrub_prepared(body).as_ref(), bytes);
            let prepared = scrubber.prepare_exchange(
                "",
                [],
                RequestBodyRef::Bytes(b""),
                [("content-type", content_type)],
                Some(bytes),
            );
            assert_eq!(
                scrubber
                    .scrub_stream(prepared.response_body.expect("body"), [bytes.len()])
                    .first()
                    .map(|event| &event.data),
                Some(&json!({"bytes":bytes.len(),"content_type":content_type}))
            );
            let prepared = scrubber.prepare_exchange(
                "",
                [],
                RequestBodyRef::Bytes(b""),
                [("content-type", content_type), ("content-encoding", "gzip")],
                Some(bytes),
            );
            assert!(prepared.response_body.expect("body").opaque().is_some());
        }
    }
    // UTF-8 bodies use the text scrubber even under a binary media type.
    let mut scrubber = Scrubber::default();
    scrubber.learn_header("x-api-key", "abc");
    let prepared = scrubber.prepare_exchange(
        "",
        [],
        RequestBodyRef::Bytes(b""),
        [("content-type", "application/octet-stream")],
        Some(b"abc"),
    );
    assert_eq!(
        scrubber
            .scrub_prepared(prepared.response_body.expect("body"))
            .as_ref(),
        REDACTED.as_bytes()
    );
}

#[test]
fn json_lines_learn_short_credentials_before_rendering_any_line() {
    let source = b"{\"api_key\":\"abc\"}\n{\"echo\":\"abc\",\"longer\":\"abcd\"}\n";
    for purpose in [Purpose::Traffic, Purpose::Cassette] {
        let mut scrubber = Scrubber::new(purpose);
        let prepared = scrubber.prepare_exchange(
            "",
            [("content-type", "application/x-ndjson")],
            RequestBodyRef::Bytes(source),
            [("content-type", "application/json")],
            Some(br#"{"echo":"abc"}"#),
        );
        let body = prepared.request_body.expect("body");
        assert!(body.opaque().is_none());
        let bytes = scrubber.scrub_prepared(body);
        let lines: Vec<Value> = std::str::from_utf8(&bytes)
            .expect("text")
            .lines()
            .map(|line| serde_json::from_str(line).expect("JSON line"))
            .collect();
        assert_eq!(lines.first(), Some(&json!({"api_key":REDACTED})));
        assert_eq!(
            lines.get(1),
            Some(&json!({"echo":REDACTED,"longer":"abcd"}))
        );
        assert_eq!(
            serde_json::from_slice::<Value>(
                &scrubber.scrub_prepared(prepared.response_body.expect("response"))
            )
            .expect("JSON"),
            json!({"echo":REDACTED})
        );
        let prepared = scrubber.prepare_exchange(
            "",
            [],
            RequestBodyRef::Bytes(b""),
            [("content-type", "application/x-ndjson")],
            Some(source),
        );
        let events =
            scrubber.scrub_stream(prepared.response_body.expect("response"), [source.len()]);
        assert_eq!(
            events.get(1).map(|event| &event.data),
            Some(&json!({"echo":REDACTED,"longer":"abcd"}))
        );
    }
}

fn inspected_multipart() -> (Vec<u8>, Vec<u8>) {
    let mut raw = b"--sample\r\nContent-Disposition: form-data; name = \"api_key\"\r\n\r\nabc\r\n--sample\r\nContent-Disposition: form-data; name=\"payload\"\r\nContent-Type: application/json\r\n\r\n{\"echo\":\"abc\",\"ordinary\":17}\r\n--sample\r\nContent-Disposition: form-data; name=\"file\"; filename=\"image.bin\"\r\nContent-Type: application/octet-stream\r\n\r\n".to_vec();
    let binary = b"\xff\x00inside--sampleX\x00\xe2\x82".to_vec();
    raw.extend_from_slice(&binary);
    raw.extend_from_slice(b"\r\n--sample--\r\n");
    (raw, binary)
}

#[cfg(feature = "http")]
#[test]
fn strict_multipart_learns_text_parts_and_preserves_binary_parts() {
    let (raw, binary) = inspected_multipart();
    for purpose in [Purpose::Traffic, Purpose::Cassette] {
        let mut scrubber = Scrubber::new(purpose);
        let prepared = scrubber.prepare_exchange(
            "",
            [("content-type", "multipart/form-data; boundary=sample")],
            RequestBodyRef::Bytes(&raw),
            [("content-type", "application/json")],
            Some(br#"{"echo":"abc","ordinary":17}"#),
        );
        let request = prepared.request_body.expect("request");
        assert!(request.opaque().is_none());
        let bytes = scrubber.scrub_prepared(request);
        let parts = parse_multipart_parts(&bytes, "sample").expect("valid scrubbed multipart");
        assert_eq!(parts.len(), 3);
        assert_eq!(
            parts.first().map(|part| part.body.as_slice()),
            Some(REDACTED.as_bytes())
        );
        assert_eq!(
            parts
                .get(1)
                .map(|part| serde_json::from_slice::<Value>(&part.body).expect("JSON")),
            Some(json!({"echo":REDACTED,"ordinary":17}))
        );
        assert_eq!(parts.get(2).map(|part| &part.body), Some(&binary));
        assert_eq!(
            serde_json::from_slice::<Value>(
                &scrubber.scrub_prepared(prepared.response_body.expect("reply"))
            )
            .expect("JSON"),
            json!({"echo":REDACTED,"ordinary":17})
        );
    }
}

#[test]
fn malformed_multipart_never_passes_capture_validation() {
    let valid = "--sample\r\nContent-Disposition: form-data; name=\"message\"\r\n\r\nhello\r\n--sample--\r\n";
    for malformed in [
        valid.replace(
            "\r\n\r\nhello",
            "\r\nAuthorization Bearer private-unparsed-credential\r\n\r\nhello",
        ),
        valid.replace("name=\"message\"", "name=\"message\"; name=\"api_key\""),
        valid.replace("name=\"message\"", "name=\"api\\_key\""),
        valid.replace("name=\"message\"", "name=\"api_key"),
        valid.replace("--sample--\r\n", ""),
        valid.replace("--sample--\r\n", "--sample--trailing\r\n"),
        valid.replace("\r\n", "\n"),
        valid.replace(
            "Content-Disposition: form-data; name=\"message\"",
            "Content-Disposition: form-data",
        ),
        valid.replace(
            "\r\n\r\nhello",
            "\r\nContent-Transfer-Encoding: base64\r\n\r\nhello",
        ),
        valid.replace("\r\n\r\nhello", "\r\nContent-Encoding: gzip\r\n\r\nhello"),
        valid.replace(
            "\r\n\r\nhello",
            "\r\nContent-Type: application/json\r\n\r\nhello",
        ),
        format!("unparsed-secret\r\n{valid}"),
        format!("{valid}unparsed-secret"),
    ] {
        let mut scrubber = Scrubber::default();
        let prepared = scrubber.prepare_exchange(
            "",
            [("content-type", "multipart/form-data; boundary=sample")],
            RequestBodyRef::Bytes(malformed.as_bytes()),
            [],
            None,
        );
        let body = prepared.request_body.expect("body");
        assert!(body.opaque().is_some(), "{malformed}");
        let metadata = scrubber.scrub_prepared(body);
        assert!(!String::from_utf8_lossy(&metadata).contains("unparsed-secret"));
    }
}

#[cfg(feature = "http")]
#[tokio::test]
async fn inspected_multipart_credentials_scrub_reply_echoes_in_both_files() {
    let (request, binary) = inspected_multipart();
    let captured = capture_scrub_case(
        "",
        &[("content-type", "multipart/form-data; boundary=sample")],
        &request,
        br#"{"echo":"abc","ordinary":17}"#,
        false,
        None,
    )
    .await;
    assert_eq!(
        captured.traffic.pointer("/response/body"),
        Some(&json!({"echo":REDACTED,"ordinary":17}))
    );
    assert_eq!(
        captured.traffic.pointer("/request/body/content_type"),
        Some(&json!("multipart/form-data; boundary=sample"))
    );
    let response: Value = serde_json::from_str(
        captured
            .fixture
            .pointer("/then/body")
            .and_then(Value::as_str)
            .expect("reply"),
    )
    .expect("JSON");
    assert_eq!(response, json!({"echo":REDACTED,"ordinary":17}));
    use base64::Engine;
    let recorded = base64::prelude::BASE64_STANDARD
        .decode(
            captured
                .fixture
                .pointer("/when/body")
                .and_then(Value::as_str)
                .expect("multipart"),
        )
        .expect("base64");
    let parts = parse_multipart_parts(&recorded, "sample").expect("parts");
    assert_eq!(parts.get(2).map(|part| &part.body), Some(&binary));
}

#[cfg(feature = "http")]
#[tokio::test]
async fn strict_multipart_with_an_inline_boundary_in_binary_data_replays() {
    use crate::http::{CassetteMode, CassetteSpec, ProviderCassette, RecordVia};
    let (request, _) = inspected_multipart();
    let captured = capture_scrub_case(
        "",
        &[("content-type", "multipart/form-data; boundary=sample")],
        &request,
        b"{}",
        false,
        None,
    )
    .await;
    let replay = ProviderCassette::try_start_at(
        RecordVia::Proxy,
        "example",
        CassetteSpec::new("scrub-exchange"),
        "https://example.test",
        CassetteMode::Replay,
        captured.path.clone(),
    )
    .await
    .expect("replay");
    let reply = rig_reqwest::reqwest::Client::new()
        .post(format!("{}/submit", replay.base_url()))
        .header("content-type", "multipart/form-data; boundary=sample")
        .body(request)
        .send()
        .await
        .expect("reply");
    assert_eq!(
        reply.status().as_u16(),
        200,
        "{}",
        reply.text().await.expect("body")
    );
    replay.try_finish().await.expect("consumed");
}

#[test]
fn valid_utf8_textual_types_and_untyped_text_are_kept_and_scrubbed() {
    for purpose in [Purpose::Traffic, Purpose::Cassette] {
        for content_type in [
            Some("text/html"),
            Some("application/xml"),
            Some("application/problem+xml"),
            Some("text/plain"),
            Some("application/javascript"),
            None,
        ] {
            for source in [
                "<html>Gateway failure: abc</html>",
                "<error>abc</error>",
                "{not JSON; abc",
                "[ordinary text] abc",
            ] {
                let mut scrubber = Scrubber::new(purpose);
                let prepared = scrubber.prepare_exchange(
                    "",
                    [("authorization", "Bearer abc")],
                    RequestBodyRef::Bytes(b""),
                    content_type.map(|value| ("content-type", value)),
                    Some(source.as_bytes()),
                );
                let body = prepared.response_body.expect("body");
                assert!(body.opaque().is_none());
                assert!(!body.is_binary());
                let rendered = scrubber.scrub_prepared(body);
                let text = std::str::from_utf8(&rendered).expect("text");
                assert!(text.contains(REDACTED), "{text}");
                assert!(!text.contains("abc"), "{text}");
            }
            if let Some(content_type) = content_type {
                let mut scrubber = Scrubber::new(purpose);
                let prepared = scrubber.prepare_exchange(
                    "",
                    [],
                    RequestBodyRef::Bytes(b""),
                    [("content-type", content_type)],
                    Some(b"invalid \xff"),
                );
                assert!(prepared.response_body.expect("body").opaque().is_some());
            }
        }
    }
}

#[cfg(feature = "http")]
#[tokio::test]
async fn utf8_binary_media_requests_teach_both_sinks_about_reply_echoes() {
    let secret = "private-upload-credential-97531";
    let request = json!({"api_key":secret,"ordinary":17}).to_string();
    for streaming in [false, true] {
        let reply = json!({"echo":secret,"ordinary":17}).to_string();
        let reply = if streaming {
            format!("data: {reply}\n\ndata: [DONE]\n\n")
        } else {
            reply
        };
        let captured = capture_scrub_case(
            "",
            &[("content-type", "application/octet-stream")],
            request.as_bytes(),
            reply.as_bytes(),
            streaming,
            None,
        )
        .await;
        assert_eq!(
            captured.traffic.pointer("/request/body/api_key"),
            Some(&json!(REDACTED))
        );
        let path = if streaming {
            "/response/events/0/data/echo"
        } else {
            "/response/body/echo"
        };
        assert_eq!(captured.traffic.pointer(path), Some(&json!(REDACTED)));
        assert!(!captured.fixture.to_string().contains(secret));
        assert!(!captured.traffic.to_string().contains(secret));
        assert!(
            captured
                .fixture
                .pointer("/then/body")
                .and_then(Value::as_str)
                .is_some()
        );
    }
}

#[cfg(feature = "http")]
#[tokio::test]
async fn opaque_requests_contain_header_and_location_echoes_in_both_files() {
    let secret = "opaque-request-credential-97531";
    let request = json!({"api_key":secret}).to_string();
    let reply = json!({"echo":secret}).to_string();
    let error_header = format!("invalid {secret}");
    let location = format!("/error/{secret}");
    for streaming in [false, true] {
        let captured = capture_scrub_case_with_headers(
            "",
            &[("content-encoding", "gzip")],
            request.as_bytes(),
            reply.as_bytes(),
            streaming,
            None,
            &[
                ("x-error", error_header.as_str()),
                ("location", location.as_str()),
            ],
        )
        .await;
        assert!(!captured.traffic.to_string().contains(secret));
        assert!(!captured.fixture.to_string().contains(secret));
        assert_eq!(
            captured.traffic.pointer("/response/headers/x-error"),
            Some(&json!(REDACTED))
        );
        assert_eq!(
            captured.traffic.pointer("/response/headers/location"),
            Some(&json!(REDACTED))
        );
        assert_eq!(
            captured.traffic.pointer("/response/headers/content-type"),
            Some(&json!(if streaming {
                "text/event-stream"
            } else {
                "application/json"
            }))
        );
        let headers = captured
            .fixture
            .pointer("/then/header")
            .and_then(Value::as_array)
            .expect("headers");
        let location = headers
            .iter()
            .find(|header| header.get("name").and_then(Value::as_str) == Some("location"))
            .expect("location");
        assert_eq!(location.get("value"), Some(&json!(REDACTED)));
        assert!(
            captured
                .fixture
                .pointer("/then/body")
                .is_none_or(Value::is_null)
        );
        assert_eq!(
            captured.fixture.pointer("/then/withheld/bytes"),
            Some(&json!(reply.len()))
        );
    }
}

#[test]
fn binary_fallback_withholds_only_learned_non_prose_secrets_and_containers() {
    for purpose in [Purpose::Traffic, Purpose::Cassette] {
        for source in [
            b"\xff\x00abc\x00".as_slice(),
            b"\xff\x00abcd\x00".as_slice(),
        ] {
            let mut scrubber = Scrubber::new(purpose);
            let prepared = scrubber.prepare_exchange(
                "",
                [("x-api-key", "abc"), ("content-type", "audio/wav")],
                RequestBodyRef::Bytes(source),
                [("content-type", "application/json")],
                Some(br#"{"message":"kept reply"}"#),
            );
            let withheld = source == b"\xff\x00abc\x00";
            assert_eq!(prepared.request_withheld().is_some(), withheld);
            assert_eq!(prepared.response_withheld().is_some(), withheld);
            if !withheld {
                assert_eq!(
                    scrubber
                        .scrub_prepared(prepared.request_body.expect("request"))
                        .as_ref(),
                    source
                );
                assert_eq!(
                    scrubber
                        .scrub_prepared(prepared.response_body.expect("response"))
                        .as_ref(),
                    br#"{"message":"kept reply"}"#
                );
            }
        }
        let mut scrubber = Scrubber::new(purpose);
        scrubber.learn_text(r#"example: {"password":"prose-credential-97531"}"#);
        let mut bytes = b"\xff\x00".to_vec();
        bytes.extend_from_slice(b"prose-credential-97531");
        let prepared = scrubber.prepare_exchange(
            "",
            [],
            RequestBodyRef::Bytes(b""),
            [("content-type", "audio/wav")],
            Some(&bytes),
        );
        assert!(prepared.response_withheld().is_none());
        assert_eq!(
            scrubber
                .scrub_prepared(prepared.response_body.expect("body"))
                .as_ref(),
            bytes
        );
        for media in [
            "application/gzip",
            "application/zip",
            "application/x-zip-compressed",
            "application/x-xz",
        ] {
            let prepared = scrubber.prepare_exchange(
                "",
                [],
                RequestBodyRef::Bytes(b""),
                [("content-type", media)],
                Some(b"even valid UTF-8"),
            );
            assert!(prepared.response_withheld().is_some());
        }
    }
}

#[cfg(feature = "http")]
#[test]
fn gemini_json_error_under_event_stream_type_is_inspected() {
    let body = r#"{"error":{"code":404,"message":"Model not found","status":"NOT_FOUND"}}"#;
    for purpose in [Purpose::Traffic, Purpose::Cassette] {
        let mut scrubber = Scrubber::new(purpose);
        let prepared = scrubber.prepare_exchange(
            "",
            [],
            RequestBodyRef::Bytes(b""),
            [("content-type", "text/event-stream")],
            Some(body.as_bytes()),
        );
        assert!(prepared.response_withheld().is_none());
        let value: Value =
            serde_json::from_slice(&scrubber.scrub_prepared(prepared.response_body.expect("body")))
                .expect("JSON");
        assert_eq!(value.pointer("/error/code"), Some(&json!(404)));
        assert_eq!(value.pointer("/error/status"), Some(&json!("NOT_FOUND")));
    }
}

#[test]
fn strict_boundary_parsing_is_case_insensitive_and_unambiguous() {
    assert_eq!(
        multipart_boundary("Multipart/Form-Data; boundary= \"abc\""),
        Some("abc".into())
    );
    for value in [
        "multipart/form-data; boundary=a; boundary=b",
        "multipart/form-data; boundary=\"abc",
        "multipart/form-data; boundary=\"a\\b\"",
    ] {
        assert!(multipart_boundary(value).is_none());
    }
}

#[test]
fn typed_and_wire_binary_parts_share_the_withholding_decision() {
    use rig_core::http_client::multipart::Part;
    let bytes = b"\xff\x00abc\x00";
    let form = MultipartForm::new().part(
        Part::bytes("file", Bytes::from_static(bytes))
            .content_type("application/octet-stream".parse().expect("type")),
    );
    let mut raw = b"--sample\r\nContent-Disposition: form-data; name=\"file\"\r\nContent-Type: application/octet-stream\r\n\r\n".to_vec();
    raw.extend_from_slice(bytes);
    raw.extend_from_slice(b"\r\n--sample--\r\n");
    for request in [
        RequestBodyRef::Bytes(&raw),
        RequestBodyRef::Multipart(&form),
    ] {
        let mut scrubber = Scrubber::default();
        let prepared = scrubber.prepare_exchange(
            "",
            [
                ("x-api-key", "abc"),
                ("content-type", "multipart/form-data; boundary=sample"),
            ],
            request,
            [("content-type", "application/json")],
            Some(b"{}"),
        );
        assert!(prepared.request_withheld().is_some());
        assert!(prepared.response_withheld().is_some());
    }
}

#[cfg(feature = "http")]
#[tokio::test]
async fn sniffed_sse_learns_multiline_response_credentials_before_either_sink_renders() {
    use rig_core::http_client::framing::Framing;
    let secret = "opaque-new-credential-97531";
    let response = format!(
        "data: {{\"client_secret\":\ndata: {{\"value\":\"{secret}\"}}}}\n\ndata: {{\"echo\":\"{secret}\"}}\n\n"
    );
    for content_type in ["text/plain", "text/event-stream"] {
        let captured = capture_scrub_case_with_headers(
            "",
            &[],
            b"{}",
            response.as_bytes(),
            true,
            None,
            &[("content-type", content_type)],
        )
        .await;
        assert!(!captured.traffic.to_string().contains(secret));
        assert!(!captured.fixture.to_string().contains(secret));
        assert_eq!(
            captured
                .traffic
                .pointer("/response/events/0/data/client_secret/value"),
            Some(&json!(REDACTED))
        );
        assert_eq!(
            captured.traffic.pointer("/response/events/1/data/echo"),
            Some(&json!(REDACTED))
        );
        let body = captured
            .fixture
            .pointer("/then/body")
            .and_then(Value::as_str)
            .expect("SSE");
        let events = Framing::Sse.split(body.as_bytes());
        let echo: Value =
            serde_json::from_str(&events.get(1).expect("echo frame").as_str()).expect("JSON");
        assert_eq!(echo.get("echo"), Some(&json!(REDACTED)));
    }
}

#[cfg(feature = "http")]
#[tokio::test]
async fn utf8_and_unreadable_secret_headers_cannot_leak_echoes_in_either_file() {
    let secret = "opaque-credential-é-123456";
    let reply = json!({"echo":secret}).to_string();
    for unreadable in [false, true] {
        for streaming in [false, true] {
            let value = if unreadable {
                b"\xffopaque-credential".as_slice()
            } else {
                secret.as_bytes()
            };
            let mut headers = HeaderMap::new();
            headers.insert(
                "x-api-key",
                rig_core::http_client::HeaderValue::from_bytes(value).expect("header"),
            );
            let response = if streaming {
                format!("data: {reply}\n\n")
            } else {
                reply.clone()
            };
            let captured = capture_scrub_case_with_header_map(
                "",
                &headers,
                b"{}",
                response.as_bytes(),
                streaming,
                None,
                &[("x-error", secret)],
            )
            .await;
            assert!(!captured.traffic.to_string().contains(secret));
            assert!(!captured.fixture.to_string().contains(secret));
            if unreadable {
                assert_eq!(
                    captured.traffic.pointer("/response/headers/x-error"),
                    Some(&json!(REDACTED))
                );
                assert!(
                    captured
                        .fixture
                        .pointer("/when/body")
                        .is_none_or(Value::is_null)
                );
                assert!(
                    captured
                        .fixture
                        .pointer("/then/body")
                        .is_none_or(Value::is_null)
                );
                let path = if streaming {
                    "/response/events/0/data/bytes"
                } else {
                    "/response/body/bytes"
                };
                assert_eq!(captured.traffic.pointer(path), Some(&json!(response.len())));
            } else {
                let path = if streaming {
                    "/response/events/0/data/echo"
                } else {
                    "/response/body/echo"
                };
                assert_eq!(captured.traffic.pointer(path), Some(&json!(REDACTED)));
                assert!(
                    captured
                        .fixture
                        .pointer("/then/body")
                        .and_then(Value::as_str)
                        .is_some()
                );
            }
        }
    }
}

#[cfg(feature = "http")]
#[tokio::test]
async fn duplicate_json_credentials_and_parents_withhold_both_values_in_both_files() {
    let first = "first-private-credential-12345";
    let last = "last-private-credential-67890";
    let direct = format!(r#"{{"client_secret":"{first}","client_secret":"{last}"}}"#);
    let parent =
        format!(r#"{{"auth":{{"client_secret":"{first}"}},"auth":{{"client_secret":"{last}"}}}}"#);
    let response = json!({"echo":[first,last]}).to_string();
    let error_header = format!("{first} {last}");
    for (content_type, request) in [
        ("application/json", direct.clone()),
        ("application/json", parent),
        ("application/x-ndjson", format!("{direct}\n")),
        ("text/event-stream", format!("data: {direct}\n\n")),
    ] {
        for streaming in [false, true] {
            let captured = capture_scrub_case_with_headers(
                "",
                &[("content-type", content_type)],
                request.as_bytes(),
                response.as_bytes(),
                streaming,
                None,
                &[("x-error", error_header.as_str())],
            )
            .await;
            for output in [&captured.traffic, &captured.fixture] {
                assert!(!output.to_string().contains(first));
                assert!(!output.to_string().contains(last));
            }
            assert!(
                captured
                    .fixture
                    .pointer("/when/body")
                    .is_none_or(Value::is_null)
            );
            assert!(
                captured
                    .fixture
                    .pointer("/then/body")
                    .is_none_or(Value::is_null)
            );
            assert_eq!(
                captured.traffic.pointer("/response/headers/x-error"),
                Some(&json!(REDACTED))
            );
        }
    }
}

#[test]
fn frame_boundaries_use_the_same_inspected_format_and_reject_partial_records() {
    for content_type in [None, Some("text/plain"), Some("text/event-stream")] {
        assert!(stopped_at_frame_boundary(
            content_type.map(|value| ("content-type", value)),
            b"data: {\"done\":true}\n\n"
        ));
        assert!(!stopped_at_frame_boundary(
            content_type.map(|value| ("content-type", value)),
            b"data: {\"done\":true}\n"
        ));
    }
    for content_type in ["application/x-ndjson", "application/jsonl"] {
        assert!(stopped_at_frame_boundary(
            [("content-type", content_type)],
            b"{\"done\":true}\n"
        ));
        assert!(!stopped_at_frame_boundary(
            [("content-type", content_type)],
            b"{\"done\":true}"
        ));
        assert!(!stopped_at_frame_boundary(
            [("content-type", content_type)],
            b"{\"done\":true,\"done\":false}\n"
        ));
        assert!(!stopped_at_frame_boundary(
            [("content-type", content_type)],
            b"{\"done\":"
        ));
    }
}

#[test]
fn unique_json_preserves_scalar_types_and_rejects_nested_duplicates() {
    let source = r#"{"array":[null,true,-17,18446744073709551615,1.25,1e200,"é"],"ordinary":3}"#;
    assert_eq!(
        decode_json(source).expect("unique JSON"),
        serde_json::from_str::<Value>(source).expect("reference JSON")
    );
    for source in [r#"{"key":1,"key":2}"#, r#"[{"parent":{"key":1,"key":2}}]"#] {
        assert_eq!(decode_json(source), Err(JsonFailure::DuplicateKey));
    }
    assert_eq!(decode_json("{not JSON}"), Err(JsonFailure::Invalid));
}

#[cfg(feature = "http")]
#[tokio::test]
async fn untyped_invalid_utf8_requests_withhold_both_sinks_and_keep_no_secret_bytes() {
    use base64::Engine;
    let secret = "opaque-material-97531";
    let mut request = json!({"password":secret}).to_string().into_bytes();
    request.extend_from_slice(&[0xe2, 0x82]);
    for streaming in [false, true] {
        let response = json!({"echo":secret}).to_string();
        let response = if streaming {
            format!("data: {response}\n\n")
        } else {
            response
        };
        let captured = capture_scrub_case_with_header_map(
            "",
            &HeaderMap::new(),
            &request,
            response.as_bytes(),
            streaming,
            None,
            &[("x-error", secret)],
        )
        .await;
        assert!(!captured.traffic.to_string().contains(secret));
        assert!(!captured.fixture.to_string().contains(secret));
        assert_eq!(
            captured.traffic.pointer("/request/body"),
            Some(&json!({"bytes":request.len(),"content_type":null}))
        );
        let path = if streaming {
            "/response/events/0/data/bytes"
        } else {
            "/response/body/bytes"
        };
        assert_eq!(captured.traffic.pointer(path), Some(&json!(response.len())));
        assert!(
            captured
                .fixture
                .pointer("/when/body")
                .is_none_or(Value::is_null)
        );
        assert!(
            captured
                .fixture
                .pointer("/then/body")
                .is_none_or(Value::is_null)
        );
        for side in ["when", "then"] {
            let side = captured.fixture.get(side).expect("side");
            if let Some(body) = side.get("body").and_then(Value::as_str) {
                let bytes = if side.get("body_encoding").and_then(Value::as_str) == Some("base64") {
                    base64::prelude::BASE64_STANDARD
                        .decode(body)
                        .expect("base64")
                } else {
                    body.as_bytes().to_vec()
                };
                assert!(
                    !bytes
                        .windows(secret.len())
                        .any(|part| part == secret.as_bytes())
                );
            }
        }
    }
}

#[test]
fn json_number_decoding_uses_serde_value_semantics() {
    let source = r#"{"max_tokens":1024,"t":0.5,"long_float":0.12345678901234567890123456789,"big_integer":1844674407370955161600123}"#;
    let decoded = decode_json(source).expect("JSON");
    let reference: Value = serde_json::from_str(source).expect("serde JSON");
    assert_eq!(decoded, reference);
    assert_eq!(decoded.get("t").and_then(Value::as_f64), Some(0.5));
    for name in ["t", "long_float", "big_integer"] {
        assert!(decoded.get(name).is_some_and(Value::is_number));
    }
}

#[test]
fn frame_boundaries_use_all_recordable_reply_headers() {
    for (content_type, body) in [
        ("text/event-stream", b"data: {\"done\":true}\n\n".as_slice()),
        ("application/x-ndjson", b"{\"done\":true}\n".as_slice()),
    ] {
        let mut headers = HeaderMap::new();
        headers.insert("content-type", content_type.parse().expect("type"));
        assert!(stopped_at_frame_boundary(
            recordable_pairs(header_pairs(&headers)),
            body
        ));
        headers.insert("content-encoding", "gzip".parse().expect("encoding"));
        assert!(!stopped_at_frame_boundary(
            recordable_pairs(header_pairs(&headers)),
            body
        ));
        headers.remove("content-encoding");
        headers.append("content-type", content_type.parse().expect("type"));
        assert!(!stopped_at_frame_boundary(
            recordable_pairs(header_pairs(&headers)),
            body
        ));
        headers.insert(
            "content-type",
            rig_core::http_client::HeaderValue::from_bytes(b"\xff").expect("header"),
        );
        assert!(!stopped_at_frame_boundary(
            recordable_pairs(header_pairs(&headers)),
            body
        ));
    }
}

#[cfg(feature = "http")]
#[tokio::test]
async fn untyped_file_parts_keep_audio_and_replies_in_both_sinks() {
    use base64::Engine;
    use rig_core::http_client::multipart::Part;

    let audio = b"RIFF\xff\x00audio\xe2\x82";
    let form = MultipartForm::new()
        .part(Part::bytes("file", Bytes::from_static(audio)).filename("speech.wav"));
    let (boundary, raw) = form.encode();
    assert!(!String::from_utf8_lossy(&raw).contains("Content-Type:"));
    let mut headers = HeaderMap::new();
    headers.insert(
        "content-type",
        format!("multipart/form-data; boundary={boundary}")
            .parse()
            .expect("type"),
    );
    for request in [
        RequestBodyRef::Multipart(&form),
        RequestBodyRef::Bytes(&raw),
    ] {
        let typed = matches!(&request, RequestBodyRef::Multipart(_));
        let request_headers = if typed {
            HeaderMap::new()
        } else {
            headers.clone()
        };
        let captured = capture_scrub_request_with_header_map(
            "",
            &request_headers,
            request,
            br#"{"text":"transcribed speech","usage":{"seconds":1}}"#,
            false,
            None,
            &[],
        )
        .await;
        assert_eq!(
            captured.traffic.pointer("/response/body/text"),
            Some(&json!("transcribed speech"))
        );
        assert!(captured.fixture.pointer("/when/withheld").is_none());
        assert!(captured.fixture.pointer("/then/withheld").is_none());
        let reply: Value = serde_json::from_str(
            captured
                .fixture
                .pointer("/then/body")
                .and_then(Value::as_str)
                .expect("reply"),
        )
        .expect("JSON");
        assert_eq!(reply.get("text"), Some(&json!("transcribed speech")));
        if typed {
            assert_eq!(
                captured.traffic.pointer("/request/parts/0/name"),
                Some(&json!("file"))
            );
            assert_eq!(
                captured.traffic.pointer("/request/parts/0/filename"),
                Some(&json!("speech.wav"))
            );
        }
        let encoded = captured
            .fixture
            .pointer("/when/body")
            .and_then(Value::as_str)
            .expect("multipart body");
        let recorded = base64::engine::general_purpose::STANDARD
            .decode(encoded)
            .expect("base64");
        let content_type = captured
            .fixture
            .pointer("/when/header")
            .and_then(Value::as_array)
            .expect("headers")
            .iter()
            .find(|header| header.get("name").and_then(Value::as_str) == Some("content-type"))
            .and_then(|header| header.get("value"))
            .and_then(Value::as_str)
            .expect("type");
        let boundary = multipart_boundary(content_type).expect("boundary");
        let parts = parse_multipart_parts(&recorded, &boundary).expect("parts");
        assert_eq!(
            parts.first().map(|part| part.body.as_slice()),
            Some(audio.as_slice())
        );
    }
}

#[test]
fn untyped_non_file_parts_still_fail_closed_and_file_secrets_are_withheld() {
    use rig_core::http_client::multipart::Part;
    let bytes = b"\xff\x00abc\x00";
    for filename in [None, Some("speech.wav")] {
        let mut part = Part::bytes("file", Bytes::from_static(bytes));
        if let Some(filename) = filename {
            part = part.filename(filename);
        }
        let form = MultipartForm::new().part(part);
        let (boundary, raw) = form.encode();
        let media = format!("multipart/form-data; boundary={boundary}");
        for purpose in [Purpose::Traffic, Purpose::Cassette] {
            for learned in [false, true] {
                for request in [
                    RequestBodyRef::Multipart(&form),
                    RequestBodyRef::Bytes(&raw),
                ] {
                    let mut scrubber = Scrubber::new(purpose);
                    let prepared = scrubber.prepare_exchange(
                        "",
                        [("content-type", media.as_str())]
                            .into_iter()
                            .chain(learned.then_some(("x-api-key", "abc"))),
                        request,
                        [("content-type", "application/json")],
                        Some(b"{}"),
                    );
                    let withheld = filename.is_none() || learned;
                    assert_eq!(prepared.request_withheld().is_some(), withheld);
                    assert_eq!(prepared.response_withheld().is_some(), withheld);
                }
            }
        }
    }
}

#[cfg(feature = "http")]
#[tokio::test]
async fn recorded_query_echoes_replay_with_request_credentials_and_reject_other_changes() {
    use crate::http::{CassetteError, CassetteMode, CassetteSpec, ProviderCassette, RecordVia};
    let secret = "opaque-query-secret-97531";
    for from_header in [false, true] {
        let query = if from_header {
            format!("?echo={secret}&page_token=p2&tag=one&tag=one")
        } else {
            format!("?echo={secret}&api_key={secret}&page_token=p2&tag=one&tag=one")
        };
        let credential = format!("Bearer {secret}");
        let headers: Vec<_> = from_header
            .then_some(("authorization", credential.as_str()))
            .into_iter()
            .collect();
        let captured = capture_scrub_case(&query, &headers, b"{}", b"{}", false, None).await;
        assert!(!captured.fixture.to_string().contains(secret));
        assert!(!captured.traffic.to_string().contains(secret));
        let pairs = captured
            .fixture
            .pointer("/when/query_param")
            .and_then(Value::as_array)
            .expect("query");
        assert_eq!(
            pairs
                .iter()
                .find(|pair| pair.get("name").and_then(Value::as_str) == Some("echo"))
                .and_then(|pair| pair.get("value")),
            Some(&json!(REDACTED))
        );
        for shape in [false, true] {
            for wrong in [false, true] {
                let spec = CassetteSpec::new("scrub-exchange");
                let spec = if shape {
                    spec.shape_matched()
                } else {
                    spec.exact_matched()
                };
                let replay = ProviderCassette::try_start_at(
                    RecordVia::Proxy,
                    "example",
                    spec,
                    "https://example.test",
                    CassetteMode::Replay,
                    captured.path.clone(),
                )
                .await
                .expect("replay");
                let query = if wrong {
                    query.replace("page_token=p2", "page_token=p1")
                } else {
                    query.clone()
                };
                let mut request = rig_reqwest::reqwest::Client::builder()
                    .no_proxy()
                    .build()
                    .expect("client")
                    .post(format!("{}/submit{query}", replay.base_url()))
                    .header("content-type", "application/json")
                    .body("{}");
                for (name, value) in &headers {
                    request = request.header(*name, *value);
                }
                let response = request.send().await.expect("reply");
                assert_eq!(response.status().as_u16(), if wrong { 404 } else { 200 });
                response.bytes().await.expect("body");
                if wrong {
                    assert!(matches!(
                        replay.try_finish().await,
                        Err(CassetteError::ReplayMismatch { .. })
                    ));
                } else {
                    replay.try_finish().await.expect("same request replays");
                }
            }
        }
    }
}

#[cfg(feature = "http")]
#[tokio::test]
async fn whole_untyped_bodies_ignore_file_part_metadata_in_both_sinks() {
    let request = b"\xffuninspectable-whole-body";
    let reply = br#"{"message":"must be withheld"}"#;
    for (name, value) in [
        ("filename", "speech.wav"),
        ("content-disposition", "form-data; filename=\"speech.wav\""),
    ] {
        let mut headers = HeaderMap::new();
        headers.insert(name, value.parse().expect("header"));
        let captured =
            capture_scrub_case_with_header_map("", &headers, request, reply, false, None, &[])
                .await;
        assert_eq!(
            captured.traffic.pointer("/request/body/bytes"),
            Some(&json!(request.len()))
        );
        assert_eq!(
            captured.traffic.pointer("/response/body/bytes"),
            Some(&json!(reply.len()))
        );
        assert_eq!(
            captured.fixture.pointer("/when/withheld/bytes"),
            Some(&json!(request.len()))
        );
        assert_eq!(
            captured.fixture.pointer("/then/withheld/bytes"),
            Some(&json!(reply.len()))
        );
        assert!(
            captured
                .fixture
                .pointer("/when/body")
                .is_none_or(Value::is_null)
        );
        assert!(
            captured
                .fixture
                .pointer("/then/body")
                .is_none_or(Value::is_null)
        );
    }
}

#[cfg(feature = "http")]
#[tokio::test]
async fn pretty_streamed_json_keeps_secret_field_context_in_both_sinks() {
    let reply = b"{\n  \"password\":\n  \"xy\"\n}";
    let mut unary = None;
    for streaming in [false, true] {
        let captured = capture_scrub_case_with_headers(
            "",
            &[],
            b"{}",
            reply,
            streaming,
            None,
            &[("content-type", "application/json")],
        )
        .await;
        let expected = json!({"password":REDACTED});
        let traffic_body = if streaming {
            let events = captured
                .traffic
                .pointer("/response/events")
                .and_then(Value::as_array)
                .expect("events");
            assert_eq!(events.len(), 1);
            events
                .first()
                .and_then(|event| event.get("data"))
                .expect("data")
        } else {
            captured.traffic.pointer("/response/body").expect("body")
        };
        assert_eq!(traffic_body, &expected);
        let fixture_body: Value = serde_json::from_str(
            captured
                .fixture
                .pointer("/then/body")
                .and_then(Value::as_str)
                .expect("body"),
        )
        .expect("JSON");
        assert_eq!(fixture_body, expected);
        if streaming {
            assert_eq!(unary.as_ref(), Some(traffic_body));
        } else {
            unary = Some(traffic_body.clone());
        }
    }
}

#[cfg(feature = "http")]
#[tokio::test]
async fn learned_credentials_in_multipart_delimiters_withhold_both_sinks() {
    let secret = "opaque-boundary-credential-97531";
    for boundary in [secret.to_owned(), format!("prefix{secret}suffix")] {
        let raw = format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"one\"\r\n\r\nhello\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"two\"\r\n\r\nworld\r\n--{boundary}--\r\n"
        );
        let media = format!("multipart/form-data; boundary={boundary}");
        let credential = format!("Bearer {secret}");
        for request_side in [false, true] {
            let mut headers = vec![("authorization", credential.as_str())];
            if request_side {
                headers.push(("content-type", media.as_str()));
            }
            let reply_headers = [("content-type", media.as_str())];
            let captured = capture_scrub_case_with_headers(
                "",
                &headers,
                if request_side { raw.as_bytes() } else { b"{}" },
                if request_side { b"{}" } else { raw.as_bytes() },
                false,
                None,
                if request_side { &[] } else { &reply_headers },
            )
            .await;
            assert!(!captured.traffic.to_string().contains(secret));
            assert!(!captured.fixture.to_string().contains(secret));
            if request_side {
                assert_eq!(
                    captured.traffic.pointer("/request/body/bytes"),
                    Some(&json!(raw.len()))
                );
                assert!(
                    captured
                        .fixture
                        .pointer("/when/body")
                        .is_none_or(Value::is_null)
                );
                assert_eq!(
                    captured.fixture.pointer("/when/withheld/bytes"),
                    Some(&json!(raw.len()))
                );
            }
            let response_size = if request_side { 2 } else { raw.len() };
            assert_eq!(
                captured.traffic.pointer("/response/body/bytes"),
                Some(&json!(response_size))
            );
            assert!(
                captured
                    .fixture
                    .pointer("/then/body")
                    .is_none_or(Value::is_null)
            );
            assert_eq!(
                captured.fixture.pointer("/then/withheld/bytes"),
                Some(&json!(response_size))
            );
        }
    }
}

#[cfg(feature = "http")]
#[tokio::test]
async fn recorded_path_and_header_echoes_replay_but_ordinary_changes_miss() {
    use crate::http::{CassetteError, CassetteMode, CassetteSpec, ProviderCassette, RecordVia};
    let secret = "opaque-path-header-credential-97531";
    let credential = format!("Bearer {secret}");
    for path_echo in [true, false] {
        let suffix = if path_echo {
            format!("/{secret}?page_token=p2")
        } else {
            "?page_token=p2".to_owned()
        };
        let accept = if path_echo {
            "application/json; mode=right".to_owned()
        } else {
            format!("application/json; echo={secret}; mode=right")
        };
        let captured = capture_scrub_case(
            &suffix,
            &[
                ("authorization", credential.as_str()),
                ("accept", accept.as_str()),
            ],
            b"{}",
            b"{}",
            false,
            None,
        )
        .await;
        assert!(!captured.fixture.to_string().contains(secret));
        for shape in [false, true] {
            for wrong in [false, true] {
                let spec = CassetteSpec::new("scrub-exchange").unordered();
                let spec = if shape {
                    spec.shape_matched()
                } else {
                    spec.exact_matched()
                };
                let replay = ProviderCassette::try_start_at(
                    RecordVia::Proxy,
                    "example",
                    spec,
                    "https://example.test",
                    CassetteMode::Replay,
                    captured.path.clone(),
                )
                .await
                .expect("replay");
                let path = if wrong && path_echo {
                    "different"
                } else {
                    "submit"
                };
                let accept = if wrong && !path_echo {
                    accept.replace("mode=right", "mode=wrong")
                } else {
                    accept.clone()
                };
                let response = rig_reqwest::reqwest::Client::builder()
                    .no_proxy()
                    .build()
                    .expect("client")
                    .post(format!("{}/{path}{suffix}", replay.base_url()))
                    .header("authorization", &credential)
                    .header("accept", accept)
                    .header("content-type", "application/json")
                    .body("{}")
                    .send()
                    .await
                    .expect("reply");
                assert_eq!(response.status().as_u16(), if wrong { 404 } else { 200 });
                response.bytes().await.expect("body");
                if wrong {
                    assert!(matches!(
                        replay.try_finish().await,
                        Err(CassetteError::ReplayMismatch { .. })
                    ));
                } else {
                    replay.try_finish().await.expect("same request replays");
                }
            }
        }
    }
}

#[cfg(feature = "http")]
#[tokio::test]
async fn encoded_url_and_form_echoes_are_private_in_json_and_sse_in_both_sinks() {
    use rig_core::http_client::framing::Framing;
    for (secret, encoded) in [
        ("opaque+credential/97531", "opaque%2Bcredential%2F97531"),
        ("opaque+credential/97531", "opaque%2bcredential%2f97531"),
        ("opaque credential/97531", "opaque+credential%2f97531"),
        (
            "opaque-é+credential/97531",
            "opaque-%c3%a9%2Bcredential%2f97531",
        ),
    ] {
        for from_header in [false, true] {
            let credential = format!("Bearer {secret}");
            let headers: Vec<_> = from_header
                .then_some(("authorization", credential.as_str()))
                .into_iter()
                .collect();
            let request = if from_header {
                json!({})
            } else {
                json!({"client_secret": secret})
            }
            .to_string();
            let form = format!("e%63ho=prefix-{encoded}-tail&keep=a%2fb+%20");
            let payload =
                json!({"url":format!("https://example.test/?{form}#echo={encoded}"), "form": form});
            let expected_form = "e%63ho=prefix-[REDACTED]-tail&keep=a%2fb+%20";
            let expected = json!({"url":format!("https://example.test/?{expected_form}#echo=[REDACTED]"),"form":expected_form});
            for streaming in [false, true] {
                let response = if streaming {
                    format!("data: {payload}\n\n")
                } else {
                    payload.to_string()
                };
                let captured = capture_scrub_case(
                    "",
                    &headers,
                    request.as_bytes(),
                    response.as_bytes(),
                    streaming,
                    None,
                )
                .await;
                assert!(!captured.traffic.to_string().contains(secret));
                assert!(!captured.fixture.to_string().contains(secret));
                assert!(!captured.traffic.to_string().contains(encoded));
                assert!(!captured.fixture.to_string().contains(encoded));
                let actual = captured
                    .traffic
                    .pointer(if streaming {
                        "/response/events/0/data"
                    } else {
                        "/response/body"
                    })
                    .expect("traffic payload");
                assert_eq!(actual, &expected);
                let body = captured
                    .fixture
                    .pointer("/then/body")
                    .and_then(Value::as_str)
                    .expect("cassette body");
                let actual: Value = if streaming {
                    serde_json::from_str(
                        &Framing::Sse
                            .split(body.as_bytes())
                            .first()
                            .expect("event")
                            .as_str(),
                    )
                    .expect("JSON event")
                } else {
                    serde_json::from_str(body).expect("JSON body")
                };
                assert_eq!(actual, expected);
            }
        }
    }
}

#[test]
fn encoded_form_values_keep_spelling_and_distinguish_plus_from_space() {
    for purpose in [Purpose::Traffic, Purpose::Cassette] {
        let mut scrubber = Scrubber::new(purpose);
        let prepared = scrubber.prepare_exchange(
            "", [("x-api-key", "opaque credential/97531")], RequestBodyRef::Bytes(b""),
            [("content-type", "application/x-www-form-urlencoded")],
            Some(b"echo=prefix-opaque+credential%2f97531-tail&different=opaque%2Bcredential%2F97531&keep=%2f%2F+%20"),
        );
        assert_eq!(
            scrubber
                .scrub_prepared(prepared.response_body.expect("body"))
                .as_ref(),
            b"echo=prefix-[REDACTED]-tail&different=opaque%2Bcredential%2F97531&keep=%2f%2F+%20"
        );
    }
}

#[cfg(feature = "http")]
#[tokio::test]
async fn json_header_cookies_learn_individual_values_for_both_sinks() {
    let first = "zq9Xv7Lm2Pw4Rt8K";
    let second = "session-value-97531";
    let request = json!({"headers":{"Cookie":format!("sid={first}; other={second}; theme=dark")}})
        .to_string();
    let response = format!("sessions {first} and {second} expired");
    let captured = capture_scrub_case_with_headers(
        "",
        &[],
        request.as_bytes(),
        response.as_bytes(),
        false,
        None,
        &[("content-type", "text/plain")],
    )
    .await;
    for secret in [first, second] {
        assert!(!captured.traffic.to_string().contains(secret));
        assert!(!captured.fixture.to_string().contains(secret));
    }
    assert_eq!(
        captured.traffic.pointer("/response/body"),
        Some(&json!("sessions [REDACTED] and [REDACTED] expired"))
    );
    assert_eq!(
        captured.fixture.pointer("/then/body"),
        Some(&json!("sessions [REDACTED] and [REDACTED] expired"))
    );
}

#[cfg(feature = "http")]
#[tokio::test]
async fn raw_multipart_header_credentials_withhold_both_sinks_before_normalization() {
    let secret = "zq9X;v7Lm2Pw4Rt8K";
    let raw = format!(
        "--sample\r\nContent-Disposition: form-data; name=\"file\"\r\nX-Label: {secret}\r\n\r\nhello\r\n--sample--\r\n"
    );
    let media = "multipart/form-data; boundary=sample";
    let reply_headers = [("content-type", media)];
    for request_side in [false, true] {
        let mut headers = vec![("x-api-key", secret)];
        if request_side {
            headers.push(("content-type", media));
        }
        let captured = capture_scrub_case_with_headers(
            "",
            &headers,
            if request_side { raw.as_bytes() } else { b"{}" },
            if request_side { b"{}" } else { raw.as_bytes() },
            false,
            None,
            if request_side { &[] } else { &reply_headers },
        )
        .await;
        assert!(!captured.traffic.to_string().contains(secret));
        assert!(!captured.fixture.to_string().contains(secret));
        if request_side {
            assert_eq!(
                captured.traffic.pointer("/request/body/bytes"),
                Some(&json!(raw.len()))
            );
            assert_eq!(
                captured.fixture.pointer("/when/withheld/bytes"),
                Some(&json!(raw.len()))
            );
        }
        assert_eq!(
            captured.traffic.pointer("/response/body/bytes"),
            Some(&json!(if request_side { 2 } else { raw.len() }))
        );
        assert!(
            captured
                .fixture
                .pointer("/then/body")
                .is_none_or(Value::is_null)
        );
    }
}
