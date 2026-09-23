use super::*;

/// Scrubbing must redact, never invent. An empty wire value carries
/// nothing sensitive, and minting a placeholder for it makes a recording
/// claim the provider sent a token where it sent `""` — which silently
/// changes replay semantics for code that reads the field (found via
/// Anthropic's `content_block_start.signature`, empty on the wire).
#[test]
fn scrubbing_an_empty_value_invents_nothing() {
    let mut scrubber = CassetteScrubber::new(CassettePolicy::default());
    assert_eq!(scrubber.placeholder("", "signature_"), "");
    // A real value still redacts, and stays stable across occurrences.
    let first = scrubber.placeholder("sig-abc", "signature_");
    assert!(!first.is_empty());
    assert_eq!(scrubber.placeholder("sig-abc", "signature_"), first);
    // The empty value did not consume a counter slot.
    assert!(
        first.ends_with('1'),
        "the first real value should take placeholder 1, got {first}"
    );
}

/// The marker exists so a bulk re-record sweep abandons the tests whose
/// fixture was hand-built, instead of overwriting it. Replay, where those
/// fixtures are the point, must be unaffected.
#[test]
fn the_recording_marker_stops_only_a_recording_run() {
    assert!(skips_recording(CassetteMode::Record, "hand-derived"));
    assert!(!skips_recording(CassetteMode::Replay, "hand-derived"));
    // The ambient mode of the test process is replay, so a marked test runs.
    assert!(!skip_when_recording("hand-derived"));
}

fn query_pair(name: &str, value: &str) -> NameValue {
    NameValue {
        name: name.to_string(),
        value: value.to_string(),
    }
}

fn cassette_request(path: &str) -> CassetteRequest {
    CassetteRequest {
        path: path.to_string(),
        method: "POST".to_string(),
        query_param: Vec::new(),
        header: Vec::new(),
        body: None,
        body_encoding: BodyEncoding::Utf8,
    }
}

fn cassette_response() -> CassetteResponse {
    CassetteResponse {
        status: 200,
        header: Vec::new(),
        body: None,
        body_encoding: BodyEncoding::Utf8,
    }
}

fn incoming_request(path: &str, body: impl Into<Bytes>) -> IncomingRequest {
    IncomingRequest {
        method: Method::POST,
        uri: path.parse().expect("test URI should parse"),
        headers: axum::http::HeaderMap::new(),
        body: body.into(),
    }
}

#[test]
fn query_matches_exact_pairs_in_any_order() {
    let expected = [query_pair("a", "1"), query_pair("b", "2")];

    assert!(query_matches(Some("b=2&a=1"), &expected));
}

#[test]
fn query_matches_counts_duplicate_pairs() {
    let expected = [query_pair("a", "1"), query_pair("a", "1")];

    assert!(query_matches(Some("a=1&a=1"), &expected));
    assert!(!query_matches(Some("a=1"), &expected));
    assert!(!query_matches(Some("a=1&a=2"), &expected));
}

#[test]
fn query_matches_rejects_extra_actual_params() {
    let expected = [query_pair("a", "1")];

    assert!(!query_matches(Some("a=1&b=2"), &expected));
}

#[test]
fn query_matches_rejects_missing_actual_params() {
    let expected = [query_pair("a", "1"), query_pair("b", "2")];

    assert!(!query_matches(Some("a=1"), &expected));
}

#[test]
fn query_matches_empty_expected_only_matches_empty_actual_query() {
    assert!(query_matches(None, &[]));
    assert!(query_matches(Some(""), &[]));
    assert!(!query_matches(Some("a=1"), &[]));
}

#[test]
fn bodyless_cassette_request_only_matches_empty_actual_body() {
    let policy = CassettePolicy::default();
    let headers = axum::http::HeaderMap::new();

    assert!(body_matches(
        policy,
        &headers,
        &[],
        &[],
        None,
        BodyEncoding::Utf8
    ));
    assert!(!body_matches(
        policy,
        &headers,
        &[],
        br#"{"unexpected":true}"#,
        None,
        BodyEncoding::Utf8
    ));
}

#[test]
fn replay_matching_is_ordered_by_default() {
    let policy = CassettePolicy::default();
    let request = incoming_request("/v1/second", Bytes::new());
    let interactions = vec![
        ReplayInteraction {
            when: cassette_request("/v1/first"),
            then: cassette_response(),
            consumed: false,
        },
        ReplayInteraction {
            when: cassette_request("/v1/second"),
            then: cassette_response(),
            consumed: false,
        },
    ];

    assert_eq!(
        matching_interaction_index(policy, &interactions, &request),
        None
    );
}

#[test]
fn replay_matching_can_be_explicitly_unordered() {
    let policy = CassettePolicy {
        replay_matching: ReplayMatching::Unordered,
        ..CassettePolicy::default()
    };
    let request = incoming_request("/v1/second", Bytes::new());
    let interactions = vec![
        ReplayInteraction {
            when: cassette_request("/v1/first"),
            then: cassette_response(),
            consumed: false,
        },
        ReplayInteraction {
            when: cassette_request("/v1/second"),
            then: cassette_response(),
            consumed: false,
        },
    ];

    assert_eq!(
        matching_interaction_index(policy, &interactions, &request),
        Some(1)
    );
}

#[test]
fn replay_matching_requires_bearer_provider_auth_without_recording_its_value() {
    for provider in ["openai", "doubleword"] {
        let policy = CassettePolicy::for_scenario(
            provider,
            "agent/completion_smoke",
            ReplayMatching::Ordered,
        );
        let interaction = ReplayInteraction {
            when: cassette_request("/v1/responses"),
            then: cassette_response(),
            consumed: false,
        };
        let interactions = vec![interaction];
        let request_without_auth = incoming_request("/v1/responses", Bytes::new());
        let mut request_with_auth = incoming_request("/v1/responses", Bytes::new());
        request_with_auth.headers.insert(
            axum::http::header::AUTHORIZATION,
            HeaderValue::from_static("Bearer [REDACTED]"),
        );

        assert_eq!(
            matching_interaction_index(policy, &interactions, &request_without_auth),
            None,
            "{provider} replay should require bearer authentication"
        );
        assert_eq!(
            missing_required_headers(policy, &request_without_auth.headers),
            vec!["authorization"]
        );
        assert_eq!(
            matching_interaction_index(policy, &interactions, &request_with_auth),
            Some(0)
        );
    }
}

#[test]
fn gemini_interactions_policy_requires_api_key_header() {
    let policy = CassettePolicy::for_scenario(
        "gemini",
        "interactions_api/tool_result",
        ReplayMatching::Ordered,
    );
    let mut request = incoming_request("/v1beta/models", Bytes::new());

    assert_eq!(
        missing_required_headers(policy, &request.headers),
        vec!["x-goog-api-key"]
    );

    request
        .headers
        .insert("x-goog-api-key", HeaderValue::from_static("[REDACTED]"));

    assert!(required_headers_present(policy, &request.headers));
}

#[tokio::test]
async fn direct_recorder_omits_sigv4_headers() {
    let policy =
        CassettePolicy::for_scenario("bedrock", "agent/completion_smoke", ReplayMatching::Ordered);
    let interactions = Arc::new(Mutex::new(Vec::new()));
    let ledger_dir = assert_fs::TempDir::new().expect("ledger directory");
    let recorder = DirectRecorder {
        interactions: interactions.clone(),
        policy,
        ledger: Arc::new(relay::LedgerTarget {
            path: ledger_dir.path().join("ledger.jsonl"),
            provider: "bedrock".to_owned(),
            scenario: "agent/completion_smoke".to_owned(),
            origin: "https://bedrock-runtime.us-east-1.amazonaws.com".to_owned(),
        }),
    };

    recorder
        .record_http_interaction(
            DirectHttpRequest {
                method: "POST",
                uri: "https://bedrock-runtime.us-east-1.amazonaws.com/model/example/invoke",
                headers: [
                    ("authorization", "AWS4-HMAC-SHA256 Credential=AKIAEXAMPLE"),
                    ("x-amz-date", "20260709T000000Z"),
                    ("x-amz-security-token", "session-token"),
                    ("content-type", "application/json"),
                ],
                body: br#"{"ok":true}"#,
            },
            DirectHttpResponse {
                status: 200,
                headers: [
                    ("content-type", "application/json"),
                    ("x-amzn-requestid", "request-id"),
                ],
                body: br#"{"ok":true}"#,
            },
        )
        .await;

    let interactions = interactions.lock().await;
    let interaction = interactions
        .first()
        .expect("interaction should be recorded");
    assert_eq!(interaction.when.header.len(), 1);
    assert_eq!(interaction.when.header[0].name, "content-type");

    // The response keeps `x-amzn-requestid` verbatim (the SDK reads the AWS
    // request id off that header and nowhere else) and still drops
    // everything outside the allowlist.
    let response_headers = interaction
        .then
        .header
        .iter()
        .map(|header| (header.name.as_str(), header.value.as_str()))
        .collect::<Vec<_>>();
    assert_eq!(
        response_headers,
        vec![
            ("content-type", "application/json"),
            ("x-amzn-requestid", "request-id"),
        ]
    );
}

#[test]
fn replay_miss_diagnostics_scrub_actual_request_details() {
    let policy = CassettePolicy::default();
    let request = incoming_request(
        "/v1/miss?key=AIzaSyExampleSecretToken123456&api_key=raw-api-key",
        r#"{"url":"https://example.test/file?api_key=body-secret&access_token=body-token","id":"resp_12345678"}"#,
    );
    let interactions = vec![ReplayInteraction {
        when: cassette_request("/v1/other"),
        then: cassette_response(),
        consumed: false,
    }];

    let message = replay_miss_message(policy, &request, &interactions);

    assert!(!message.contains("AIzaSyExampleSecretToken123456"));
    assert!(!message.contains("raw-api-key"));
    assert!(!message.contains("body-secret"));
    assert!(!message.contains("body-token"));
    assert!(message.contains(REDACTED));
    // A response id is not a secret; the diagnostic names it as sent.
    assert!(message.contains("resp_12345678"));
}

#[test]
fn replay_miss_diagnostics_report_missing_required_headers_without_values() {
    let policy = CassettePolicy::for_scenario(
        "anthropic",
        "agent/completion_smoke",
        ReplayMatching::Ordered,
    );
    let request = incoming_request("/v1/messages", Bytes::new());
    let interactions = vec![ReplayInteraction {
        when: cassette_request("/v1/messages"),
        then: cassette_response(),
        consumed: false,
    }];

    let message = replay_miss_message(policy, &request, &interactions);

    assert!(message.contains("x-api-key"));
    assert!(message.contains("missing_required_headers"));
    assert!(!message.contains("Bearer"));
    assert!(!message.contains("secret"));
}

#[test]
fn replay_finish_fails_when_replay_misses_were_recorded() {
    let misses = vec![ReplayMiss {
        diagnostic:
            r#"{"actual_path":"/v1/miss","message":"Request did not match any route or mock"}"#
                .to_string(),
    }];

    let result = std::panic::catch_unwind(|| {
        assert_replay_finished(Path::new("fixture.yaml"), &[], &misses);
    });

    assert!(result.is_err());
    let message = replay_completion_failure_message(Path::new("fixture.yaml"), &[], &misses)
        .expect("recorded replay miss should produce a failure message");
    assert!(message.contains("unexpected replay request"));
    assert!(message.contains("/v1/miss"));
}

#[tokio::test]
async fn replay_request_records_unexpected_misses() {
    let state = Arc::new(Mutex::new(ReplayState {
        cassette_path: PathBuf::from("fixture.yaml"),
        interactions: vec![ReplayInteraction {
            when: cassette_request("/v1/expected"),
            then: cassette_response(),
            consumed: false,
        }],
        misses: Vec::new(),
        policy: CassettePolicy::default(),
    }));

    let response = replay_request(
        State(state.clone()),
        Method::POST,
        "/v1/unexpected".parse().expect("test URI should parse"),
        axum::http::HeaderMap::new(),
        Bytes::new(),
    )
    .await;

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let state = state.lock().await;
    assert_eq!(state.misses.len(), 1);
    assert!(state.misses[0].diagnostic.contains("/v1/unexpected"));
}

#[test]
fn sse_body_chunks_reconstruct_original_body() {
    let body = "data: {\"text\":\"hello\"}\n\ndata: {\"text\":\"snowman ☃\"}\n\n";
    let chunks = sse_body_chunks(body);
    let reconstructed = chunks
        .iter()
        .map(|chunk| std::str::from_utf8(chunk).expect("chunk should be UTF-8"))
        .collect::<String>();

    assert_eq!(reconstructed, body);
}

#[test]
fn sse_response_body_uses_multiple_fragmented_chunks() {
    let chunks = sse_body_chunks("data: one\n\ndata: two\n\n");

    assert!(
        chunks.len() > 2,
        "SSE replay should fragment events, got {chunks:?}"
    );
}

#[test]
fn sse_response_body_can_split_inside_event() {
    let body = "data: one\n\ndata: two\n\n";
    let chunks = sse_body_chunks(body);
    let first_event_end = body
        .find("\n\n")
        .map(|index| index + "\n\n".len())
        .expect("fixture should contain an event separator");

    let mut boundary = 0;
    let has_inside_event_boundary = chunks.iter().any(|chunk| {
        boundary += chunk.len();
        boundary > 0 && boundary < first_event_end
    });

    assert!(
        has_inside_event_boundary,
        "SSE replay should split inside an event, got {chunks:?}"
    );
}

#[test]
fn cassette_response_rejects_invalid_response_header_name() {
    let mut response = cassette_response();
    response.header.push(NameValue {
        name: "bad header".to_string(),
        value: "ok".to_string(),
    });

    let result = std::panic::catch_unwind(|| {
        super::cassette_response(&response, Path::new("fixture.yaml"));
    });

    assert!(result.is_err());
}

#[test]
fn cassette_response_rejects_invalid_response_header_value() {
    let mut response = cassette_response();
    response.header.push(NameValue {
        name: "x-valid-name".to_string(),
        value: "bad\nvalue".to_string(),
    });

    let result = std::panic::catch_unwind(|| {
        super::cassette_response(&response, Path::new("fixture.yaml"));
    });

    assert!(result.is_err());
}

#[tokio::test]
async fn cassette_response_preserves_non_sse_body() {
    let response = CassetteResponse {
        status: 200,
        header: vec![NameValue {
            name: "content-type".to_string(),
            value: "application/json".to_string(),
        }],
        body: Some(r#"{"ok":true}"#.to_string()),
        body_encoding: BodyEncoding::Utf8,
    };

    let response = super::cassette_response(&response, Path::new("fixture.yaml"));
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("non-SSE cassette body should collect");

    assert_eq!(body, Bytes::from_static(br#"{"ok":true}"#));
}

#[tokio::test]
async fn cassette_response_decodes_base64_body() {
    let response = CassetteResponse {
        status: 200,
        header: vec![NameValue {
            name: "content-type".to_string(),
            value: "application/vnd.amazon.eventstream".to_string(),
        }],
        body: Some(BASE64_STANDARD.encode([0, 159, 146, 150])),
        body_encoding: BodyEncoding::Base64,
    };

    let response = super::cassette_response(&response, Path::new("fixture.yaml"));
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("base64 cassette body should collect");

    assert_eq!(body, Bytes::from_static(&[0, 159, 146, 150]));
}

#[test]
fn recorded_body_base64_encodes_non_utf8_bytes() {
    let recorded = recorded_body(&[0, 159, 146, 150]);

    assert_eq!(recorded.encoding, BodyEncoding::Base64);
    assert_eq!(recorded.body.as_deref(), Some("AJ+Slg=="));
}

#[test]
#[cfg(feature = "bedrock")]
fn scrubber_reframes_binary_event_stream_payloads() {
    let generated_id = "tooluse_123456789";
    let message = EventStreamMessage::new(format!(
        r#"{{"contentBlockStart":{{"start":{{"toolUse":{{"toolUseId":"{generated_id}"}}}}}}}}"#
    ));
    let mut event_stream = Vec::new();
    write_message_to(&message, &mut event_stream).expect("event stream should encode");
    let cassette = format!(
        "when:\n  path: /model/test/converse-stream\n  method: POST\nthen:\n  status: 200\n  body: {}\n  body_encoding: base64\n",
        BASE64_STANDARD.encode(event_stream)
    );

    let scrubbed = scrub_cassette_contents(&cassette);
    assert!(
        cassette_safety_failures(Path::new("fixture.yaml"), &scrubbed).is_empty(),
        "scrubbed event stream should pass cassette safety"
    );

    let interaction = parse_cassette_interactions(Path::new("fixture.yaml"), &scrubbed)
        .into_iter()
        .next()
        .expect("cassette interaction");
    let encoded = interaction.then.body.expect("response body");
    let mut reframed = Bytes::from(
        BASE64_STANDARD
            .decode(encoded)
            .expect("base64 body should decode"),
    );
    let message = read_message_from(&mut reframed).expect("event stream should remain valid");
    let payload = std::str::from_utf8(message.payload()).expect("JSON payload should be UTF-8");
    assert!(
        payload.contains(generated_id),
        "the id is recorded verbatim"
    );
}

#[test]
fn body_matching_compares_base64_bytes() {
    let policy = CassettePolicy::default();
    let headers = axum::http::HeaderMap::new();
    let expected = BASE64_STANDARD.encode([0, 159, 146, 150]);

    assert!(body_matches(
        policy,
        &headers,
        &[],
        &[0, 159, 146, 150],
        Some(&expected),
        BodyEncoding::Base64
    ));
}

#[test]
fn safety_detects_aws_keys_in_base64_bodies_without_echoing_value() {
    let leaked = "AKIA1234567890ABCDEF";
    let cassette = format!(
        "when:\n  path: /model/test/converse-stream\n  method: POST\nthen:\n  status: 200\n  body: {}\n  body_encoding: base64\n",
        BASE64_STANDARD.encode(format!(r#"{{"leaked":"{leaked}"}}"#))
    );

    let failures = cassette_safety_failures(Path::new("fixture.yaml"), &cassette);

    assert!(
        failures
            .iter()
            .any(|failure| failure.contains("AWS access key-shaped token(s)"))
    );
    assert!(failures.iter().all(|failure| !failure.contains(leaked)));
}

#[test]
fn safety_detects_token_shaped_openai_keys_without_echoing_value() {
    let cassette = scrub_cassette_contents(
        r#"when:
  path: /v1/responses
  method: POST
then:
  status: 200
  body: '{"leaked":"sk-proj-abcdefghijklmnopqrstuvwxyz1234567890"}'
"#,
    );

    let failures = cassette_safety_failures(Path::new("fixture.yaml"), &cassette);

    assert!(
        failures
            .iter()
            .any(|failure| failure.contains("OpenAI API key-shaped token(s)"))
    );
    assert!(
        failures
            .iter()
            .all(|failure| !failure.contains("sk-proj-abcdefghijklmnopqrstuvwxyz1234567890"))
    );
}

#[test]
fn safety_detects_token_shaped_anthropic_keys_without_echoing_value() {
    let cassette = scrub_cassette_contents(
        r#"when:
  path: /v1/messages
  method: POST
then:
  status: 200
  body: '{"leaked":"sk-ant-api03-abcdefghijklmnopqrstuvwxyz1234567890"}'
"#,
    );

    let failures = cassette_safety_failures(Path::new("fixture.yaml"), &cassette);

    assert!(
        failures
            .iter()
            .any(|failure| failure.contains("Anthropic API key-shaped token(s)"))
    );
    assert!(
        failures
            .iter()
            .all(|failure| !failure.contains("sk-ant-api03-abcdefghijklmnopqrstuvwxyz1234567890"))
    );
    assert!(
        failures
            .iter()
            .all(|failure| !failure.contains("OpenAI API key-shaped token(s)"))
    );
}

#[test]
fn safety_detects_google_api_keys_without_echoing_value() {
    let cassette = scrub_cassette_contents(
        r#"when:
  path: /v1beta/models/gemini
  method: POST
then:
  status: 200
  body: '{"leaked":"AIzaSyExampleSecretToken1234567890"}'
"#,
    );

    let failures = cassette_safety_failures(Path::new("fixture.yaml"), &cassette);

    assert!(
        failures
            .iter()
            .any(|failure| failure.contains("Google API key-shaped token(s)"))
    );
    assert!(
        failures
            .iter()
            .all(|failure| !failure.contains("AIzaSyExampleSecretToken1234567890"))
    );
}

#[test]
fn safety_does_not_flag_short_sk_substrings() {
    let cassette = scrub_cassette_contents(
        r#"when:
  path: /v1/responses
  method: POST
then:
  status: 200
  body: '{"text":"the sk-etch marker is harmless"}'
"#,
    );

    assert!(cassette_safety_failures(Path::new("fixture.yaml"), &cassette).is_empty());
}

/// Provider ids are recorded verbatim: a file id, a message id and the
/// repeated file id in a later path and query all survive, and scrubbing
/// the result again changes nothing.
#[test]
fn scrubber_records_provider_ids_verbatim() {
    let cassette = r#"when:
  path: /v1/files
  method: POST
then:
  status: 200
  body: '{"id":"file_011Cb1W1wnAxQP1a6AuVcPx5","type":"file","created_at":"2026-05-14T00:18:05Z"}'
---
when:
  path: /v1/messages
  method: POST
  body: '{"source":{"type":"file","file_id":"file_011Cb1W1wnAxQP1a6AuVcPx5"}}'
then:
  status: 200
  body: '{"id":"msg_01D9wgWnWe16jLatSL7ce5Gm","content":[{"type":"text","text":"rig-file-id-page-two-verifier-8c27"}]}'
---
when:
  path: /v1/files/file_011Cb1W1wnAxQP1a6AuVcPx5
  method: DELETE
  query_param:
  - name: resource
    value: file_011Cb1W1wnAxQP1a6AuVcPx5
then:
  status: 200
  body: '{"id":"file_011Cb1W1wnAxQP1a6AuVcPx5","type":"file_deleted"}'
"#;

    let scrubbed = scrub_cassette_contents(cassette);

    assert_eq!(scrubbed.matches("file_011Cb1W1wnAxQP1a6AuVcPx5").count(), 5);
    assert!(scrubbed.contains("msg_01D9wgWnWe16jLatSL7ce5Gm"));
    assert!(!scrubbed.contains("REDACTED_"));
    assert!(scrubbed.contains("rig-file-id-page-two-verifier-8c27"));
    assert_eq!(scrub_cassette_contents(&scrubbed), scrubbed);
}

#[test]
fn scrubber_records_bedrock_tool_use_ids_verbatim() {
    let cassette = r#"when:
  path: /model/amazon.nova-lite-v1%3A0/converse
  method: POST
then:
  status: 200
  body: '{"output":{"message":{"content":[{"toolUse":{"toolUseId":"tooluse_A3wBRw65LYralyPMOxFhuj","name":"subtract","input":{"x":2,"y":5}}}]}}}'
---
when:
  path: /model/amazon.nova-lite-v1%3A0/converse
  method: POST
  body: '{"messages":[{"content":[{"toolResult":{"toolUseId":"tooluse_A3wBRw65LYralyPMOxFhuj","content":[{"text":"-3"}]}}]}]}'
then:
  status: 200
  body: '{"output":{"message":{"content":[{"text":"Done"}]}}}'
"#;

    let scrubbed = scrub_cassette_contents(cassette);

    assert_eq!(
        scrubbed.matches("tooluse_A3wBRw65LYralyPMOxFhuj").count(),
        2
    );
    assert!(!scrubbed.contains("REDACTED_"));
    assert_eq!(scrub_cassette_contents(&scrubbed), scrubbed);
}

/// An SSE recording keeps its ids and fingerprint; only the volatile date
/// header goes.
#[test]
fn scrubber_keeps_sse_json_payloads_and_drops_volatile_headers() {
    let cassette = r#"when:
  path: /v1/chat/completions
  method: POST
then:
  status: 200
  header:
  - name: date
    value: Thu, 14 May 2026 00:00:00 GMT
  - name: content-type
    value: text/event-stream
  body: "data: {\"id\":\"chatcmpl-DfEFWCScgKdeItzBxcAl2DTWsWPwj\",\"created\":1778718594,\"choices\":[{\"delta\":{\"tool_calls\":[{\"id\":\"call_vJUubymOrhXJwTYjJvSnqzAe\",\"type\":\"function\"}]}}],\"system_fingerprint\":\"fp_c27f75025a\"}\n\ndata: [DONE]\n"
"#;

    let scrubbed = scrub_cassette_contents(cassette);

    assert!(scrubbed.contains("chatcmpl-DfEFWCScgKdeItzBxcAl2DTWsWPwj"));
    assert!(scrubbed.contains("call_vJUubymOrhXJwTYjJvSnqzAe"));
    assert!(scrubbed.contains("fp_c27f75025a"));
    assert!(!scrubbed.contains("date"));
    assert!(scrubbed.contains("data: [DONE]"));
    assert!(scrubbed.contains("content-type"));
}

#[test]
fn scrubber_keeps_public_model_ids() {
    let cassette = r#"when:
  path: /v1/models
  method: GET
then:
  status: 200
  body: '{"data":[{"type":"model","id":"gpt-5.2"},{"type":"model","id":"claude-sonnet-4-6"}]}'
"#;

    let scrubbed = scrub_cassette_contents(cassette);

    assert!(scrubbed.contains("gpt-5.2"));
    assert!(scrubbed.contains("claude-sonnet-4-6"));
    assert!(!scrubbed.contains("id_REDACTED"));
}

/// Opaque replay fields are recorded verbatim, so a fixture can seed a live
/// call with the provider's own signature or ciphertext.
#[test]
fn scrubber_records_opaque_replay_fields_verbatim() {
    let cassette = r#"when:
  path: /v1/messages
  method: POST
  body: '{"messages":[{"role":"assistant","content":[{"type":"thinking","thinking":"t","signature":"ErUBCkYIBRgCIkA7/sig+bytes=="},{"type":"redacted_thinking","data":"EqQBCgY/redacted+bytes=="}]}]}'
then:
  status: 200
  body: '{"output":[{"type":"reasoning","id":"rs_0123456789abcdef","encrypted_content":"gAAAAAB/encrypted+bytes==","summary":[]}],"candidates":[{"content":{"parts":[{"text":"x","thoughtSignature":"CqMBCgY/thought+bytes=="}]}}],"data":[{"b64_json":"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="}],"images":["iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="],"id":"gen_1","obfuscation":"aB3d","safety_identifier":"u_9f8e7d","prompt_cache_key":"rig-key","system_fingerprint":"fp_c27f75025a","url":"https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbCdEf","name":"cachedContents/n3v1qk0nqz9k","nextPageToken":"cjwKDoIBCwifnJDUBhDo0c8VCipCKHZi"}'
"#;

    let scrubbed = scrub_cassette_contents(cassette);

    for verbatim in [
        "ErUBCkYIBRgCIkA7/sig+bytes==",
        "EqQBCgY/redacted+bytes==",
        "rs_0123456789abcdef",
        "gAAAAAB/encrypted+bytes==",
        "CqMBCgY/thought+bytes==",
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg==",
        "aB3d",
        "u_9f8e7d",
        "rig-key",
        "fp_c27f75025a",
        "grounding-api-redirect/AbCdEf",
        "cachedContents/n3v1qk0nqz9k",
        "cjwKDoIBCwifnJDUBhDo0c8VCipCKHZi",
    ] {
        assert!(
            scrubbed.contains(verbatim),
            "{verbatim} was scrubbed: {scrubbed}"
        );
    }
    assert!(!scrubbed.contains("REDACTED_"), "{scrubbed}");
    assert!(
        cassette_safety_failures(Path::new("fixture.yaml"), &scrubbed).is_empty(),
        "verbatim opaque fields pass the safety scan"
    );
}

/// A fixture recorded under the old policy is already in recorded form:
/// scrubbing leaves every placeholder in place, so the committed corpus
/// passes the safety scan unchanged.
#[test]
fn scrubbing_is_the_identity_on_a_legacy_placeholder_fixture() {
    let legacy = r#"when:
  path: /v1/responses
  method: POST
  body: '{"input":[{"type":"reasoning","id":"rs_REDACTED_1","encrypted_content":"encrypted_content_REDACTED_2"},{"type":"function_call","call_id":"call_REDACTED_1","id":"fc_REDACTED_1"}],"prompt_cache_key":"id_REDACTED_1"}'
then:
  status: 200
  header:
  - name: content-type
    value: application/json
  - name: x-request-id
    value: req_REDACTED_1
  body: '{"id":"resp_REDACTED_1","output":[{"type":"reasoning","id":"rs_REDACTED_2","encrypted_content":"encrypted_content_REDACTED_3","summary":[]}],"safety_identifier":"id_REDACTED_2","system_fingerprint":"fp_REDACTED_1","obfuscation":"obfuscation_REDACTED_1","name":"cachedContents/cached-REDACTED_1","data":[{"b64_json":"aGVsbG8="}]}'
"#;

    fn placeholders(text: &str) -> Vec<&str> {
        let mut found: Vec<&str> = text
            .split(|ch: char| !(ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-')))
            .filter(|token| token.contains("REDACTED_"))
            .collect();
        found.sort_unstable();
        found
    }

    let scrubbed = scrub_cassette_contents(legacy);
    assert_eq!(placeholders(&scrubbed), placeholders(legacy));
    assert!(scrubbed.contains("aGVsbG8="));
    assert!(scrubbed.contains("cachedContents/cached-REDACTED_1"));
    assert_eq!(scrub_cassette_contents(&scrubbed), scrubbed);
    assert!(cassette_safety_failures(Path::new("fixture.yaml"), &scrubbed).is_empty());
}

/// A standard-base64 blob puts `+` or `/` before a key-shaped run as often
/// as any letter; neither starts a key.
#[test]
fn safety_scans_ignore_key_prefixes_after_base64_punctuation() {
    let cassette = scrub_cassette_contents(
        r#"when:
  path: /v1/messages
  method: POST
then:
  status: 200
  body: '{"signature":"Eq/AIzaSyExampleSecretToken1234567890ABCDEFGHIJ+AKIA0123456789ABCDEF=="}'
"#,
    );
    assert!(
        cassette_safety_failures(Path::new("fixture.yaml"), &cassette).is_empty(),
        "{cassette}"
    );
}

/// Long base64 runs, which verbatim signatures, ciphertext and images now
/// are, contain every four-letter prefix eventually; a key detector must
/// require the prefix to start a token.
#[test]
fn safety_scans_ignore_key_prefixes_inside_base64_blobs() {
    let cassette = scrub_cassette_contents(
        r#"when:
  path: /v1/messages
  method: POST
then:
  status: 200
  body: '{"signature":"ErUBCkYIBRgCIkAIzaSyExampleSecretToken1234567890ABCDEFGHIJAKIA0123456789ABCDEFGHASIA0123456789ABCDEFsk-proj-notakeyhereatall0123456789abcdef","x":1}'
"#,
    );

    assert!(
        cassette_safety_failures(Path::new("fixture.yaml"), &cassette).is_empty(),
        "{cassette}"
    );
    // The same prefixes starting a token are still keys.
    let leaked = scrub_cassette_contents(
        r#"when:
  path: /v1/messages
  method: POST
then:
  status: 200
  body: '{"leaked":"AIzaSyExampleSecretToken1234567890ABCDEFGHIJ","aws":"AKIA0123456789ABCDEF"}'
"#,
    );
    let failures = cassette_safety_failures(Path::new("fixture.yaml"), &leaked);
    assert!(
        failures.iter().any(|f| f.contains("Google API key")),
        "{failures:?}"
    );
    assert!(failures.iter().any(|f| f.contains("AWS")), "{failures:?}");
}

#[test]
fn scrubber_removes_volatile_headers_and_sensitive_query_params() {
    let cassette = r#"when:
  path: /v1beta/models
  method: GET
  query_param:
  - name: key
    value: AIzaSySecret
then:
  status: 200
  header:
  - name: content-type
    value: application/json
  - name: x-request-id
    value: req_abc123456789
  - name: retry-after
    value: '2'
  - name: Retry-After
    value: 'Sunday, 06-Nov-94 08:49:37 GMT'
  - name: retry-after
    value: opaque-private-credential
  - name: set-cookie
    value: __cf_bm=secret
  body: '{}'
"#;

    let scrubbed = scrub_cassette_contents(cassette);

    assert!(scrubbed.contains("value: '[REDACTED]'"));
    assert!(!scrubbed.contains("AIzaSySecret"));
    // `x-request-id` is allowlisted (provider transport request ids are
    // feature data) and its value is recorded as sent.
    assert!(scrubbed.contains("x-request-id"));
    assert!(scrubbed.contains("req_abc123456789"));
    assert!(scrubbed.contains("name: retry-after\n    value: '2'"));
    assert!(scrubbed.contains("Sun, 06 Nov 1994 08:49:37 GMT"));
    assert!(!scrubbed.contains("opaque-private-credential"));
    assert_eq!(scrubbed, scrub_cassette_contents(&scrubbed));
    assert!(!scrubbed.contains("set-cookie"));
    assert!(scrubbed.contains("content-type"));
}

#[test]
fn scrubber_removes_sensitive_query_params_embedded_in_body_text() {
    let cassette = r#"when:
  path: /v1beta/models
  method: POST
  body: '{"url":"https://example.test/download?KEY=AIzaSyExampleSecretToken123456&API_KEY=body-secret&apikey=body-secret-2&access_token=body-token"}'
then:
  status: 200
  body: '{}'
"#;

    let scrubbed = scrub_cassette_contents(cassette);

    assert!(!scrubbed.contains("AIzaSyExampleSecretToken123456"));
    assert!(!scrubbed.contains("body-secret"));
    assert!(!scrubbed.contains("body-secret-2"));
    assert!(!scrubbed.contains("body-token"));
    assert_eq!(scrubbed.matches(REDACTED).count(), 4);
}

/// OAuth tokens in a recorded body are placeholdered at record time: a token
/// exchange or a refresh reply can never reach a committed cassette with its
/// material.
#[test]
fn oauth_token_fields_are_placeholdered_at_record_time() {
    let cassette = scrub_cassette_contents(
        r#"when:
  path: /oauth/token
  method: POST
then:
  status: 200
  body: '{"access_token":"ya29.a0AfB_secret-material","id_token":"eyJhbGciOi.secret","refresh_token":"1//0g-secret","token_type":"Bearer"}'
"#,
    );
    for secret in [
        "ya29.a0AfB_secret-material",
        "eyJhbGciOi.secret",
        "1//0g-secret",
    ] {
        assert!(
            !cassette.contains(secret),
            "{secret} survived scrubbing: {cassette}"
        );
    }
    assert!(
        cassette.contains("access_token"),
        "the key is kept, the value is not: {cassette}"
    );
    assert_eq!(
        scrub_cassette_contents(&cassette),
        cassette,
        "scrubbing is idempotent"
    );
}
