//! What a recording session keeps when the test fails or the provider
//! answers with an account failure: the fixture is never written, the
//! exchanges go to the attempt root, and the ledger names what was created.

use super::*;
use futures::FutureExt;
use serde_json::json;
use std::panic::AssertUnwindSafe;

fn client() -> rig_reqwest::reqwest::Client {
    rig_reqwest::reqwest::Client::builder()
        .no_proxy()
        .build()
        .expect("HTTP client")
}

struct Session {
    _dir: assert_fs::TempDir,
    fixture: PathBuf,
    attempts: PathBuf,
}

impl Session {
    fn new() -> Self {
        let dir = assert_fs::TempDir::new().expect("session directory");
        let fixture = dir.path().join("fixtures").join("example.yaml");
        let attempts = dir.path().join("attempts");
        Self {
            _dir: dir,
            fixture,
            attempts,
        }
    }

    async fn start(&self, spec: CassetteSpec, upstream: &httpmock::MockServer) -> ProviderCassette {
        ProviderCassette::try_start_with_attempts(
            RecordVia::Proxy,
            "openai",
            spec,
            &format!("{}/v1", upstream.base_url()),
            CassetteMode::Record,
            self.fixture.clone(),
            self.attempts.clone(),
        )
        .await
        .expect("recording session starts")
    }

    fn attempt_files(&self) -> Vec<PathBuf> {
        let mut files = Vec::new();
        let mut stack = vec![self.attempts.clone()];
        while let Some(dir) = stack.pop() {
            let Ok(entries) = fs::read_dir(&dir) else {
                continue;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                } else if path.extension().is_some_and(|ext| ext == "yaml") {
                    files.push(path);
                }
            }
        }
        files
    }
}

fn panic_message(payload: &PanicPayload) -> String {
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_owned()))
        .unwrap_or_default()
}

#[tokio::test]
async fn a_failed_recording_is_kept_as_an_attempt_and_its_state_is_cleaned_up() {
    let upstream = httpmock::MockServer::start_async().await;
    upstream
        .mock_async(|when, then| {
            when.method("POST").path("/v1/responses");
            then.status(200)
                .header("content-type", "application/json")
                .json_body(json!({"id": "resp_failed_run", "object": "response"}));
        })
        .await;
    let delete = upstream
        .mock_async(|when, then| {
            when.method("DELETE").path("/v1/responses/resp_failed_run");
            then.status(200);
        })
        .await;
    let session = Session::new();
    let cassette = session
        .start(CassetteSpec::new("guard/failed_run"), &upstream)
        .await;

    client()
        .post(format!("{}/responses", cassette.base_url()))
        .json(&json!({"model": "gpt-4.1-nano", "input": "hi"}))
        .send()
        .await
        .expect("recorded exchange");

    // The test body panicked after the provider created state.
    let failure: PanicPayload = Box::new("the assertion after the call failed");
    let resumed = AssertUnwindSafe(cassette.finish_after_test(Err(failure)))
        .catch_unwind()
        .await
        .expect_err("the test's panic is resumed");
    assert_eq!(
        panic_message(&resumed),
        "the assertion after the call failed"
    );

    assert!(
        !session.fixture.exists(),
        "a failed recording never becomes the fixture"
    );
    let attempts = session.attempt_files();
    assert_eq!(attempts.len(), 1, "{attempts:?}");
    let kept = fs::read_to_string(&attempts[0]).expect("attempt readable");
    assert!(kept.contains("resp_failed_run"), "{kept}");
    assert!(
        attempts[0].starts_with(session.attempts.join("openai").join("guard")),
        "{attempts:?}"
    );

    let ledger_path = session.attempts.join(ledger::LEDGER_FILE);
    let outstanding = ledger::outstanding(&ledger_path);
    assert_eq!(outstanding.len(), 1);
    assert_eq!(outstanding[0].id, "resp_failed_run");
    assert_eq!(
        outstanding[0].delete_url,
        format!("{}/v1/responses/resp_failed_run", upstream.base_url())
    );

    let report = ledger::clean_up(&ledger_path, |_| Some(Vec::new())).await;
    assert_eq!(report.deleted, 1);
    delete.assert_async().await;
    assert!(ledger::outstanding(&ledger_path).is_empty());
}

#[tokio::test]
async fn an_undeclared_quota_reply_is_refused_and_kept_as_an_attempt() {
    let upstream = httpmock::MockServer::start_async().await;
    upstream
        .mock_async(|when, then| {
            when.method("POST").path("/v1/messages");
            then.status(400)
                .header("content-type", "application/json")
                .body(r#"{"type":"error","error":{"type":"invalid_request_error","message":"You have reached your specified workspace API usage limits. You will regain access on 2026-10-01 at 00:00 UTC."}}"#);
        })
        .await;
    let session = Session::new();
    let cassette = session
        .start(CassetteSpec::new("guard/quota"), &upstream)
        .await;
    let status = client()
        .post(format!("{}/messages", cassette.base_url()))
        .json(&json!({"max_tokens": 1}))
        .send()
        .await
        .expect("recorded exchange")
        .status();
    assert_eq!(status, StatusCode::BAD_REQUEST);

    let refused = AssertUnwindSafe(cassette.finish())
        .catch_unwind()
        .await
        .expect_err("the recording is refused");
    let message = panic_message(&refused);
    assert!(message.contains("undeclared Quota failure"), "{message}");
    assert!(!session.fixture.exists());
    assert_eq!(session.attempt_files().len(), 1);
}

#[tokio::test]
async fn a_declared_auth_failure_is_recorded() {
    let upstream = httpmock::MockServer::start_async().await;
    upstream
        .mock_async(|when, then| {
            when.method("GET").path("/v1/models");
            then.status(401)
                .header("content-type", "application/json")
                .body(r#"{"error":{"message":"Incorrect API key provided","type":"invalid_request_error","code":"invalid_api_key"}}"#);
        })
        .await;

    // Declared on the spec.
    let session = Session::new();
    let cassette = session
        .start(
            CassetteSpec::new("guard/auth").expects_account_failure(AccountFailure::Auth),
            &upstream,
        )
        .await;
    client()
        .get(format!("{}/models", cassette.base_url()))
        .send()
        .await
        .expect("recorded exchange");
    cassette.finish().await;
    assert!(session.fixture.exists());

    // Declared by handing out the rejected key.
    let session = Session::new();
    let cassette = session
        .start(CassetteSpec::new("guard/auth"), &upstream)
        .await;
    let _ = cassette.bogus_api_key();
    client()
        .get(format!("{}/models", cassette.base_url()))
        .send()
        .await
        .expect("recorded exchange");
    cassette.finish().await;
    assert!(session.fixture.exists());

    // Undeclared, the same reply is refused.
    let session = Session::new();
    let cassette = session
        .start(CassetteSpec::new("guard/auth"), &upstream)
        .await;
    client()
        .get(format!("{}/models", cassette.base_url()))
        .send()
        .await
        .expect("recorded exchange");
    let refused = AssertUnwindSafe(cassette.finish())
        .catch_unwind()
        .await
        .expect_err("an undeclared auth failure is refused");
    assert!(panic_message(&refused).contains("undeclared Auth failure"));
    assert!(!session.fixture.exists());
}

#[tokio::test]
async fn a_stored_response_is_recorded_only_when_the_session_deletes_it() {
    let upstream = httpmock::MockServer::start_async().await;
    upstream
        .mock_async(|when, then| {
            when.method("POST").path("/v1/responses");
            then.status(200)
                .header("content-type", "application/json")
                .json_body(json!({"id": "resp_kept", "object": "response"}));
        })
        .await;
    upstream
        .mock_async(|when, then| {
            when.method("DELETE").path("/v1/responses/resp_kept");
            then.status(200)
                .header("content-type", "application/json")
                .json_body(json!({"id": "resp_kept", "deleted": true}));
        })
        .await;
    let create = |cassette: &ProviderCassette, body: serde_json::Value| {
        client()
            .post(format!("{}/responses", cassette.base_url()))
            .json(&body)
            .send()
    };

    // Stored and left behind: refused.
    let session = Session::new();
    let cassette = session
        .start(CassetteSpec::new("guard/stored"), &upstream)
        .await;
    create(&cassette, json!({"input": "hi"}))
        .await
        .expect("recorded exchange");
    let refused = AssertUnwindSafe(cassette.finish())
        .catch_unwind()
        .await
        .expect_err("stored state is refused");
    assert!(panic_message(&refused).contains("resp_kept"));
    assert!(!session.fixture.exists());

    // `store: false`: recorded.
    let session = Session::new();
    let cassette = session
        .start(CassetteSpec::new("guard/stored"), &upstream)
        .await;
    create(&cassette, json!({"input": "hi", "store": false}))
        .await
        .expect("recorded exchange");
    cassette.finish().await;
    assert!(session.fixture.exists());

    // Stored, then deleted in the same session: recorded.
    let session = Session::new();
    let cassette = session
        .start(CassetteSpec::new("guard/stored"), &upstream)
        .await;
    create(&cassette, json!({"input": "hi"}))
        .await
        .expect("recorded exchange");
    client()
        .delete(format!("{}/responses/resp_kept", cassette.base_url()))
        .send()
        .await
        .expect("recorded delete");
    cassette.finish().await;
    assert!(session.fixture.exists());
}

#[test]
fn only_an_accepted_delete_removes_a_stored_response() {
    let exchange = |method: &str, path: &str, request: &str, status: u16, reply: &str| {
        format!(
            "when:\n  path: {path}\n  method: {method}\n  body: '{request}'\nthen:\n  status: {status}\n  body: '{reply}'\n"
        )
    };
    let create = exchange("POST", "/v1/responses", "{}", 200, r#"{"id":"resp_1"}"#);
    let refused = exchange("DELETE", "/v1/responses/resp_1", "", 500, "{}");
    let accepted = exchange("DELETE", "/v1/responses/resp_1", "", 200, "{}");
    assert_eq!(
        cassette_stored_state("openai", &format!("{create}---\n{refused}")),
        ["resp_1"]
    );
    assert!(cassette_stored_state("openai", &format!("{create}---\n{accepted}")).is_empty());
    let stateless = exchange(
        "POST",
        "/v1/responses",
        r#"{"store":false}"#,
        200,
        r#"{"id":"resp_2"}"#,
    );
    assert!(cassette_stored_state("xai", &stateless).is_empty());
    assert!(cassette_stored_state("openrouter", &create).is_empty());
}

#[test]
fn the_attempt_root_is_in_the_target_directory_the_binary_was_built_in() {
    let target = Path::new("/work/target");
    assert_eq!(
        target_dir_of(&target.join("debug/deps/rig_cassette-0123")),
        Some(target.to_path_buf())
    );
    assert_eq!(
        target_dir_of(&target.join("release/examples/cassette_tool")),
        Some(target.to_path_buf())
    );
    assert_eq!(target_dir_of(Path::new("/usr/local/bin/tool")), None);
}

#[tokio::test]
async fn proxy_recordings_scrub_original_header_credentials_from_error_replies() {
    let upstream = httpmock::MockServer::start_async().await;
    let bearer = "opaque-proxy-credential-012345";
    let custom = "custom-proxy-credential-678901";
    upstream.mock_async(|when, then| {
        when.method("GET").path("/v1/models")
            .header("authorization", format!("Bearer {bearer}"))
            .header("x-custom-key", custom);
        then.status(401).header("content-type", "application/json")
            .json_body(json!({"error": {"message": format!("invalid {bearer} {custom} abc12345"), "type": "authentication_error"}}));
    }).await;
    let session = Session::new();
    let cassette = session
        .start(
            CassetteSpec::new("guard/proxy_credentials")
                .expects_account_failure(AccountFailure::Auth),
            &upstream,
        )
        .await;
    let response = client()
        .get(format!("{}/models", cassette.base_url()))
        .header("authorization", format!("Bearer {bearer}"))
        .header("x-custom-key", custom)
        .header("x-api-key", "abc12345")
        .send()
        .await
        .expect("proxy reply");
    assert_eq!(response.status().as_u16(), 401);
    cassette.try_finish().await.expect("scrubbed recording");
    let recorded = fs::read_to_string(&session.fixture).expect("recorded cassette");
    for secret in [bearer, custom, "abc12345"] {
        assert!(!recorded.contains(secret), "{recorded}");
    }
    let interactions = parse_cassette_interactions(&session.fixture, &recorded);
    assert_eq!(interactions.len(), 1);
    assert!(interactions.iter().all(|interaction| {
        interaction
            .when
            .header
            .iter()
            .all(|header| contains_case_insensitive(RECORDED_REQUEST_HEADERS, &header.name))
    }));
    assert_eq!(scrub_cassette_contents(&recorded), recorded);
}

/// A stream that broke midway would replay as a clean, truncated reply, so
/// it refuses the recording even when the test tolerated the error; the
/// response its first event created stays in the ledger for cleanup.
#[tokio::test]
async fn a_stream_that_failed_midway_refuses_the_recording() {
    use futures::StreamExt;
    use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};

    let upstream = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("upstream listens");
    let origin = format!("http://{}", upstream.local_addr().expect("address"));
    tokio::spawn(async move {
        let (mut socket, _) = upstream.accept().await.expect("connection");
        let mut request = [0_u8; 4096];
        let _ = socket.read(&mut request).await;
        socket
            .write_all(
                b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: 400\r\n\r\n\
                  data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_broken\"}}\n\n",
            )
            .await
            .expect("partial reply");
        // Closing short of the promised length breaks the stream.
    });
    let session = Session::new();
    let cassette = ProviderCassette::try_start_with_attempts(
        RecordVia::Proxy,
        "openai",
        CassetteSpec::new("guard/broken_stream"),
        &format!("{origin}/v1"),
        CassetteMode::Record,
        session.fixture.clone(),
        session.attempts.clone(),
    )
    .await
    .expect("recording session starts");

    let response = client()
        .post(format!("{}/responses", cassette.base_url()))
        .body(r#"{"stream":true}"#)
        .send()
        .await
        .expect("the reply starts");
    assert_eq!(response.status().as_u16(), 200);
    let chunks: Vec<_> = response.bytes_stream().collect().await;
    assert!(
        chunks.iter().any(Result::is_err),
        "the stream should fail: {chunks:?}"
    );

    let error = cassette
        .try_finish()
        .await
        .expect_err("a broken stream refuses the recording");
    let CassetteError::RecordingRefused { refusals, .. } = &error else {
        panic!("expected a refusal: {error}");
    };
    assert_eq!(refusals.len(), 1, "{refusals:?}");
    assert!(
        refusals[0].contains("POST /v1/responses failed midway"),
        "{refusals:?}"
    );
    assert!(!session.fixture.exists());
    let outstanding = ledger::outstanding(&session.attempts.join(ledger::LEDGER_FILE));
    assert_eq!(outstanding.len(), 1, "{outstanding:?}");
    assert_eq!(outstanding[0].id, "resp_broken");
}

/// An upstream that answers its connections in order with `replies`, each
/// raw, then holds the connection open for `hold`.
async fn raw_upstream(replies: Vec<Vec<u8>>, hold: std::time::Duration) -> String {
    use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("upstream listens");
    let origin = format!("http://{}", listener.local_addr().expect("address"));
    tokio::spawn(async move {
        for reply in replies {
            let (mut socket, _) = listener.accept().await.expect("connection");
            let mut request = [0_u8; 4096];
            let _ = socket.read(&mut request).await;
            socket.write_all(&reply).await.expect("reply");
            tokio::spawn(async move {
                tokio::time::sleep(hold).await;
                drop(socket);
            });
        }
    });
    origin
}

impl Session {
    async fn start_direct(&self, origin: &str) -> ProviderCassette {
        ProviderCassette::try_start_with_attempts(
            RecordVia::Direct,
            "openai",
            CassetteSpec::new("guard/direct"),
            &format!("{origin}/v1"),
            CassetteMode::Record,
            self.fixture.clone(),
            self.attempts.clone(),
        )
        .await
        .expect("recording session starts")
    }
}

fn post(origin: &str, path: &str) -> rig_core::http_client::Request<Bytes> {
    rig_core::http_client::Request::post(format!("{origin}/v1/{path}"))
        .body(Bytes::from_static(b"{}"))
        .expect("a request")
}

/// A success replays as a whole reply, so one whose body read failed, or
/// whose body nobody read, refuses the recording instead of becoming a
/// clean empty success.
#[tokio::test]
async fn a_success_whose_body_failed_or_went_unread_refuses_the_recording() {
    use rig_core::http_client::HttpClientExt;

    let origin = raw_upstream(
        vec![
            b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: 100\r\nconnection: close\r\n\r\n{\"cut"
                .to_vec(),
            b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: 2\r\n\r\n{}".to_vec(),
        ],
        std::time::Duration::ZERO,
    )
    .await;
    let session = Session::new();
    let cassette = session.start_direct(&origin).await;
    let http = cassette.direct_http();

    let cut = http
        .send::<_, Bytes>(post(&origin, "failed"))
        .await
        .expect("the reply starts");
    assert!(cut.into_body().await.is_err(), "the body read should fail");
    let unread = http
        .send::<_, Bytes>(post(&origin, "unread"))
        .await
        .expect("a reply");
    drop(unread);

    let error = cassette
        .try_finish()
        .await
        .expect_err("incomplete successes refuse the recording");
    let CassetteError::RecordingRefused { refusals, .. } = &error else {
        panic!("expected a refusal: {error}");
    };
    assert_eq!(refusals.len(), 2, "{refusals:?}");
    assert!(
        refusals[0].contains("the reply to POST /v1/failed failed while its body was read"),
        "{refusals:?}"
    );
    assert!(
        refusals[1].contains("the reply to POST /v1/unread was never read"),
        "{refusals:?}"
    );
    assert!(!session.fixture.exists());
}

/// A non-success reply the transport captured is the reply itself and is
/// kept; a binary stream stopped partway (here, inside a UTF-8 character)
/// cannot be replayed whole and refuses.
#[tokio::test]
async fn a_captured_error_is_kept_and_a_cut_binary_stream_refuses() {
    use futures::StreamExt as _;
    use rig_core::http_client::HttpClientExt;

    let mut cut =
        b"HTTP/1.1 200 OK\r\ncontent-type: application/octet-stream\r\ncontent-length: 100\r\n\r\n"
            .to_vec();
    // The first two bytes of a three-byte character.
    cut.extend_from_slice(&[0xe4, 0xbd]);
    let origin = raw_upstream(
        vec![
            b"HTTP/1.1 404 Not Found\r\ncontent-type: application/json\r\ncontent-length: 20\r\nconnection: close\r\n\r\n{\"error\":\"no route\"}"
                .to_vec(),
            cut,
        ],
        std::time::Duration::from_secs(5),
    )
    .await;
    let session = Session::new();
    let cassette = session.start_direct(&origin).await;
    let http = cassette.direct_http();

    let missing = http.send::<_, Bytes>(post(&origin, "missing")).await;
    assert_eq!(
        missing
            .err()
            .and_then(|error| error.non_success_status())
            .map(|status| status.as_u16()),
        Some(404)
    );
    let streamed = http
        .send_streaming(post(&origin, "audio"))
        .await
        .expect("the stream starts");
    let mut body = streamed.into_body();
    let first = body.next().await.expect("a chunk").expect("bytes");
    assert_eq!(first.as_ref(), [0xe4, 0xbd]);
    drop(body);

    let error = cassette
        .try_finish()
        .await
        .expect_err("a cut binary stream refuses the recording");
    let CassetteError::RecordingRefused {
        refusals, kept_at, ..
    } = &error
    else {
        panic!("expected a refusal: {error}");
    };
    assert_eq!(refusals.len(), 1, "{refusals:?}");
    assert!(
        refusals[0].contains("the streamed reply to POST /v1/audio stopped before it was complete"),
        "{refusals:?}"
    );
    let kept = fs::read_to_string(kept_at.as_ref().expect("the attempt is kept"))
        .expect("the attempt reads");
    let interactions = parse_cassette_interactions(&session.fixture, &kept);
    assert_eq!(interactions.len(), 1);
    assert_eq!(interactions[0].then.status, 404);
    assert_eq!(
        interactions[0].then.body.as_deref(),
        Some(r#"{"error":"no route"}"#)
    );
}

/// The session scrubs every exchange with one scrubber, so placeholders are
/// numbered across exchanges: two OAuth tokens stay two placeholders, and a
/// token seen again keeps its own.
#[tokio::test]
async fn oauth_tokens_keep_their_identity_across_a_sessions_exchanges() {
    let upstream = httpmock::MockServer::start_async().await;
    for (grant, token) in [
        ("first", "opaque-token-aaaaaaaa"),
        ("second", "opaque-token-bbbbbbbb"),
        ("third", "opaque-token-aaaaaaaa"),
    ] {
        upstream
            .mock_async(|when, then| {
                when.method("POST")
                    .path("/v1/oauth/token")
                    .body(format!(r#"{{"grant":"{grant}"}}"#));
                then.status(200)
                    .header("content-type", "application/json")
                    .json_body(json!({"access_token": token, "token_type": "Bearer"}));
            })
            .await;
    }
    let session = Session::new();
    let cassette = session
        .start(CassetteSpec::new("guard/oauth_identity"), &upstream)
        .await;
    for grant in ["first", "second", "third"] {
        let status = client()
            .post(format!("{}/oauth/token", cassette.base_url()))
            .body(format!(r#"{{"grant":"{grant}"}}"#))
            .send()
            .await
            .expect("proxied")
            .status();
        assert_eq!(status.as_u16(), 200);
    }
    cassette
        .try_finish()
        .await
        .expect("the recording is written");

    let recorded = fs::read_to_string(&session.fixture).expect("the fixture");
    let tokens: Vec<String> = parse_cassette_interactions(&session.fixture, &recorded)
        .iter()
        .filter_map(|interaction| interaction.then.body.as_deref())
        .filter_map(|body| serde_json::from_str::<Value>(body).ok())
        .filter_map(|body| {
            body.get("access_token")
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
        .collect();
    assert_eq!(tokens, ["id_REDACTED_1", "id_REDACTED_2", "id_REDACTED_1"]);
}

/// A stream nobody read, and a request that got no reply at all, have no
/// reply replay could serve: each refuses the recording.
#[tokio::test]
async fn an_unread_stream_and_a_request_without_reply_refuse_the_recording() {
    use rig_core::http_client::HttpClientExt;

    let origin = raw_upstream(
        vec![b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\r\ndata: {}\n\n".to_vec()],
        std::time::Duration::from_secs(5),
    )
    .await;
    let closed = {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("a free port");
        format!("http://{}", listener.local_addr().expect("address"))
    };
    let session = Session::new();
    let cassette = session.start_direct(&origin).await;
    let http = cassette.direct_http();

    let unread = http
        .send_streaming(post(&origin, "unread"))
        .await
        .expect("the stream starts");
    drop(unread);
    assert!(
        http.send::<_, Bytes>(post(&closed, "unreachable"))
            .await
            .is_err()
    );

    let error = cassette
        .try_finish()
        .await
        .expect_err("both refuse the recording");
    let CassetteError::RecordingRefused { refusals, .. } = &error else {
        panic!("expected a refusal: {error}");
    };
    assert_eq!(refusals.len(), 2, "{refusals:?}");
    assert!(
        refusals[0].contains("the streamed reply to POST /v1/unread was never read"),
        "{refusals:?}"
    );
    assert!(
        refusals[1].contains("the request POST /v1/unreachable got no reply"),
        "{refusals:?}"
    );
}

/// A proxied stream still under way when the test ends: `finish` waits for
/// it, so its whole reply reaches the fixture, and refuses the recording
/// when it does not end in time.
#[tokio::test]
async fn finish_waits_for_an_exchange_still_under_way() {
    use futures::StreamExt as _;

    let head = b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: close\r\n\r\ndata: {\"n\":1}\n\n";
    // The first stream ends 300 ms after its first event; the second never.
    let origin = raw_upstream(vec![head.to_vec()], std::time::Duration::from_millis(300)).await;
    let session = Session::new();
    let cassette = ProviderCassette::try_start_with_attempts(
        RecordVia::Proxy,
        "openai",
        CassetteSpec::new("guard/in_flight"),
        &format!("{origin}/v1"),
        CassetteMode::Record,
        session.fixture.clone(),
        session.attempts.clone(),
    )
    .await
    .expect("recording session starts");
    let response = client()
        .post(format!("{}/chat/completions", cassette.base_url()))
        .body(r#"{"stream":true}"#)
        .send()
        .await
        .expect("the stream starts");
    let mut stream = response.bytes_stream();
    stream.next().await.expect("an event").expect("bytes");
    let reader = tokio::spawn(async move { while stream.next().await.is_some() {} });

    cassette
        .try_finish()
        .await
        .expect("the stream ends in time");
    reader.await.expect("the reader");
    let recorded = fs::read_to_string(&session.fixture).expect("the fixture");
    assert!(recorded.contains(r#"data: {"n":1}"#), "{recorded}");

    let origin = raw_upstream(vec![head.to_vec()], std::time::Duration::from_secs(30)).await;
    let session = Session::new();
    let cassette = ProviderCassette::try_start_with_attempts(
        RecordVia::Proxy,
        "openai",
        CassetteSpec::new("guard/in_flight"),
        &format!("{origin}/v1"),
        CassetteMode::Record,
        session.fixture.clone(),
        session.attempts.clone(),
    )
    .await
    .expect("recording session starts");
    let response = client()
        .post(format!("{}/chat/completions", cassette.base_url()))
        .body(r#"{"stream":true}"#)
        .send()
        .await
        .expect("the stream starts");
    let mut stream = response.bytes_stream();
    stream.next().await.expect("an event").expect("bytes");
    let held = tokio::spawn(async move { while stream.next().await.is_some() {} });

    let error = cassette
        .try_finish()
        .await
        .expect_err("a stream that never ends refuses the recording");
    let CassetteError::RecordingRefused { refusals, .. } = &error else {
        panic!("expected a refusal: {error}");
    };
    assert!(
        refusals
            .iter()
            .any(|why| why.contains("1 exchanges were still under way")),
        "{refusals:?}"
    );
    held.abort();
    assert!(!session.fixture.exists());
}

/// A request cancelled before its headers, after one that completed: the
/// recording is refused, so the fixture keeps what it held instead of a
/// cassette that replay would meet with a request it cannot serve.
#[tokio::test]
async fn a_request_cancelled_before_its_headers_refuses_the_recording() {
    use rig_core::http_client::HttpClientExt;

    let origin = raw_upstream(
        vec![
            b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: 2\r\nconnection: close\r\n\r\n{}"
                .to_vec(),
            // The second connection never gets its headers.
            Vec::new(),
        ],
        std::time::Duration::from_secs(5),
    )
    .await;
    let session = Session::new();
    fs::create_dir_all(session.fixture.parent().expect("fixture parent")).expect("fixtures");
    fs::write(&session.fixture, "the fixture before\n").expect("a fixture");
    let cassette = session.start_direct(&origin).await;
    let http = cassette.direct_http();

    let done = http
        .send::<_, Bytes>(post(&origin, "done"))
        .await
        .expect("a reply");
    done.into_body().await.expect("a body");
    let cancelled = tokio::time::timeout(
        std::time::Duration::from_millis(200),
        http.send::<_, Bytes>(post(&origin, "cancelled")),
    )
    .await;
    assert!(
        cancelled.is_err(),
        "the request should wait for its headers"
    );

    let error = cassette
        .try_finish()
        .await
        .expect_err("a cancelled request refuses the recording");
    let CassetteError::RecordingRefused { refusals, .. } = &error else {
        panic!("expected a refusal: {error}");
    };
    assert_eq!(refusals.len(), 1, "{refusals:?}");
    assert!(
        refusals[0].contains(
            "the request POST /v1/cancelled got no reply (it was cancelled before its headers)"
        ),
        "{refusals:?}"
    );
    assert_eq!(
        fs::read_to_string(&session.fixture).expect("the fixture"),
        "the fixture before\n"
    );
}

/// A stream the test stopped is kept only when it is an event stream that
/// stopped between two events, as a provider's reader does after its final
/// event; a text stream stopped partway, or an event stream cut inside an
/// event, refuses.
#[tokio::test]
async fn a_stopped_stream_is_kept_only_between_events() {
    use futures::StreamExt as _;
    use rig_core::http_client::HttpClientExt;

    let origin = raw_upstream(
        vec![
            b"HTTP/1.1 200 OK\r\ncontent-type: text/plain\r\ncontent-length: 100\r\n\r\npartial text".to_vec(),
            b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\r\ndata: {\"a\":1}\n\ndata: {\"b\":".to_vec(),
            b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\r\ndata: {\"a\":1}\n\n".to_vec(),
        ],
        std::time::Duration::from_secs(5),
    )
    .await;
    let session = Session::new();
    let cassette = session.start_direct(&origin).await;
    let http = cassette.direct_http();
    for (path, sent) in [
        ("text", "partial text"),
        ("cut", "data: {\"a\":1}\n\ndata: {\"b\":"),
        ("between", "data: {\"a\":1}\n\n"),
    ] {
        let streamed = http
            .send_streaming(post(&origin, path))
            .await
            .expect("the stream starts");
        let mut body = streamed.into_body();
        let mut read = Vec::new();
        while read.len() < sent.len() {
            read.extend_from_slice(&body.next().await.expect("a chunk").expect("bytes"));
        }
        assert_eq!(read, sent.as_bytes());
        drop(body);
    }

    let error = cassette
        .try_finish()
        .await
        .expect_err("the cut streams refuse the recording");
    let CassetteError::RecordingRefused {
        refusals, kept_at, ..
    } = &error
    else {
        panic!("expected a refusal: {error}");
    };
    assert_eq!(refusals.len(), 2, "{refusals:?}");
    assert!(
        refusals[0].contains("the streamed reply to POST /v1/text stopped before it was complete"),
        "{refusals:?}"
    );
    assert!(
        refusals[1].contains("the streamed reply to POST /v1/cut stopped before it was complete"),
        "{refusals:?}"
    );
    let kept = fs::read_to_string(kept_at.as_ref().expect("the attempt is kept"))
        .expect("the attempt reads");
    let interactions = parse_cassette_interactions(&session.fixture, &kept);
    assert_eq!(interactions.len(), 1);
    assert_eq!(interactions[0].when.path, "/v1/between");
}

/// `direct_http` sends through the session's transport, so an error reply
/// reaches the fixture as the provider sent it, over 64 KiB or binary, while
/// the provider gets rig-reqwest's error for it: the same error text replay
/// gives for the fixture.
#[tokio::test]
async fn direct_error_replies_reach_the_fixture_byte_for_byte() {
    use base64::Engine as _;
    use rig_core::http_client::HttpClientExt;

    let oversized = "not found ".repeat(100 * 1024 / 10);
    let binary: Vec<u8> = (0..=255_u8).cycle().take(4096).collect();
    let upstream = httpmock::MockServer::start_async().await;
    upstream
        .mock_async(|when, then| {
            when.method("POST").path("/v1/large");
            then.status(404)
                .header("content-type", "text/plain")
                .body(oversized.clone());
        })
        .await;
    upstream
        .mock_async(|when, then| {
            when.method("POST").path("/v1/binary");
            then.status(422)
                .header("content-type", "application/octet-stream")
                .body(binary.clone());
        })
        .await;
    let origin = upstream.base_url();
    let session = Session::new();
    let cassette = session.start_direct(&origin).await;
    let http = cassette.direct_http();

    let large = http
        .send::<_, Bytes>(post(&origin, "large"))
        .await
        .err()
        .expect("a provider error");
    assert_eq!(
        large.non_success_status().map(|status| status.as_u16()),
        Some(404)
    );
    let expected = format!(
        "{}\n[Response body truncated after 64 KiB]",
        oversized.get(..64 * 1024).expect("a long body")
    );
    assert_eq!(large.non_success_body(), Some(expected.as_str()));
    let streamed = http.send_streaming(post(&origin, "binary")).await;
    assert_eq!(
        streamed
            .err()
            .and_then(|error| error.non_success_status())
            .map(|status| status.as_u16()),
        Some(422)
    );

    cassette
        .try_finish()
        .await
        .expect("the recording is written");
    let recorded = fs::read_to_string(&session.fixture).expect("the fixture");
    let interactions = parse_cassette_interactions(&session.fixture, &recorded);
    assert_eq!(interactions.len(), 2);
    assert_eq!(
        interactions[0].then.body.as_deref(),
        Some(oversized.as_str())
    );
    let kept = base64::prelude::BASE64_STANDARD
        .decode(interactions[1].then.body.as_deref().unwrap_or_default())
        .expect("a base64 body");
    assert_eq!(kept, binary);

    // Replay serves the fixture to rig-reqwest, whose error text is the one
    // the recording gave.
    let replay = ProviderCassette::try_start_with_attempts(
        RecordVia::Direct,
        "openai",
        CassetteSpec::new("guard/direct"),
        &format!("{origin}/v1"),
        CassetteMode::Replay,
        session.fixture.clone(),
        session.attempts.clone(),
    )
    .await
    .expect("the replay starts");
    let replayed = replay
        .direct_http()
        .send::<_, Bytes>(
            rig_core::http_client::Request::post(format!("{}/large", replay.base_url()))
                .header("authorization", "Bearer replay-key")
                .body(Bytes::from_static(b"{}"))
                .expect("a request"),
        )
        .await
        .err()
        .expect("the replayed error");
    assert_eq!(replayed.non_success_body(), Some(expected.as_str()));
    let _ = replay
        .direct_http()
        .send_streaming(
            rig_core::http_client::Request::post(format!("{}/binary", replay.base_url()))
                .header("authorization", "Bearer replay-key")
                .body(Bytes::from_static(b"{}"))
                .expect("a request"),
        )
        .await;
    replay
        .try_finish()
        .await
        .expect("the replay plays everything");
}

/// A credential a later exchange teaches leaves the earlier ones too: when the
/// recording is read, the session's scrubber scrubs every kept exchange again
/// with all it learned, such as a token an early reply held that a later
/// request sends as its key.
#[tokio::test]
async fn a_credential_learned_later_is_scrubbed_from_earlier_exchanges() {
    let token = "tid=opaque-copilot-0123456789;exp=1792900000";
    let upstream = httpmock::MockServer::start_async().await;
    upstream
        .mock_async(|when, then| {
            when.method("GET").path("/v1/copilot_internal/v2/token");
            then.status(200)
                .header("content-type", "application/json")
                .json_body(json!({"token": token, "expires_at": 1792900000}));
        })
        .await;
    upstream
        .mock_async(|when, then| {
            when.method("POST").path("/v1/chat/completions");
            then.status(200)
                .header("content-type", "application/json")
                .json_body(json!({"id": "chatcmpl_1"}));
        })
        .await;
    let session = Session::new();
    let cassette = session
        .start(CassetteSpec::new("guard/learned_later"), &upstream)
        .await;
    let issued = client()
        .get(format!("{}/copilot_internal/v2/token", cassette.base_url()))
        .send()
        .await
        .expect("proxied");
    assert_eq!(issued.status().as_u16(), 200);
    let used = client()
        .post(format!("{}/chat/completions", cassette.base_url()))
        .header("authorization", format!("Bearer {token}"))
        .body("{}")
        .send()
        .await
        .expect("proxied");
    assert_eq!(used.status().as_u16(), 200);

    cassette
        .try_finish()
        .await
        .expect("the recording is written");
    let recorded = fs::read_to_string(&session.fixture).expect("the fixture");
    assert!(
        !recorded.contains("opaque-copilot-0123456789"),
        "{recorded}"
    );
}

/// A proxied event stream whose provider sent its final event and keeps the
/// connection open ends when its reader goes, so `finish` does not wait for
/// the provider: the recording is written at once.
#[tokio::test]
async fn an_event_stream_ends_when_its_reader_goes() {
    use futures::StreamExt as _;

    let origin = raw_upstream(
        vec![b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\r\ndata: {\"type\":\"done\"}\n\n".to_vec()],
        std::time::Duration::from_secs(30),
    )
    .await;
    let session = Session::new();
    let cassette = ProviderCassette::try_start_with_attempts(
        RecordVia::Proxy,
        "openai",
        CassetteSpec::new("guard/reader_goes"),
        &format!("{origin}/v1"),
        CassetteMode::Record,
        session.fixture.clone(),
        session.attempts.clone(),
    )
    .await
    .expect("recording session starts");
    let response = client()
        .post(format!("{}/chat/completions", cassette.base_url()))
        .body(r#"{"stream":true}"#)
        .send()
        .await
        .expect("the stream starts");
    let mut stream = response.bytes_stream();
    let event = stream.next().await.expect("an event").expect("bytes");
    assert_eq!(event.as_ref(), b"data: {\"type\":\"done\"}\n\n");
    drop(stream);

    let started = std::time::Instant::now();
    cassette
        .try_finish()
        .await
        .expect("the recording is written");
    assert!(
        started.elapsed() < std::time::Duration::from_secs(3),
        "finish waited {:?} for the provider",
        started.elapsed()
    );
    let recorded = fs::read_to_string(&session.fixture).expect("the fixture");
    assert!(recorded.contains(r#"data: {"type":"done"}"#), "{recorded}");
}

/// A redirect keeps its `location` in the cassette, and one to the provider
/// itself is kept as its path, so replay follows both to the replay server:
/// a relative redirect and a same-origin absolute one each record two
/// exchanges, and replay plays all four.
#[tokio::test]
async fn redirects_record_and_replay_through_the_cassette() {
    let upstream = httpmock::MockServer::start_async().await;
    let absolute = format!("{}/v1/target-b", upstream.base_url());
    for (from, to, body) in [
        ("/v1/redirect-relative", "/v1/target-a".to_owned(), "a"),
        ("/v1/redirect-absolute", absolute.clone(), "b"),
    ] {
        upstream
            .mock_async(|when, then| {
                when.method("GET").path(from);
                then.status(302).header("location", to.as_str());
            })
            .await;
        let target = if body == "a" {
            "/v1/target-a"
        } else {
            "/v1/target-b"
        };
        upstream
            .mock_async(|when, then| {
                when.method("GET").path(target);
                then.status(200).body(body);
            })
            .await;
    }
    let session = Session::new();
    let follow = |base: String| async move {
        let mut bodies = Vec::new();
        for path in ["redirect-relative", "redirect-absolute"] {
            let reply = client()
                .get(format!("{base}/{path}"))
                .header("authorization", "Bearer redirect-key")
                .send()
                .await
                .expect("followed");
            assert_eq!(reply.status().as_u16(), 200);
            bodies.push(reply.text().await.expect("body"));
        }
        bodies
    };

    let recording = session
        .start(CassetteSpec::new("guard/redirects"), &upstream)
        .await;
    assert_eq!(follow(recording.base_url()).await, ["a", "b"]);
    recording
        .try_finish()
        .await
        .expect("the recording is written");
    let recorded = fs::read_to_string(&session.fixture).expect("the fixture");
    let locations: Vec<String> = parse_cassette_interactions(&session.fixture, &recorded)
        .iter()
        .flat_map(|interaction| interaction.then.header.iter())
        .filter(|header| header.name.eq_ignore_ascii_case("location"))
        .map(|header| header.value.clone())
        .collect();
    assert_eq!(locations, ["/v1/target-a", "/v1/target-b"]);

    let replay = ProviderCassette::try_start_with_attempts(
        RecordVia::Proxy,
        "openai",
        CassetteSpec::new("guard/redirects"),
        &format!("{}/v1", upstream.base_url()),
        CassetteMode::Replay,
        session.fixture.clone(),
        session.attempts.clone(),
    )
    .await
    .expect("the replay starts");
    assert_eq!(follow(replay.base_url()).await, ["a", "b"]);
    replay
        .try_finish()
        .await
        .expect("the replay plays all four exchanges");
}
