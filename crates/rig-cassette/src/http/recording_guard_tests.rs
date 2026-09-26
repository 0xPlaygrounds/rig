//! What a recording session keeps when the test fails or the provider
//! answers with an account failure: the fixture is never written, the
//! exchanges go to the attempt root, and the ledger names what was created.

use super::*;
use serde_json::json;

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
        ProviderCassette::start_with_attempts(
            Transport::Proxy,
            "openai",
            spec,
            &format!("{}/v1", upstream.base_url()),
            CassetteMode::Record,
            self.fixture.clone(),
            self.attempts.clone(),
        )
        .await
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

fn openai_direct_recorder(ledger: &Path) -> DirectRecorder {
    DirectRecorder {
        interactions: Arc::new(Mutex::new(Vec::new())),
        policy: CassettePolicy::for_scenario("openai", "direct/stream", ReplayMatching::Ordered),
        ledger: Arc::new(relay::LedgerTarget {
            path: ledger.to_path_buf(),
            provider: "openai".to_owned(),
            scenario: "direct/stream".to_owned(),
            origin: "https://api.openai.com".to_owned(),
        }),
    }
}

/// A streamed Responses reply: the creation event names the stored response.
const CREATED_STREAM: &str = "event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_direct\"}}\n\n\
event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_direct\"}}\n\n";

async fn stream_through_direct_client(
    ledger: &Path,
    request_body: &'static str,
) -> (Vec<Vec<String>>, usize) {
    use futures::StreamExt as _;

    let stub = httpmock::MockServer::start_async().await;
    stub.mock_async(|when, then| {
        when.method("POST").path("/v1/responses");
        then.status(200)
            .header("content-type", "text/event-stream")
            .body(CREATED_STREAM);
    })
    .await;
    let client = DirectRecordingHttpClient::new(Some(openai_direct_recorder(ledger)));
    let request = HttpRequest::builder()
        .method("POST")
        .uri(format!("{}/v1/responses", stub.base_url()))
        .body(Bytes::from_static(request_body.as_bytes()))
        .expect("request");
    let response = client.send_streaming(request).await.expect("stream opens");
    let mut stream = response.into_body();
    // The ledger's ids as each chunk reaches the caller, before the next poll.
    let mut seen_per_chunk = Vec::new();
    let mut bytes = 0;
    while let Some(chunk) = stream.next().await {
        bytes += chunk.expect("chunk").len();
        seen_per_chunk.push(
            ledger::outstanding(ledger)
                .into_iter()
                .map(|resource| resource.id)
                .collect(),
        );
    }
    (seen_per_chunk, bytes)
}

#[tokio::test]
async fn the_direct_path_logs_a_streamed_creation_before_the_caller_reads_it() {
    let dir = assert_fs::TempDir::new().expect("ledger directory");
    let ledger_path = dir.path().join(ledger::LEDGER_FILE);
    let (seen, bytes) = stream_through_direct_client(&ledger_path, "{}").await;
    assert_eq!(
        bytes,
        CREATED_STREAM.len(),
        "the reply passes through whole"
    );
    assert!(!seen.is_empty(), "at least one chunk");
    // Every chunk the caller got, the first included, was preceded by the
    // ledger line of the response it names.
    for (index, ids) in seen.iter().enumerate() {
        assert_eq!(ids, &["resp_direct"], "chunk {index}");
    }
    let outstanding = ledger::outstanding(&ledger_path);
    assert_eq!(
        outstanding[0].delete_url,
        "https://api.openai.com/v1/responses/resp_direct"
    );
    assert_eq!(outstanding[0].scenario, "direct/stream");

    // `store: false` creates nothing, streamed or not.
    let dir = assert_fs::TempDir::new().expect("ledger directory");
    let stateless = dir.path().join(ledger::LEDGER_FILE);
    let (seen, _) = stream_through_direct_client(&stateless, r#"{"store":false}"#).await;
    assert!(seen.iter().all(Vec::is_empty), "{seen:?}");
}

#[test]
fn a_streamed_event_split_across_chunks_is_logged_when_it_completes() {
    let dir = assert_fs::TempDir::new().expect("ledger directory");
    let ledger_path = dir.path().join(ledger::LEDGER_FILE);
    let mut tap = StreamLedgerTap {
        recorder: openai_direct_recorder(&ledger_path),
        method: "POST".to_owned(),
        uri: "https://api.openai.com/v1/responses".to_owned(),
        request_body: Bytes::from_static(b"{}"),
        status: 200,
        pending: Vec::new(),
    };
    let (head, tail) = CREATED_STREAM.split_at(40);
    tap.feed(head.as_bytes());
    assert!(
        ledger::outstanding(&ledger_path).is_empty(),
        "half an event names nothing"
    );
    tap.feed(tail.as_bytes());
    let ids: Vec<String> = ledger::outstanding(&ledger_path)
        .into_iter()
        .map(|resource| resource.id)
        .collect();
    assert_eq!(ids, ["resp_direct"]);
    assert!(tap.pending.is_empty(), "every complete event was consumed");
}

#[test]
fn a_stream_fed_a_byte_at_a_time_is_still_logged() {
    let dir = assert_fs::TempDir::new().expect("ledger directory");
    let ledger_path = dir.path().join(ledger::LEDGER_FILE);
    let mut tap = StreamLedgerTap {
        recorder: openai_direct_recorder(&ledger_path),
        method: "POST".to_owned(),
        uri: "https://api.openai.com/v1/responses".to_owned(),
        request_body: Bytes::from_static(b"{}"),
        status: 200,
        pending: Vec::new(),
    };
    // CRLF-delimited, so every terminator spans several one-byte chunks.
    let crlf = CREATED_STREAM.replace('\n', "\r\n");
    for byte in crlf.as_bytes() {
        tap.feed(std::slice::from_ref(byte));
    }
    let ids: Vec<String> = ledger::outstanding(&ledger_path)
        .into_iter()
        .map(|resource| resource.id)
        .collect();
    assert_eq!(ids, ["resp_direct"]);
    assert!(tap.pending.is_empty(), "every complete event was consumed");
}

#[test]
fn every_stream_but_a_known_binary_one_is_tapped() {
    let headers = |content_type: &str| {
        let mut headers = http_client::HeaderMap::new();
        headers.insert(
            axum::http::header::CONTENT_TYPE,
            HeaderValue::from_str(content_type).expect("header"),
        );
        headers
    };
    assert!(StreamLedgerTap::taps(&headers(
        "text/event-stream; charset=utf-8"
    )));
    assert!(StreamLedgerTap::taps(&headers("application/json")));
    // A stream that names no type may still name what it created.
    assert!(StreamLedgerTap::taps(&http_client::HeaderMap::new()));
    for binary in [
        "audio/mpeg",
        "image/png",
        "video/mp4",
        "application/octet-stream",
    ] {
        assert!(!StreamLedgerTap::taps(&headers(binary)), "{binary}");
    }
}

fn json_tap(ledger_path: &Path) -> StreamLedgerTap {
    StreamLedgerTap {
        recorder: openai_direct_recorder(ledger_path),
        method: "POST".to_owned(),
        uri: "https://api.openai.com/v1/responses".to_owned(),
        request_body: Bytes::from_static(b"{}"),
        status: 200,
        pending: Vec::new(),
    }
}

#[test]
fn a_stream_dropped_before_its_end_still_logs_what_it_held() {
    let dir = assert_fs::TempDir::new().expect("ledger directory");
    let ledger_path = dir.path().join(ledger::LEDGER_FILE);
    // A JSON (not SSE) reply is only checked once complete: the caller reads
    // all of it but drops the stream before polling its end.
    let mut tap = json_tap(&ledger_path);
    tap.feed(br#"{"id":"resp_dropped","object":"response"}"#);
    assert!(ledger::outstanding(&ledger_path).is_empty());
    drop(tap);
    let ids: Vec<String> = ledger::outstanding(&ledger_path)
        .into_iter()
        .map(|resource| resource.id)
        .collect();
    assert_eq!(ids, ["resp_dropped"]);
}

#[test]
fn a_poisoned_tap_keeps_logging() {
    let dir = assert_fs::TempDir::new().expect("ledger directory");
    let ledger_path = dir.path().join(ledger::LEDGER_FILE);
    let shared = Arc::new(std::sync::Mutex::new(json_tap(&ledger_path)));
    let poisoner = shared.clone();
    let _ = std::thread::spawn(move || {
        let _guard = poisoner.lock();
        panic!("poison the tap's lock");
    })
    .join();
    assert!(shared.is_poisoned());
    StreamLedgerTap::locked(&shared).feed(CREATED_STREAM.as_bytes());
    let ids: Vec<String> = ledger::outstanding(&ledger_path)
        .into_iter()
        .map(|resource| resource.id)
        .collect();
    assert_eq!(ids, ["resp_direct"]);
}
