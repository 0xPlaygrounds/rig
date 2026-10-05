//! `try_start_at` and `try_finish`: every failure the panicking entry points
//! report comes back as a `CassetteError` variant, and nothing panics
//! afterwards.

use super::*;
use serde_json::json;

const UPSTREAM: &str = "https://example.invalid/v1";

struct Scratch {
    dir: assert_fs::TempDir,
}

impl Scratch {
    fn new() -> Self {
        Self {
            dir: assert_fs::TempDir::new().expect("scratch directory"),
        }
    }

    fn fixture(&self) -> PathBuf {
        self.dir.path().join("fixtures").join("example.yaml")
    }

    fn attempts(&self) -> PathBuf {
        self.dir.path().join("attempts")
    }

    /// Write `contents` to the fixture path.
    fn write_fixture(&self, contents: &str) -> PathBuf {
        let path = self.fixture();
        fs::create_dir_all(path.parent().expect("fixture parent")).expect("fixture directory");
        fs::write(&path, contents).expect("write fixture");
        path
    }

    /// A fixture of `count` ordered interactions, each matching the body
    /// `{"input": <index>}`.
    fn write_answers(&self, count: usize) -> PathBuf {
        let interactions: Vec<_> = (0..count).map(answer).collect();
        self.write_fixture(&serialize_cassette_interactions(&interactions))
    }

    async fn try_start(&self, mode: CassetteMode) -> Result<ProviderCassette, CassetteError> {
        self.try_start_with(mode, self.fixture()).await
    }

    async fn try_start_with(
        &self,
        mode: CassetteMode,
        fixture: PathBuf,
    ) -> Result<ProviderCassette, CassetteError> {
        ProviderCassette::try_start_with_attempts(
            RecordVia::Direct,
            "example",
            CassetteSpec::new("try-session"),
            UPSTREAM,
            mode,
            fixture,
            self.attempts(),
        )
        .await
    }
}

fn answer(index: usize) -> CassetteInteraction {
    CassetteInteraction {
        when: CassetteRequest {
            path: "/v1/answer".into(),
            method: "POST".into(),
            query_param: Vec::new(),
            header: vec![NameValue {
                name: "content-type".into(),
                value: "application/json".into(),
            }],
            body: Some(json!({ "input": index }).to_string()),
            body_encoding: BodyEncoding::Utf8,
        },
        then: CassetteResponse {
            status: 200,
            header: Vec::new(),
            body: Some(json!({ "answer": index }).to_string()),
            body_encoding: BodyEncoding::Utf8,
        },
    }
}

/// Post `{"input": input}` to a replay session and return the status.
async fn post(cassette: &ProviderCassette, input: usize) -> StatusCode {
    post_body(cassette, json!({ "input": input })).await
}

/// Post `body` to the session and return the status the replay answered with.
async fn post_body(cassette: &ProviderCassette, body: serde_json::Value) -> StatusCode {
    reqwest::Client::builder()
        .no_proxy()
        .build()
        .expect("HTTP client")
        .post(format!("{}/answer", cassette.base_url()))
        .json(&body)
        .send()
        .await
        .expect("local replay response")
        .status()
}

/// Hand a direct-recording session one exchange answered with `status` and
/// `body`.
async fn record(cassette: &ProviderCassette, status: u16, body: &str) {
    cassette
        .direct_recorder()
        .expect("a direct-recording session")
        .record_http_interaction(
            DirectHttpRequest {
                method: "POST",
                uri: "https://example.invalid/v1/answer",
                headers: [("content-type", "application/json")],
                body: br#"{"input":0}"#,
            },
            DirectHttpResponse {
                status,
                headers: [("content-type", "application/json")],
                body: body.as_bytes(),
            },
        )
        .await;
}

#[tokio::test]
async fn an_invalid_base_url_is_an_error() {
    let scratch = Scratch::new();
    let error = ProviderCassette::try_start_at(
        RecordVia::Proxy,
        "example",
        CassetteSpec::new("try-session"),
        "not a url",
        CassetteMode::Replay,
        scratch.fixture(),
    )
    .await
    .expect_err("the URL does not parse");

    assert!(
        matches!(&error, CassetteError::InvalidBaseUrl { url, .. } if url == "not a url"),
        "{error:?}"
    );
    assert_eq!(error.path(), scratch.fixture());
}

#[tokio::test]
async fn a_missing_fixture_is_an_error() {
    let scratch = Scratch::new();
    let error = ProviderCassette::try_start_at(
        RecordVia::Proxy,
        "example",
        CassetteSpec::new("try-session"),
        UPSTREAM,
        CassetteMode::Replay,
        scratch.fixture(),
    )
    .await
    .expect_err("there is nothing to replay");

    assert!(
        matches!(error, CassetteError::MissingFixture { ref path } if *path == scratch.fixture()),
        "{error:?}"
    );
    assert!(error.to_string().starts_with("missing provider cassette"));
}

#[tokio::test]
async fn an_unreadable_fixture_is_an_error() {
    let scratch = Scratch::new();
    fs::create_dir_all(scratch.fixture()).expect("a directory where the fixture belongs");

    let error = scratch
        .try_start(CassetteMode::Replay)
        .await
        .expect_err("a directory is not a fixture");

    assert!(
        matches!(error, CassetteError::UnreadableFixture { .. }),
        "{error:?}"
    );
}

#[tokio::test]
async fn a_malformed_fixture_is_an_error() {
    let scratch = Scratch::new();
    scratch.write_fixture("when: [not, an, interaction]\n");

    let error = scratch
        .try_start(CassetteMode::Replay)
        .await
        .expect_err("the fixture does not deserialize");

    assert!(
        matches!(error, CassetteError::MalformedFixture { .. }),
        "{error:?}"
    );
}

#[tokio::test]
async fn an_interaction_the_server_cannot_serve_is_an_error() {
    let scratch = Scratch::new();
    let mut broken = answer(1);
    broken.then.header.push(NameValue {
        name: "not a header".into(),
        value: "value".into(),
    });
    scratch.write_fixture(&serialize_cassette_interactions(&[answer(0), broken]));

    let error = scratch
        .try_start(CassetteMode::Replay)
        .await
        .expect_err("the second response is not valid HTTP");

    assert!(
        matches!(&error, CassetteError::InvalidInteraction { index: 1, reason, .. }
            if reason.contains("not a header")),
        "{error:?}"
    );
}

#[tokio::test]
async fn malformed_clock_readings_are_an_error() {
    let scratch = Scratch::new();
    let fixture = scratch.write_answers(1);
    fs::write(clock_sidecar(&fixture), "not json").expect("write clock readings");

    let error = scratch
        .try_start(CassetteMode::Replay)
        .await
        .expect_err("the clock readings do not parse");

    assert!(
        matches!(&error, CassetteError::MalformedClock { sidecar, .. }
            if *sidecar == clock_sidecar(&fixture)),
        "{error:?}"
    );
}

#[tokio::test]
async fn a_fully_played_replay_finishes() {
    let scratch = Scratch::new();
    scratch.write_answers(1);
    let cassette = scratch
        .try_start(CassetteMode::Replay)
        .await
        .expect("the fixture replays");
    assert_eq!(post(&cassette, 0).await, StatusCode::OK);

    cassette
        .try_finish()
        .await
        .expect("every interaction played");
}

#[tokio::test]
async fn an_unplayed_interaction_is_an_error() {
    let scratch = Scratch::new();
    scratch.write_answers(2);
    let cassette = scratch
        .try_start(CassetteMode::Replay)
        .await
        .expect("the fixture replays");
    assert_eq!(post(&cassette, 0).await, StatusCode::OK);

    let error = cassette
        .try_finish()
        .await
        .expect_err("the second interaction never played");

    let CassetteError::ReplayMismatch {
        unused_interactions,
        unexpected_requests,
        ..
    } = &error
    else {
        panic!("expected a replay mismatch: {error:?}");
    };
    assert_eq!(unused_interactions, &["[1] POST /v1/answer"]);
    assert!(unexpected_requests.is_empty());
    assert!(error.to_string().contains("left unused interactions"));
}

#[tokio::test]
async fn a_refused_request_is_an_error() {
    let scratch = Scratch::new();
    scratch.write_answers(1);
    let cassette = scratch
        .try_start(CassetteMode::Replay)
        .await
        .expect("the fixture replays");
    // Its field is unrecorded, so exact and shape matching both refuse it.
    assert_eq!(
        post_body(&cassette, json!({ "unrecorded": 7 })).await,
        StatusCode::NOT_FOUND
    );
    assert_eq!(post(&cassette, 0).await, StatusCode::OK);

    let error = cassette
        .try_finish()
        .await
        .expect_err("the replay refused a request");

    let CassetteError::ReplayMismatch {
        unused_interactions,
        unexpected_requests,
        ..
    } = &error
    else {
        panic!("expected a replay mismatch: {error:?}");
    };
    assert!(unused_interactions.is_empty());
    assert_eq!(unexpected_requests.len(), 1);
}

#[tokio::test]
async fn unused_clock_readings_are_an_error() {
    let scratch = Scratch::new();
    let fixture = scratch.write_answers(1);
    fs::write(clock_sidecar(&fixture), r#"{"readings":[1,2]}"#).expect("write clock readings");
    let cassette = scratch
        .try_start(CassetteMode::Replay)
        .await
        .expect("the fixture replays");
    assert_eq!(cassette.clock().now(), 1);
    assert_eq!(post(&cassette, 0).await, StatusCode::OK);

    let error = cassette
        .try_finish()
        .await
        .expect_err("one clock reading was never read");

    assert!(
        matches!(
            error,
            CassetteError::UnusedClockReadings {
                used: 1,
                recorded: 2,
                ..
            }
        ),
        "{error:?}"
    );
}

#[tokio::test]
async fn an_empty_recording_is_an_error() {
    let scratch = Scratch::new();
    let cassette = scratch
        .try_start(CassetteMode::Record)
        .await
        .expect("recording starts");

    let error = cassette
        .try_finish()
        .await
        .expect_err("nothing was exchanged");

    assert!(
        matches!(error, CassetteError::EmptyRecording { .. }),
        "{error:?}"
    );
    assert!(!scratch.fixture().exists());
}

#[tokio::test]
async fn a_refused_recording_is_an_error_that_says_where_it_was_kept() {
    let scratch = Scratch::new();
    let cassette = scratch
        .try_start(CassetteMode::Record)
        .await
        .expect("recording starts");
    record(
        &cassette,
        401,
        r#"{"error":{"message":"Incorrect API key provided","type":"invalid_request_error","code":"invalid_api_key"}}"#,
    )
    .await;

    let error = cassette
        .try_finish()
        .await
        .expect_err("an undeclared auth failure is refused");

    let CassetteError::RecordingRefused {
        refusals, kept_at, ..
    } = &error
    else {
        panic!("expected a refused recording: {error:?}");
    };
    assert!(
        refusals[0].contains("undeclared Auth failure"),
        "{refusals:?}"
    );
    let kept_at = kept_at.as_deref().expect("the recording was kept");
    assert!(kept_at.starts_with(scratch.attempts()), "{kept_at:?}");
    assert!(kept_at.exists());
    assert!(error.to_string().contains(&kept_at.display().to_string()));
    assert!(!scratch.fixture().exists());
}

#[tokio::test]
async fn an_unsafe_recording_is_an_error() {
    let scratch = Scratch::new();
    let cassette = scratch
        .try_start(CassetteMode::Record)
        .await
        .expect("recording starts");
    // Scrubbing redacts credentials, not prose that names one.
    record(&cassette, 200, r#"{"hint":"export OPENAI_API_KEY first"}"#).await;

    let error = cassette
        .try_finish()
        .await
        .expect_err("the recording still names a credential");

    assert!(
        matches!(&error, CassetteError::UnsafeRecording { failures, .. } if !failures.is_empty()),
        "{error:?}"
    );
    assert!(!scratch.fixture().exists());
}

#[tokio::test]
async fn a_fixture_that_cannot_be_written_is_an_error() {
    let scratch = Scratch::new();
    let blocker = scratch.dir.path().join("blocker");
    fs::write(&blocker, "a file, not a directory").expect("write blocker");
    let cassette = scratch
        .try_start_with(CassetteMode::Record, blocker.join("example.yaml"))
        .await
        .expect("recording starts");
    record(&cassette, 200, r#"{"answer":0}"#).await;

    let error = cassette
        .try_finish()
        .await
        .expect_err("the fixture's parent is a file");

    assert!(
        matches!(error, CassetteError::WriteFixture { .. }),
        "{error:?}"
    );
}

#[tokio::test]
async fn clock_readings_that_cannot_be_written_are_an_error() {
    let scratch = Scratch::new();
    let fixture = scratch.fixture();
    fs::create_dir_all(clock_sidecar(&fixture)).expect("a directory where the readings belong");
    let cassette = scratch
        .try_start(CassetteMode::Record)
        .await
        .expect("recording starts");
    cassette.clock().now();
    record(&cassette, 200, r#"{"answer":0}"#).await;

    let error = cassette
        .try_finish()
        .await
        .expect_err("the readings path is a directory");

    assert!(
        matches!(error, CassetteError::ClockWrite { .. }),
        "{error:?}"
    );
    assert!(fixture.exists(), "the fixture itself was written");
}

#[tokio::test]
async fn nothing_panics_after_a_failed_try_finish() {
    let scratch = Scratch::new();
    scratch.write_answers(2);
    let cassette = scratch
        .try_start(CassetteMode::Replay)
        .await
        .expect("the fixture replays");

    // `try_finish` consumes and drops the unplayed session; its drop guard
    // must stay silent because the error already reports the mismatch.
    let error = cassette
        .try_finish()
        .await
        .expect_err("neither interaction played");
    drop(error);
}
