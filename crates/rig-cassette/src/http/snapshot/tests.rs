//! Request snapshots: the diff and its application, how bodies are read,
//! and replay sessions that check and write them.
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic
)]

use super::*;
use crate::http::{
    BodyEncoding, CassetteInteraction, CassetteMode, CassetteRequest, CassetteResponse,
    CassetteSpec, NameValue, ProviderCassette, RecordVia, serialize_cassette_interactions,
};
use proptest::prelude::*;
use reqwest::StatusCode;
use std::fs;

const UPSTREAM: &str = "https://example.invalid/v1";
const BOUNDARY: &str = "rigboundary";

#[test]
fn the_snapshot_sits_beside_its_fixture() {
    assert_eq!(
        request_snapshot(Path::new("cassettes/openai/a.b.yaml")),
        Path::new("cassettes/openai/a.b.requests.json")
    );
}

#[test]
fn the_mode_is_off_unless_set() {
    let path = Path::new("fixture.yaml");
    let mode = |value: Option<&str>| SnapshotMode::parse(value.map(str::to_owned), path);
    assert_eq!(mode(None).ok(), Some(SnapshotMode::Off));
    assert_eq!(mode(Some("")).ok(), Some(SnapshotMode::Off));
    assert_eq!(mode(Some("OFF")).ok(), Some(SnapshotMode::Off));
    assert_eq!(mode(Some("Check")).ok(), Some(SnapshotMode::Check));
    assert_eq!(mode(Some("write")).ok(), Some(SnapshotMode::Write));
    let error = mode(Some("rewrite")).expect_err("not a mode");
    assert!(
        matches!(&error, CassetteError::InvalidSnapshotMode { value, .. } if value == "rewrite"),
        "{error}"
    );
    assert_eq!(error.path(), path);
}

#[test]
fn equal_bodies_have_no_changes() {
    let body = json!({ "model": "m", "messages": [{ "role": "user" }] });
    assert!(diff(&body, &body).is_empty());
}

#[test]
fn changes_name_the_pointer_and_both_values() {
    let recorded = json!({
        "a/b": 1,
        "gone": true,
        "messages": [{ "content": [{ "text": "hi", "type": "text" }], "role": "system" }],
    });
    let sent = json!({
        "a/b": 2,
        "messages": [{ "content": "hi", "role": "system" }],
        "new": null,
    });
    let changes = serde_json::to_value(diff(&recorded, &sent)).expect("changes serialize");
    assert_eq!(
        changes,
        json!([
            { "path": "/a~1b", "recorded": 1, "sent": 2 },
            { "path": "/gone", "recorded": true },
            {
                "path": "/messages/0/content",
                "recorded": [{ "text": "hi", "type": "text" }],
                "sent": "hi",
            },
            { "path": "/new", "sent": null },
        ])
    );
}

#[test]
fn an_inserted_item_is_one_splice() {
    let recorded = json!({ "messages": ["a", "b", "d"] });
    let sent = json!({ "messages": ["a", "b", "c", "d"] });
    let changes = diff(&recorded, &sent);
    assert_eq!(
        serde_json::to_value(&changes).expect("changes serialize"),
        json!([{ "path": "/messages", "splice": 2, "recorded": [], "sent": ["c"] }])
    );
    assert_eq!(apply(&recorded, &changes).ok(), Some(sent));
}

#[test]
fn a_null_value_survives_the_file() {
    let changes = diff(&json!({ "a": null }), &json!({ "a": 1 }));
    let text = serde_json::to_string(&changes).expect("changes serialize");
    let read: Vec<Change> = serde_json::from_str(&text).expect("changes parse");
    assert_eq!(read, changes);
    assert_eq!(
        read.first().and_then(|change| change.recorded.clone()),
        Some(Value::Null)
    );
}

#[test]
fn a_change_that_no_longer_fits_the_recording_is_refused() {
    let changes = diff(&json!({ "a": 1 }), &json!({ "a": 2 }));
    let error = apply(&json!({ "a": 3 }), &changes).expect_err("stale change");
    assert!(error.starts_with("/a: "), "{error}");
    let added = diff(&json!({}), &json!({ "a": 2 }));
    assert!(apply(&json!({ "a": 2 }), &added).is_err());
    let spliced = diff(&json!(["a", "b"]), &json!(["a"]));
    assert!(apply(&json!(["a", "c"]), &spliced).is_err());
}

#[test]
fn a_difference_reads_as_one_line_per_change() {
    let lines = describe(
        &json!({ "messages": ["a"], "model": "m" }),
        &json!({ "messages": ["a", "b"], "stream": true }),
    );
    assert_eq!(
        lines,
        [
            r#"  /messages from item 1: snapshot [], sent ["b"]"#,
            r#"  /model: snapshot "m", not sent"#,
            "  /stream: not in the snapshot, sent true",
        ]
    );
    let many = describe(&json!((0..30).collect::<Vec<_>>()), &json!(vec![0; 30]));
    assert_eq!(many.len(), LISTED_DIFFERENCES + 1);
    assert_eq!(
        many.last().map(String::as_str),
        Some("  ... and 9 more difference(s)")
    );
}

fn json_value() -> impl Strategy<Value = Value> {
    let leaf = prop_oneof![
        Just(Value::Null),
        any::<bool>().prop_map(Value::from),
        (0u8..4).prop_map(Value::from),
        "[a-c/~]{0,2}".prop_map(Value::from),
    ];
    leaf.prop_recursive(4, 24, 4, |inner| {
        prop_oneof![
            prop::collection::vec(inner.clone(), 0..5).prop_map(Value::Array),
            prop::collection::btree_map("[a-c/~]{1,2}", inner, 0..4)
                .prop_map(|map| Value::Object(map.into_iter().collect())),
        ]
    })
}

proptest! {
    #[test]
    fn applying_the_diff_gives_what_was_sent(recorded in json_value(), sent in json_value()) {
        let changes = diff(&recorded, &sent);
        prop_assert_eq!(apply(&recorded, &changes).ok(), Some(sent.clone()));
        prop_assert_eq!(changes.is_empty(), recorded == sent);
    }
}

fn multipart(parts: &[(&str, &[u8])]) -> Vec<u8> {
    let mut body = Vec::new();
    for (name, bytes) in parts {
        body.extend_from_slice(
            format!("--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n")
                .as_bytes(),
        );
        body.extend_from_slice(bytes);
        body.extend_from_slice(b"\r\n");
    }
    body.extend_from_slice(format!("--{BOUNDARY}--\r\n").as_bytes());
    body
}

fn multipart_type() -> String {
    format!("multipart/form-data; boundary={BOUNDARY}")
}

#[test]
fn bodies_read_as_scrubbed_canonical_json_text_or_parts() {
    let policy = CassettePolicy::default();
    assert_eq!(body_view(policy, None, b""), Value::Null);
    assert_eq!(
        serde_json::to_string(&body_view(policy, None, br#"{"b":1,"a":{"d":2,"c":3}}"#))
            .expect("view serializes"),
        r#"{"a":{"c":3,"d":2},"b":1}"#
    );
    assert_eq!(
        body_view(policy, None, br#"{"created_at":"2026-01-01T00:00:00Z"}"#),
        json!({ "created_at": "1970-01-01T00:00:00Z" })
    );
    assert_eq!(body_view(policy, None, b"plain text"), json!("plain text"));
    assert_eq!(
        body_view(policy, None, &[0xff, 0x00]),
        json!({ "bytes": 2, "fnv1a64": format!("{:016x}", fnv1a64(&[0xff, 0x00])) })
    );
    let large = vec![b'x'; VERBATIM_PART_LIMIT + 1];
    let body = multipart(&[("model", b"whisper-1"), ("file", &large)]);
    assert_eq!(
        body_view(policy, Some(&multipart_type()), &body),
        json!({ "multipart": [
            {
                "headers": { "content-disposition": "form-data; name=\"model\"" },
                "body": "whisper-1",
            },
            {
                "headers": { "content-disposition": "form-data; name=\"file\"" },
                "body": { "bytes": large.len(), "fnv1a64": format!("{:016x}", fnv1a64(&large)) },
            },
        ]})
    );
}

/// A scratch fixture with one interaction on `/v1/answer` whose recorded
/// request has `content_type` and `body` (`None` records no body).
struct Scratch {
    dir: assert_fs::TempDir,
}

impl Scratch {
    fn new(content_type: &str, body: Option<&str>) -> Self {
        let scratch = Self {
            dir: assert_fs::TempDir::new().expect("scratch directory"),
        };
        let interaction = CassetteInteraction {
            when: CassetteRequest {
                path: "/v1/answer".into(),
                method: "POST".into(),
                query_param: Vec::new(),
                header: vec![NameValue {
                    name: "content-type".into(),
                    value: content_type.into(),
                }],
                body: body.map(str::to_owned),
                body_encoding: BodyEncoding::Utf8,
            },
            then: CassetteResponse {
                status: 200,
                header: Vec::new(),
                body: Some("{}".into()),
                body_encoding: BodyEncoding::Utf8,
            },
        };
        fs::write(
            scratch.fixture(),
            serialize_cassette_interactions(&[interaction]),
        )
        .expect("write fixture");
        scratch
    }

    fn fixture(&self) -> PathBuf {
        self.dir.path().join("example.yaml")
    }

    fn snapshot(&self) -> PathBuf {
        request_snapshot(&self.fixture())
    }

    async fn start(&self, mode: SnapshotMode) -> Result<ProviderCassette, CassetteError> {
        self.start_with(CassetteSpec::new("snapshot"), mode).await
    }

    async fn start_with(
        &self,
        spec: CassetteSpec,
        mode: SnapshotMode,
    ) -> Result<ProviderCassette, CassetteError> {
        ProviderCassette::try_start_session(
            RecordVia::Proxy,
            "example",
            spec,
            UPSTREAM,
            CassetteMode::Replay,
            self.fixture(),
            self.dir.path().join("attempts"),
            mode,
        )
        .await
    }
}

/// Post `body` as `content_type` and return the status and response body.
async fn post(
    cassette: &ProviderCassette,
    content_type: &str,
    body: Vec<u8>,
) -> (StatusCode, String) {
    let response = reqwest::Client::builder()
        .no_proxy()
        .build()
        .expect("HTTP client")
        .post(format!("{}/answer", cassette.base_url()))
        .header("content-type", content_type)
        .body(body)
        .send()
        .await
        .expect("local replay response");
    let status = response.status();
    (status, response.text().await.expect("response body"))
}

#[tokio::test]
async fn a_request_equal_to_its_recording_needs_no_snapshot() {
    let scratch = Scratch::new("application/json", Some(r#"{"input":1}"#));
    let cassette = scratch
        .start(SnapshotMode::Check)
        .await
        .expect("session starts");
    let (status, _) = post(&cassette, "application/json", br#"{"input":1}"#.to_vec()).await;
    assert_eq!(status, StatusCode::OK);
    cassette.try_finish().await.expect("replay passes");
    assert!(!scratch.snapshot().exists());
}

#[tokio::test]
async fn write_pins_a_multipart_body_the_recording_omitted_and_check_holds_it() {
    let scratch = Scratch::new(&multipart_type(), None);
    let audio = multipart(&[("model", b"whisper-1"), ("file", b"RIFF")]);

    let cassette = scratch
        .start(SnapshotMode::Write)
        .await
        .expect("session starts");
    post(&cassette, &multipart_type(), audio.clone()).await;
    cassette.try_finish().await.expect("replay passes");
    let written: Value =
        serde_json::from_str(&fs::read_to_string(scratch.snapshot()).expect("snapshot written"))
            .expect("snapshot parses");
    assert_eq!(
        written,
        json!({ "interactions": [{ "index": 0, "changes": [{
            "path": "",
            "recorded": null,
            "sent": { "multipart": [
                {
                    "headers": { "content-disposition": "form-data; name=\"model\"" },
                    "body": "whisper-1",
                },
                {
                    "headers": { "content-disposition": "form-data; name=\"file\"" },
                    "body": "RIFF",
                },
            ]},
        }]}]})
    );

    let cassette = scratch
        .start(SnapshotMode::Check)
        .await
        .expect("session starts");
    post(&cassette, &multipart_type(), audio).await;
    cassette.try_finish().await.expect("the snapshot holds");

    // The recording matches any multipart body, so only the snapshot sees this.
    let cassette = scratch
        .start(SnapshotMode::Check)
        .await
        .expect("session starts");
    let (status, _) = post(
        &cassette,
        &multipart_type(),
        multipart(&[("model", b"whisper-2"), ("file", b"RIFF")]),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let error = cassette
        .try_finish()
        .await
        .expect_err("the snapshot differs");
    assert!(
        matches!(error, CassetteError::SnapshotMismatch { .. }),
        "{error}"
    );
    assert_eq!(error.path(), scratch.fixture());
    let message = error.to_string();
    assert!(
        message.contains(r#"/multipart/0/body: snapshot "whisper-1", sent "whisper-2""#),
        "{message}"
    );
}

#[tokio::test]
async fn write_removes_a_snapshot_the_requests_no_longer_need() {
    let scratch = Scratch::new("application/json", Some(r#"{"input":1}"#));
    fs::write(scratch.snapshot(), r#"{"interactions": []}"#).expect("write snapshot");
    let cassette = scratch
        .start(SnapshotMode::Write)
        .await
        .expect("session starts");
    post(&cassette, "application/json", br#"{"input":1}"#.to_vec()).await;
    cassette.try_finish().await.expect("replay passes");
    assert!(!scratch.snapshot().exists());
}

#[tokio::test]
async fn a_session_dropped_after_playing_everything_still_writes() {
    let scratch = Scratch::new(&multipart_type(), None);
    let cassette = scratch
        .start(SnapshotMode::Write)
        .await
        .expect("session starts");
    post(
        &cassette,
        &multipart_type(),
        multipart(&[("file", b"RIFF")]),
    )
    .await;
    drop(cassette);
    assert!(scratch.snapshot().exists());
}

#[tokio::test]
async fn a_refused_request_reports_its_snapshot_difference() {
    let scratch = Scratch::new("application/json", Some(r#"{"input":1,"model":"m"}"#));
    let cassette = scratch
        .start(SnapshotMode::Check)
        .await
        .expect("session starts");
    let (status, body) = post(
        &cassette,
        "application/json",
        br#"{"input":2,"model":"m"}"#.to_vec(),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let body: Value = serde_json::from_str(&body).expect("miss body is JSON");
    assert_eq!(
        body["snapshot_diff"],
        json!("interaction 0:\n  /input: snapshot 1, sent 2")
    );
    let error = cassette
        .try_finish()
        .await
        .expect_err("replay refused a request");
    assert!(
        error
            .to_string()
            .contains("snapshot difference at interaction 0:\n  /input: snapshot 1, sent 2"),
        "{error}"
    );
}

#[tokio::test]
async fn an_unusable_snapshot_stops_the_session_at_start() {
    let scratch = Scratch::new("application/json", Some(r#"{"input":1}"#));
    for (contents, reason) in [
        ("not json", "expected ident"),
        (
            r#"{"interactions": [{"index": 3, "changes": []}]}"#,
            "interaction 3 is not in",
        ),
        (
            r#"{"interactions": [{"index": 0, "changes": [{"path": "/input", "recorded": 5, "sent": 6}]}]}"#,
            "/input: the recording holds another value here",
        ),
    ] {
        fs::write(scratch.snapshot(), contents).expect("write snapshot");
        let error = scratch
            .start(SnapshotMode::Check)
            .await
            .expect_err("the snapshot cannot be used");
        assert!(
            matches!(error, CassetteError::InvalidSnapshot { .. }),
            "{error}"
        );
        assert_eq!(error.path(), scratch.fixture());
        assert!(error.to_string().contains(reason), "{error}");
    }
    // Off and Write never read it.
    for mode in [SnapshotMode::Off, SnapshotMode::Write] {
        let cassette = scratch.start(mode).await.expect("session starts");
        post(&cassette, "application/json", br#"{"input":1}"#.to_vec()).await;
        cassette.try_finish().await.expect("replay passes");
    }
}

#[tokio::test]
async fn a_snapshot_that_cannot_be_removed_fails_the_write() {
    let scratch = Scratch::new("application/json", Some(r#"{"input":1}"#));
    fs::create_dir(scratch.snapshot()).expect("a directory where the snapshot goes");
    let cassette = scratch
        .start(SnapshotMode::Write)
        .await
        .expect("session starts");
    post(&cassette, "application/json", br#"{"input":1}"#.to_vec()).await;
    let error = cassette.try_finish().await.expect_err("removal fails");
    assert!(
        matches!(error, CassetteError::WriteSnapshot { .. }),
        "{error}"
    );
    assert_eq!(error.path(), scratch.fixture());
}

/// Shape matching serves a request that keeps its recording's coarse shape;
/// the snapshot check then decides whether its bytes are the reviewed ones.
#[tokio::test]
async fn under_shape_matching_only_the_snapshot_admits_a_changed_request() {
    let scratch = Scratch::new("application/json", Some(r#"{"input":"one","model":"m"}"#));
    let shaped = CassetteSpec::new("snapshot").shape_matched();
    let changed = br#"{"input":["one","two"],"model":"m"}"#.to_vec();

    let cassette = scratch
        .start_with(shaped, SnapshotMode::Check)
        .await
        .expect("session starts");
    let (status, _) = post(&cassette, "application/json", changed.clone()).await;
    assert_eq!(status, StatusCode::OK, "the coarse shape matches");
    let error = cassette
        .try_finish()
        .await
        .expect_err("no snapshot holds the change");
    assert!(
        matches!(error, CassetteError::SnapshotMismatch { .. }),
        "{error}"
    );
    assert!(
        error
            .to_string()
            .contains(r#"/input: snapshot "one", sent ["one","two"]"#),
        "{error}"
    );

    let cassette = scratch
        .start_with(shaped, SnapshotMode::Write)
        .await
        .expect("session starts");
    post(&cassette, "application/json", changed.clone()).await;
    cassette
        .try_finish()
        .await
        .expect("write records the change");
    let written: Value =
        serde_json::from_str(&fs::read_to_string(scratch.snapshot()).expect("snapshot written"))
            .expect("snapshot parses");
    assert_eq!(
        written,
        json!({ "interactions": [{ "index": 0, "changes": [
            { "path": "/input", "recorded": "one", "sent": ["one", "two"] },
        ]}]})
    );

    let cassette = scratch
        .start_with(shaped, SnapshotMode::Check)
        .await
        .expect("session starts");
    post(&cassette, "application/json", changed.clone()).await;
    cassette
        .try_finish()
        .await
        .expect("the snapshot holds the change");

    // With snapshots off nothing pins the bytes: unordered shape matching
    // serves the closest recording and the session passes.
    let cassette = scratch
        .start_with(shaped.unordered(), SnapshotMode::Off)
        .await
        .expect("session starts");
    let (status, _) = post(&cassette, "application/json", changed.clone()).await;
    assert_eq!(status, StatusCode::OK);
    cassette
        .try_finish()
        .await
        .expect("nothing checks the bytes");

    // Exact matching still refuses the change, snapshot or not.
    let cassette = scratch
        .start_with(
            CassetteSpec::new("snapshot").exact_matched(),
            SnapshotMode::Check,
        )
        .await
        .expect("session starts");
    let (status, _) = post(&cassette, "application/json", changed).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    cassette
        .try_finish()
        .await
        .expect_err("exact matching refused the request");
}

#[tokio::test]
async fn under_shape_matching_a_new_field_or_model_is_refused() {
    let scratch = Scratch::new("application/json", Some(r#"{"input":"one","model":"m"}"#));
    let shaped = CassetteSpec::new("snapshot").shape_matched();
    for body in [
        br#"{"input":"one","model":"m","seed":1}"#.to_vec(),
        br#"{"input":"one","model":"n"}"#.to_vec(),
    ] {
        let cassette = scratch
            .start_with(shaped, SnapshotMode::Check)
            .await
            .expect("session starts");
        let (status, message) = post(&cassette, "application/json", body).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        let message: Value = serde_json::from_str(&message).expect("miss body is JSON");
        assert_eq!(message["candidates"][0]["shape_matches"], json!(false));
        cassette
            .try_finish()
            .await
            .expect_err("replay refused the request");
    }
}
