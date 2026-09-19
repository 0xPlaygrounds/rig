//! What a replay session refuses, end to end through `ProviderCassette`:
//! an interaction left unplayed, a miss the caller swallowed, and a session
//! dropped without `finish` while it holds either.

use super::*;
use futures::FutureExt;
use serde_json::json;
use std::panic::AssertUnwindSafe;

/// A fixture of `answers.len()` ordered interactions on one route, each
/// matching the body `{"input": <index>}`.
fn write_fixture(root: &Path, scenario: &str, answers: &[&str]) {
    let path = cassette_path(root, "example", scenario);
    fs::create_dir_all(path.parent().expect("fixture parent")).expect("create fixture directory");
    let interactions: Vec<_> = answers
        .iter()
        .enumerate()
        .map(|(index, answer)| CassetteInteraction {
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
                body: Some((*answer).into()),
                body_encoding: BodyEncoding::Utf8,
            },
        })
        .collect();
    fs::write(path, serialize_cassette_interactions(&interactions)).expect("write fixture");
}

async fn start(root: &Path, scenario: &'static str) -> ProviderCassette {
    ProviderCassette::start(root, "example", scenario, "https://example.invalid/v1").await
}

/// Post `{"input": input}` to the session and return the status the replay
/// answered with: the caller of a provider client sees nothing else.
async fn post(cassette: &ProviderCassette, input: usize) -> StatusCode {
    reqwest::Client::builder()
        .no_proxy()
        .build()
        .expect("HTTP client")
        .post(format!("{}/answer", cassette.base_url()))
        .json(&json!({ "input": input }))
        .send()
        .await
        .expect("local replay response")
        .status()
}

fn panic_message(payload: PanicPayload) -> String {
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_owned()))
        .expect("a string panic payload")
}

#[tokio::test]
async fn finish_refuses_an_interaction_left_unplayed() {
    let scratch = assert_fs::TempDir::new().expect("temporary fixtures");
    let root = scratch.path().join("fixtures/cassettes");
    write_fixture(
        &root,
        "pair",
        &[r#"{"answer":"first"}"#, r#"{"answer":"second"}"#],
    );
    let cassette = start(&root, "pair").await;
    assert_eq!(post(&cassette, 0).await, StatusCode::OK);

    let outcome = AssertUnwindSafe(cassette.finish()).catch_unwind().await;

    let message = panic_message(outcome.expect_err("an unplayed interaction fails finish"));
    assert!(
        message.contains("left unused interactions"),
        "names the failure: {message}"
    );
    assert!(
        message.contains("[1] POST /v1/answer"),
        "names the interaction: {message}"
    );
}

#[tokio::test]
async fn finish_refuses_a_miss_the_caller_swallowed() {
    let scratch = assert_fs::TempDir::new().expect("temporary fixtures");
    let root = scratch.path().join("fixtures/cassettes");
    write_fixture(&root, "single", &[r#"{"answer":"only"}"#]);
    let cassette = start(&root, "single").await;
    // A request the recording never saw: the replay refuses it, and a
    // provider client would surface that as an ordinary HTTP failure the
    // test body may well tolerate.
    assert_eq!(post(&cassette, 7).await, StatusCode::NOT_FOUND);
    assert_eq!(post(&cassette, 0).await, StatusCode::OK);

    let outcome = AssertUnwindSafe(cassette.finish()).catch_unwind().await;

    let message = panic_message(outcome.expect_err("a swallowed miss fails finish"));
    assert!(
        message.contains("received unexpected replay request(s)"),
        "{message}"
    );
    assert!(
        !message.contains("left unused interactions"),
        "the recording itself was fully played: {message}"
    );
}

#[tokio::test]
async fn dropping_an_unfinished_session_reports_what_finish_would_have() {
    let scratch = assert_fs::TempDir::new().expect("temporary fixtures");
    let root = scratch.path().join("fixtures/cassettes");
    write_fixture(
        &root,
        "pair",
        &[r#"{"answer":"first"}"#, r#"{"answer":"second"}"#],
    );

    let outcome = AssertUnwindSafe(async {
        let cassette = start(&root, "pair").await;
        assert_eq!(post(&cassette, 0).await, StatusCode::OK);
        // Returned without `finish`: the second interaction was never played.
        drop(cassette);
    })
    .catch_unwind()
    .await;

    let message = panic_message(outcome.expect_err("the drop guard fails the test"));
    assert!(message.contains("left unused interactions"), "{message}");
    assert!(
        message.contains("dropped without `finish`"),
        "names the cause: {message}"
    );
}

#[tokio::test]
async fn dropping_a_fully_played_session_is_silent() {
    let scratch = assert_fs::TempDir::new().expect("temporary fixtures");
    let root = scratch.path().join("fixtures/cassettes");
    write_fixture(&root, "single", &[r#"{"answer":"only"}"#]);

    let cassette = start(&root, "single").await;
    assert_eq!(post(&cassette, 0).await, StatusCode::OK);
    drop(cassette);
}

#[tokio::test]
async fn a_panicking_test_body_is_reported_not_the_guard() {
    let scratch = assert_fs::TempDir::new().expect("temporary fixtures");
    let root = scratch.path().join("fixtures/cassettes");
    write_fixture(
        &root,
        "pair",
        &[r#"{"answer":"first"}"#, r#"{"answer":"second"}"#],
    );

    // The body panics while the session is unplayed: the session unwinds
    // with it, and the guard stays silent so the body's own panic is what
    // the test reports.
    let outcome = AssertUnwindSafe(async {
        let cassette = start(&root, "pair").await;
        assert_eq!(post(&cassette, 0).await, StatusCode::OK);
        panic!("the assertion the test was about");
    })
    .catch_unwind()
    .await;

    let message = panic_message(outcome.expect_err("the body panics"));
    assert_eq!(message, "the assertion the test was about");
}

#[tokio::test]
async fn a_fallible_tests_own_error_survives_an_unplayed_session() {
    let scratch = assert_fs::TempDir::new().expect("temporary fixtures");
    let root = scratch.path().join("fixtures/cassettes");
    write_fixture(
        &root,
        "pair",
        &[r#"{"answer":"first"}"#, r#"{"answer":"second"}"#],
    );
    let cassette = start(&root, "pair").await;
    assert_eq!(post(&cassette, 0).await, StatusCode::OK);

    // The fallible wrapper returns the test's error; the session it leaves
    // unplayed must not turn that into a panic.
    let outcome = AssertUnwindSafe(cassette.finish_after_test_result(Ok(Err("the test's error"))))
        .catch_unwind()
        .await;

    assert_eq!(
        outcome.expect("no panic replaces the test's error"),
        Err("the test's error")
    );
}
