//! `start_at` and `checkpoint_recording`: a consumer that stages candidates
//! records into an explicit path outside the fixture layout, keeps partial
//! snapshots while the live run is in progress, and replays an exact path
//! without consulting the environment.

use super::*;
use serde_json::json;

fn client() -> reqwest::Client {
    reqwest::Client::builder()
        .no_proxy()
        .build()
        .expect("HTTP client")
}

fn interactions_in(path: &Path) -> Vec<CassetteInteraction> {
    let yaml = fs::read_to_string(path).expect("cassette should be readable");
    serde_yaml::Deserializer::from_str(&yaml)
        .map(CassetteInteraction::deserialize)
        .collect::<Result<Vec<_>, _>>()
        .expect("cassette should parse")
}

#[tokio::test]
async fn recording_to_an_explicit_path_keeps_partial_snapshots_before_finalizing() {
    let upstream = httpmock::MockServer::start_async().await;
    upstream
        .mock_async(|when, then| {
            when.path("/v1/answer");
            then.status(200)
                .json_body(json!({"text": "recorded exchange"}));
        })
        .await;
    let candidate = assert_fs::TempDir::new().expect("candidate directory");
    let path = candidate.path().join("provider.yaml");
    let partial = candidate.path().join("provider.partial.yaml");
    let cassette = ProviderCassette::start_at(
        Transport::Proxy,
        "example",
        CassetteSpec::new("explicit-destination"),
        &format!("{}/v1", upstream.base_url()),
        CassetteMode::Record,
        path.clone(),
    )
    .await;

    // Nothing has been exchanged: no snapshot, and no empty file either.
    assert!(!cassette.checkpoint_recording(&partial).await);
    assert!(!partial.exists());

    let status = client()
        .post(format!("{}/answer", cassette.base_url()))
        .json(&json!({"input": 0}))
        .send()
        .await
        .expect("proxied response")
        .status();
    assert_eq!(status, StatusCode::OK);

    assert!(cassette.checkpoint_recording(&partial).await);
    let snapshot = interactions_in(&partial);
    assert_eq!(snapshot.len(), 1);
    assert_eq!(snapshot[0].when.path, "/v1/answer");
    assert!(!path.exists(), "a snapshot is not the finalized recording");

    cassette.finish().await;
    assert_eq!(interactions_in(&path).len(), 1);
    assert!(partial.exists(), "finalization leaves the snapshot alone");
}

#[tokio::test]
async fn replaying_an_explicit_path_ignores_the_fixture_layout_and_never_snapshots() {
    let candidate = assert_fs::TempDir::new().expect("candidate directory");
    let path = candidate.path().join("provider.yaml");
    let interaction = CassetteInteraction {
        when: CassetteRequest {
            path: "/v1/answer".into(),
            method: "POST".into(),
            query_param: Vec::new(),
            header: vec![NameValue {
                name: "content-type".into(),
                value: "application/json".into(),
            }],
            body: Some(json!({"input": 0}).to_string()),
            body_encoding: BodyEncoding::Utf8,
        },
        then: CassetteResponse {
            status: 200,
            header: Vec::new(),
            body: Some(json!({"text": "replayed"}).to_string()),
            body_encoding: BodyEncoding::Utf8,
        },
    };
    fs::write(&path, serialize_cassette_interactions(&[interaction])).expect("write fixture");
    let cassette = ProviderCassette::start_at(
        Transport::Proxy,
        "example",
        CassetteSpec::new("explicit-destination"),
        "https://example.invalid/v1",
        CassetteMode::Replay,
        path.clone(),
    )
    .await;

    let partial = candidate.path().join("provider.partial.yaml");
    assert!(!cassette.checkpoint_recording(&partial).await);
    assert!(!partial.exists());

    let status = client()
        .post(format!("{}/answer", cassette.base_url()))
        .json(&json!({"input": 0}))
        .send()
        .await
        .expect("replayed response")
        .status();
    assert_eq!(status, StatusCode::OK);
    cassette.finish().await;
}
