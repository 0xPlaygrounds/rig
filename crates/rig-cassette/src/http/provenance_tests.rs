//! The recording guard: a live capture is a claim about the provider's wire,
//! so only a scenario declared [`Provenance::Live`] may take the recording
//! path. A derived or scripted scenario — and an undeclared one — is refused
//! before the session reaches the upstream or creates a file.

use super::*;
use futures::FutureExt;
use serde_json::json;
use std::panic::AssertUnwindSafe;

fn client() -> reqwest::Client {
    reqwest::Client::builder()
        .no_proxy()
        .build()
        .expect("HTTP client")
}

/// Record one exchange against a local endpoint and return the upstream's
/// hit count together with whether the destination was written.
async fn record_attempt(spec: CassetteSpec) -> Result<(usize, bool), String> {
    let upstream = httpmock::MockServer::start_async().await;
    let endpoint = upstream
        .mock_async(|when, then| {
            when.any_request();
            then.status(200).json_body(json!({"answer": "ok"}));
        })
        .await;
    let destination = assert_fs::TempDir::new().expect("destination directory");
    let path = destination.path().join("provider.yaml");
    let base_url = format!("{}/v1", upstream.base_url());

    let attempt = AssertUnwindSafe(async {
        let cassette = ProviderCassette::start_at(
            Transport::Proxy,
            "example",
            spec,
            &base_url,
            CassetteMode::Record,
            path.clone(),
        )
        .await;
        let status = client()
            .post(format!("{}/answer", cassette.base_url()))
            .json(&json!({"input": 0}))
            .send()
            .await
            .expect("proxied response")
            .status();
        assert_eq!(status, StatusCode::OK);
        cassette.finish().await;
    })
    .catch_unwind()
    .await;

    let observed = (endpoint.calls_async().await, path.exists());
    match attempt {
        Ok(()) => Ok(observed),
        Err(payload) => {
            let message = payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| {
                    payload
                        .downcast_ref::<&str>()
                        .map(|text| (*text).to_owned())
                })
                .unwrap_or_else(|| "non-string panic".to_owned());
            assert_eq!(
                observed,
                (0, false),
                "a refused recording must not reach the upstream or write a fixture: {message}"
            );
            Err(message)
        }
    }
}

#[tokio::test]
async fn a_declared_live_scenario_records() {
    let recorded =
        record_attempt(CassetteSpec::new("guard/live").with_provenance(Provenance::Live))
            .await
            .expect("a live scenario may be recorded");
    assert_eq!(recorded, (1, true));
}

#[tokio::test]
async fn a_derived_scenario_is_refused_before_the_upstream() {
    let refusal =
        record_attempt(CassetteSpec::new("guard/derived").with_provenance(Provenance::Derived))
            .await
            .expect_err("a derived scenario may not be recorded");
    assert!(
        refusal.contains("refusing to live-record example/guard/derived")
            && refusal.contains("declared derived"),
        "{refusal}"
    );
}

#[tokio::test]
async fn a_scripted_scenario_is_refused_before_the_upstream() {
    let refusal =
        record_attempt(CassetteSpec::new("guard/scripted").with_provenance(Provenance::Scripted))
            .await
            .expect_err("a scripted scenario may not be recorded");
    assert!(refusal.contains("declared scripted"), "{refusal}");
}

#[tokio::test]
async fn an_undeclared_scenario_is_refused_before_the_upstream() {
    let refusal = record_attempt(CassetteSpec::new("guard/undeclared"))
        .await
        .expect_err("an undeclared scenario may not be recorded");
    assert!(refusal.contains("no provenance is declared"), "{refusal}");
}

/// Replay is unaffected: provenance gates capture, not playback, so a
/// fabricated scenario still serves its committed bytes.
#[tokio::test]
async fn replay_serves_every_provenance() {
    let fixtures = assert_fs::TempDir::new().expect("fixture directory");
    let path = fixtures.path().join("provider.yaml");
    let interactions = [CassetteInteraction {
        when: CassetteRequest {
            path: "/v1/answer".into(),
            method: "POST".into(),
            query_param: Vec::new(),
            header: Vec::new(),
            body: Some(r#"{"input":0}"#.into()),
            body_encoding: BodyEncoding::Utf8,
        },
        then: CassetteResponse {
            status: 200,
            header: Vec::new(),
            body: Some(r#"{"answer":"ok"}"#.into()),
            body_encoding: BodyEncoding::Utf8,
        },
    }];
    fs::write(&path, serialize_cassette_interactions(&interactions)).expect("write fixture");

    for provenance in [Provenance::Live, Provenance::Derived, Provenance::Scripted] {
        let cassette = ProviderCassette::start_at(
            Transport::Proxy,
            "example",
            CassetteSpec::new("guard/replayed").with_provenance(provenance),
            "https://example.invalid/v1",
            CassetteMode::Replay,
            path.clone(),
        )
        .await;
        let response = client()
            .post(format!("{}/answer", cassette.base_url()))
            .json(&json!({"input": 0}))
            .send()
            .await
            .expect("replayed response");
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.text().await.expect("body"), r#"{"answer":"ok"}"#);
        cassette.finish().await;
    }
}
