//! Guard failures are injected locally; these exercise the harness, not a provider.

#![allow(clippy::unwrap_used)]

use super::*;

#[tokio::test]
async fn slow_ecs_validation_leaves_time_to_finalize_completed_recording() {
    let upstream = httpmock::MockServer::start_async().await;
    upstream
        .mock_async(|when, then| {
            when.path("/completed");
            then.status(200)
                .json_body(serde_json::json!({"text":"completed local exchange"}));
        })
        .await;
    let candidate = assert_fs::TempDir::new().unwrap();
    let path = candidate.path().join("provider.yaml");
    let cassette = ProviderCassette::start_at(
        "openai",
        CassetteSpec::new("slow-validation-capture-probe"),
        &upstream.base_url(),
        CassetteMode::Record,
        path.clone(),
    )
    .await;
    assert!(
        !cassette
            .checkpoint_recording(&path.with_file_name("provider.partial.yaml"))
            .await
    );
    assert!(!path.with_file_name("provider.partial.yaml").exists());
    let base = cassette.base_url();
    let budget = Budget::new(Limits {
        seconds: 16,
        ..Limits::default()
    });
    let deadline = execution_deadline(CassetteMode::Record, &budget);
    let mut case = super::super::cases()
        .into_iter()
        .find(|case| case.id == "synthetic-approve")
        .unwrap();
    case.repair = true;
    case.fault = super::super::Fault::RepairTimeout;
    let before = Instant::now();
    let execution = async {
        let response = reqwest::Client::new()
            .post(format!("{base}/completed"))
            .json(&serde_json::json!({"model":"synthetic","max_output_tokens":1}))
            .send()
            .await
            .unwrap();
        assert!(response.status().is_success());
        let _ = response.bytes().await.unwrap();
        execute_with_deadline(&case, super::super::Scripted, Some(deadline)).await
    };
    let error = complete_capture(cassette, &path, &budget, CassetteMode::Record, execution)
        .await
        .unwrap_err();
    assert!(
        before.elapsed() < Duration::from_secs(8),
        "finalization reserve consumed: {:?}",
        before.elapsed()
    );
    assert!(error.to_string().contains("deadline"), "{error}");
    assert!(
        error.to_string().contains("finalization completed"),
        "{error}"
    );
    for recorded in [&path, &path.with_file_name("provider.partial.yaml")] {
        let text = super::super::artifacts::safe_cassette(recorded).unwrap();
        assert!(text.contains("completed local exchange"));
    }
    assert!(!candidate.path().join("capture.json").exists());
}

#[tokio::test]
async fn scrubbed_partial_traffic_survives_a_finalization_failure() {
    use futures::FutureExt;
    let upstream = httpmock::MockServer::start_async().await;
    upstream
        .mock_async(|when, then| {
            when.path("/failure");
            then.status(500)
                .json_body(serde_json::json!({"error":"controlled upstream failure"}));
        })
        .await;
    let candidate = assert_fs::TempDir::new().unwrap();
    let path = candidate.path().join("provider.yaml");
    let partial = candidate.path().join("provider.partial.yaml");
    let cassette = ProviderCassette::start_at(
        "openai",
        CassetteSpec::new("partial-failure-probe"),
        &upstream.base_url(),
        CassetteMode::Record,
        path.clone(),
    )
    .await;
    let response = reqwest::Client::new()
        .post(format!("{}/failure", cassette.base_url()))
        .json(&serde_json::json!({"model":"synthetic","max_output_tokens":1}))
        .send()
        .await
        .unwrap();
    assert_eq!(
        response.status(),
        reqwest::StatusCode::INTERNAL_SERVER_ERROR
    );
    let _ = response.bytes().await.unwrap();
    assert!(cassette.checkpoint_recording(&partial).await);
    let before = super::super::artifacts::safe_cassette(&partial).unwrap();
    assert!(before.contains("controlled upstream failure"));
    // Deliberately make the final destination unwritable as a file.
    std::fs::create_dir(&path).unwrap();
    assert!(
        std::panic::AssertUnwindSafe(cassette.finish())
            .catch_unwind()
            .await
            .is_err()
    );
    assert_eq!(
        super::super::artifacts::safe_cassette(&partial).unwrap(),
        before
    );
    assert!(!candidate.path().join("capture.json").exists());
    assert!(!candidate.path().join("provenance.json").exists());
    let mut case = super::super::cases()
        .into_iter()
        .find(|case| case.id == "openai-unary")
        .unwrap();
    case.repair = true;
    assert!(super::super::artifacts::check_capture(&case, None, &partial).is_err());
    assert!(
        super::super::artifacts::check_capture(
            &case,
            Some(&serde_json::json!({"source":"live-provider","execution_succeeded":false})),
            &partial
        )
        .is_err()
    );
}

#[tokio::test]
async fn external_destinations_are_refused_before_budget_or_network_work() {
    let guard = Guard {
        budget: Budget::new(Limits::default()),
        authority: "127.0.0.1:49123".into(),
    };
    for destination in [
        "https://api.openai.com/v1/responses",
        "http://127.0.0.1:49124/responses",
        "https://127.0.0.1:49123/responses",
        "http://127.0.0.1:49123@evil.invalid/responses",
    ] {
        assert!(
            guard
                .before_request_headers(
                    &Method::POST,
                    &destination.parse().unwrap(),
                    &mut HeaderMap::new()
                )
                .await
                .is_err()
        );
    }
    assert_eq!(guard.budget.used(), 0);
}

#[tokio::test]
async fn request_and_elapsed_limits_refuse_work_before_sending() {
    let limits = Limits {
        requests: 1,
        ..Limits::default()
    };
    let mut guard = Guard {
        budget: Budget::new(limits),
        authority: "127.0.0.1:49123".into(),
    };
    let uri = "http://127.0.0.1:49123/responses".parse().unwrap();
    assert!(
        guard
            .before_request_headers(&Method::POST, &uri, &mut HeaderMap::new())
            .await
            .is_ok()
    );
    assert!(
        guard
            .before_request_headers(&Method::POST, &uri, &mut HeaderMap::new())
            .await
            .is_err()
    );
    assert_eq!(guard.budget.used(), 1);
    guard.budget.deadline = Instant::now();
    assert!(
        guard
            .before_request_headers(&Method::POST, &uri, &mut HeaderMap::new())
            .await
            .is_err()
    );
    assert_eq!(guard.budget.used(), 1);
}

#[tokio::test]
async fn all_provider_token_fields_are_bounded_and_a_missing_limit_is_an_error() {
    let guard = Guard {
        budget: Budget::new(Limits::default()),
        authority: "127.0.0.1:49123".into(),
    };
    let uri = "http://127.0.0.1:49123/responses".parse().unwrap();
    for (body, allowed) in [
        (serde_json::json!({"max_tokens":512}), true),
        (serde_json::json!({"max_output_tokens":512}), true),
        (serde_json::json!({"max_completion_tokens":512}), true),
        (
            serde_json::json!({"generationConfig":{"maxOutputTokens":512}}),
            true,
        ),
        (serde_json::json!({"max_tokens":513}), false),
        (serde_json::json!({"max_tokens":0}), false),
        (serde_json::json!({}), false),
    ] {
        let result = guard
            .before_request_body(
                &Method::POST,
                &uri,
                &HeaderMap::new(),
                Bytes::from(body.to_string()),
            )
            .await;
        assert_eq!(result.is_ok(), allowed, "body {body}");
    }
}
