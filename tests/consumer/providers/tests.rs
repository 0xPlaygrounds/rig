//! Guard failures are injected locally; these exercise the harness, not a provider.

#![allow(clippy::unwrap_used)]

use super::*;

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
