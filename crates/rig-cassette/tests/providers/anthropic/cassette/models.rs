//! Anthropic model listing smoke test.

use super::super::support::{with_anthropic_cassette, with_anthropic_cassette_bogus_key};
use rig::error::ProviderError;
use rig::model::ModelLister;

#[tokio::test]
async fn list_models_smoke() {
    with_anthropic_cassette("models/list_models_smoke", |client| async move {
        let models = match client.models().list_all().await {
            Ok(models) => models,
            Err(error) => {
                panic!(
                    "listing Anthropic models should succeed\nDisplay: {error}\nDebug: {error:#?}"
                )
            }
        };

        assert!(
            !models.is_empty(),
            "expected Anthropic to return at least one model\nModel list: {models:#?}"
        );
    })
    .await;
}

/// rig#2079 — the shared fetch path classifies a rejected listing as
/// the provider's reply, with its status and a route naming provider and path.
///
/// Anthropic's listing is the paginated one, so this also pins that a failure
/// on the *first* page surfaces as an error rather than an empty list.
#[tokio::test]
async fn list_models_rejected_key_reports_api_error_with_context() {
    with_anthropic_cassette_bogus_key(
        "models/list_models_rejected_key_reports_api_error_with_context",
        |client| async move {
            let error = client
                .models()
                .list_all()
                .await
                .expect_err("a bogus key must not list models");

            let ProviderError::ProviderResponse(response) = &error else {
                panic!(
                    "a rejected listing must keep the provider's reply\nDisplay: {error}\n\
                     Debug: {error:#?}"
                );
            };
            let status_code = response.status.map(|status| status.as_u16());
            let message = error.to_string();

            assert_eq!(status_code, Some(401), "unexpected status: {error:#?}");
            // `provider=` carries the wire's stable descriptor name, which is
            // the same lowercase token `CompletionResponse::provider` reports
            // and telemetry records. The deleted client layer decorated this
            // message with its own capitalised display name instead, so there
            // were two spellings of one provider's identity; there is now one.
            for expected in ["provider=anthropic", "path=/v1/models", "status 401"] {
                assert!(
                    message.contains(expected),
                    "the error must carry {expected}; got {message}"
                );
            }
        },
    )
    .await;
}
