//! Anthropic model listing smoke test.

use super::super::support::{with_anthropic_cassette, with_anthropic_cassette_bogus_key};
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
/// `ApiError` carrying provider, path, status and a body preview.
///
/// Anthropic's listing is the paginated one, so this also pins that a failure
/// on the *first* page surfaces as an error rather than an empty list.
#[tokio::test]
async fn list_models_rejected_key_reports_api_error_with_context() {
    use rig::model::ModelListingError;

    with_anthropic_cassette_bogus_key(
        "models/list_models_rejected_key_reports_api_error_with_context",
        |client| async move {
            let error = client
                .models()
                .list_all()
                .await
                .expect_err("a bogus key must not list models");

            let ModelListingError::ApiError {
                status_code,
                message,
            } = &error
            else {
                panic!(
                    "a rejected listing must classify as ApiError\nDisplay: {error}\n\
                     Debug: {error:#?}"
                );
            };

            assert_eq!(*status_code, 401, "unexpected status: {error:#?}");
            // `provider=` carries the wire's stable descriptor name, which is
            // the same lowercase token `CompletionResponse::provider` reports
            // and telemetry records. The deleted client layer decorated this
            // message with its own capitalised display name instead, so there
            // were two spellings of one provider's identity; there is now one.
            for expected in ["provider=anthropic", "path=/v1/models", "status=401"] {
                assert!(
                    message.contains(expected),
                    "the error must carry {expected}; got {message}"
                );
            }
        },
    )
    .await;
}
