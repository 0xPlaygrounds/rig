//! OpenAI model listing smoke test.

use rig::model::ModelLister;

use super::super::support::{with_openai_cassette, with_openai_cassette_bogus_key};

#[tokio::test]
async fn list_models_smoke() {
    with_openai_cassette("models/list_models_smoke", |client| async move {
        let models = match client.openai.models().list_all().await {
            Ok(models) => models,
            Err(error) => {
                panic!("listing OpenAI models should succeed\nDisplay: {error}\nDebug: {error:#?}")
            }
        };

        assert!(
            !models.is_empty(),
            "expected OpenAI to return at least one model\nModel list: {models:#?}"
        );
    })
    .await;
}

/// rig#2079 — the shared fetch path classifies a rejected listing as
/// `ApiError` carrying provider, path, status and a body preview.
///
/// OpenAI drives the plain (unpaginated) shared helper, so this pins the
/// triage that `get_json` now shares with `get_bytes` rather than duplicating.
#[tokio::test]
async fn list_models_rejected_key_reports_api_error_with_context() {
    use rig::model::ModelListingError;

    with_openai_cassette_bogus_key(
        "models/list_models_rejected_key_reports_api_error_with_context",
        |client| async move {
            let error = client
                .openai
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
            // `provider=` carries the wire's stable descriptor name — the same
            // lowercase token `CompletionResponse::provider` reports and
            // telemetry records. The deleted client layer decorated this
            // message with its own capitalised display name instead, so one
            // provider's identity had two spellings; there is now one.
            //
            // `path=` is likewise the path actually sent, `/v1/models`, which
            // is what the fixture records. The client reported `/models`, its
            // route relative to a base URL that already carried `/v1` — a
            // value no request ever had.
            for expected in ["provider=openai", "path=/v1/models", "status=401"] {
                assert!(
                    message.contains(expected),
                    "the error must carry {expected}; got {message}"
                );
            }
        },
    )
    .await;
}
