//! Gemini model listing smoke test.

use rig::client::ModelListingClient;

use super::super::support::{with_gemini_cassette, with_gemini_cassette_bogus_key};

#[tokio::test]
async fn list_models_smoke() {
    with_gemini_cassette("models/list_models_smoke", |client| async move {
        let models = client
            .list_models()
            .await
            .expect("listing Gemini models should succeed");

        println!("Gemini returned {} models", models.len());

        assert!(
            !models.is_empty(),
            "expected Gemini to return at least one model\nModel list: {models:#?}"
        );

        // rig#2322 — Gemini reports an output ceiling for every model and rig
        // used to drop it during conversion. Nothing in the library knew the
        // real limit, which is how a hardcoded 4096 default survived: the
        // recorded listing says 65536 for this model, ~16x larger.
        let flash = models
            .iter()
            .find(|model| model.id == "gemini-2.5-flash")
            .expect("the recorded listing should contain gemini-2.5-flash");

        assert_eq!(
            flash.max_output_tokens,
            Some(65_536),
            "the model's output ceiling must survive listing conversion"
        );
        assert_eq!(
            flash.context_length,
            Some(1_048_576),
            "the input window must stay distinct from the output ceiling"
        );

        // Not merely "some model reported one": every entry the wire annotates
        // must round-trip, so a conversion that dropped the field for a subset
        // still fails.
        assert!(
            models
                .iter()
                .filter(|model| model.max_output_tokens.is_some())
                .count()
                > 40,
            "the recorded listing annotates ~50 models with outputTokenLimit; \
             far fewer surviving means the conversion is dropping them"
        );
    })
    .await;
}

/// rig#2079 — a failed Gemini listing must carry provider, path, status and a
/// body preview.
///
/// Gemini used to build its own request and let a transport-level send error
/// convert directly, so a rejected listing surfaced as a bare `RequestError`
/// whose only content was the stringified status. Its own
/// `!status.is_success()` branch never ran: the reqwest transport reports a
/// non-2xx *as an error* before returning a response, the same dead-arm shape
/// rig#2315 fixed for `verify`. Routing through the shared fetch classifies it
/// as `ApiError` with the context every other lister already produced.
///
/// Recorded with a deliberately invalid key so the 401 is Gemini's own.
#[tokio::test]
async fn list_models_rejected_key_reports_api_error_with_context() {
    use rig::model::ModelListingError;

    with_gemini_cassette_bogus_key(
        "models/list_models_rejected_key_reports_api_error_with_context",
        |client| async move {
            let error = client
                .list_models()
                .await
                .expect_err("a bogus key must not list models");

            let ModelListingError::ApiError {
                status_code,
                message,
            } = &error
            else {
                panic!(
                    "a rejected listing must classify as ApiError, not a bare transport \
                     error\nDisplay: {error}\nDebug: {error:#?}"
                );
            };

            assert_eq!(*status_code, 400, "unexpected status: {error:#?}");
            for expected in [
                "provider=Gemini",
                "path=/v1beta/models?pageSize=1000",
                "status=400",
            ] {
                assert!(
                    message.contains(expected),
                    "the error must carry {expected}; got {message}"
                );
            }
        },
    )
    .await;
}
