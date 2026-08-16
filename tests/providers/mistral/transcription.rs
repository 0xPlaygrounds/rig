//! Cassette-backed coverage for Mistral's transcription capability.
//!
//! Migrated from the `#[ignore]`d live smoke test this file used to hold, and
//! extended into the matrix for one defect: Mistral was the only provider whose
//! transcription hand-rolled the send/status/decode tail instead of using the
//! shared driver, and the tail it built dropped the failed response's headers —
//! so `Retry-After` was unreachable on a rate-limited transcription, against
//! the contract rig#2210 states and the shared drivers' own tests guard.
//!
//! ## Matrix
//!
//! The bug's input space is the cross of {transport shape} × {outcome}, and it
//! is fully enumerable: a transport either hands a non-success response back
//! (a custom `HttpClientExt`) or raises it as an error already carrying its
//! headers (the bundled reqwest client), and an outcome is either success or
//! failure. Only the first column can lose headers, and that is exactly the
//! column a recording cannot reach — the cassette proxy is driven through
//! reqwest, which takes the other branch. So the regression cell is a unit
//! test with a recording transport, and the recorded cells cover the
//! capability's live surface around it.
//!
//! | # | cell | model | shape | status |
//! |---|---|---|---|---|
//! | 1 | `voxtral_mini_transcribes_the_audio_fixture` | `voxtral-mini` | success | recorded |
//! | 2 | `voxtral_small_is_not_a_transcription_model` | `voxtral-small` | 400 — the catalog says `audio_transcription: false` | recorded |
//! | 3 | `timestamp_granularities_populate_segments` | `voxtral-mini` | success, `segments` non-empty | recorded |
//! | 4 | `a_language_hint_is_accepted` | `voxtral-mini` | success, `language` form field | recorded |
//! | 4b | `an_array_of_timestamp_granularities_is_rejected` | `voxtral-mini` | 422 — the array form of an `additional_params` value | recorded |
//! | 5 | `an_unknown_model_is_rejected_with_its_body` | invalid | 4xx | recorded |
//! | 6 | `bogus_key_transcription_keeps_status_and_body` | `voxtral-mini` | 401 | recorded |
//! | 7 | `transcription_non_success_preserves_response_headers` | — | non-success, response handed back | unit — the only shape that can lose headers |
//! | 8 | `transcription_non_success_preserves_status_and_body` | — | non-success, response handed back | unit — the same shape's status/body half |
//! | 9 | `transcription_form_carries_the_documented_fields` | — | request form | unit — a multipart request records with **no** body (the proxy stores bodies as strings), so a fixture cannot pin the outbound form |
//!
//! Cells 7–9 live next to the provider in
//! `crates/rig-core/src/providers/mistral/transcription.rs`.

use anyhow::Result;
use axum::http;
use rig::prelude::TranscriptionClient;
use rig::providers::mistral;
use rig::transcription::TranscriptionModel;

use super::support::{with_mistral_cassette_bogus_key_result, with_mistral_cassette_result};
use crate::support::{AUDIO_FIXTURE_PATH, assert_nonempty_response};

#[tokio::test]
async fn voxtral_mini_transcribes_the_audio_fixture() -> Result<()> {
    with_mistral_cassette_result(
        "transcription/voxtral_mini_transcribes_the_audio_fixture",
        |client| async move {
            let model = client.transcription_model(mistral::VOXTRAL_MINI);
            let response = model
                .transcription_request()
                .load_file(AUDIO_FIXTURE_PATH)?
                .send()
                .await?;

            assert_nonempty_response(&response.text);
            anyhow::ensure!(
                response.response.model.starts_with("voxtral-mini"),
                "the response names the model that served it: {}",
                response.response.model
            );
            // Voxtral charges the clip's audio outside `prompt_tokens`, the
            // same accounting the completion path folds into input usage
            // (#2337) — so the seconds it billed for are part of the record.
            let usage = &response.response.usage;
            anyhow::ensure!(
                usage
                    .prompt_audio_seconds
                    .is_some_and(|seconds| seconds > 0),
                "a transcription reports the audio it charged for: {usage}"
            );
            Ok::<_, anyhow::Error>(())
        },
    )
    .await
}

/// `VOXTRAL_SMALL` sits beside `VOXTRAL_MINI` in the transcription module, so
/// it reads as the larger transcription model. It is not one: the live catalog
/// reports `audio_transcription: false` for it (and `audio: true`,
/// `completion_chat: true` — it is the audio-*chat* model the multimodal
/// matrix drives), and the transcription endpoint rejects it outright. Pinned
/// so the constant's documentation is backed by the wire rather than by
/// inference from where it is declared.
#[tokio::test]
async fn voxtral_small_is_not_a_transcription_model() -> Result<()> {
    with_mistral_cassette_result(
        "transcription/voxtral_small_is_not_a_transcription_model",
        |client| async move {
            let model = client.transcription_model(mistral::VOXTRAL_SMALL);
            let error = model
                .transcription_request()
                .load_file(AUDIO_FIXTURE_PATH)?
                .send()
                .await
                .map(|_| ())
                .expect_err("Mistral does not serve voxtral-small for transcription");

            let body = error
                .provider_response_body()
                .expect("the rejection body must survive");
            anyhow::ensure!(
                body.contains("Invalid model"),
                "expected Mistral's invalid-model rejection, got {body}"
            );
            Ok::<_, anyhow::Error>(())
        },
    )
    .await
}

/// `segments` is required by Mistral's schema but comes back empty unless the
/// request asks for timestamps, so the smoke cell only ever decodes `[]`. This
/// is the cell that decodes a populated one.
///
/// Mistral documents `timestamp_granularities` as incompatible with
/// `language`, so the two are never set together.
#[tokio::test]
async fn timestamp_granularities_populate_segments() -> Result<()> {
    with_mistral_cassette_result(
        "transcription/timestamp_granularities_populate_segments",
        |client| async move {
            let model = client.transcription_model(mistral::VOXTRAL_MINI);
            let response = model
                .transcription_request()
                .load_file(AUDIO_FIXTURE_PATH)?
                // A bare string, not a JSON array: the shared multipart builder
                // sends a non-string `additional_params` value JSON-encoded, and
                // Mistral answers `["segment"]` with a 422 naming the enum it
                // wanted. The array form is pinned by the cell below.
                .additional_params(serde_json::json!({"timestamp_granularities": "segment"}))
                .send()
                .await?;

            let segments = &response.response.segments;
            anyhow::ensure!(
                !segments.is_empty(),
                "asking for segment timestamps must produce segments"
            );
            for segment in segments {
                anyhow::ensure!(
                    segment.start < segment.end,
                    "a segment spans time: {} .. {}",
                    segment.start,
                    segment.end
                );
                anyhow::ensure!(!segment.text.is_empty(), "a segment carries its text");
                anyhow::ensure!(
                    !segment.segment_type.is_empty(),
                    "Mistral tags each segment with its kind"
                );
            }
            Ok::<_, anyhow::Error>(())
        },
    )
    .await
}

/// What a caller gets for passing `timestamp_granularities` the way OpenAI's
/// API documents it — as an array. The shared multipart builder JSON-encodes a
/// non-string `additional_params` value, and Mistral's form parser wants the
/// bare enum, so the request is a 422 that names the values it accepts.
/// Recorded rather than fixed: rig has one multipart encoding for every
/// provider on the shared driver, the string form works, and the error is
/// specific enough to act on.
#[tokio::test]
async fn an_array_of_timestamp_granularities_is_rejected() -> Result<()> {
    with_mistral_cassette_result(
        "transcription/an_array_of_timestamp_granularities_is_rejected",
        |client| async move {
            let model = client.transcription_model(mistral::VOXTRAL_MINI);
            let error = model
                .transcription_request()
                .load_file(AUDIO_FIXTURE_PATH)?
                .additional_params(serde_json::json!({"timestamp_granularities": ["segment"]}))
                .send()
                .await
                .map(|_| ())
                .expect_err("Mistral's form parser wants the bare enum value");

            anyhow::ensure!(
                error.provider_response_status() == Some(http::StatusCode::UNPROCESSABLE_ENTITY),
                "expected a 422, got {:?}",
                error.provider_response_status()
            );
            let body = error
                .provider_response_body()
                .expect("the rejection body must survive");
            anyhow::ensure!(
                body.contains("timestamp_granularities"),
                "the rejection must name the field it is about: {body}"
            );
            Ok::<_, anyhow::Error>(())
        },
    )
    .await
}

#[tokio::test]
async fn a_language_hint_is_accepted() -> Result<()> {
    with_mistral_cassette_result(
        "transcription/a_language_hint_is_accepted",
        |client| async move {
            let model = client.transcription_model(mistral::VOXTRAL_MINI);
            let response = model
                .transcription_request()
                .load_file(AUDIO_FIXTURE_PATH)?
                .language("en".to_string())
                .send()
                .await?;

            assert_nonempty_response(&response.text);
            Ok::<_, anyhow::Error>(())
        },
    )
    .await
}

/// The failure path with a real body: the rejection reaches the caller with its
/// status and Mistral's own message rather than a bare transport error.
#[tokio::test]
async fn an_unknown_model_is_rejected_with_its_body() -> Result<()> {
    with_mistral_cassette_result(
        "transcription/an_unknown_model_is_rejected_with_its_body",
        |client| async move {
            let model = client.transcription_model("voxtral-does-not-exist");
            let error = model
                .transcription_request()
                .load_file(AUDIO_FIXTURE_PATH)?
                .send()
                .await
                .map(|_| ())
                .expect_err("an unknown model must be rejected");

            anyhow::ensure!(
                error
                    .provider_response_status()
                    .is_some_and(|status| status.is_client_error()),
                "expected a 4xx, got {:?}",
                error.provider_response_status()
            );
            anyhow::ensure!(
                error
                    .provider_response_body()
                    .is_some_and(|body| !body.is_empty()),
                "the provider's own message must survive the failure"
            );
            Ok::<_, anyhow::Error>(())
        },
    )
    .await
}

#[tokio::test]
async fn bogus_key_transcription_keeps_status_and_body() -> Result<()> {
    with_mistral_cassette_bogus_key_result(
        "transcription/bogus_key_transcription_keeps_status_and_body",
        |client| async move {
            let model = client.transcription_model(mistral::VOXTRAL_MINI);
            let error = model
                .transcription_request()
                .load_file(AUDIO_FIXTURE_PATH)?
                .send()
                .await
                .map(|_| ())
                .expect_err("an invalid key must fail");

            anyhow::ensure!(
                error.provider_response_status() == Some(http::StatusCode::UNAUTHORIZED),
                "expected a 401, got {:?}",
                error.provider_response_status()
            );
            anyhow::ensure!(
                error
                    .provider_response_body()
                    .is_some_and(|body| !body.is_empty()),
                "the rejection body must survive"
            );
            Ok::<_, anyhow::Error>(())
        },
    )
    .await
}
