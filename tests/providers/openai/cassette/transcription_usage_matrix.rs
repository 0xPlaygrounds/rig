//! Edge matrix for the `usage` an OpenAI transcription reports.
//!
//! **Bug.** `/v1/audio/transcriptions` answers with the transcript *and* what
//! it cost, in one of two shapes depending on how the model bills:
//!
//! ```json
//! {"text": "…", "usage": {"type":"duration","seconds":6}}
//! {"text": "…", "usage": {"type":"tokens","input_tokens":54,"output_tokens":16,"total_tokens":70}}
//! ```
//!
//! Rig's response type modeled only `{ text }`. The normalized
//! `transcription::TranscriptionResponse` has no usage slot either, so the
//! accounting was dropped even from the raw provider response — the surface
//! that exists precisely to carry provider-specific fields. A caller had no
//! way to learn what a transcription cost.
//!
//! **What `origin/main` does with these fixtures.** Nothing: the fix is
//! purely additive — it changes nothing rig sends, and the recorded bytes are
//! the same either way, which is the whole point (the data was always on the
//! wire). `main` simply has no `usage` field to read, so the cells that read
//! one do not compile there. That is honest evidence the API is new, not
//! evidence of a behavior regression, and there is no behavior to regress.
//!
//! **What these cells can and cannot see.** The harness cannot match a
//! multipart request body — a non-UTF-8 upload is exported with no body at
//! all, and any multipart request then matches (`tests/common/cassettes.rs`).
//! So these cells pin the *response* mapping and that it survives the whole
//! client stack; they cannot pin what went out. The outbound multipart shape,
//! including `additional_params` flattening, is pinned separately by
//! `providers::internal::transcription`'s own unit tests
//! (`flattens_additional_params_onto_the_form` and its siblings). The model
//! and client axes below likewise select a fixture rather than exercising a
//! branch — `Client` and `CompletionsClient` build the identical request — and
//! are here because the decode runs through each of them end to end.
//!
//! **Why this matrix is smaller than the others.** The *response* space it
//! covers is small: the endpoint reports usage in exactly two shapes, and both
//! are recorded, through both client extensions, alongside a richer response
//! format and a rejected request. The shapes rig must tolerate but no live
//! model produces — a third `type`, a payload carrying both a duration and
//! token counts, a partial token payload, an absent `usage`, an explicit
//! `null` — are unit cells beside the fix, because no recording can produce
//! them. Groq, Azure OpenAI, Venice and HuggingFace share this response type;
//! their own usage shapes are outside this matrix.
//!
//! | # | cell | model | client | usage shape | status |
//! |---|------|-------|--------|-------------|--------|
//! | 1 | `whisper_reports_duration_usage` | whisper-1 | responses | duration | recorded |
//! | 2 | `gpt_4o_transcribe_reports_token_usage` | gpt-4o-transcribe | responses | tokens | recorded |
//! | 3 | `gpt_4o_mini_transcribe_reports_token_usage` | gpt-4o-mini-transcribe | responses | tokens | recorded |
//! | 4 | `completions_client_reports_duration_usage` | whisper-1 | completions | duration | recorded |
//! | 5 | `completions_client_reports_token_usage` | gpt-4o-transcribe | completions | tokens | recorded |
//! | 6 | `verbose_json_still_reports_duration_usage` | whisper-1 | responses | duration (richer body) | recorded |
//! | 7 | `transcript_still_reaches_the_normalized_response` | whisper-1 | responses | duration | recorded |
//! | 8 | `rejected_request_surfaces_the_provider_body` | whisper-1 | responses | none (400) | recorded |
//!
//! Unit cells beside the fix
//! (`usage_decodes_both_billing_shapes_and_keeps_unknown_ones`): both live
//! shapes with and without the input-token breakdown, a payload whose `type`
//! says tokens while also carrying `seconds` (which must not decode as a
//! duration and drop the counts), a partial token payload, an unmodeled third
//! shape carried verbatim, an absent `usage`, and an explicit `null`.

use rig::client::transcription::TranscriptionClient;
use rig::providers::openai::{self, TranscriptionUsage};
use rig::transcription::TranscriptionModel;
use serde_json::json;

use super::super::support::with_openai_transcription_cassette;
use crate::support::AUDIO_FIXTURE_PATH;

fn audio() -> Vec<u8> {
    std::fs::read(AUDIO_FIXTURE_PATH).expect("audio fixture should be readable")
}

/// The provider's own payload, recovered from the normalized response's
/// `raw` field — the same value `raw_transcription` would have returned.
fn raw(response: &rig::transcription::TranscriptionResponse) -> openai::TranscriptionResponse {
    serde_json::from_value(response.raw.clone())
        .expect("raw payload should round-trip to OpenAI's own transcription type")
}

/// The transcript itself, so no cell asserts usage while ignoring whether the
/// endpoint still did its job.
fn assert_transcribed(text: &str) {
    assert!(
        text.split_whitespace().count() > 3,
        "the transcript must survive alongside the usage: {text:?}"
    );
}

// ---------------------------------------------------------------------------
// Duration-billed and token-billed models, through both clients.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn whisper_reports_duration_usage() {
    with_openai_transcription_cassette(
        "transcription_usage_matrix/whisper_reports_duration_usage",
        |client| async move {
            let response = client
                .transcription_model(openai::WHISPER_1)
                .transcription_request()
                .data(audio())
                .filename(Some("audio.mp3".to_owned()))
                .send()
                .await
                .expect("transcription should succeed");

            assert_transcribed(&response.text);
            match raw(&response).usage {
                Some(TranscriptionUsage::Duration { seconds, .. }) => assert!(seconds > 0.0),
                other => panic!("whisper-1 bills by duration, got {other:?}"),
            }
        },
    )
    .await;
}

#[tokio::test]
async fn gpt_4o_transcribe_reports_token_usage() {
    with_openai_transcription_cassette(
        "transcription_usage_matrix/gpt_4o_transcribe_reports_token_usage",
        |client| async move {
            let response = client
                .transcription_model("gpt-4o-transcribe")
                .transcription_request()
                .data(audio())
                .filename(Some("audio.mp3".to_owned()))
                .send()
                .await
                .expect("transcription should succeed");

            assert_transcribed(&response.text);
            match raw(&response).usage {
                Some(TranscriptionUsage::Tokens {
                    input_tokens,
                    input_token_details,
                    output_tokens,
                    total_tokens,
                    ..
                }) => {
                    assert!(input_tokens > 0 && output_tokens > 0);
                    assert_eq!(total_tokens, input_tokens + output_tokens);
                    // Audio and text input tokens bill at different rates, so
                    // the breakdown is part of what a turn cost.
                    let details = input_token_details.expect("input token breakdown");
                    assert_eq!(details.audio_tokens + details.text_tokens, input_tokens);
                }
                other => panic!("the gpt-4o-transcribe family bills by token, got {other:?}"),
            }
        },
    )
    .await;
}

#[tokio::test]
async fn gpt_4o_mini_transcribe_reports_token_usage() {
    with_openai_transcription_cassette(
        "transcription_usage_matrix/gpt_4o_mini_transcribe_reports_token_usage",
        |client| async move {
            let response = client
                .transcription_model("gpt-4o-mini-transcribe")
                .transcription_request()
                .data(audio())
                .filename(Some("audio.mp3".to_owned()))
                .send()
                .await
                .expect("transcription should succeed");

            assert_transcribed(&response.text);
            assert!(matches!(
                raw(&response).usage,
                Some(TranscriptionUsage::Tokens { .. })
            ));
        },
    )
    .await;
}

/// The Chat Completions client reaches the same shared transcription model, so
/// both extensions must decode the same payload.
#[tokio::test]
async fn completions_client_reports_duration_usage() {
    with_openai_transcription_cassette(
        "transcription_usage_matrix/completions_client_reports_duration_usage",
        |client| async move {
            let response = client
                .completions_api()
                .transcription_model(openai::WHISPER_1)
                .transcription_request()
                .data(audio())
                .filename(Some("audio.mp3".to_owned()))
                .send()
                .await
                .expect("transcription should succeed");

            assert_transcribed(&response.text);
            assert!(matches!(
                raw(&response).usage,
                Some(TranscriptionUsage::Duration { .. })
            ));
        },
    )
    .await;
}

#[tokio::test]
async fn completions_client_reports_token_usage() {
    with_openai_transcription_cassette(
        "transcription_usage_matrix/completions_client_reports_token_usage",
        |client| async move {
            let response = client
                .completions_api()
                .transcription_model("gpt-4o-transcribe")
                .transcription_request()
                .data(audio())
                .filename(Some("audio.mp3".to_owned()))
                .send()
                .await
                .expect("transcription should succeed");

            assert_transcribed(&response.text);
            assert!(matches!(
                raw(&response).usage,
                Some(TranscriptionUsage::Tokens { .. })
            ));
        },
    )
    .await;
}

/// A richer response format wraps `usage` in a much larger payload (adding
/// `segments`, `language`, and a top-level `duration`) without changing what
/// the usage itself is. The request that asks for it is pinned by the shared
/// multipart unit tests, not here — see the module doc.
#[tokio::test]
async fn verbose_json_still_reports_duration_usage() {
    with_openai_transcription_cassette(
        "transcription_usage_matrix/verbose_json_still_reports_duration_usage",
        |client| async move {
            let response = client
                .transcription_model(openai::WHISPER_1)
                .transcription_request()
                .data(audio())
                .filename(Some("audio.mp3".to_owned()))
                .additional_params(json!({ "response_format": "verbose_json" }))
                .send()
                .await
                .expect("transcription should succeed");

            assert_transcribed(&response.text);
            // The variant, not merely its presence: this response also carries
            // a top-level `duration` float, so it is the payload where a
            // variant-selection regression would surface first.
            assert!(
                matches!(
                    raw(&response).usage,
                    Some(TranscriptionUsage::Duration { .. })
                ),
                "a richer response format must not cost or reshape the usage: {:?}",
                raw(&response).usage
            );
        },
    )
    .await;
}

/// Reading the usage must not come at the transcript's expense: the normalized
/// response's `text` is still the transcript, not a stringified payload.
#[tokio::test]
async fn transcript_still_reaches_the_normalized_response() {
    with_openai_transcription_cassette(
        "transcription_usage_matrix/transcript_still_reaches_the_normalized_response",
        |client| async move {
            let response = client
                .transcription_model(openai::WHISPER_1)
                .transcription_request()
                .data(audio())
                .filename(Some("audio.mp3".to_owned()))
                .send()
                .await
                .expect("transcription should succeed");

            assert_eq!(response.text, raw(&response).text);
            assert_eq!(response.provider, "openai");
            assert_transcribed(&response.text);
        },
    )
    .await;
}

/// A rejected request has no usage to report. It must surface the provider's
/// own body rather than fail somewhere in the decode over the missing field —
/// the error path does not construct a response at all, and this pins that the
/// new field did not change that.
#[tokio::test]
async fn rejected_request_surfaces_the_provider_body() {
    with_openai_transcription_cassette(
        "transcription_usage_matrix/rejected_request_surfaces_the_provider_body",
        |client| async move {
            let Err(error) = client
                .transcription_model(openai::WHISPER_1)
                .transcription_request()
                .data(audio())
                .filename(Some("audio.mp3".to_owned()))
                .additional_params(json!({ "response_format": "rig-invalid" }))
                .send()
                .await
            else {
                panic!("an invalid response_format must be rejected")
            };

            let body = error.provider_response_body().unwrap_or_default();
            assert!(body.contains("response_format"), "{body}");
        },
    )
    .await;
}
