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
//! **How these cells fail on `origin/main`.** They do not fail as mock
//! misses — the fix changes nothing rig sends. They fail on the assertion:
//! `TranscriptionResponse` has no `usage` field on `main`, so the cells do not
//! compile there at all. The recorded fixtures are unchanged bytes either way,
//! which is the point: the data was always on the wire.
//!
//! **Why this matrix is smaller than the others.** The input space is small
//! and fully enumerated rather than sampled. The endpoint's response has
//! exactly two documented usage shapes, and rig reaches it through exactly two
//! client extensions over one shared model type
//! (`internal::transcription::OpenAiTranscriptionModel`, also used by Groq and
//! Azure). The cross-product below is therefore complete: {duration-billed,
//! token-billed} × {Responses client, Chat Completions client}, plus the
//! request-shaping dimension that decides *which* shape comes back
//! (`additional_params`), plus the two absence cases. The shapes rig must
//! tolerate but no live model produces — a third `type`, an explicit `null` —
//! are unit cells beside the fix, because no recording can produce them.
//!
//! | # | cell | model | client | usage shape | status |
//! |---|------|-------|--------|-------------|--------|
//! | 1 | `whisper_reports_duration_usage` | whisper-1 | responses | duration | recorded |
//! | 2 | `gpt_4o_transcribe_reports_token_usage` | gpt-4o-transcribe | responses | tokens | recorded |
//! | 3 | `gpt_4o_mini_transcribe_reports_token_usage` | gpt-4o-mini-transcribe | responses | tokens | recorded |
//! | 4 | `completions_client_reports_duration_usage` | whisper-1 | completions | duration | recorded |
//! | 5 | `completions_client_reports_token_usage` | gpt-4o-transcribe | completions | tokens | recorded |
//! | 6 | `verbose_json_still_reports_usage` | whisper-1 | responses | duration + segments | recorded |
//! | 7 | `transcript_still_reaches_the_normalized_response` | whisper-1 | responses | duration | recorded |
//! | 8 | `error_turn_reports_no_usage` | whisper-1 | responses | none (400) | recorded |
//!
//! Unit cells beside the fix
//! (`usage_decodes_both_billing_shapes_and_keeps_unknown_ones`): both live
//! shapes, an unmodeled third shape carried verbatim instead of failing the
//! transcription, an absent `usage`, and an explicit `null`.

use rig::client::transcription::TranscriptionClient;
use rig::providers::openai::{self, TranscriptionUsage};
use rig::transcription::TranscriptionModel;
use serde_json::json;

use super::super::support::with_openai_transcription_cassette;
use crate::support::AUDIO_FIXTURE_PATH;

fn audio() -> Vec<u8> {
    std::fs::read(AUDIO_FIXTURE_PATH).expect("audio fixture should be readable")
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
            match response.response.usage {
                Some(TranscriptionUsage::Duration { seconds }) => assert!(seconds > 0.0),
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
            match response.response.usage {
                Some(TranscriptionUsage::Tokens {
                    input_tokens,
                    output_tokens,
                    total_tokens,
                }) => {
                    assert!(input_tokens > 0 && output_tokens > 0);
                    assert_eq!(total_tokens, input_tokens + output_tokens);
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
                response.response.usage,
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
                response.response.usage,
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
                response.response.usage,
                Some(TranscriptionUsage::Tokens { .. })
            ));
        },
    )
    .await;
}

/// The request-shaping dimension: `verbose_json` changes the payload around
/// `usage` (adding `segments`, `language`, `duration`) without changing that
/// the usage is reported.
#[tokio::test]
async fn verbose_json_still_reports_usage() {
    with_openai_transcription_cassette(
        "transcription_usage_matrix/verbose_json_still_reports_usage",
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
            assert!(
                response.response.usage.is_some(),
                "a richer response format must not cost the usage"
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

            assert_eq!(response.text, response.response.text);
            assert_transcribed(&response.text);
        },
    )
    .await;
}

/// A rejected request has no usage to report, and must still surface the
/// provider's own body rather than a decode failure over the missing field.
#[tokio::test]
async fn error_turn_reports_no_usage() {
    with_openai_transcription_cassette(
        "transcription_usage_matrix/error_turn_reports_no_usage",
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
