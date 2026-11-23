//! Edge matrix for `additional_params` on OpenAI text-to-speech.
//!
//! `AudioGenerationRequestBuilder::additional_params` must reach the
//! `/v1/audio/speech` body through the default
//! `RawAudioGenerationProvider::audio_generation_request_body`, as it does for
//! providers that override the body. The parameters change the response:
//! `response_format` changes the returned container, and `instructions` steers
//! delivery on the `gpt-4o-mini-tts` family. The harness matches the recorded
//! request body, so a cell whose parameter is dropped is a mock miss.
//!
//! These scenarios use the direct recorder rather than the httpmock proxy,
//! because the endpoint answers with raw audio and the proxy exports bodies as
//! strings. That recorder captures nothing for a non-success response, so an
//! invalid `response_format` cannot be recorded; the container cell proves the
//! parameter arrived instead. Inputs are one or two words so the payloads stay
//! small. The merge itself is covered by a unit test beside the provider.

use rig::providers::openai;
use serde_json::json;

use super::super::support::with_openai_audio_cassette;
use rig::audio_generation::AudioGenerationRequestBuilder;

const TEXT: &str = "hello";
const VOICE: &str = "alloy";

/// The container an audio payload is in, by magic bytes — the only assertion
/// that can tell whether a caller's `response_format` actually took effect.
fn container(audio: &[u8]) -> &'static str {
    match audio {
        [0xff, 0xf3 | 0xf2 | 0xfb, ..] | [b'I', b'D', b'3', ..] => "mp3",
        [b'R', b'I', b'F', b'F', ..] => "wav",
        [b'f', b'L', b'a', b'C', ..] => "flac",
        [b'O', b'g', b'g', b'S', ..] => "ogg",
        _ => "unknown",
    }
}

#[tokio::test]
async fn response_format_flac_changes_the_container() {
    with_openai_audio_cassette(
        "audio_params_matrix/response_format_flac_changes_the_container",
        |client| async move {
            let response = client
                .openai
                .audio_generation(openai::TTS_1)
                .call(
                    AudioGenerationRequestBuilder::new(TEXT, VOICE)
                        .additional_params(json!({ "response_format": "flac" }))
                        .build(),
                )
                .await
                .expect("speech synthesis should succeed");

            assert_eq!(container(&response.audio), "flac");
        },
    )
    .await;
}

/// `instructions` is the other parameter this endpoint takes, and only the
/// `gpt-4o-mini-tts` family acts on it. `tts-1` accepts and ignores it, so a
/// rejection has nothing to observe.
#[tokio::test]
async fn instructions_reach_the_tts_model() {
    with_openai_audio_cassette(
        "audio_params_matrix/instructions_reach_the_tts_model",
        |client| async move {
            let response = client
                .openai
                .audio_generation("gpt-4o-mini-tts")
                .call(
                    AudioGenerationRequestBuilder::new(TEXT, VOICE)
                        .additional_params(json!({ "instructions": "Speak slowly and warmly." }))
                        .build(),
                )
                .await
                .expect("speech synthesis should succeed");

            assert!(!response.audio.is_empty());
        },
    )
    .await;
}

/// A non-object payload merges nothing and leaves the derived body as it was.
#[tokio::test]
async fn non_object_additional_params_are_a_no_op() {
    with_openai_audio_cassette(
        "audio_params_matrix/non_object_additional_params_are_a_no_op",
        |client| async move {
            let response = client
                .openai
                .audio_generation(openai::TTS_1)
                .call(
                    AudioGenerationRequestBuilder::new(TEXT, VOICE)
                        .additional_params(json!("not-an-object"))
                        .build(),
                )
                .await
                .expect("speech synthesis should succeed");

            assert_eq!(container(&response.audio), "mp3");
        },
    )
    .await;
}
