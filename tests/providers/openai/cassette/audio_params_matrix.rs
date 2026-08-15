//! Edge matrix for `additional_params` on OpenAI text-to-speech.
//!
//! **Bug.** `AudioGenerationRequestBuilder::additional_params` was never
//! merged into the `/v1/audio/speech` body by the *default*
//! `RawAudioGenerationProvider::audio_generation_request_body`, so it was
//! silently inert for whoever inherited that default — OpenAI included. Every
//! provider that overrides the body already merged it (xAI, OpenRouter,
//! Venice), which is what makes this a defect rather than a design: three
//! siblings honor the field and the default does not.
//!
//! The parameters demonstrably change the response, so this is not cosmetic:
//! `response_format: "wav"` returns a RIFF payload where the default returns
//! MP3, and `instructions` steers delivery on the `gpt-4o-mini-tts` family.
//!
//! **How these cells fail on `origin/main`.** The harness matches the recorded
//! request body, so every cell carrying a caller parameter is a mock miss on
//! `main`, which omits it. The two cells that assert the returned *container*
//! (`wav`, `flac`) would fail on the bytes as well.
//!
//! These scenarios record through the direct recorder rather than the httpmock
//! proxy — the endpoint answers with raw audio, and the proxy exports bodies as
//! strings, so a recorded speech response would replay as zero bytes. Inputs
//! are one or two words so the recorded payloads stay small.
//!
//! | # | cell | model | params | asserts | status |
//! |---|------|-------|--------|---------|--------|
//! | 1 | `default_body_returns_mp3` | tts-1 | none | MP3 magic | recorded |
//! | 2 | `response_format_wav_changes_the_container` | tts-1 | response_format | RIFF magic | recorded |
//! | 3 | `response_format_flac_changes_the_container` | tts-1 | response_format | fLaC magic | recorded |
//! | 4 | `instructions_reach_the_tts_model` | gpt-4o-mini-tts | instructions | 200 + audio | recorded |
//! | 5 | `completions_client_shares_the_fixed_body` | tts-1 | response_format | RIFF magic | recorded |
//! | 6 | `additional_params_can_override_voice` | tts-1 | voice | 200 + audio | recorded |
//! | 7 | `non_object_additional_params_are_a_no_op` | tts-1 | `"not-an-object"` | MP3 magic | recorded |
//!
//! **Two cells were designed and dropped**, with reasons, rather than left as
//! silent gaps:
//!
//! * *an invalid `response_format` rejected by the endpoint* — the natural way
//!   to prove a parameter reached the wire, and the form the sibling image
//!   matrix uses throughout. It cannot be recorded here: these scenarios need
//!   the direct recorder (binary response bodies), and that path captures no
//!   interaction for a non-success response, so the cassette comes out empty.
//!   Cells 2, 3 and 5 prove the same thing more strongly anyway — the returned
//!   *container* changes — and the error path itself is covered beside the
//!   provider by `audio_generation_non_success_preserves_status_and_body`.
//! * *`instructions` rejected by a model that does not take it* — `tts-1`
//!   accepts and silently ignores the field (verified live: `200`, MP3 body),
//!   so there is no observable outcome to assert. Cell 4 covers the model that
//!   does honor it.
//!
//! A unit cell beside the fix (`request_body_merges_additional_params_last`)
//! covers the merge itself, including overriding each derived key and a
//! non-object payload — shapes that need no network at all.

use rig::audio_generation::AudioGenerationModel;
use rig::client::audio_generation::AudioGenerationClient;
use rig::providers::openai;
use serde_json::json;

use super::super::support::with_openai_audio_cassette;

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
async fn default_body_returns_mp3() {
    with_openai_audio_cassette(
        "audio_params_matrix/default_body_returns_mp3",
        |client| async move {
            let response = client
                .audio_generation_model(openai::TTS_1)
                .audio_generation_request()
                .text(TEXT)
                .voice(VOICE)
                .send()
                .await
                .expect("speech synthesis should succeed");

            assert_eq!(container(&response.audio), "mp3");
        },
    )
    .await;
}

#[tokio::test]
async fn response_format_wav_changes_the_container() {
    with_openai_audio_cassette(
        "audio_params_matrix/response_format_wav_changes_the_container",
        |client| async move {
            let response = client
                .audio_generation_model(openai::TTS_1)
                .audio_generation_request()
                .text(TEXT)
                .voice(VOICE)
                .additional_params(json!({ "response_format": "wav" }))
                .send()
                .await
                .expect("speech synthesis should succeed");

            assert_eq!(
                container(&response.audio),
                "wav",
                "the caller's response_format must reach the endpoint"
            );
        },
    )
    .await;
}

#[tokio::test]
async fn response_format_flac_changes_the_container() {
    with_openai_audio_cassette(
        "audio_params_matrix/response_format_flac_changes_the_container",
        |client| async move {
            let response = client
                .audio_generation_model(openai::TTS_1)
                .audio_generation_request()
                .text(TEXT)
                .voice(VOICE)
                .additional_params(json!({ "response_format": "flac" }))
                .send()
                .await
                .expect("speech synthesis should succeed");

            assert_eq!(container(&response.audio), "flac");
        },
    )
    .await;
}

/// `instructions` is the other parameter this endpoint takes, and only the
/// `gpt-4o-mini-tts` family acts on it — `tts-1` accepts and ignores it, which
/// is why the rejection cell it would have anchored was dropped (see above).
#[tokio::test]
async fn instructions_reach_the_tts_model() {
    with_openai_audio_cassette(
        "audio_params_matrix/instructions_reach_the_tts_model",
        |client| async move {
            let response = client
                .audio_generation_model("gpt-4o-mini-tts")
                .audio_generation_request()
                .text(TEXT)
                .voice(VOICE)
                .additional_params(json!({ "instructions": "Speak slowly and warmly." }))
                .send()
                .await
                .expect("speech synthesis should succeed");

            assert!(!response.audio.is_empty());
        },
    )
    .await;
}

#[tokio::test]
async fn completions_client_shares_the_fixed_body() {
    with_openai_audio_cassette(
        "audio_params_matrix/completions_client_shares_the_fixed_body",
        |client| async move {
            let response = client
                .completions_api()
                .audio_generation_model(openai::TTS_1)
                .audio_generation_request()
                .text(TEXT)
                .voice(VOICE)
                .additional_params(json!({ "response_format": "wav" }))
                .send()
                .await
                .expect("speech synthesis should succeed");

            assert_eq!(container(&response.audio), "wav");
        },
    )
    .await;
}

/// Merged last, so a caller can override a key the builder derives.
#[tokio::test]
async fn additional_params_can_override_voice() {
    with_openai_audio_cassette(
        "audio_params_matrix/additional_params_can_override_voice",
        |client| async move {
            let response = client
                .audio_generation_model(openai::TTS_1)
                .audio_generation_request()
                .text(TEXT)
                .voice(VOICE)
                .additional_params(json!({ "voice": "nova" }))
                .send()
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
                .audio_generation_model(openai::TTS_1)
                .audio_generation_request()
                .text(TEXT)
                .voice(VOICE)
                .additional_params(json!("not-an-object"))
                .send()
                .await
                .expect("speech synthesis should succeed");

            assert_eq!(container(&response.audio), "mp3");
        },
    )
    .await;
}
