//! Cassette-backed Venice text-to-speech smoke test.

use rig::audio_generation::AudioGenerationModel;
use rig::client::audio_generation::AudioGenerationClient;
use rig::providers::venice;

use super::super::support::with_venice_direct_cassette;

/// Venice's `response_format` has no field on Rig's audio request, so it
/// travels through `additional_params` — which is also what this pins.
///
/// Recorded, not assumed: with no `response_format` Venice answers this path
/// with RIFF/WAV even though its model metadata advertises `mp3` as the
/// default format, so the format is requested explicitly (and keeps the
/// fixture small).
#[tokio::test]
async fn audio_generation_smoke() {
    with_venice_direct_cassette(
        "audio_generation/audio_generation_smoke",
        |client| async move {
            let model = client.audio_generation_model(venice::TTS_KOKORO);
            let response = model
                .audio_generation_request()
                .text("Rig speaks.")
                .voice("af_sky")
                .speed(1.0)
                .additional_params(serde_json::json!({ "response_format": "mp3" }))
                .send()
                .await
                .expect("Venice speech synthesis should succeed");

            assert!(
                response.audio.len() > 1024,
                "expected synthesized audio bytes, got {} bytes",
                response.audio.len()
            );
            assert_eq!(
                &response.audio[..3],
                b"ID3",
                "expected an MP3 payload for the requested response format"
            );
        },
    )
    .await;
}
