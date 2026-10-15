//! Mistral transcription, recorded: Voxtral Mini transcribes the audio
//! fixture, and Voxtral Small, an audio-input chat model, is refused.

use rig::providers::mistral;
use rig::transcription::TranscriptionRequestBuilder;

use super::support::with_mistral_cassette_result;
use crate::support::AUDIO_FIXTURE_PATH;

fn request() -> rig::transcription::TranscriptionRequest {
    TranscriptionRequestBuilder::from_file(AUDIO_FIXTURE_PATH)
        .expect("should be able to load audio fixture")
        .build()
}

#[tokio::test]
async fn voxtral_mini_transcribes_the_audio_fixture() -> anyhow::Result<()> {
    with_mistral_cassette_result(
        "transcription/voxtral_mini_transcribes_the_audio_fixture",
        |client| async move {
            let response = client
                .transcription(mistral::VOXTRAL_MINI)
                .call(request())
                .await?;
            anyhow::ensure!(!response.text.trim().is_empty(), "no transcript");
            Ok(())
        },
    )
    .await
}

#[tokio::test]
async fn voxtral_small_is_not_a_transcription_model() -> anyhow::Result<()> {
    with_mistral_cassette_result(
        "transcription/voxtral_small_is_not_a_transcription_model",
        |client| async move {
            let Err(error) = client
                .transcription(mistral::VOXTRAL_SMALL)
                .call(request())
                .await
            else {
                anyhow::bail!("the transcription endpoint accepted an audio-chat model");
            };
            let status = error
                .provider_response_status()
                .map(|status| status.as_u16());
            anyhow::ensure!(
                status == Some(400),
                "expected a 400, got {status:?}: {error}"
            );
            Ok(())
        },
    )
    .await
}
