//! MiniMax speech and voice-cloning smoke tests.

use rig::audio_generation::AudioGenerationRequestBuilder;
use rig::providers::minimax;
use rig::providers::minimax::voice_clone::{CloneAudio, VoiceClone};

use crate::support::{AUDIO_TEXT, assert_nonempty_bytes};

#[tokio::test]
#[ignore = "requires MINIMAX_API_KEY"]
async fn audio_generation_smoke() {
    let model = minimax::from_env()
        .expect("MINIMAX_API_KEY should be set")
        .audio_generation(minimax::SPEECH_2_8_TURBO);

    let response = model
        .call(AudioGenerationRequestBuilder::new(AUDIO_TEXT, "English_Graceful_Lady").build())
        .await
        .expect("audio generation should succeed");

    assert_nonempty_bytes(&response.audio);
}

/// Clones a voice from a sample the speech endpoint reads, which needs at
/// least 10 seconds of audio, so the sample text is long.
#[tokio::test]
#[ignore = "requires MINIMAX_API_KEY with voice-cloning access; creates a voice"]
async fn voice_cloning_smoke() {
    let client = minimax::from_env().expect("MINIMAX_API_KEY should be set");
    let sample = client
        .audio_generation(minimax::SPEECH_2_8_TURBO)
        .call(
            AudioGenerationRequestBuilder::new([AUDIO_TEXT; 8].join(" "), "English_Graceful_Lady")
                .build(),
        )
        .await
        .expect("the sample should synthesize")
        .audio;

    let cloning = minimax::voice_cloning(&client);
    let file = cloning
        .upload(CloneAudio::sample("sample.mp3", sample))
        .await
        .expect("the sample should upload");
    let voice_id = format!("RigSmoke{}", file.file_id);
    let voice = cloning
        .clone_voice(VoiceClone::new(file.file_id, voice_id.clone()))
        .await
        .expect("the voice should clone");
    assert_eq!(voice.voice_id, voice_id);

    let speech = client
        .audio_generation(minimax::SPEECH_2_8_TURBO)
        .call(AudioGenerationRequestBuilder::new(AUDIO_TEXT, voice.voice_id).build())
        .await
        .expect("the cloned voice should speak");
    assert_nonempty_bytes(&speech.audio);
}
