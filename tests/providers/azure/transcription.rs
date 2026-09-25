//! Migrated from `examples/transcription.rs`.

use rig::providers::openai::wire::{AZURE, OpenAI};
use rig::transcription::TranscriptionRequestBuilder;

use crate::support::{AUDIO_FIXTURE_PATH, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires AZURE_API_KEY or AZURE_TOKEN, plus AZURE_API_VERSION and AZURE_ENDPOINT"]
async fn transcription_smoke() {
    let azure = OpenAI::from_env_with(&AZURE).expect("config should build from env");
    let model = rig::model(azure.transcription("whisper"));
    let response = model
        .call(
            TranscriptionRequestBuilder::from_file(AUDIO_FIXTURE_PATH)
                .expect("should be able to load audio fixture")
                .build(),
            None,
        )
        .await
        .expect("transcription should succeed");

    assert_nonempty_response(&response.text);
}
