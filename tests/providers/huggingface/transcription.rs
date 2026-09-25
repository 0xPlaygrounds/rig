//! Migrated from `examples/transcription.rs`.

use rig::providers::openai::wire::{HUGGINGFACE, OpenAI};
use rig::transcription::TranscriptionRequestBuilder;
use rig::wire::Wire as _;

use crate::support::{AUDIO_FIXTURE_PATH, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires HUGGINGFACE_API_KEY"]
async fn transcription_smoke() {
    let provider = OpenAI::from_env_with(&HUGGINGFACE).expect("config should build from env");
    let model = provider
        .transcription("whisper-large-v3")
        .on(rig::transport());
    let response = model
        .call(
            TranscriptionRequestBuilder::from_file(AUDIO_FIXTURE_PATH)
                .expect("should be able to load audio fixture")
                .build(),
        )
        .await
        .expect("transcription should succeed");

    assert_nonempty_response(&response.text);
}
