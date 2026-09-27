//! Migrated from `examples/transcription.rs`.

use rig::providers::openai::{self};
use rig::transcription::TranscriptionRequestBuilder;
use rig_test_support::cassette_models::OpenAiModels;

use crate::support::{AUDIO_FIXTURE_PATH, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn transcription_smoke() {
    let client = OpenAiModels::from_env().expect("config should build from env");
    let model = client.transcription(openai::WHISPER_1);
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
