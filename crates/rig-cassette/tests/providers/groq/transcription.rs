//! Migrated from `examples/transcription.rs`.

use rig::providers::groq;
use rig::providers::openai::wire::GROQ;
use rig::transcription::TranscriptionRequestBuilder;
use rig_test_support::cassette_models::OpenAiModels;

use crate::support::{AUDIO_FIXTURE_PATH, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires GROQ_API_KEY"]
async fn transcription_smoke() {
    let bound = OpenAiModels::from_env_for(&GROQ).expect("GROQ_API_KEY should be set");
    let model = bound.transcription(groq::WHISPER_LARGE_V3);
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
