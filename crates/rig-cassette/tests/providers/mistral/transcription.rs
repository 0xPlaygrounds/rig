//! Migrated from `examples/transcription.rs`.

use rig::providers::mistral;
use rig::providers::openai::wire::MISTRAL;
use rig::transcription::TranscriptionRequestBuilder;
use rig_test_support::cassette_models::OpenAiModels;

use crate::support::{AUDIO_FIXTURE_PATH, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires MISTRAL_API_KEY"]
async fn transcription_smoke() {
    let client = OpenAiModels::from_env_for(&MISTRAL).expect("MISTRAL_API_KEY should be set");
    let model = client.transcription(mistral::VOXTRAL_MINI);
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
