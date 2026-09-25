//! Migrated from `examples/transcription.rs`.

use rig::providers::openai::{self, wire::OpenAI};
use rig::transcription::TranscriptionRequestBuilder;

use crate::support::{AUDIO_FIXTURE_PATH, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn transcription_smoke() {
    let client = OpenAI::from_env().expect("config should build from env");
    let model = rig::model(client.transcription(openai::WHISPER_1));
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
