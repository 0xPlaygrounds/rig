//! Migrated from `examples/transcription.rs`.

use rig::prelude::*;
use rig::providers::openai::{self, wire::OpenAI};
use rig::transcription::TranscriptionRequestBuilder;

use crate::support::{AUDIO_FIXTURE_PATH, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn transcription_smoke() {
    let client = OpenAI::from_env()
        .expect("config should build from env")
        .bound()
        .expect("transport should build");
    let model = client.transcription(openai::WHISPER_1);
    let response = TranscriptionRequestBuilder::from_file(model, AUDIO_FIXTURE_PATH)
        .expect("should be able to load audio fixture")
        .send()
        .await
        .expect("transcription should succeed");

    assert_nonempty_response(&response.text);
}
