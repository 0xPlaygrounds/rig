//! Migrated from `examples/transcription.rs`.

use rig::prelude::*;
use rig::providers::groq;
use rig::providers::openai::wire::{GROQ, OpenAI};
use rig::transcription::TranscriptionRequestBuilder;

use crate::support::{AUDIO_FIXTURE_PATH, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires GROQ_API_KEY"]
async fn transcription_smoke() {
    let bound = OpenAI::from_env_with(&GROQ)
        .expect("GROQ_API_KEY should be set")
        .bound()
        .expect("transport should build");
    let model = bound.transcription(groq::WHISPER_LARGE_V3);
    let response = TranscriptionRequestBuilder::from_file(model, AUDIO_FIXTURE_PATH)
        .expect("should be able to load audio fixture")
        .send()
        .await
        .expect("transcription should succeed");

    assert_nonempty_response(&response.text);
}
