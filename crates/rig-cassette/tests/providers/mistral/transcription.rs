//! Migrated from `examples/transcription.rs`.

use rig::prelude::*;
use rig::providers::mistral;
use rig::providers::openai::wire::{MISTRAL, OpenAI};
use rig::transcription::TranscriptionRequestBuilder;

use crate::support::{AUDIO_FIXTURE_PATH, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires MISTRAL_API_KEY"]
async fn transcription_smoke() {
    let client = OpenAI::from_env_with(&MISTRAL)
        .expect("MISTRAL_API_KEY should be set")
        .bound()
        .expect("client should build");
    let model = client.transcription(mistral::VOXTRAL_MINI);
    let response = TranscriptionRequestBuilder::from_file(model, AUDIO_FIXTURE_PATH)
        .expect("should be able to load audio fixture")
        .send()
        .await
        .expect("transcription should succeed");

    assert_nonempty_response(&response.text);
}
