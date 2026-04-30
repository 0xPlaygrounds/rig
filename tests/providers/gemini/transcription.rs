//! Migrated from `examples/transcription.rs`.

use rig::client::ProviderClient;
use rig::prelude::TranscriptionClient;
use rig::providers::gemini;
use rig::transcription::TranscriptionModel;

use crate::support::{AUDIO_FIXTURE_PATH, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires GEMINI_API_KEY"]
async fn transcription_smoke() {
    let client = gemini::Client::from_env().expect("client should build");
    let model = client.transcription_model(gemini::completion::GEMINI_3_FLASH_PREVIEW);
    let response = model
        .transcription_request()
        .load_file(AUDIO_FIXTURE_PATH)
        .expect("should be able to load audio fixture")
        .send()
        .await
        .expect("transcription should succeed");

    assert_nonempty_response(&response.text);
}
