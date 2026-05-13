//! Migrated from `examples/transcription.rs`.

use rig::prelude::TranscriptionClient;
use rig::providers::gemini;
use rig::transcription::TranscriptionModel;

use crate::support::{AUDIO_FIXTURE_PATH, assert_nonempty_response};

#[tokio::test]
async fn transcription_smoke() {
    let (cassette, client) =
        super::super::support::gemini_cassette("transcription/transcription_smoke").await;
    let model = client.transcription_model(gemini::completion::GEMINI_3_FLASH_PREVIEW);
    let response = model
        .transcription_request()
        .load_file(AUDIO_FIXTURE_PATH)
        .expect("should be able to load audio fixture")
        .send()
        .await
        .expect("transcription should succeed");

    assert_nonempty_response(&response.text);

    cassette.finish().await;
}
