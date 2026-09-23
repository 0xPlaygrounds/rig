//! Migrated from `examples/transcription.rs`.

use rig::providers::gemini;
use rig::transcription::TranscriptionRequestBuilder;

use super::super::support::with_gemini_cassette;
use crate::support::{AUDIO_FIXTURE_PATH, assert_nonempty_response};

#[tokio::test]
async fn transcription_smoke() {
    with_gemini_cassette("transcription/transcription_smoke", |client| async move {
        let model = client.transcription(gemini::completion::GEMINI_3_FLASH_PREVIEW);
        let response = TranscriptionRequestBuilder::from_file(model, AUDIO_FIXTURE_PATH)
            .expect("should be able to load audio fixture")
            .send()
            .await
            .expect("transcription should succeed");

        assert_nonempty_response(&response.text);
    })
    .await;
}
