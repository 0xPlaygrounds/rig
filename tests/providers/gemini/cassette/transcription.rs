//! Migrated from `examples/transcription.rs`.

use rig::providers::gemini;
use rig::transcription::TranscriptionRequest;

use super::super::support::with_gemini_cassette;
use crate::support::{AUDIO_FIXTURE_PATH, assert_nonempty_response};

#[tokio::test]
async fn transcription_smoke() {
    with_gemini_cassette("transcription/transcription_smoke", |client| async move {
        let response = gemini::functions::transcribe(
            &client.config(gemini::completion::GEMINI_3_FLASH_PREVIEW),
            &client.http(),
            TranscriptionRequest::from_file(AUDIO_FIXTURE_PATH)
                .expect("should be able to load audio fixture"),
        )
        .await
        .expect("transcription should succeed");

        assert_nonempty_response(&response.text);
    })
    .await;
}
