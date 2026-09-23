//! Cassette-backed Venice transcription smoke test.

use rig::providers::venice;
use rig::transcription::TranscriptionRequestBuilder;

use crate::support::{AUDIO_FIXTURE_PATH, assert_nonempty_response};

use super::super::support::with_venice_cassette;

#[tokio::test]
async fn transcription_smoke() {
    with_venice_cassette("transcription/transcription_smoke", |client| async move {
        let model = client.transcription(venice::WHISPER_LARGE_V3);
        let response = TranscriptionRequestBuilder::from_file(model, AUDIO_FIXTURE_PATH)
            .expect("should be able to load audio fixture")
            .send()
            .await
            .expect("transcription should succeed");

        assert_nonempty_response(&response.text);
    })
    .await;
}
