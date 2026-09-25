//! Cassette-backed OpenRouter transcription smoke test.

use rig::providers::openrouter;
use rig::transcription::TranscriptionRequestBuilder;

use crate::support::{AUDIO_FIXTURE_PATH, assert_nonempty_response};

use super::super::support::with_openrouter_cassette;

#[tokio::test]
async fn transcription_smoke() {
    with_openrouter_cassette("transcription/transcription_smoke", |client| async move {
        let model = rig::model(client.transcription(openrouter::WHISPER_1));
        let response = model
            .call(
                TranscriptionRequestBuilder::from_file(AUDIO_FIXTURE_PATH)
                    .expect("should be able to load audio fixture")
                    .build(),
            )
            .await
            .expect("transcription should succeed");

        assert_nonempty_response(&response.text);
    })
    .await;
}
