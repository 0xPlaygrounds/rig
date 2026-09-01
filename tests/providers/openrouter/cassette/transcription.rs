//! Cassette-backed OpenRouter transcription smoke test.

use rig::prelude::TranscriptionClient;
use rig::providers::openrouter;
use rig::transcription::TranscriptionModel;

use crate::support::{AUDIO_FIXTURE_PATH, assert_nonempty_response};

use super::super::support::with_openrouter_cassette;

#[tokio::test]
async fn transcription_smoke() {
    with_openrouter_cassette("transcription/transcription_smoke", |client| async move {
        let model = client.transcription_model(openrouter::WHISPER_1);
        let response = model
            .transcription_request()
            .load_file(AUDIO_FIXTURE_PATH)
            .expect("should be able to load audio fixture")
            .send()
            .await
            .expect("transcription should succeed");

        assert_nonempty_response(&response.text);
    })
    .await;
}
