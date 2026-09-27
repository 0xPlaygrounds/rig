//! Migrated from `examples/transcription.rs`.

use rig::providers::azure;
use rig::transcription::TranscriptionRequestBuilder;

use crate::support::{AUDIO_FIXTURE_PATH, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires AZURE_API_KEY or AZURE_TOKEN, plus AZURE_API_VERSION and AZURE_ENDPOINT"]
async fn transcription_smoke() {
    let azure = azure::from_env().expect("config should build from env");
    let model = azure.transcription("whisper");
    let response = model
        .call(
            TranscriptionRequestBuilder::from_file(AUDIO_FIXTURE_PATH)
                .expect("should be able to load audio fixture")
                .build(),
        )
        .await
        .expect("transcription should succeed");

    assert_nonempty_response(&response.text);
}
