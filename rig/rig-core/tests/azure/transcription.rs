//! Migrated from `examples/transcription.rs`.

use rig::client::ProviderClient;
use rig::prelude::TranscriptionClient;
use rig::providers::azure;
use rig::transcription::TranscriptionModel;

use crate::support::{AUDIO_FIXTURE_PATH, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires AZURE_OPENAI_API_KEY and related Azure env vars"]
async fn transcription_smoke() {
    let client = azure::Client::from_env();
    let model = client.transcription_model("whisper");
    let response = model
        .transcription_request()
        .load_file(AUDIO_FIXTURE_PATH)
        .send()
        .await
        .expect("transcription should succeed");

    assert_nonempty_response(&response.text);
}
