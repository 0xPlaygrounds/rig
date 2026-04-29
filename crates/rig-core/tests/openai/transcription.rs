//! Migrated from `examples/transcription.rs`.

use rig_core::client::ProviderClient;
use rig_core::prelude::TranscriptionClient;
use rig_core::providers::openai;
use rig_core::transcription::TranscriptionModel;

use crate::support::{AUDIO_FIXTURE_PATH, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn transcription_smoke() {
    let client = openai::Client::from_env().expect("client should build");
    let model = client.transcription_model(openai::WHISPER_1);
    let response = model
        .transcription_request()
        .load_file(AUDIO_FIXTURE_PATH)
        .expect("should be able to load audio fixture")
        .send()
        .await
        .expect("transcription should succeed");

    assert_nonempty_response(&response.text);
}
