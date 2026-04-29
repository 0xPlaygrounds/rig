//! Migrated from `examples/transcription.rs`.

use rig_core::client::ProviderClient;
use rig_core::prelude::TranscriptionClient;
use rig_core::providers::huggingface;
use rig_core::transcription::TranscriptionModel;

use crate::support::{AUDIO_FIXTURE_PATH, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires HUGGINGFACE_API_KEY"]
async fn transcription_smoke() {
    let client = huggingface::Client::from_env().expect("client should build");
    let model = client.transcription_model("whisper-large-v3");
    let response = model
        .transcription_request()
        .load_file(AUDIO_FIXTURE_PATH)
        .expect("should be able to load audio fixture")
        .send()
        .await
        .expect("transcription should succeed");

    assert_nonempty_response(&response.text);
}
