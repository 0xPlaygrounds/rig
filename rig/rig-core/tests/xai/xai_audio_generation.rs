//! Migrated from `examples/xai_audio_generation.rs`.

use rig::audio_generation::AudioGenerationModel;
use rig::client::ProviderClient;
use rig::client::audio_generation::AudioGenerationClient;
use rig::providers::xai;

use crate::support::{AUDIO_TEXT, assert_nonempty_bytes};

#[tokio::test]
#[ignore = "requires XAI_API_KEY"]
async fn provider_specific_audio_generation() {
    let client = xai::Client::from_env();
    let model = client.audio_generation_model(xai::TTS_1);
    let response = model
        .audio_generation_request()
        .text(AUDIO_TEXT)
        .voice("eve")
        .additional_params(serde_json::json!({ "language": "en" }))
        .send()
        .await
        .expect("audio generation should succeed");

    assert_nonempty_bytes(&response.audio);
}
