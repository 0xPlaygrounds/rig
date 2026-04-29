//! OpenAI audio generation smoke test.

use rig_core::audio_generation::AudioGenerationModel;
use rig_core::client::ProviderClient;
use rig_core::client::audio_generation::AudioGenerationClient;
use rig_core::providers::openai;

use crate::support::{AUDIO_TEXT, assert_nonempty_bytes};

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn audio_generation_smoke() {
    let client = openai::Client::from_env().expect("client should build");
    let model = client.audio_generation_model(openai::TTS_1);

    let response = model
        .audio_generation_request()
        .text(AUDIO_TEXT)
        .voice("alloy")
        .send()
        .await
        .expect("audio generation should succeed");

    assert_nonempty_bytes(&response.audio);
}
