//! OpenRouter audio generation (TTS) smoke test.

use rig::audio_generation::AudioGenerationModel;
use rig::client::ProviderClient;
use rig::prelude::AudioGenerationClient;
use rig::providers::openrouter;

use crate::support::{AUDIO_TEXT, assert_nonempty_bytes};

#[tokio::test]
#[ignore = "requires OPENROUTER_API_KEY"]
async fn audio_generation_smoke() {
    let client = openrouter::Client::from_env().expect("client should build");
    let model = client.audio_generation_model(openrouter::GPT_4O_MINI_TTS);
    let response = model
        .audio_generation_request()
        .text(AUDIO_TEXT)
        .voice("alloy")
        .send()
        .await
        .expect("audio generation should succeed");

    assert_nonempty_bytes(&response.audio);
}
