//! OpenAI audio generation smoke test.

use rig::audio_generation::AudioGenerationModel;
use rig::audio_generation::AudioGenerationRequest;
use rig::client::ProviderClient;
use rig::client::audio_generation::AudioGenerationClient;
use rig::providers::openai;

use crate::support::{AUDIO_TEXT, assert_nonempty_bytes};

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn audio_generation_smoke() {
    let client = openai::Client::from_env().expect("client should build");
    let model = client.audio_generation_model(openai::TTS_1);

    let response = model
        .audio_generation(AudioGenerationRequest::new(AUDIO_TEXT, "alloy"))
        .await
        .expect("audio generation should succeed");

    assert_nonempty_bytes(&response.audio);
}
