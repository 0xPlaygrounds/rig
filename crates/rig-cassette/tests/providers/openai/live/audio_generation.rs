//! OpenAI audio generation smoke test.

use rig::providers::openai::{self, wire::OpenAI};
use rig::wire::Wire as _;

use crate::support::{AUDIO_TEXT, assert_nonempty_bytes};
use rig::audio_generation::AudioGenerationRequestBuilder;

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn audio_generation_smoke() {
    let client = OpenAI::from_env().expect("config should build from env");
    let model = client.audio_generation(openai::TTS_1).on(rig::transport());

    let response = model
        .call(AudioGenerationRequestBuilder::new(AUDIO_TEXT, "alloy").build())
        .await
        .expect("audio generation should succeed");

    assert_nonempty_bytes(&response.audio);
}
