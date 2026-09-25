//! Hyperbolic audio generation smoke test.

use rig::providers::openai::wire::{HYPERBOLIC, OpenAI};

use crate::support::{AUDIO_TEXT, assert_nonempty_bytes};
use rig::audio_generation::AudioGenerationRequestBuilder;

#[tokio::test]
#[ignore = "requires HYPERBOLIC_API_KEY"]
async fn audio_generation_smoke() {
    let provider = OpenAI::from_env_with(&HYPERBOLIC).expect("config should build from env");
    let model = rig::model(provider.audio_generation("EN"));

    let response = model
        .call(
            AudioGenerationRequestBuilder::new(AUDIO_TEXT, "EN-US").build(),
            None,
        )
        .await
        .expect("audio generation should succeed");

    assert_nonempty_bytes(&response.audio);
}
