//! Hyperbolic audio generation smoke test.

use rig::providers::openai::wire::{HYPERBOLIC, OpenAI};
use rig_test_support::endpoint::Endpoint;

use crate::support::{AUDIO_TEXT, assert_nonempty_bytes};
use rig::audio_generation::AudioGenerationRequestBuilder;

#[tokio::test]
#[ignore = "requires HYPERBOLIC_API_KEY"]
async fn audio_generation_smoke() {
    let provider = Endpoint::new(
        OpenAI::from_env_with(&HYPERBOLIC).expect("config should build from env"),
        rig::rig_reqwest::bundled().expect("transport should build"),
    );
    let model = provider.audio_generation("EN");

    let response = model
        .call(
            AudioGenerationRequestBuilder::new(AUDIO_TEXT, "EN-US").build(),
            None,
        )
        .await
        .expect("audio generation should succeed");

    assert_nonempty_bytes(&response.audio);
}
