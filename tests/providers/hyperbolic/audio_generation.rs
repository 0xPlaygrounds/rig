//! Hyperbolic audio generation smoke test.

use rig::audio_generation::AudioGenerationModel;
use rig::prelude::*;
use rig::providers::openai::wire::{HYPERBOLIC, OpenAI};

use crate::support::{AUDIO_TEXT, assert_nonempty_bytes};

#[tokio::test]
#[ignore = "requires HYPERBOLIC_API_KEY"]
async fn audio_generation_smoke() {
    let provider = OpenAI::from_env_with(&HYPERBOLIC)
        .expect("config should build from env")
        .bound()
        .expect("transport should build");
    let model = provider.audio_generation("EN");

    let response = model
        .audio_generation_request(AUDIO_TEXT, "EN-US")
        .send()
        .await
        .expect("audio generation should succeed");

    assert_nonempty_bytes(&response.audio);
}
