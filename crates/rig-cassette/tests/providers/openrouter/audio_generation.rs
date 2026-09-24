//! OpenRouter audio generation (TTS) smoke test.

use rig::prelude::*;
use rig_test_support::endpoint::Endpoint;

use rig::providers::openai::wire::{OPENROUTER, OpenAI};
use rig::providers::openrouter;

use crate::support::{AUDIO_TEXT, assert_nonempty_bytes};

#[tokio::test]
#[ignore = "requires OPENROUTER_API_KEY"]
async fn audio_generation_smoke() {
    let bound = Endpoint::new(
        OpenAI::from_env_with(&OPENROUTER).expect("OPENROUTER_API_KEY"),
        rig::rig_reqwest::bundled().expect("transport should build"),
    );
    let model = bound.audio_generation(openrouter::GPT_4O_MINI_TTS);
    let response = model
        .audio_generation_request(AUDIO_TEXT, "alloy")
        .send()
        .await
        .expect("audio generation should succeed");

    assert_nonempty_bytes(&response.audio);
}
