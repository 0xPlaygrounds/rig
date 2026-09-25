//! OpenRouter audio generation (TTS) smoke test.

use rig_test_support::endpoint::Endpoint;

use rig::providers::openai::wire::{OPENROUTER, OpenAI};
use rig::providers::openrouter;

use crate::support::{AUDIO_TEXT, assert_nonempty_bytes};
use rig::audio_generation::AudioGenerationRequestBuilder;

#[tokio::test]
#[ignore = "requires OPENROUTER_API_KEY"]
async fn audio_generation_smoke() {
    let bound = Endpoint::new(
        OpenAI::from_env_with(&OPENROUTER).expect("OPENROUTER_API_KEY"),
        rig::rig_reqwest::shared(),
    );
    let model = bound.audio_generation(openrouter::GPT_4O_MINI_TTS);
    let response = model
        .call(
            AudioGenerationRequestBuilder::new(AUDIO_TEXT, "alloy").build(),
            None,
        )
        .await
        .expect("audio generation should succeed");

    assert_nonempty_bytes(&response.audio);
}
