//! OpenRouter audio generation (TTS) smoke test.

use rig::providers::openai::wire::{OPENROUTER, OpenAI};
use rig::providers::openrouter;

use crate::support::{AUDIO_TEXT, assert_nonempty_bytes};
use rig::audio_generation::AudioGenerationRequestBuilder;

#[tokio::test]
#[ignore = "requires OPENROUTER_API_KEY"]
async fn audio_generation_smoke() {
    let bound = OpenAI::from_env_with(&OPENROUTER).expect("OPENROUTER_API_KEY");
    let model = rig::model(bound.audio_generation(openrouter::GPT_4O_MINI_TTS));
    let response = model
        .call(AudioGenerationRequestBuilder::new(AUDIO_TEXT, "alloy").build())
        .await
        .expect("audio generation should succeed");

    assert_nonempty_bytes(&response.audio);
}
