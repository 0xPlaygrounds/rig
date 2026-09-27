//! xAI audio generation smoke test covering provider-specific additional parameters.

use rig::providers::xai;
use rig_test_support::cassette_models::OpenAiModels;
use serde_json::json;

use crate::support::{AUDIO_TEXT, assert_nonempty_bytes};
use rig::audio_generation::AudioGenerationRequestBuilder;

#[tokio::test]
#[ignore = "requires XAI_API_KEY"]
async fn audio_generation_smoke() {
    // xAI's text-to-speech route is OpenAI-shaped (`/v1/tts`, xAI's own body),
    // so the chat-side configuration is what serves it.
    let client = OpenAiModels::from_env_for(&xai::DIALECT).expect("XAI_API_KEY");
    let model = client.audio_generation(xai::TTS_1);

    let response = model
        .call(
            AudioGenerationRequestBuilder::new(AUDIO_TEXT, "eve")
                .additional_params(json!({
                    "language": "en",
                }))
                .build(),
        )
        .await
        .expect("audio generation should succeed");

    assert_nonempty_bytes(&response.audio);
}
