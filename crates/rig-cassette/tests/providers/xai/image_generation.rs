//! xAI image generation smoke test covering provider-specific additional parameters.

use rig::providers::openai;
use rig::providers::xai;
use rig_test_support::cassette_models::OpenAiModels;
use serde_json::json;

use super::support::with_xai_cassette;
use crate::support::{IMAGE_PROMPT, assert_image_bytes};
use rig::image_generation::ImageGenerationRequestBuilder;

#[tokio::test]
async fn image_generation_smoke() {
    with_xai_cassette(
        "image_generation/image_generation_smoke",
        |client| async move {
            // xAI's images route is OpenAI-shaped, so the chat-side
            // configuration serves it — rebuilt here from the cassette's
            // credential and base URL so the fixture still replays.
            let responses = client;
            let model = OpenAiModels::new(
                openai::wire::OpenAIConfig::with_key(&xai::DIALECT, responses.config.api_key)
                    .with_base_url(responses.config.base_url),
                responses.http,
            )
            .image_generation(xai::image_generation::GROK_IMAGINE_IMAGE_PRO);

            let response = model
                .call(
                    ImageGenerationRequestBuilder::new(IMAGE_PROMPT)
                        .additional_params(json!({
                            "resolution": "2k",
                            "aspect_ratio": "4:3",
                        }))
                        .build(),
                )
                .await
                .expect("image generation should succeed");

            assert_image_bytes(&response.image);
        },
    )
    .await;
}
