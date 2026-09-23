//! xAI image generation smoke test covering provider-specific additional parameters.

use rig::image_generation::ImageGenerationModel;
use rig::providers::openai;
use rig::providers::xai;
use serde_json::json;

use super::support::with_xai_cassette;
use crate::support::{IMAGE_PROMPT, assert_image_bytes};

#[tokio::test]
async fn image_generation_smoke() {
    with_xai_cassette(
        "image_generation/image_generation_smoke",
        |client| async move {
            // xAI's images route is OpenAI-shaped, so the chat-side
            // configuration serves it — rebuilt here from the cassette's
            // credential and base URL so the fixture still replays.
            let model = client
                .map_wire(|responses| {
                    openai::wire::OpenAI::with_key(&xai::DIALECT, responses.api_key)
                        .with_base_url(responses.base_url)
                })
                .image_generation(xai::image_generation::GROK_IMAGINE_IMAGE_PRO);

            let response = model
                .image_generation_request(IMAGE_PROMPT)
                .additional_params(json!({
                    "resolution": "2k",
                    "aspect_ratio": "4:3",
                }))
                .send()
                .await
                .expect("image generation should succeed");

            assert_image_bytes(&response.image);
        },
    )
    .await;
}
