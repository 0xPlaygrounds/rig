//! xAI image generation smoke test covering provider-specific additional parameters.

use rig::client::image_generation::ImageGenerationClient;
use rig::image_generation::ImageGenerationModel;
use rig::providers::xai;
use serde_json::json;

use super::support::with_xai_cassette;
use crate::support::{IMAGE_PROMPT, assert_nonempty_bytes};

#[tokio::test]
async fn image_generation_smoke() {
    with_xai_cassette(
        "image_generation/image_generation_smoke",
        |client| async move {
            let model =
                client.image_generation_model(xai::image_generation::GROK_IMAGINE_IMAGE_PRO);

            let response = model
                .image_generation_request()
                .prompt(IMAGE_PROMPT)
                .additional_params(json!({
                    "resolution": "2k",
                    "aspect_ratio": "4:3",
                }))
                .send()
                .await
                .expect("image generation should succeed");

            assert_nonempty_bytes(&response.image);
        },
    )
    .await;
}
