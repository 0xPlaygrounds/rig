//! xAI image generation smoke test covering provider-specific additional parameters.

use rig::client::ProviderClient;
use rig::client::image_generation::ImageGenerationClient;
use rig::image_generation::ImageGenerationModel;
use rig::providers::xai;
use serde_json::json;

use crate::support::{IMAGE_PROMPT, assert_nonempty_bytes};

#[tokio::test]
#[ignore = "requires XAI_API_KEY"]
async fn image_generation_smoke() {
    let client = xai::Client::from_env();
    let model = client.image_generation_model(xai::image_generation::GROK_IMAGINE_IMAGE_PRO);

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
}
