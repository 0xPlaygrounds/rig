//! Migrated from `examples/xai_image_generation.rs`.

use rig::client::ProviderClient;
use rig::client::image_generation::ImageGenerationClient;
use rig::image_generation::ImageGenerationModel;
use rig::providers::xai;

use crate::support::{IMAGE_PROMPT, assert_nonempty_bytes};

#[tokio::test]
#[ignore = "requires XAI_API_KEY"]
async fn provider_specific_image_generation() {
    let client = xai::Client::from_env();
    let model = client.image_generation_model(xai::image_generation::GROK_IMAGINE_IMAGE_PRO);
    let response = model
        .image_generation_request()
        .prompt(IMAGE_PROMPT)
        .additional_params(serde_json::json!({
            "resolution": "2k",
            "aspect_ratio": "4:3",
        }))
        .send()
        .await
        .expect("image generation should succeed");

    assert_nonempty_bytes(&response.image);
}
