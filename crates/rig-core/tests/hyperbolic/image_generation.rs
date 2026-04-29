//! Hyperbolic image generation smoke test.

use rig_core::client::ProviderClient;
use rig_core::client::image_generation::ImageGenerationClient;
use rig_core::image_generation::ImageGenerationModel;
use rig_core::providers::hyperbolic;

use crate::support::{IMAGE_PROMPT, assert_nonempty_bytes};

#[tokio::test]
#[ignore = "requires HYPERBOLIC_API_KEY"]
async fn image_generation_smoke() {
    let client = hyperbolic::Client::from_env().expect("client should build");
    let model = client.image_generation_model(hyperbolic::SDXL_TURBO);

    let response = model
        .image_generation_request()
        .prompt(IMAGE_PROMPT)
        .width(1024)
        .height(1024)
        .send()
        .await
        .expect("image generation should succeed");

    assert_nonempty_bytes(&response.image);
}
