//! Hyperbolic image generation smoke test.

use rig::client::ProviderClient;
use rig::client::image_generation::ImageGenerationClient;
use rig::image_generation::ImageGenerationModel;
use rig::image_generation::ImageGenerationRequest;
use rig::providers::hyperbolic;

use crate::support::{IMAGE_PROMPT, assert_nonempty_bytes};

#[tokio::test]
#[ignore = "requires HYPERBOLIC_API_KEY"]
async fn image_generation_smoke() {
    let client = hyperbolic::Client::from_env().expect("client should build");
    let model = client.image_generation_model(hyperbolic::SDXL_TURBO);

    let response = model
        .image_generation(
            ImageGenerationRequest::new(IMAGE_PROMPT)
                .with_width(1024)
                .with_height(1024),
        )
        .await
        .expect("image generation should succeed");

    assert_nonempty_bytes(&response.image);
}
