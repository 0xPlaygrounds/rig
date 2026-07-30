//! Hugging Face image generation smoke test.

use rig::client::ProviderClient;
use rig::client::image_generation::ImageGenerationClient;
use rig::image_generation::ImageGenerationModel;
use rig::image_generation::ImageGenerationRequest;
use rig::providers::huggingface;

use crate::support::{IMAGE_PROMPT, assert_nonempty_bytes};

#[tokio::test]
#[ignore = "requires HUGGINGFACE_API_KEY"]
async fn image_generation_smoke() {
    let client = huggingface::Client::from_env().expect("client should build");
    let model = client.image_generation_model("stabilityai/stable-diffusion-3-medium-diffusers");

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
