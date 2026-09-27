//! Hugging Face image generation smoke test.

use rig::providers::huggingface;

use crate::support::{IMAGE_PROMPT, assert_nonempty_bytes};
use rig::image_generation::ImageGenerationRequestBuilder;

#[tokio::test]
#[ignore = "requires HUGGINGFACE_API_KEY"]
async fn image_generation_smoke() {
    let provider = huggingface::from_env().expect("config should build from env");
    let model = provider.image_generation("stabilityai/stable-diffusion-3-medium-diffusers");

    let response = model
        .call(
            ImageGenerationRequestBuilder::new(IMAGE_PROMPT)
                .width(1024)
                .height(1024)
                .build(),
        )
        .await
        .expect("image generation should succeed");

    assert_nonempty_bytes(&response.image);
}
