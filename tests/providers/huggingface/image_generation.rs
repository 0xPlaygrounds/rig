//! Hugging Face image generation smoke test.

use rig::http_runtime::HttpRuntime;
use rig::image_generation::ImageGenerationRequest;
use rig::providers::huggingface;

use crate::support::{IMAGE_PROMPT, assert_nonempty_bytes};

#[tokio::test]
#[ignore = "requires HUGGINGFACE_API_KEY"]
async fn image_generation_smoke() {
    let cfg =
        huggingface::functions::Config::from_env("stabilityai/stable-diffusion-3-medium-diffusers")
            .expect("config should build");
    let rt = HttpRuntime::new();

    let response = huggingface::functions::generate_image(
        &cfg,
        &rt,
        ImageGenerationRequest::new(IMAGE_PROMPT)
            .with_width(1024)
            .with_height(1024),
    )
    .await
    .expect("image generation should succeed");

    assert_nonempty_bytes(&response.image);
}
