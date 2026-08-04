//! OpenAI image generation smoke test.

use rig::image_generation::ImageGenerationRequest;
use rig::providers::openai;

use crate::support::{IMAGE_PROMPT, assert_nonempty_bytes};

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn image_generation_smoke() {
    let cfg = openai::functions::Config::from_env(openai::DALL_E_2).expect("config should build");
    let rt = rig::http_runtime::HttpRuntime::new();

    let response = openai::functions::generate_image(
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

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn gpt_image_2_image_generation_smoke() {
    let cfg =
        openai::functions::Config::from_env(openai::GPT_IMAGE_2).expect("config should build");
    let rt = rig::http_runtime::HttpRuntime::new();

    let response = openai::functions::generate_image(
        &cfg,
        &rt,
        ImageGenerationRequest::new(IMAGE_PROMPT)
            .with_width(1024)
            .with_height(1024),
    )
    .await
    .expect("gpt-image-2 image generation should succeed");

    assert_nonempty_bytes(&response.image);

    let output_path = std::env::temp_dir().join("rig-openai-gpt-image-2-smoke.png");
    std::fs::write(&output_path, &response.image).expect("generated image should save to disk");
    println!("saved generated image to {}", output_path.display());
}
