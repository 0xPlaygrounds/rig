//! OpenAI image generation smoke test.

use rig::providers::openai::{self, wire::OpenAI};
use rig::wire::Wire as _;

use crate::support::{IMAGE_PROMPT, assert_nonempty_bytes};
use rig::image_generation::ImageGenerationRequestBuilder;

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn image_generation_smoke() {
    let client = OpenAI::from_env().expect("config should build from env");
    let model = client
        .image_generation(openai::DALL_E_2)
        .on(rig::transport());

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

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn gpt_image_2_image_generation_smoke() {
    let client = OpenAI::from_env().expect("config should build from env");
    let model = client
        .image_generation(openai::GPT_IMAGE_2)
        .on(rig::transport());

    let response = model
        .call(
            ImageGenerationRequestBuilder::new(IMAGE_PROMPT)
                .width(1024)
                .height(1024)
                .build(),
        )
        .await
        .expect("gpt-image-2 image generation should succeed");

    assert_nonempty_bytes(&response.image);

    let output_path = std::env::temp_dir().join("rig-openai-gpt-image-2-smoke.png");
    std::fs::write(&output_path, &response.image).expect("generated image should save to disk");
    println!("saved generated image to {}", output_path.display());
}
