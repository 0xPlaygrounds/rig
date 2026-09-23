//! OpenAI image generation smoke test.

use rig::image_generation::ImageGenerationModel;
use rig::prelude::*;
use rig::providers::openai::{self, wire::OpenAI};

use crate::support::{IMAGE_PROMPT, assert_nonempty_bytes};

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn image_generation_smoke() {
    let client = OpenAI::from_env()
        .expect("config should build from env")
        .bound()
        .expect("transport should build");
    let model = client.image_generation(openai::DALL_E_2);

    let response = model
        .image_generation_request(IMAGE_PROMPT)
        .width(1024)
        .height(1024)
        .send()
        .await
        .expect("image generation should succeed");

    assert_nonempty_bytes(&response.image);
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn gpt_image_2_image_generation_smoke() {
    let client = OpenAI::from_env()
        .expect("config should build from env")
        .bound()
        .expect("transport should build");
    let model = client.image_generation(openai::GPT_IMAGE_2);

    let response = model
        .image_generation_request(IMAGE_PROMPT)
        .width(1024)
        .height(1024)
        .send()
        .await
        .expect("gpt-image-2 image generation should succeed");

    assert_nonempty_bytes(&response.image);

    let output_path = std::env::temp_dir().join("rig-openai-gpt-image-2-smoke.png");
    std::fs::write(&output_path, &response.image).expect("generated image should save to disk");
    println!("saved generated image to {}", output_path.display());
}
