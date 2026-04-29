//! OpenAI image generation smoke test.

use rig_core::client::ProviderClient;
use rig_core::client::image_generation::ImageGenerationClient;
use rig_core::image_generation::ImageGenerationModel;
use rig_core::providers::openai;

use crate::support::{IMAGE_PROMPT, assert_nonempty_bytes};

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn image_generation_smoke() {
    let client = openai::Client::from_env().expect("client should build");
    let model = client.image_generation_model(openai::DALL_E_2);

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

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn gpt_image_2_image_generation_smoke() {
    let client = openai::Client::from_env().expect("client should build");
    let model = client.image_generation_model(openai::GPT_IMAGE_2);

    let response = model
        .image_generation_request()
        .prompt(IMAGE_PROMPT)
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
