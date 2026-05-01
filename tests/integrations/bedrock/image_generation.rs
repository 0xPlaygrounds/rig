//! AWS Bedrock image generation smoke test inspired by OpenAI image generation tests.

use rig::image_generation::ImageGenerationModel;
use rig::prelude::ImageGenerationClient;

use super::{
    BEDROCK_IMAGE_MODEL, client,
    support::{IMAGE_PROMPT, assert_nonempty_bytes},
};

#[tokio::test]
#[ignore = "requires AWS credentials and Bedrock image generation model access"]
async fn image_generation_smoke() {
    let model = client().image_generation_model(BEDROCK_IMAGE_MODEL);
    let response = model
        .image_generation_request()
        .prompt(IMAGE_PROMPT)
        .width(512)
        .height(512)
        .send()
        .await
        .expect("image generation request should succeed");

    assert_nonempty_bytes(&response.image);
}
