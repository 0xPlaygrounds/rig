//! AWS Bedrock image generation smoke test inspired by OpenAI image generation tests.

use super::{
    BEDROCK_IMAGE_MODEL, client,
    support::{IMAGE_PROMPT, assert_nonempty_bytes},
};
use rig::image_generation::ImageGenerationRequestBuilder;

#[tokio::test]
#[ignore = "requires AWS credentials and Bedrock image generation model access"]
async fn image_generation_smoke() {
    let model = client().image_generation(BEDROCK_IMAGE_MODEL);
    let response = model
        .call(
            ImageGenerationRequestBuilder::new(IMAGE_PROMPT)
                .width(512)
                .height(512)
                .build(),
        )
        .await
        .expect("image generation request should succeed");

    assert_nonempty_bytes(&response.image);
}
