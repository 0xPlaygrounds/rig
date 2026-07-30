//! AWS Bedrock image generation smoke test inspired by OpenAI image generation tests.

use rig::image_generation::ImageGenerationRequest;

use super::{
    BEDROCK_IMAGE_MODEL, aws_client,
    support::{IMAGE_PROMPT, assert_nonempty_bytes},
};

#[tokio::test]
#[ignore = "requires AWS credentials and Bedrock image generation model access"]
async fn image_generation_smoke() {
    let response = rig::bedrock::functions::generate_image(
        &aws_client().await,
        BEDROCK_IMAGE_MODEL,
        ImageGenerationRequest::new(IMAGE_PROMPT)
            .with_width(512)
            .with_height(512),
    )
    .await
    .expect("image generation request should succeed");

    assert_nonempty_bytes(&response.image);
}
