//! Hyperbolic image generation smoke test.

use rig::providers::hyperbolic;
use rig::providers::openai::wire::{HYPERBOLIC, OpenAI};

use crate::support::{IMAGE_PROMPT, assert_nonempty_bytes};
use rig::image_generation::ImageGenerationRequestBuilder;

#[tokio::test]
#[ignore = "requires HYPERBOLIC_API_KEY"]
async fn image_generation_smoke() {
    let provider = OpenAI::from_env_with(&HYPERBOLIC).expect("config should build from env");
    let model = rig::model(provider.image_generation(hyperbolic::SDXL_TURBO));

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
