//! MiniMax image generation smoke test.

use rig::image_generation::ImageGenerationRequestBuilder;
use rig::providers::minimax;

use crate::support::{IMAGE_PROMPT, assert_nonempty_bytes};

#[tokio::test]
#[ignore = "requires MINIMAX_API_KEY"]
async fn image_generation_smoke() {
    let model = minimax::from_env()
        .expect("MINIMAX_API_KEY should be set")
        .image_generation(minimax::IMAGE_01);

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
