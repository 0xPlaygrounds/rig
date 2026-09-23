//! A generated image fed back as input on Venice: see `common/image_inputs.rs`.

use rig::providers::venice;

use super::super::support::with_venice_cassette;
use crate::image_inputs;

/// A Venice vision model reads the image Venice generated, as user content.
#[tokio::test]
async fn generated_image_as_user_content() {
    const SCENARIO: &str = "image_input_matrix/generated_image_as_user_content";
    with_venice_cassette(
        "image_input_matrix/generated_image_as_user_content",
        |client| async move {
            let bytes = image_inputs::generate(
                &client.image_generation(venice::image_generation::Z_IMAGE_TURBO),
                None,
                Some(serde_json::json!({ "format": "png", "seed": 42 })),
            )
            .await;
            image_inputs::as_user_content(
                &client.completion(venice::QWEN3_VL_235B_A22B),
                &bytes,
                None,
            )
            .await;
        },
    )
    .await;
    image_inputs::assert_recorded("venice", SCENARIO, image_inputs::Slot::UserContent);
}
