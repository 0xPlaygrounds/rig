//! Cassette-backed Venice image generation smoke test.
//!
//! Venice's image wire is its own: `POST /image/generate` with `width`/
//! `height`, answering with `{ id, images: [base64], … }`. The recorded
//! request body is what pins that shape — a regression to OpenAI's
//! `/images/generations` body would fail as a mock miss.

use rig::client::image_generation::ImageGenerationClient;
use rig::image_generation::ImageGenerationModel;
use rig::providers::venice;

use super::super::support::with_venice_cassette;

#[tokio::test]
async fn image_generation_smoke() {
    with_venice_cassette(
        "image_generation/image_generation_smoke",
        |client| async move {
            let model = client.image_generation_model(venice::VENICE_SD35);
            let response = model
                .image_generation_request()
                .prompt("A lighthouse on a rocky cliff at sunrise, clean illustrative style.")
                .width(256)
                .height(256)
                .additional_params(serde_json::json!({
                    "format": "webp",
                    "seed": 42,
                    "steps": 4,
                    "safe_mode": true,
                    "embed_exif_metadata": false,
                }))
                .send()
                .await
                .expect("Venice image generation should succeed");

            assert!(
                !response.image.is_empty(),
                "expected decoded image bytes from the base64 payload"
            );
            assert!(
                !response.response.id.is_empty(),
                "expected Venice to report a generation id"
            );
            assert_eq!(
                response.response.images.len(),
                1,
                "expected exactly one image for a single-variant request"
            );
        },
    )
    .await;
}
