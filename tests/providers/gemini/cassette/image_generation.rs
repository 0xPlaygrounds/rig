//! Gemini image generation cassette tests.

use rig::client::image_generation::ImageGenerationClient;
use rig::image_generation::ImageGenerationModel;
use rig::providers::gemini;

#[tokio::test]
async fn nano_banana_image_generation_smoke() {
    super::super::support::with_gemini_cassette(
        "image_generation/nano_banana_image_generation_smoke",
        |client| async move {
            let model = client.image_generation_model(gemini::GEMINI_2_5_FLASH_IMAGE);
            let response = model
                .image_generation_request()
                .prompt("Generate a simple flat icon of a yellow banana on a white background.")
                .width(256)
                .height(256)
                .send()
                .await
                .expect("Nano Banana image generation should succeed");

            assert!(
                response.image.len() > 100,
                "expected non-empty generated image bytes"
            );
            assert_eq!(
                response.response.model_version.as_deref(),
                Some(gemini::GEMINI_2_5_FLASH_IMAGE),
                "expected Gemini response to identify the image model"
            );
        },
    )
    .await;

    // rig#2322 — the image request literal seeded itself with
    // `..Default::default()`, so every Gemini image generation shipped a
    // hardcoded `maxOutputTokens: 4096` and `temperature: 1.0` that no caller
    // asked for. Image output is billed and limited in tokens, so an injected
    // cap here is a real truncation risk, not a harmless extra field.
    super::super::support::assert_recorded_sampling_fields(
        "image_generation/nano_banana_image_generation_smoke",
        &[],
    );
}
