//! xAI image generation smoke test covering provider-specific additional parameters.

use rig::http_runtime::HttpRuntime;
use rig::image_generation::ImageGenerationRequest;
use rig::providers::xai;
use serde_json::json;

use super::support::with_xai_cassette;
use crate::support::{IMAGE_PROMPT, assert_nonempty_bytes};

#[tokio::test]
async fn image_generation_smoke() {
    with_xai_cassette(
        "image_generation/image_generation_smoke",
        |env| async move {
            let cfg = env.config(xai::image_generation::GROK_IMAGINE_IMAGE_PRO);
            let rt = HttpRuntime::new();

            let response = xai::functions::generate_image(
                &cfg,
                &rt,
                ImageGenerationRequest::new(IMAGE_PROMPT).with_additional_params(json!({
                    "resolution": "2k",
                    "aspect_ratio": "4:3",
                })),
            )
            .await
            .expect("image generation should succeed");

            assert_nonempty_bytes(&response.image);
        },
    )
    .await;
}
