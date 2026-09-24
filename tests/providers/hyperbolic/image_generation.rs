//! Hyperbolic image generation smoke test.

use rig::providers::hyperbolic;
use rig::providers::openai::wire::{HYPERBOLIC, OpenAI};
use rig_test_support::endpoint::Endpoint;

use crate::support::{IMAGE_PROMPT, assert_nonempty_bytes};

#[tokio::test]
#[ignore = "requires HYPERBOLIC_API_KEY"]
async fn image_generation_smoke() {
    let provider = Endpoint::new(
        OpenAI::from_env_with(&HYPERBOLIC).expect("config should build from env"),
        rig::rig_reqwest::bundled().expect("transport should build"),
    );
    let model = provider.image_generation(hyperbolic::SDXL_TURBO);

    let response = model
        .image_generation_request(IMAGE_PROMPT)
        .width(1024)
        .height(1024)
        .send()
        .await
        .expect("image generation should succeed");

    assert_nonempty_bytes(&response.image);
}
