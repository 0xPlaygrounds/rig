//! Hugging Face image generation smoke test.

use rig::image_generation::ImageGenerationModel;
use rig::prelude::*;
use rig::providers::openai::wire::{HUGGINGFACE, OpenAI};

use crate::support::{IMAGE_PROMPT, assert_nonempty_bytes};

#[tokio::test]
#[ignore = "requires HUGGINGFACE_API_KEY"]
async fn image_generation_smoke() {
    let provider = OpenAI::from_env_with(&HUGGINGFACE)
        .expect("config should build from env")
        .bound()
        .expect("transport should build");
    let model = provider.image_generation("stabilityai/stable-diffusion-3-medium-diffusers");

    let response = model
        .image_generation_request(IMAGE_PROMPT)
        .width(1024)
        .height(1024)
        .send()
        .await
        .expect("image generation should succeed");

    assert_nonempty_bytes(&response.image);
}
