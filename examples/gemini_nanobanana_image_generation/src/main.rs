//! Generates an image with Gemini's "nano banana" image model.
//! Requires `GEMINI_API_KEY`.
//!
//! Image generation is a free function over plain data: a
//! `gemini::functions::Config` naming the image model plus an `HttpRuntime`
//! for transport. There is no image-generation model type and no client.

use anyhow::Result;
use rig::http_runtime::HttpRuntime;
use rig::image_generation::ImageGenerationRequest;
use rig::providers::gemini;

#[tokio::main]
async fn main() -> Result<()> {
    let cfg = gemini::functions::Config::from_env(gemini::GEMINI_2_5_FLASH_IMAGE)?;
    let rt = HttpRuntime::new();

    let response = gemini::functions::generate_image(
        &cfg,
        &rt,
        ImageGenerationRequest::new(
            "Generate a simple flat icon of a yellow banana on a white background.",
        )
        .with_width(512)
        .with_height(512),
    )
    .await?;

    let output_path = "/tmp/rig-nanobanana.png";
    std::fs::write(output_path, response.image)?;

    println!("Wrote generated image to {output_path}");

    Ok(())
}
