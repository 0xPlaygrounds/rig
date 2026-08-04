//! Generating an image with Bedrock.
//!
//! The `ImageGenerationModel` trait and its client are gone: image generation
//! is the free function `functions::generate_image` over a plain
//! `functions::ImageConfig` and an AWS client built from it.
use rig_bedrock::functions;
use rig_bedrock::image::AMAZON_NOVA_CANVAS;
use rig_core::image_generation::ImageGenerationRequest;
use std::fs::File;
use std::io::Write;
use std::path::Path;

const DEFAULT_PATH: &str = "./output.png";

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let cfg = functions::ImageConfig::new(AMAZON_NOVA_CANVAS);
    let client = functions::client_from_config(&cfg.client_config()).await;

    let response = functions::generate_image(
        &client,
        &cfg.model,
        ImageGenerationRequest::new(
            "A castle sitting upon a large mountain, overlooking the water.",
        )
        .with_width(512)
        .with_height(512),
    )
    .await?;

    // save image
    let mut file = File::create_new(Path::new(DEFAULT_PATH))?;
    file.write_all(&response.image)?;

    Ok(())
}
