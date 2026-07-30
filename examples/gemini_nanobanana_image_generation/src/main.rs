use anyhow::Result;
use rig::client::{ProviderClient, image_generation::ImageGenerationClient};
use rig::image_generation::ImageGenerationModel;
use rig::image_generation::ImageGenerationRequest;
use rig::providers::gemini;

#[tokio::main]
async fn main() -> Result<()> {
    let client = gemini::Client::from_env()?;
    let model = client.image_generation_model(gemini::GEMINI_2_5_FLASH_IMAGE);

    let response = model
        .image_generation(
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
