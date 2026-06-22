use anyhow::Result;
use rig::client::{ProviderClient, image_generation::ImageGenerationClient};
use rig::image_generation::ImageGenerationModel;
use rig::providers::gemini;

#[tokio::main]
async fn main() -> Result<()> {
    let client = gemini::Client::from_env()?;
    let model = client.image_generation_model(gemini::GEMINI_2_5_FLASH_IMAGE);

    let response = model
        .image_generation_request()
        .prompt("Generate a simple flat icon of a yellow banana on a white background.")
        .width(512)
        .height(512)
        .send()
        .await?;

    let output_path = "/tmp/rig-nanobanana.png";
    std::fs::write(output_path, response.image)?;

    println!("Wrote generated image to {output_path}");

    Ok(())
}
