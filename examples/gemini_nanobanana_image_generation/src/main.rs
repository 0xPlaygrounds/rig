use anyhow::Result;
use rig::image_generation::ImageGenerationModel;
use rig::prelude::*;
use rig::providers::gemini::{self, Gemini};

#[tokio::main]
async fn main() -> Result<()> {
    let client = Gemini::from_env()?.bound()?;
    let model = client.image_generation(gemini::GEMINI_2_5_FLASH_IMAGE);

    let response = model
        .image_generation_request(
            "Generate a simple flat icon of a yellow banana on a white background.",
        )
        .width(512)
        .height(512)
        .send()
        .await?;

    let output_path = "/tmp/rig-nanobanana.png";
    std::fs::write(output_path, response.image)?;

    println!("Wrote generated image to {output_path}");

    Ok(())
}
