use anyhow::Result;
use rig::image_generation::ImageGenerationRequestBuilder;
use rig::prelude::*;
use rig::providers::gemini::{self, Gemini};

#[tokio::main]
async fn main() -> Result<()> {
    let client = Gemini::from_env()?;
    let http = rig::rig_reqwest::bundled()?;
    let model = Model::new(
        client.image_generation(gemini::GEMINI_2_5_FLASH_IMAGE),
        http,
    );

    let response = model
        .call(
            ImageGenerationRequestBuilder::new(
                "Generate a simple flat icon of a yellow banana on a white background.",
            )
            .width(512)
            .height(512)
            .build(),
            None,
        )
        .await?;

    let output_path = "/tmp/rig-nanobanana.png";
    std::fs::write(output_path, response.image)?;

    println!("Wrote generated image to {output_path}");

    Ok(())
}
