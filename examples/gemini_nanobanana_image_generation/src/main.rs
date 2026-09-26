use anyhow::Result;
use rig::image_generation::ImageGenerationRequestBuilder;
use rig::providers::gemini::{self, Gemini};

#[tokio::main]
async fn main() -> Result<()> {
    let client = Gemini::from_env()?;
    let model = rig::model(client.image_generation(gemini::GEMINI_2_5_FLASH_IMAGE));

    let response = model
        .call(
            ImageGenerationRequestBuilder::new(
                "Generate a simple flat icon of a yellow banana on a white background.",
            )
            .width(512)
            .height(512)
            .build(),
        )
        .await?;

    let output_path = "/tmp/rig-nanobanana.png";
    std::fs::write(output_path, response.image)?;

    println!("Wrote generated image to {output_path}");

    Ok(())
}
