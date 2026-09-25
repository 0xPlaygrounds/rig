use rig_bedrock::client::BedrockRuntime;
use rig_bedrock::image::{AMAZON_NOVA_CANVAS, Images};
use rig_core::Model;
use rig_core::image_generation::ImageGenerationRequestBuilder;
use std::fs::File;
use std::io::Write;
use std::path::Path;

const DEFAULT_PATH: &str = "./output.png";

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let image_generation_model =
        Model::new(Images::new(AMAZON_NOVA_CANVAS), BedrockRuntime::from_env());
    let response = image_generation_model
        .call(
            ImageGenerationRequestBuilder::new(
                "A castle sitting upon a large mountain, overlooking the water.",
            )
            .width(512)
            .height(512)
            .build(),
            None,
        )
        .await?;

    // save image
    let mut file = File::create_new(Path::new(DEFAULT_PATH))?;
    file.write_all(&response.image)?;

    Ok(())
}
