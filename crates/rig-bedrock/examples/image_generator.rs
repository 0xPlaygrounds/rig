use rig_bedrock::client::BedrockRuntime;
use rig_bedrock::image::AMAZON_NOVA_CANVAS;
use rig_core::RigError;
use rig_core::image_generation::ImageGenerationRequestBuilder;
use std::fs::File;
use std::io::Write;
use std::path::Path;

const DEFAULT_PATH: &str = "./output.png";

#[tokio::main]
async fn main() -> Result<(), RigError> {
    let image_generation_model = BedrockRuntime::from_env().image_generation(AMAZON_NOVA_CANVAS);
    let response = image_generation_model
        .call(
            ImageGenerationRequestBuilder::new(
                "A castle sitting upon a large mountain, overlooking the water.",
            )
            .width(512)
            .height(512)
            .build(),
        )
        .await?;

    // save image
    let mut file = File::create_new(Path::new(DEFAULT_PATH)).map_err(RigError::other)?;
    file.write_all(&response.image).map_err(RigError::other)?;

    Ok(())
}
