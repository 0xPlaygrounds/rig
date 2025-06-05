use rig::image_generation::ImageGenerationModel;
use rig::prelude::*;
use rig::providers::hyperbolic;
use std::env::args;
use std::fs::File;
use std::io::Write;
use std::path::Path;

const DEFAULT_PATH: &str = "./output.png";

#[tokio::main]
async fn main() {
    let arguments: Vec<String> = args().collect();

    let path = if arguments.len() > 1 {
        arguments[1].clone()
    } else {
        DEFAULT_PATH.to_string()
    };

    let path = Path::new(&path);
    let mut file = File::create_new(path).expect("Failed to create file");

    let hyperbolic = hyperbolic::Client::from_env();
    let stable_diffusion = hyperbolic.image_generation_model(hyperbolic::SDXL_TURBO);

    let response = stable_diffusion
        .image_generation_request()
        .prompt("A castle sitting upon a large mountain, overlooking the water.")
        .width(1024)
        .height(1024)
        .send()
        .await
        .expect("Failed to generate image");

    let _ = file.write(&response.image);
}
