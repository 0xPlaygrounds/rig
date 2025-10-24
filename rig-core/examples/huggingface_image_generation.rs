use rig::client::ImageGenerationClient;
use rig::image_generation::ImageGenerationModel;
use rig::providers::huggingface;
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

    let huggingface = huggingface::Client::from_env();
    let dalle = huggingface.image_generation_model(huggingface::STABLE_DIFFUSION_3);

    let response = dalle
        .image_generation_request()
        .prompt("A castle sitting upon a large mountain, overlooking the water.")
        .width(1024)
        .height(1024)
        .send()
        .await
        .expect("Failed to generate image");

    let _ = file.write(&response.image);
}
