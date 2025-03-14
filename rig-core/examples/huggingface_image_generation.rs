use rig::image_generation::ImageGenerationModel;
use rig::providers::{huggingface, openai};
use std::env::args;
use std::fs::File;
use std::io::Write;
use std::path::Path;

#[tokio::main]
async fn main() {
    let huggingface = huggingface::Client::from_env();

    let dalle = huggingface.image_generation_model(huggingface::STABLE_DIFFUSION_3);

    let response = dalle
        .image_generation_request()
        .prompt("A castle sitting upon a large mountain, overlooking the water.")
        .size((1024, 1024))
        .send()
        .await
        .expect("Failed to generate image");

    let arguments: Vec<String> = args().collect();

    let path = Path::new(&arguments[1]);
    let mut file = File::create_new(path).expect("Failed to create file");
    let _ = file.write(&response.image);
}
