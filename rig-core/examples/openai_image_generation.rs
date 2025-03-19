use rig::image_generation::ImageGenerationModel;
use rig::providers::openai;
use std::env::args;
use std::fs::File;
use std::io::Write;
use std::path::Path;

#[tokio::main]
async fn main() {
    let arguments: Vec<String> = args().collect();

    let path = arguments.get(1).unwrap_or(&"./output.png".to_string());
    let path = Path::new(path);
    let mut file = File::create_new(path).expect("Failed to create file");

    let openai = openai::Client::from_env();

    let dalle = openai.image_generation_model(openai::DALL_E_2);

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
