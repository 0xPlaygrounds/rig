//! xAI-specific image generation example showing provider-specific
//! `additional_params`. The generic image-generation example lives in
//! `openai_image_generation.rs`.

use rig::image_generation::ImageGenerationModel;
use rig::prelude::*;
use rig::providers::xai;
use serde_json::json;
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

    let client = xai::Client::from_env();
    let model = client.image_generation_model(xai::image_generation::GROK_IMAGINE_IMAGE_PRO);

    let response = model
        .image_generation_request()
        .prompt("A lone explorer discovering an underwater lost city lit by bioluminescent coral.")
        .additional_params(json!({
            "resolution": "2k",
            "aspect_ratio": "4:3",
        }))
        .send()
        .await
        .expect("Failed to generate image");

    let _ = file.write(&response.image);
}
