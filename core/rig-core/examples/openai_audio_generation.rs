use rig::audio_generation::AudioGenerationModel;
use rig::client::audio_generation::AudioGenerationClient;
use rig::prelude::*;
use rig::providers::openai;
use std::env::args;
use std::fs::File;
use std::io::Write;
use std::path::Path;

const DEFAULT_PATH: &str = "./output.mp3";

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

    let openai = openai::Client::from_env();
    let tts = openai.audio_generation_model(openai::TTS_1);

    let response = tts
        .audio_generation_request()
        .text("The quick brown fox jumps over the lazy dog")
        .voice("alloy")
        .send()
        .await
        .expect("Failed to generate image");

    let _ = file.write(&response.audio);
}
