//! xAI-specific audio generation example showing provider-specific
//! `additional_params`. The generic audio-generation example lives in
//! `openai_audio_generation.rs`.

use rig::audio_generation::AudioGenerationModel;
use rig::client::audio_generation::AudioGenerationClient;
use rig::prelude::*;
use rig::providers::xai;
use serde_json::json;
use std::env::args;
use std::fs::File;
use std::io::Write;
use std::path::Path;

const DEFAULT_PATH: &str = "./output.wav";

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
    let tts = client.audio_generation_model(xai::TTS_1);

    let response = tts
        .audio_generation_request()
        .text("The quick brown fox jumps over the lazy dog")
        .voice("eve")
        .additional_params(json!({
            "language": "en",
        }))
        .send()
        .await
        .expect("Failed to generate audio");

    let _ = file.write(&response.audio);
}
