use std::env::args;

use rig::{providers::openai, transcription::TranscriptionModel};

#[tokio::main]
async fn main() {
    // Load the path from the first command line argument
    let args= args().collect::<Vec<_>>();

    if args.len() <= 1 {
        println!("No file was specified!");
        return;
    }

    let file_path = args[1].clone();

    // Create an OAI client
    let openai = openai::Client::from_env();

    // Create the whisper transcription model
    let whisper = openai.transcription_model(openai::WHISPER_1);

    let response = whisper
        .transcription_request()
        .load_file(&file_path)
        .send()
        .await
        .expect("Failed to transcribe file");

    let text = response.text;

    println!("Whisper-1: {text}")
}
