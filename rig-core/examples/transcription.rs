use rig::prelude::*;
use rig::providers::huggingface;
use rig::{
    providers::{azure, gemini, groq, openai},
    transcription::TranscriptionModel,
};
use std::env::args;

#[tokio::main]
async fn main() {
    // Load the path from the first command line argument
    let args = args().collect::<Vec<_>>();

    if args.len() <= 1 {
        println!("No file was specified!");
        return;
    }

    let file_path = args[1].clone();
    println!("Transcribing {}", &file_path);
    whisper(&file_path).await;
    gemini(&file_path).await;
    azure(&file_path).await;
    groq(&file_path).await;
    huggingface(&file_path).await;
}

async fn whisper(file_path: &str) {
    // Create an OAI client
    let openai = openai::Client::from_env();
    // Create the whisper transcription model
    let whisper = openai.transcription_model(openai::WHISPER_1);
    let response = whisper
        .transcription_request()
        .load_file(file_path)
        .send()
        .await
        .expect("Failed to transcribe file");
    let text = response.text;
    println!("Whisper-1: {text}")
}

async fn gemini(file_path: &str) {
    // Create an OAI client
    let gemini = gemini::Client::from_env();
    // Create the whisper transcription model
    let gemini = gemini.transcription_model(gemini::transcription::GEMINI_2_0_FLASH);
    let response = gemini
        .transcription_request()
        .load_file(file_path)
        .send()
        .await
        .expect("Failed to transcribe file");
    let text = response.text;
    println!("Gemini: {text}")
}

async fn azure(file_path: &str) {
    let azure = azure::Client::from_env();
    let whisper = azure.transcription_model("whisper");
    let response = whisper
        .transcription_request()
        .load_file(file_path)
        .send()
        .await
        .expect("Failed to transcribe file");
    let text = response.text;
    println!("Azure Whisper-1: {text}")
}

async fn groq(file_path: &str) {
    let groq = groq::Client::from_env();
    // Create the whisper transcription model
    let whisper = groq.transcription_model(groq::WHISPER_LARGE_V3);
    let response = whisper
        .transcription_request()
        .load_file(file_path)
        .send()
        .await
        .expect("Failed to transcribe file");
    let text = response.text;
    println!("Groq Whisper-Large-V3: {text}")
}

async fn huggingface(file_path: &str) {
    let huggingface = huggingface::Client::from_env();
    let whisper = huggingface.transcription_model(huggingface::WHISPER_LARGE_V3);
    let response = whisper
        .transcription_request()
        .load_file(file_path)
        .send()
        .await
        .expect("Failed to transcribe file");
    let text = response.text;
    println!("Huggingface Whisper-Large-V3: {text}")
}
