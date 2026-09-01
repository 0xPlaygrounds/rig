use rig::prelude::*;
use rig::providers::{huggingface, mistral};
use rig::{
    providers::{azure, gemini, groq, openai},
    transcription::TranscriptionModel,
};
use std::env::args;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let args = args().collect::<Vec<_>>();

    if args.len() <= 1 {
        println!("No file was specified!");
        return Ok(());
    }

    let file_path = args
        .get(1)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("No file was specified"))?;
    println!("Transcribing {}", &file_path);
    whisper(&file_path).await?;
    gemini(&file_path).await?;
    azure(&file_path).await?;
    groq(&file_path).await?;
    huggingface(&file_path).await?;
    mistral(&file_path).await?;

    Ok(())
}

async fn whisper(file_path: &str) -> Result<(), anyhow::Error> {
    let openai = openai::Client::from_env()?;
    let whisper = openai.transcription_model(openai::WHISPER_1);
    let response = whisper
        .transcription_request()
        .load_file(file_path)?
        .send()
        .await?;
    println!("Whisper-1: {}", response.text);
    Ok(())
}

async fn gemini(file_path: &str) -> Result<(), anyhow::Error> {
    let gemini = gemini::Client::from_env()?;
    let model = gemini.transcription_model(gemini::completion::GEMINI_3_FLASH_PREVIEW);
    let response = model
        .transcription_request()
        .load_file(file_path)?
        .send()
        .await?;
    println!("Gemini: {}", response.text);
    Ok(())
}

async fn azure(file_path: &str) -> Result<(), anyhow::Error> {
    let azure = azure::Client::from_env()?;
    let whisper = azure.transcription_model("whisper");
    let response = whisper
        .transcription_request()
        .load_file(file_path)?
        .send()
        .await?;
    println!("Azure Whisper-1: {}", response.text);
    Ok(())
}

async fn groq(file_path: &str) -> Result<(), anyhow::Error> {
    let groq = groq::Client::from_env()?;
    let whisper = groq.transcription_model(groq::WHISPER_LARGE_V3);
    let response = whisper
        .transcription_request()
        .load_file(file_path)?
        .send()
        .await?;
    println!("Groq Whisper-Large-V3: {}", response.text);
    Ok(())
}

async fn huggingface(file_path: &str) -> Result<(), anyhow::Error> {
    let huggingface = huggingface::Client::from_env()?;
    let whisper = huggingface.transcription_model("whisper-large-v3");
    let response = whisper
        .transcription_request()
        .load_file(file_path)?
        .send()
        .await?;
    println!("HuggingFace Whisper-Large-V3: {}", response.text);
    Ok(())
}

async fn mistral(file_path: &str) -> Result<(), anyhow::Error> {
    let client = mistral::Client::from_env()?;
    let model = client.transcription_model(mistral::VOXTRAL_MINI);
    let response = model
        .transcription_request()
        .load_file(file_path)?
        .send()
        .await?;
    println!("Mistral: {}", response.text);
    Ok(())
}
