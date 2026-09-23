use rig::prelude::*;
use rig::providers::gemini::Gemini;
use rig::providers::openai::wire::{AZURE, GROQ, HUGGINGFACE, MISTRAL, OpenAI};
use rig::providers::{gemini, groq, mistral, openai};
use rig::transcription::TranscriptionRequestBuilder;
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
    let openai = OpenAI::from_env()?.bound()?;
    let whisper = openai.transcription(openai::WHISPER_1);
    let response = TranscriptionRequestBuilder::from_file(whisper, file_path)?
        .send()
        .await?;
    println!("Whisper-1: {}", response.text);
    Ok(())
}

async fn gemini(file_path: &str) -> Result<(), anyhow::Error> {
    let gemini = Gemini::from_env()?.bound()?;
    let model = gemini.transcription(gemini::completion::GEMINI_3_FLASH_PREVIEW);
    let response = TranscriptionRequestBuilder::from_file(model, file_path)?
        .send()
        .await?;
    println!("Gemini: {}", response.text);
    Ok(())
}

async fn azure(file_path: &str) -> Result<(), anyhow::Error> {
    let azure = OpenAI::from_env_with(&AZURE)?.bound()?;
    let whisper = azure.transcription("whisper");
    let response = TranscriptionRequestBuilder::from_file(whisper, file_path)?
        .send()
        .await?;
    println!("Azure Whisper-1: {}", response.text);
    Ok(())
}

async fn groq(file_path: &str) -> Result<(), anyhow::Error> {
    let groq = OpenAI::from_env_with(&GROQ)?.bound()?;
    let whisper = groq.transcription(groq::WHISPER_LARGE_V3);
    let response = TranscriptionRequestBuilder::from_file(whisper, file_path)?
        .send()
        .await?;
    println!("Groq Whisper-Large-V3: {}", response.text);
    Ok(())
}

async fn huggingface(file_path: &str) -> Result<(), anyhow::Error> {
    let huggingface = OpenAI::from_env_with(&HUGGINGFACE)?.bound()?;
    let whisper = huggingface.transcription("whisper-large-v3");
    let response = TranscriptionRequestBuilder::from_file(whisper, file_path)?
        .send()
        .await?;
    println!("HuggingFace Whisper-Large-V3: {}", response.text);
    Ok(())
}

async fn mistral(file_path: &str) -> Result<(), anyhow::Error> {
    let client = OpenAI::from_env_with(&MISTRAL)?.bound()?;
    let model = client.transcription(mistral::VOXTRAL_MINI);
    let response = TranscriptionRequestBuilder::from_file(model, file_path)?
        .send()
        .await?;
    println!("Mistral: {}", response.text);
    Ok(())
}
