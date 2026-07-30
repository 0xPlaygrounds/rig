//! Demonstrates transcribing one audio file with every provider that speaks
//! the transcription API.
//!
//! `TranscriptionModel` (and the client that handed it out) is gone. A
//! transcription is now the same shape as any other provider call: a plain
//! `<provider>::functions::Config` naming the model, a shared
//! [`HttpRuntime`], and the provider's free `transcribe` function.
//!
//! Requires the matching provider keys (`OPENAI_API_KEY`, `GEMINI_API_KEY`,
//! `AZURE_ENDPOINT`/`AZURE_API_VERSION`/an Azure credential, `GROQ_API_KEY`,
//! `HUGGINGFACE_API_KEY`, `MISTRAL_API_KEY`). Pass the audio file as the first
//! argument.

use rig::prelude::*;
use rig::providers::{azure, gemini, groq, huggingface, mistral, openai};
use rig::transcription::TranscriptionRequest;
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

    // One transport, shared by every provider call below: configs are data,
    // the live HTTP handle lives here.
    let rt = HttpRuntime::new();

    whisper(&rt, &file_path).await?;
    gemini(&rt, &file_path).await?;
    azure(&rt, &file_path).await?;
    groq(&rt, &file_path).await?;
    huggingface(&rt, &file_path).await?;
    mistral(&rt, &file_path).await?;

    Ok(())
}

async fn whisper(rt: &HttpRuntime, file_path: &str) -> Result<(), anyhow::Error> {
    let cfg = openai::functions::Config::from_env(openai::WHISPER_1)?;
    let response =
        openai::functions::transcribe(&cfg, rt, TranscriptionRequest::from_file(file_path)?)
            .await?;
    println!("Whisper-1: {}", response.text);
    Ok(())
}

async fn gemini(rt: &HttpRuntime, file_path: &str) -> Result<(), anyhow::Error> {
    let cfg = gemini::functions::Config::from_env(gemini::completion::GEMINI_3_FLASH_PREVIEW)?;
    let response =
        gemini::functions::transcribe(&cfg, rt, TranscriptionRequest::from_file(file_path)?)
            .await?;
    println!("Gemini: {}", response.text);
    Ok(())
}

async fn azure(rt: &HttpRuntime, file_path: &str) -> Result<(), anyhow::Error> {
    let cfg = azure::functions::Config::from_env("whisper")?;
    let response =
        azure::functions::transcribe(&cfg, rt, TranscriptionRequest::from_file(file_path)?).await?;
    println!("Azure Whisper-1: {}", response.text);
    Ok(())
}

async fn groq(rt: &HttpRuntime, file_path: &str) -> Result<(), anyhow::Error> {
    let cfg = groq::functions::Config::from_env(groq::WHISPER_LARGE_V3)?;
    let response =
        groq::functions::transcribe(&cfg, rt, TranscriptionRequest::from_file(file_path)?).await?;
    println!("Groq Whisper-Large-V3: {}", response.text);
    Ok(())
}

async fn huggingface(rt: &HttpRuntime, file_path: &str) -> Result<(), anyhow::Error> {
    let cfg = huggingface::functions::Config::from_env("whisper-large-v3")?;
    let response =
        huggingface::functions::transcribe(&cfg, rt, TranscriptionRequest::from_file(file_path)?)
            .await?;
    println!("HuggingFace Whisper-Large-V3: {}", response.text);
    Ok(())
}

async fn mistral(rt: &HttpRuntime, file_path: &str) -> Result<(), anyhow::Error> {
    let cfg = mistral::functions::Config::from_env(mistral::VOXTRAL_MINI)?;
    let response =
        mistral::functions::transcribe(&cfg, rt, TranscriptionRequest::from_file(file_path)?)
            .await?;
    println!("Mistral: {}", response.text);
    Ok(())
}
