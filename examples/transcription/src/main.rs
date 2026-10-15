use rig::providers::gemini::Gemini;
use rig::providers::openai::OpenAI;
use rig::providers::{gemini, groq, mistral, openai, openrouter};
use rig::transcription::TranscriptionRequestBuilder;
use serde_json::json;
use std::env::args;

/// Transcribes the file with every provider, or with the one named after it:
/// `cargo run -p transcription -- audio.mp3 [openai|openrouter|gemini|azure|groq|huggingface|mistral]`.
#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let mut args = args().skip(1);
    let Some(file_path) = args.next() else {
        println!("No file was specified!");
        return Ok(());
    };
    let only = args.next();
    let wanted = |name: &str| only.as_deref().is_none_or(|only| only == name);

    println!("Transcribing {file_path}");
    if wanted("openai") {
        whisper(&file_path).await?;
    }
    if wanted("openrouter") {
        openrouter_speakers(&file_path).await?;
    }
    if wanted("gemini") {
        gemini(&file_path).await?;
    }
    if wanted("azure") {
        azure(&file_path).await?;
    }
    if wanted("groq") {
        groq(&file_path).await?;
    }
    if wanted("huggingface") {
        huggingface(&file_path).await?;
    }
    if wanted("mistral") {
        mistral(&file_path).await?;
    }

    Ok(())
}

async fn whisper(file_path: &str) -> Result<(), anyhow::Error> {
    let openai = OpenAI::from_env()?;
    let whisper = openai.transcription(openai::WHISPER_1);
    let response = whisper
        .call(TranscriptionRequestBuilder::from_file(file_path)?.build())
        .await?;
    println!("Whisper-1: {}", response.text);
    Ok(())
}

/// Timed, speaker-labeled segments from a diarizing model behind OpenRouter.
async fn openrouter_speakers(file_path: &str) -> Result<(), anyhow::Error> {
    let model = openrouter::from_env()?.transcription("microsoft/mai-transcribe-2");
    let request = TranscriptionRequestBuilder::from_file(file_path)?
        .additional_params(json!({
            "response_format": "verbose_json",
            "timestamp_granularities": ["segment", "word"],
            "provider": {"options": {"azure": {"diarization": {"enabled": true}}}},
        }))
        .build();
    let response = model.call(request).await?;
    println!("OpenRouter MAI-Transcribe-2: {}", response.text);
    for segment in &response.segments {
        println!(
            "  [{:>6.2}s - {:>6.2}s] speaker {}: {}",
            segment.start,
            segment.end,
            segment.speaker.as_deref().unwrap_or("?"),
            segment.text.trim()
        );
    }
    println!("  {} timed words", response.words.len());
    Ok(())
}

async fn gemini(file_path: &str) -> Result<(), anyhow::Error> {
    let gemini = Gemini::from_env()?;
    let model = gemini.transcription(gemini::completion::GEMINI_3_FLASH_PREVIEW);
    let response = model
        .call(TranscriptionRequestBuilder::from_file(file_path)?.build())
        .await?;
    println!("Gemini: {}", response.text);
    Ok(())
}

async fn azure(file_path: &str) -> Result<(), anyhow::Error> {
    let azure = rig::providers::azure::from_env()?;
    let whisper = azure.transcription("whisper");
    let response = whisper
        .call(TranscriptionRequestBuilder::from_file(file_path)?.build())
        .await?;
    println!("Azure Whisper-1: {}", response.text);
    Ok(())
}

async fn groq(file_path: &str) -> Result<(), anyhow::Error> {
    let groq = groq::from_env()?;
    let whisper = groq.transcription(groq::WHISPER_LARGE_V3);
    let response = whisper
        .call(TranscriptionRequestBuilder::from_file(file_path)?.build())
        .await?;
    println!("Groq Whisper-Large-V3: {}", response.text);
    Ok(())
}

async fn huggingface(file_path: &str) -> Result<(), anyhow::Error> {
    let huggingface = rig::providers::huggingface::from_env()?;
    let whisper = huggingface.transcription("whisper-large-v3");
    let response = whisper
        .call(TranscriptionRequestBuilder::from_file(file_path)?.build())
        .await?;
    println!("HuggingFace Whisper-Large-V3: {}", response.text);
    Ok(())
}

async fn mistral(file_path: &str) -> Result<(), anyhow::Error> {
    let client = mistral::from_env()?;
    let model = client.transcription(mistral::VOXTRAL_MINI);
    let response = model
        .call(TranscriptionRequestBuilder::from_file(file_path)?.build())
        .await?;
    println!("Mistral: {}", response.text);
    Ok(())
}
