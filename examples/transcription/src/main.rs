use rig::RigError;
use rig::error::ErrorKind;
use rig::providers::gemini::Gemini;
use rig::providers::openai::OpenAI;
use rig::providers::{gemini, groq, mistral, openai};
use rig::transcription::TranscriptionRequestBuilder;
use std::env::args;

#[tokio::main]
async fn main() -> Result<(), RigError> {
    let args = args().collect::<Vec<_>>();

    if args.len() <= 1 {
        println!("No file was specified!");
        return Ok(());
    }

    let file_path = args
        .get(1)
        .cloned()
        .ok_or_else(|| RigError::new(ErrorKind::Other, "No file was specified"))?;
    println!("Transcribing {}", &file_path);
    whisper(&file_path).await?;
    gemini(&file_path).await?;
    azure(&file_path).await?;
    groq(&file_path).await?;
    huggingface(&file_path).await?;
    mistral(&file_path).await?;

    Ok(())
}

async fn whisper(file_path: &str) -> Result<(), RigError> {
    let openai = OpenAI::from_env()?;
    let whisper = openai.transcription(openai::WHISPER_1);
    let response = whisper
        .call(
            TranscriptionRequestBuilder::from_file(file_path)
                .map_err(RigError::other)?
                .build(),
        )
        .await?;
    println!("Whisper-1: {}", response.text);
    Ok(())
}

async fn gemini(file_path: &str) -> Result<(), RigError> {
    let gemini = Gemini::from_env()?;
    let model = gemini.transcription(gemini::completion::GEMINI_3_FLASH_PREVIEW);
    let response = model
        .call(
            TranscriptionRequestBuilder::from_file(file_path)
                .map_err(RigError::other)?
                .build(),
        )
        .await?;
    println!("Gemini: {}", response.text);
    Ok(())
}

async fn azure(file_path: &str) -> Result<(), RigError> {
    let azure = rig::providers::azure::from_env()?;
    let whisper = azure.transcription("whisper");
    let response = whisper
        .call(
            TranscriptionRequestBuilder::from_file(file_path)
                .map_err(RigError::other)?
                .build(),
        )
        .await?;
    println!("Azure Whisper-1: {}", response.text);
    Ok(())
}

async fn groq(file_path: &str) -> Result<(), RigError> {
    let groq = groq::from_env()?;
    let whisper = groq.transcription(groq::WHISPER_LARGE_V3);
    let response = whisper
        .call(
            TranscriptionRequestBuilder::from_file(file_path)
                .map_err(RigError::other)?
                .build(),
        )
        .await?;
    println!("Groq Whisper-Large-V3: {}", response.text);
    Ok(())
}

async fn huggingface(file_path: &str) -> Result<(), RigError> {
    let huggingface = rig::providers::huggingface::from_env()?;
    let whisper = huggingface.transcription("whisper-large-v3");
    let response = whisper
        .call(
            TranscriptionRequestBuilder::from_file(file_path)
                .map_err(RigError::other)?
                .build(),
        )
        .await?;
    println!("HuggingFace Whisper-Large-V3: {}", response.text);
    Ok(())
}

async fn mistral(file_path: &str) -> Result<(), RigError> {
    let client = mistral::from_env()?;
    let model = client.transcription(mistral::VOXTRAL_MINI);
    let response = model
        .call(
            TranscriptionRequestBuilder::from_file(file_path)
                .map_err(RigError::other)?
                .build(),
        )
        .await?;
    println!("Mistral: {}", response.text);
    Ok(())
}
