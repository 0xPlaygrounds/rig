#![cfg(not(target_family = "wasm"))]

use std::path::PathBuf;

use futures::StreamExt;
use rig_candle::{CandleModel, ModelData};
use rig_core::completion::CompletionModel;
use rig_core::streaming::StreamedAssistantContent;

/// Concatenated visible text from a buffered completion's choice.
fn choice_text(response: &rig_core::completion::CompletionResponse) -> String {
    response
        .choice
        .iter()
        .filter_map(|content| match content {
            rig_core::completion::AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect()
}

#[tokio::test(flavor = "current_thread")]
#[ignore = "requires RIG_CANDLE_MODEL_DIR with local Llama 3 safetensors or SmolLM2 GGUF artifacts"]
async fn loads_and_generates_with_a_real_local_model()
-> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let directory = PathBuf::from(std::env::var_os("RIG_CANDLE_MODEL_DIR").ok_or(
        "RIG_CANDLE_MODEL_DIR must contain config.json, tokenizer.json, and model.safetensors or model.gguf",
    )?);
    let data = ModelData {
        config: std::fs::read(directory.join("config.json"))?,
        tokenizer: std::fs::read(directory.join("tokenizer.json"))?,
        weights: if directory.join("model.gguf").is_file() {
            std::fs::read(directory.join("model.gguf"))?
        } else {
            std::fs::read(directory.join("model.safetensors"))?
        },
    };
    let builder = if directory.join("model.gguf").is_file() {
        CandleModel::builder_from_artifacts(rig_candle::ModelArtifacts::Gguf(data))
    } else {
        CandleModel::builder(data)
    };
    let model = builder.temperature(0.0).max_tokens(16).build()?;

    let is_gguf = directory.join("model.gguf").is_file();
    let prompt = if is_gguf {
        "What is the capital of France?"
    } else {
        "Reply with one short greeting."
    };
    let request = rig_core::completion::CompletionRequest::from_prompt(prompt);
    let response = model.completion(request.clone()).await?;
    let buffered_text = choice_text(&response);
    if buffered_text.is_empty() {
        return Err(std::io::Error::other("real model returned empty generated text").into());
    }
    if response.usage.input_tokens == 0 || response.usage.output_tokens == 0 {
        return Err(std::io::Error::other("real model returned zero token usage").into());
    }
    if is_gguf && !buffered_text.contains("Paris") {
        return Err(std::io::Error::other(format!(
            "SmolLM2 coherence regression: {buffered_text:?}"
        ))
        .into());
    }
    let mut stream = model.stream(request).await?;
    let mut streamed_text = String::new();
    let mut final_response = None;
    while let Some(item) = stream.next().await {
        match item? {
            StreamedAssistantContent::Text(fragment) => streamed_text.push_str(&fragment.text),
            StreamedAssistantContent::Final(raw) => final_response = Some(raw),
            _ => {}
        }
    }
    let final_response = final_response
        .ok_or_else(|| std::io::Error::other("real model stream omitted final metadata"))?;
    if streamed_text != buffered_text {
        return Err(std::io::Error::other("buffered and streamed output differed").into());
    }
    if final_response.usage.output_tokens != response.usage.output_tokens {
        return Err(std::io::Error::other("buffered and streamed usage differed").into());
    }
    Ok(())
}
