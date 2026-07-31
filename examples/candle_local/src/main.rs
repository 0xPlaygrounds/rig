//! Streams a locally loaded GGUF model with `rig-candle`.
//!
//! Candle has no `ProviderConfig` arm (model tensors are not plain
//! configuration), so this drives the loaded-model handle directly: its
//! inherent `stream` method replaces the removed `CompletionModel` trait.
//! The equivalent path-driven entry point is `rig::candle::functions::open_stream`.

use std::io::Write;

use anyhow::Context;
use futures::StreamExt;
use rig::candle::{CandleModel, ModelData};
use rig::streaming::StreamedAssistantContent;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let project_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let model_dir = match std::env::var_os("MODEL_DIR") {
        Some(directory) => std::path::PathBuf::from(directory),
        None => project_dir.join("model"),
    };
    let prompt = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    let prompt = if prompt.is_empty() {
        "Say hello in one short sentence.".to_string()
    } else {
        prompt
    };

    let model = CandleModel::from_gguf(ModelData {
        config: std::fs::read(model_dir.join("config.json"))?,
        tokenizer: std::fs::read(model_dir.join("tokenizer.json"))?,
        weights: std::fs::read(model_dir.join("model.gguf"))?,
    })?;
    let request = rig::completion::CompletionRequest::builder(prompt)
        .preamble("You are a concise and helpful assistant.")
        .temperature(0.0)
        .max_tokens(64)
        .build();
    let mut response = model.stream(request).await?;
    let mut final_response = None;
    while let Some(item) = response.next().await {
        match item? {
            StreamedAssistantContent::Text(fragment) => {
                print!("{}", fragment.text);
                std::io::stdout().flush()?;
            }
            StreamedAssistantContent::Final(final_metadata) => {
                final_response = Some(final_metadata)
            }
            _ => {}
        }
    }
    println!();
    let final_response = final_response.context("Candle stream ended without final metadata")?;
    let usage = final_response.usage;
    println!(
        "tokens: prompt={}, generated={}, total={}",
        usage.input_tokens, usage.output_tokens, usage.total_tokens
    );
    println!(
        "finish: {:?}; provider: {}; model: {}",
        final_response.finish_reason,
        final_response.provider,
        final_response.model.as_deref().unwrap_or("n/a"),
    );
    Ok(())
}
