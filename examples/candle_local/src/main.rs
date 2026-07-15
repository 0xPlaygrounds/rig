use std::io::Write;

use anyhow::Context;
use futures::StreamExt;
use rig::candle::{LlamaModel, ModelData};
use rig::streaming::{StreamedAssistantContent, StreamingCompletion};
use rig::{agent::AgentBuilder, message::Message};

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

    let model = LlamaModel::from_gguf(ModelData {
        config: std::fs::read(model_dir.join("config.json"))?,
        tokenizer: std::fs::read(model_dir.join("tokenizer.json"))?,
        weights: std::fs::read(model_dir.join("model.gguf"))?,
    })?;
    let agent = AgentBuilder::new(model)
        .preamble("You are a concise and helpful assistant.")
        .temperature(0.0)
        .max_tokens(64)
        .build();

    let mut response = agent
        .stream_completion(prompt, std::iter::empty::<Message>())
        .await?
        .stream()
        .await?;
    let mut final_response = None;
    while let Some(item) = response.next().await {
        match item? {
            StreamedAssistantContent::Text(fragment) => {
                print!("{}", fragment.text);
                std::io::stdout().flush()?;
            }
            StreamedAssistantContent::Final(raw) => final_response = Some(raw),
            _ => {}
        }
    }
    println!();
    let raw = final_response.context("Candle stream ended without final metadata")?;
    let usage = response.usage();
    println!(
        "tokens: prompt={}, generated={}, total={}",
        usage.input_tokens, usage.output_tokens, usage.total_tokens
    );
    let throughput = match raw.tokens_per_second {
        Some(value) => format!("{value:.2} tokens/s"),
        None => "n/a".to_string(),
    };
    println!(
        "finish: {:?}; requested max: {}; effective max: {}; prefill: {} ms; time to first token: {} ms; total: {} ms; throughput: {}",
        raw.finish_reason,
        raw.requested_max_tokens,
        raw.effective_max_tokens,
        raw.prefill_duration_ms,
        raw.time_to_first_token_ms
            .map_or_else(|| "n/a".to_string(), |value| value.to_string()),
        raw.generation_duration_ms,
        throughput
    );
    Ok(())
}
