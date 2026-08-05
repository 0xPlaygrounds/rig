use std::io::Write;

use anyhow::Context;
use futures::StreamExt;
use rig::candle::{CandleModel, ModelData};
use rig::completion::CompletionModel;
use rig::streaming::RawStreamingChoice;

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
    let request = model
        .completion_request(prompt)
        .preamble("You are a concise and helpful assistant.".to_string())
        .temperature(0.0)
        .max_tokens(64)
        .build();

    // The local generation metrics printed below (throughput, prefill time,
    // time-to-first-token) are Candle's own — Rig's normalized `StreamFinal`
    // carries usage and a finish reason, not these. `raw_stream` keeps the
    // provider's terminal record typed so they stay reachable; use
    // `CompletionModel::stream` when the normalized metadata is enough.
    let mut stream = model.raw_stream(request).await?;
    let mut final_response = None;
    while let Some(item) = stream.next().await {
        match item? {
            RawStreamingChoice::Message(fragment) => {
                print!("{fragment}");
                std::io::stdout().flush()?;
            }
            RawStreamingChoice::FinalResponse(final_record) => final_response = Some(final_record),
            _ => {}
        }
    }
    println!();
    let raw = final_response.context("Candle stream ended without final metadata")?;
    println!(
        "tokens: prompt={}, generated={}, total={}",
        raw.prompt_tokens,
        raw.generated_tokens,
        raw.prompt_tokens.saturating_add(raw.generated_tokens)
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
