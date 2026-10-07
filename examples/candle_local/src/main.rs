use std::io::Write;

use anyhow::Context;
use futures::StreamExt;
use rig::candle::extension::CandleExt;
use rig::candle::{CandleModel, ModelData};
use rig::completion::CompletionRequest;
use rig::streaming::{Item, StreamEvent};

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

    let candle = CandleModel::from_gguf(ModelData {
        config: std::fs::read(model_dir.join("config.json"))?,
        tokenizer: std::fs::read(model_dir.join("tokenizer.json"))?,
        weights: std::fs::read(model_dir.join("model.gguf"))?,
    })?;
    let model = candle.completion();
    let request = CompletionRequest::new(prompt)
        .preamble("You are a concise and helpful assistant.")
        .temperature(0.0)
        .max_tokens(64);

    // The local generation metrics printed below (throughput, prefill time,
    // time-to-first-token) are Candle's own; Rig's normalized response carries
    // usage and a finish reason, not these. Candle's typed reply extras read
    // them from the terminal record.
    let mut stream = model.stream(request)?;
    while let Some(item) = stream.next().await {
        if let Item::Event(StreamEvent::Text { text, .. }) = item? {
            print!("{text}");
            std::io::stdout().flush()?;
        }
    }
    let response = stream.finish().await?;
    println!();
    let extras = response
        .extras::<CandleExt>()
        .context("the response did not come from Candle")?
        .context("Candle's terminal record did not deserialize")?;
    let show = |value: Option<u64>| value.map_or_else(|| "n/a".to_string(), |n| n.to_string());
    let prompt_tokens = extras.prompt_tokens.unwrap_or_default();
    let generated_tokens = extras.generated_tokens.unwrap_or_default();
    println!(
        "tokens: prompt={prompt_tokens}, generated={generated_tokens}, total={}",
        prompt_tokens.saturating_add(generated_tokens)
    );
    let throughput = match extras.tokens_per_second {
        Some(value) => format!("{value:.2} tokens/s"),
        None => "n/a".to_string(),
    };
    println!(
        "finish: {:?}; requested max: {}; effective max: {}; prefill: {} ms; time to first token: {} ms; total: {} ms; throughput: {}",
        extras.finish_reason,
        show(extras.requested_max_tokens),
        show(extras.effective_max_tokens),
        show(extras.prefill_duration_ms),
        show(extras.time_to_first_token_ms),
        show(extras.generation_duration_ms),
        throughput
    );
    Ok(())
}
