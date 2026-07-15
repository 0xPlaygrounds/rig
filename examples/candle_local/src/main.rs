use rig::candle::{LlamaModel, ModelData};
use rig::{agent::AgentBuilder, completion::Completion, message::Message};

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

    let model = LlamaModel::from_safetensors(ModelData {
        config: std::fs::read(model_dir.join("config.json"))?,
        tokenizer: std::fs::read(model_dir.join("tokenizer.json"))?,
        weights: std::fs::read(model_dir.join("model.safetensors"))?,
    })?;
    let agent = AgentBuilder::new(model)
        .preamble("You are a concise and helpful assistant.")
        .temperature(0.0)
        .max_tokens(64)
        .build();

    let response = agent
        .completion(prompt, std::iter::empty::<Message>())
        .await?
        .send()
        .await?;
    let raw = response.raw_response;
    println!("{}", raw.text);
    println!(
        "tokens: prompt={}, generated={}, total={}",
        response.usage.input_tokens, response.usage.output_tokens, response.usage.total_tokens
    );
    let throughput = match raw.tokens_per_second {
        Some(value) => format!("{value:.2} tokens/s"),
        None => "n/a".to_string(),
    };
    println!(
        "finish: {:?}; requested max: {}; effective max: {}; duration: {} ms; throughput: {}",
        raw.finish_reason,
        raw.requested_max_tokens,
        raw.effective_max_tokens,
        raw.generation_duration_ms,
        throughput
    );
    Ok(())
}
