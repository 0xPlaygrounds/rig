#![cfg(not(target_family = "wasm"))]

use std::path::PathBuf;

use rig_candle::{LlamaModel, ModelData};
use rig_core::completion::CompletionModel;

#[tokio::test(flavor = "current_thread")]
#[ignore = "requires RIG_CANDLE_MODEL_DIR with local Llama 3 artifacts"]
async fn loads_and_generates_with_a_real_local_model()
-> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let directory = PathBuf::from(std::env::var_os("RIG_CANDLE_MODEL_DIR").ok_or(
        "RIG_CANDLE_MODEL_DIR must contain config.json, tokenizer.json, and model.safetensors",
    )?);
    let model = LlamaModel::builder(ModelData {
        config: std::fs::read(directory.join("config.json"))?,
        tokenizer: std::fs::read(directory.join("tokenizer.json"))?,
        weights: std::fs::read(directory.join("model.safetensors"))?,
    })
    .temperature(0.0)
    .max_tokens(16)
    .build()?;

    let response = model
        .completion_request("Reply with one short greeting.")
        .send()
        .await?;
    if response.raw_response.text.is_empty() {
        return Err(std::io::Error::other("real model returned empty generated text").into());
    }
    if response.usage.input_tokens == 0 || response.usage.output_tokens == 0 {
        return Err(std::io::Error::other("real model returned zero token usage").into());
    }
    Ok(())
}
