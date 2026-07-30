//! Demonstrates an autonomous extractor loop that keeps feeding its own output back in.
//! Requires `OPENAI_API_KEY`.
//! Run it to watch the extractor keep counting upward until the stop condition is met.

//! `Extractor<Counter>` no longer exists as a type you can return, so the
//! "extractor" here is the plain data one [`extract_with_options`] call needs:
//! a [`ProviderConfig`], a shared [`Runtime`], and the [`ExtractOptions`]
//! carrying the counting instructions.
use std::sync::Arc;

use anyhow::Result;
use rig::agent::AgentConfig;
use rig::extract::{ExtractOptions, extract_with_options};
use rig::prelude::*;
use rig::providers::openai;

use schemars::JsonSchema;

#[derive(Debug, serde::Deserialize, JsonSchema, serde::Serialize)]
struct Counter {
    /// The latest counter value produced by the extractor.
    number: u32,
}

const TARGET_NUMBER: u32 = 2000;
const STEP_DELAY: std::time::Duration = std::time::Duration::from_secs(1);

/// The counting extractor as data: where to call, and how to ask.
fn build_counter_extractor() -> Result<(ProviderConfig, ExtractOptions)> {
    const ROLE: &str = "
            Add a random whole number between 1 and 64 to the number you receive.
            Return only the updated number.
        ";
    let classic = ExtractOptions::classic_extractor();
    let preamble = classic.preamble.clone().unwrap_or_default();
    let options = classic.with_preamble(format!(
        "{preamble}\n=============== ADDITIONAL INSTRUCTIONS ===============\n{ROLE}"
    ));
    let cfg = openai::functions::Config::from_env(openai::GPT_4)?;
    Ok((ProviderConfig::OpenAi(cfg), options))
}

#[tokio::main]
async fn main() -> Result<()> {
    let (provider, options) = build_counter_extractor()?;
    let rt = Arc::new(Runtime::new());
    let mut current_number = 0;
    let mut step = 1;
    let mut interval = tokio::time::interval(STEP_DELAY);

    loop {
        let next_number = extract_with_options::<Counter>(
            AgentConfig::new(),
            provider.clone(),
            rt.clone(),
            current_number.to_string(),
            options.clone(),
        )
        .await?
        .value
        .number;
        println!("Step {step}: {current_number} -> {next_number}");

        current_number = next_number;
        if current_number >= TARGET_NUMBER {
            break;
        }

        step += 1;
        interval.tick().await;
    }

    println!("Finished with number: {current_number}");

    Ok(())
}
