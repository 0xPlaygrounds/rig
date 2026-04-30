//! Demonstrates an autonomous extractor loop that keeps feeding its own output back in.
//! Requires `OPENAI_API_KEY`.
//! Run it to watch the extractor keep counting upward until the stop condition is met.

use anyhow::Result;
use rig::prelude::*;
use rig::providers::openai;
use rig::providers::openai::client::Client;

use schemars::JsonSchema;

#[derive(Debug, serde::Deserialize, JsonSchema, serde::Serialize)]
struct Counter {
    /// The latest counter value produced by the extractor.
    number: u32,
}

const TARGET_NUMBER: u32 = 2000;
const STEP_DELAY: std::time::Duration = std::time::Duration::from_secs(1);

fn build_counter_extractor(
    client: &Client,
) -> rig::extractor::Extractor<openai::responses_api::ResponsesCompletionModel, Counter> {
    client
        .extractor::<Counter>(openai::GPT_4)
        .preamble(
            "
            Add a random whole number between 1 and 64 to the number you receive.
            Return only the updated number.
        ",
        )
        .build()
}

#[tokio::main]
async fn main() -> Result<()> {
    let client = Client::from_env()?;
    let extractor = build_counter_extractor(&client);
    let mut current_number = 0;
    let mut step = 1;
    let mut interval = tokio::time::interval(STEP_DELAY);

    loop {
        let next_number = extractor.extract(&current_number.to_string()).await?.number;
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
