//! Demonstrates an autonomous extractor loop that keeps feeding its own output back in.
//! Requires `OPENAI_API_KEY`.
//! Run it to watch the extractor keep counting upward until the stop condition is met.

use anyhow::Result;
use rig::extract::ExtractOptions;
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

fn counter_preamble() -> String {
    const ROLE: &str = "
            Add a random whole number between 1 and 64 to the number you receive.
            Return only the updated number.
        ";
    let classic = ExtractOptions::classic_extractor();
    let preamble = classic.preamble.clone().unwrap_or_default();
    format!("{preamble}\n=============== ADDITIONAL INSTRUCTIONS ===============\n{ROLE}")
}

#[tokio::main]
async fn main() -> Result<()> {
    let client = openai::Client::from_env()?;
    let agent = client.agent(openai::GPT_4).build();
    let preamble = counter_preamble();
    let mut current_number = 0;
    let mut step = 1;
    let mut interval = tokio::time::interval(STEP_DELAY);

    loop {
        let next_number: Counter = agent
            .extractor(current_number.to_string())
            .classic()
            .preamble(preamble.clone())
            .run()
            .await?;
        let next_number = next_number.number;
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
