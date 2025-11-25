use rig::prelude::*;
use rig::providers::openai;
use rig::providers::openai::client::Client;

use schemars::JsonSchema;

#[derive(Debug, serde::Deserialize, JsonSchema, serde::Serialize)]
struct Counter {
    /// The score of the document
    number: u32,
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create OpenAI client
    let openai_client = Client::from_env();
    let agent = openai_client.extractor::<Counter>(openai::GPT_4)
        .preamble("
            Your role is to add a random number between 1 and 64 (using only integers) to the previous number.
        ")
        .build();
    let mut number: u32 = 0;
    let mut interval = tokio::time::interval(std::time::Duration::from_secs(1));

    // Loop the agent and allow it to run autonomously. If it hits the target number (2000 or above)
    // we then terminate the loop and return the number
    // Note that the tokio interval is to avoid being rate limited
    loop {
        // Prompt the agent and print the response
        let response = agent.extract(&number.to_string()).await.unwrap();
        if response.number >= 2000 {
            break;
        } else {
            number += response.number
        }
        interval.tick().await;
    }

    println!("Finished with number: {number:?}");

    Ok(())
}
