use rig::prelude::*;
use rig::{
    completion::Prompt,
    providers::gemini::{self},
};
#[tracing::instrument(ret)]
#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_target(false)
        .init();

    // Initialize the Google Gemini client
    let client = gemini::Client::from_env();

    // Create agent with a single context prompt
    let agent = client
        .agent("gemini-2.5-flash")
        .preamble("Be creative and concise. Answer directly and clearly.")
        .temperature(0.5)
        .build();

    tracing::info!("Prompting the agent...");

    // Prompt the agent and print the response
    let response = agent
        .prompt("How much wood would a woodchuck chuck if a woodchuck could chuck wood? Infer an answer.")
        .await;

    tracing::info!("Response: {:?}", response);

    match response {
        Ok(response) => println!("{response}"),
        Err(e) => {
            tracing::error!("Error: {:?}", e);
            return Err(e.into());
        }
    }

    Ok(())
}
