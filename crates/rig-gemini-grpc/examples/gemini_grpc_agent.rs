//! An agent talking to Gemini over gRPC.
//!
//! The concrete client retains the connected tonic channel for both fluent
//! agents and low-level function calls.

use rig_agent::client::AgentClientExt;

#[tracing::instrument(ret)]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_target(false)
        .init();

    let client = rig_gemini_grpc::Client::from_env().await?;
    let agent = client
        .agent("gemini-2.5-flash")
        .preamble("Be creative and concise. Answer directly and clearly.")
        .temperature(0.5)
        .build();

    tracing::info!("Prompting the agent via gRPC...");

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
