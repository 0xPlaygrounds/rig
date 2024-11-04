use rig::{
    completion::Prompt,
    providers::gemini::{self, completion::gemini_api_types::GenerationConfig},
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
        .agent(gemini::completion::GEMINI_1_5_PRO)
        .preamble("Be creative and concise. Answer directly and clearly.")
        .temperature(0.5)
        // The `GenerationConfig` utility struct helps construct a typesafe `additional_params`
        .additional_params(serde_json::to_value(GenerationConfig {
            top_k: Some(1),
            top_p: Some(0.95),
            candidate_count: Some(1),
            ..Default::default()
        })?) // Unwrap the Result to get the Value
        .build();

    tracing::info!("Prompting the agent...");

    // Prompt the agent and print the response
    let response = agent
        .prompt("How much wood would a woodchuck chuck if a woodchuck could chuck wood? Infer an answer.")
        .await;

    tracing::info!("Response: {:?}", response);

    match response {
        Ok(response) => println!("{}", response),
        Err(e) => {
            tracing::error!("Error: {:?}", e);
            return Err(e.into());
        }
    }

    Ok(())
}
