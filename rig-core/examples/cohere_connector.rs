use std::env;

use rig::{
    completion::{Completion, Prompt},
    providers::cohere::Client as CohereClient,
};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create Cohere client
    let cohere_api_key = env::var("COHERE_API_KEY").expect("COHERE_API_KEY not set");
    let cohere_client = CohereClient::new(&cohere_api_key);

    let klimadao_agent = cohere_client
        .agent("command-r")
        .temperature(0.0)
        .additional_params(json!({
            "connectors": [{"id":"web-search", "options":{"site": "https://docs.klimadao.finance"}}]
        }))
        .build();

    // Prompt the model and print the response
    // We use `prompt` to get a simple response from the model as a String
    let response = klimadao_agent.prompt("Tell me about BCT tokens?").await?;

    println!("\n\nCoral: {:?}", response);

    // Prompt the model and get the citations
    // We use `completion` to allow use to customize the request further and
    // get a more detailed response from the model.
    // Here the response is of type CompletionResponse<cohere::CompletionResponse>
    // which contains `choice` (Message or ToolCall) as well as `raw_response`,
    // the underlying providers' raw response.
    let response = klimadao_agent
        .completion("Tell me about BCT tokens?", vec![])
        .await?
        .additional_params(json!({
            "connectors": [{"id":"web-search", "options":{"site": "https://docs.klimadao.finance"}}]
        }))
        .send()
        .await?;

    println!(
        "\n\nCoral: {:?}\n\nCitations:\n{:?}",
        response.choice, response.raw_response.citations
    );

    Ok(())
}
