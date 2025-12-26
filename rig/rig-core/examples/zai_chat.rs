use rig::{client::{CompletionClient, ProviderClient}, completion::Prompt, providers::zai};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create the z.ai client
    let client = zai::Client::from_env();

    // Create an agent
    let agent = client.agent(zai::GLM_4_5_FLASH).build();

    // Send a prompt
    let response = agent.prompt("Hello, z.ai!").await?;

    println!("Response: {}", response);

    Ok(())
}
