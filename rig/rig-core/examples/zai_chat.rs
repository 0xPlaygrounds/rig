use rig::{completion::Prompt, providers::zai};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create the z.ai client
    let client = zai::Client::from_env();

    // Create a model
    let model = client.completion_model(zai::GLM_4);

    // Send a prompt
    let response = model.prompt("Hello, z.ai!").await?;

    println!("Response: {}", response);

    Ok(())
}
