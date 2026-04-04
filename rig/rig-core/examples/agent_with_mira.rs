use rig::client::ProviderClient;
use rig::providers::mira;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let client = mira::Client::from_env();
    let models = client.list_models().await?;

    println!("Available Mira models:");
    for model in models {
        println!("- {model}");
    }

    Ok(())
}
