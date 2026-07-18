//! Runs an OpenAI model through Rig's experimental native Bevy ECS runtime.
//! Requires `OPENAI_API_KEY`.

use anyhow::Result;
use rig::{
    bevy::{BevyClientExt, LocalRuntime},
    client::ProviderClient,
    providers::openai,
};

#[tokio::main]
async fn main() -> Result<()> {
    let client = openai::Client::from_env()?;
    let definition = client
        .bevy_agent(openai::GPT_5_2)
        .preamble("Answer directly in one short paragraph.")
        .build();
    let mut runtime = LocalRuntime::new()?;
    let agent = runtime.spawn_agent(definition)?;

    let result = runtime
        .run(agent, "Why can ECS be useful for agent orchestration?")
        .await?;
    println!("{}", result.text.as_deref().unwrap_or("<no text>"));
    Ok(())
}
