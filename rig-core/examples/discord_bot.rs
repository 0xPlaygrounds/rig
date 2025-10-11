use rig::integrations::discord_bot::DiscordExt;
use rig::prelude::*;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let discord_bot_token = std::env::var("DISCORD_BOT_TOKEN")
        .expect("DISCORD_BOT_TOKEN to be set as an environment variable");
    // Create OpenAI client
    let client = rig::providers::openai::Client::from_env();

    // Create agent with a single context prompt
    let mut discord_bot = client
        .agent("gpt-4o")
        .preamble("You are a helpful assistant.")
        .build()
        .into_discord_bot(&discord_bot_token)
        .await;

    discord_bot.start().await.unwrap();

    Ok(())
}
