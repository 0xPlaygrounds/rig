mod discord_bot;

use discord_bot::DiscordExt;
use rig::prelude::*;
use rig::providers::openai::{self, OpenAI};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let discord_bot_token = std::env::var("DISCORD_BOT_TOKEN")?;
    // Create the OpenAI provider
    let client = OpenAI::from_env()?.bound()?;

    // Create agent with a single context prompt
    let mut discord_bot = client
        .agent(openai::GPT_4O)
        .preamble("You are a helpful assistant.")
        .build()
        .into_discord_bot(&discord_bot_token)
        .await?;

    discord_bot.start().await?;

    Ok(())
}
