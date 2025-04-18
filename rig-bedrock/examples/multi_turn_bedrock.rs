use rig::completion::Prompt;
use rig_bedrock::{client::ClientBuilder, completion::AMAZON_NOVA_LITE};
mod common;
use common::address_book_tool::AddressBookTool;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    tracing_subscriber::fmt().init();
    // Create agent with a single context prompt and two tools
    let agent = ClientBuilder::new()
        .build()
        .await
        .agent(AMAZON_NOVA_LITE)
        .preamble("You have access to user address tool. Never return <thinking> part")
        .max_tokens(1024)
        .tool(AddressBookTool)
        .build();

    let result = agent
        .prompt("Can you find address for this email: jane.smith@example.com")
        .multi_turn(20)
        .await?;

    println!("\n{}", result);

    let result = agent
        .prompt("Can you find address for this email: does_not_exists@example.com")
        .multi_turn(20)
        .await?;

    println!("\n{}", result);

    Ok(())
}
