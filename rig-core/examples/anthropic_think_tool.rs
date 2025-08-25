use anyhow::Result;
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::Prompt;
use rig::providers;
use rig::think_tool::ThinkTool;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    tracing_subscriber::fmt().init();

    // Create agent with the Think tool
    let agent = providers::anthropic::Client::from_env()
        .agent(providers::anthropic::CLAUDE_3_7_SONNET)
        .name("Anthropic Thinker")
        .preamble(
            "You are a helpful assistant that can solve complex problems.
            Use the 'think' tool to reason through complex problems step by step.
            When faced with a multi-step problem or when analyzing tool results,
            use the 'think' tool to organize your thoughts before responding.",
        )
        .tool(ThinkTool)
        .build();

    println!("Solving a complex problem with the Think tool");

    // Example prompt that would benefit from the Think tool
    let prompt = "I need to plan a dinner party for 8 people, including 2 vegetarians and 1 person with a gluten allergy. \
                 Can you help me create a menu that everyone can enjoy? Consider appetizers, main courses, and desserts.";

    let resp = agent.prompt(prompt).multi_turn(10).await?;

    println!("{}", resp);

    Ok(())
}
