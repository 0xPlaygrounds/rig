use rig::{completion::Prompt, providers};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let client = providers::deepseek::Client::from_env();
    let agent = client
        .agent("deepseek-chat")
        .preamble("You are a helpful assistant.")
        .build();

    let answer = agent.prompt("Tell me a joke").await?;
    println!("Answer: {}", answer);
    Ok(())
}
