use rig::agent::AgentBuilder;
use rig::providers::zhipuai::{CompletionModel, ZHIPU_CHAT};
use rig::{completion::Prompt, providers};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    println!("Running basic agent with ZhipuAI");
    basic_zhipuai().await?;

    println!("\nRunning ZhipuAI agent with context");
    context_zhipuai().await?;

    println!("\n\nAll agents ran successfully");
    Ok(())
}

fn client() -> providers::zhipuai::Client {
    providers::zhipuai::Client::from_env()
}

fn partial_agent_zhipuai() -> AgentBuilder<CompletionModel> {
    let client = client();
    client.agent(ZHIPU_CHAT)
}

async fn basic_zhipuai() -> Result<(), anyhow::Error> {
    let poet_agent = partial_agent_zhipuai()
        .preamble("You are a Chinese poet who writes poems in classical style.")
        .build();

    // Prompt the agent and print the response
    let response = poet_agent.prompt("Write a poem about the Yangtze River").await?;
    println!("{}", response);

    Ok(())
}

async fn context_zhipuai() -> Result<(), anyhow::Error> {
    let model = client().completion_model(ZHIPU_CHAT);

    // Create an agent with multiple context documents
    let agent = AgentBuilder::new(model)
        .preamble("Definition of *Qingming Festival*: A traditional Chinese festival to honor ancestors, usually in early April.")
        .build();

    // Prompt the agent and print the response
    let response = agent.prompt("What are the customs of Qingming Festival?").await?;

    println!("{}", response);

    Ok(())
}
