use rig_agent::{agent::stream_to_stdout, prelude::*};
use rig_bedrock::client::BedrockRuntime;
use rig_bedrock::completion::{AMAZON_NOVA_LITE, Converse};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create streaming agent with a single context prompt
    let agent = AgentBuilder::new(rig_core::Model::new(
        Converse::new(AMAZON_NOVA_LITE),
        BedrockRuntime::from_env()?,
    ))
    .preamble("Be precise and concise.")
    .temperature(0.5)
    .build();

    // Stream the response and print chunks as they arrive
    let mut stream = agent
        .prompt("When and where and what type is the next solar eclipse?")
        .stream();

    let _ = stream_to_stdout(&mut stream).await?;

    Ok(())
}
