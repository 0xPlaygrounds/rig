use rig_agent::{agent::stream_to_stdout, prelude::*};
use rig_bedrock::client::BedrockRuntime;
use rig_bedrock::completion::AMAZON_NOVA_LITE;
use rig_core::RigError;

#[tokio::main]
async fn main() -> Result<(), RigError> {
    // Create streaming agent with a single context prompt
    let agent = AgentBuilder::new(BedrockRuntime::from_env().completion(AMAZON_NOVA_LITE))
        .preamble("Be precise and concise.")
        .temperature(0.5)
        .build();

    // Stream the response and print chunks as they arrive
    let mut stream = agent
        .prompt("When and where and what type is the next solar eclipse?")
        .stream();

    let _ = stream_to_stdout(&mut stream)
        .await
        .map_err(RigError::other)?;

    Ok(())
}
