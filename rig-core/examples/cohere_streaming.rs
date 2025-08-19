use rig::agent::stream_to_stdout;
use rig::prelude::*;
use rig::providers::cohere;
use rig::streaming::StreamingPrompt;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create streaming agent with a single context prompt
    let agent = cohere::Client::from_env()
        .agent(cohere::COMMAND)
        .preamble("Be precise and concise.")
        .temperature(0.5)
        .build();

    // Stream the response and print chunks as they arrive
    let mut stream = agent
        .stream_prompt("When and where and what type is the next solar eclipse?")
        .await;

    let res = stream_to_stdout(&mut stream).await?;

    println!("Token usage response: {usage:?}", usage = res.usage());
    println!("Final text response: {message:?}", message = res.response());

    Ok(())
}
