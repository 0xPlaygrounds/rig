use rig::providers::openai;
use rig::streaming::{stream_to_stdout, StreamingPrompt};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create streaming agent with a single context prompt
    let agent = openai::Client::from_env()
        .agent(openai::GPT_4O)
        .preamble("Be precise and concise.")
        .temperature(0.5)
        .build();

    // Stream the response and print chunks as they arrive
    let mut stream = agent
        .stream_prompt("When and where and what type is the next solar eclipse?")
        .await?;

    stream_to_stdout(&agent, &mut stream).await?;

    if let Some(response) = stream.response {
        println!("Usage: {:?}", response.usage)
    };

    println!("Message: {:?}", stream.choice);

    Ok(())
}
