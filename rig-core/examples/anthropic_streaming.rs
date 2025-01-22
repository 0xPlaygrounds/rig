use futures::StreamExt;
use rig::{
    providers::anthropic::{self, CLAUDE_3_5_SONNET},
    streaming::StreamingPrompt,
};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create streaming agent with a single context prompt
    let agent = anthropic::Client::from_env()
        .agent(CLAUDE_3_5_SONNET)
        .preamble("Be precise and concise.")
        .temperature(0.5)
        .build();

    // Stream the response and print chunks as they arrive
    let mut stream = agent
        .stream_prompt("When and where and what type is the next solar eclipse?")
        .await?;

    print!("Response: ");
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(chunk) => {
                print!("{}", chunk);
                // Flush stdout to ensure immediate printing
                std::io::Write::flush(&mut std::io::stdout())?;
            }
            Err(e) => eprintln!("Error receiving chunk: {}", e),
        }
    }
    println!(); // New line after streaming completes

    Ok(())
}
