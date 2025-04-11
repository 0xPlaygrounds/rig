use rig::{
    providers::inception::{self, MERCURY_CODER_SMALL},
    streaming::{stream_to_stdout, StreamingPrompt},
};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create streaming agent with a single context prompt
    let agent = inception::Client::from_env()
        .agent(MERCURY_CODER_SMALL)
        .preamble("Be precise and concise.")
        .build();

    // Stream the response and print chunks as they arrive
    let mut stream = agent
        .stream_prompt("When and where and what type is the next solar eclipse?")
        .await?;

    stream_to_stdout(agent, &mut stream).await?;

    Ok(())
}
