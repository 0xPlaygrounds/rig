use rig::agent::stream_to_stdout;
use rig::message::Message;
use rig::prelude::*;
use rig::streaming::StreamingChat;

use rig::providers::{self, openai};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create OpenAI client
    let client = providers::openai::Client::from_env();

    // Create agent with a single context prompt
    let comedian_agent = client
        .agent(openai::GPT_4)
        .preamble("You are a comedian here to entertain the user using humour and jokes.")
        .build();

    let messages = vec![
        Message::user("Tell me a joke!"),
        Message::assistant("Why did the chicken cross the road?\n\nTo get to the other side!"),
    ];

    // Prompt the agent and print the response
    let mut stream = comedian_agent.stream_chat("Entertain me!", messages).await;

    let res = stream_to_stdout(&mut stream).await.unwrap();

    println!("Response: {res:?}");

    Ok(())
}
