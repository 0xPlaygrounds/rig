//! Demonstrates a fully driven streamed run with prior conversation history,
//! consuming only its committed final response.
//! Requires `OPENAI_API_KEY`.
//! Run it to obtain a continuation of an existing exchange.

use anyhow::Result;
use rig::prelude::*;
use rig::providers::openai;

const PREAMBLE: &str = "You are a comedian here to entertain the user using humour and jokes.";
const PROMPT: &str = "Entertain me!";

fn sample_history() -> Vec<Message> {
    vec![
        Message::user("Tell me a joke!"),
        Message::assistant("Why did the chicken cross the road?\n\nTo get to the other side!"),
    ]
}

#[tokio::main]
async fn main() -> Result<()> {
    let agent = openai::Client::from_env()?
        .agent(openai::GPT_4)
        .preamble(PREAMBLE)
        .build();

    let history = sample_history();
    let response = agent
        .runner(PROMPT)
        .history(&history)
        .stream_run()
        .into_final_response()
        .await?;
    println!("{}", response.output());

    Ok(())
}
