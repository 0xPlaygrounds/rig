//! Demonstrates `stream_chat` with prior conversation history.
//! Requires `OPENAI_API_KEY`.
//! Run it to see a streamed continuation of an existing exchange.

use anyhow::{Result, anyhow};
use futures::StreamExt;
use rig::agent::{MultiTurnStreamItem, StreamingResult};
use rig::client::{CompletionClient, ProviderClient};
use rig::message::Message;
use rig::providers::openai;
use rig::streaming::StreamingChat;

const PREAMBLE: &str = "You are a comedian here to entertain the user using humour and jokes.";
const PROMPT: &str = "Entertain me!";

async fn collect_stream_final_response<R>(stream: &mut StreamingResult<R>) -> Result<String> {
    let mut final_response = None;

    while let Some(item) = stream.next().await {
        if let MultiTurnStreamItem::FinalResponse(response) = item? {
            final_response = Some(response.response().to_owned());
        }
    }

    final_response.ok_or_else(|| anyhow!("stream finished without a final response"))
}

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
    let mut stream = agent.stream_chat(PROMPT, &history).await;
    let response = collect_stream_final_response(&mut stream).await?;
    println!("{response}");

    Ok(())
}
