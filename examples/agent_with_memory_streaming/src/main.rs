//! Demonstrates keeping a conversation across streamed prompts.
//!
//! The caller owns the history: each run starts from it, and the messages the
//! final response carries (the prompt first) are appended to it afterwards.
//! Saving it between processes is `rig-memory`'s job.
//!
//! Requires `OPENAI_API_KEY`.

use anyhow::{Result, anyhow};
use futures::StreamExt;
use rig::agent::{MultiTurnStreamItem, PromptResponse, StreamingResult};
use rig::completion::Message;
use rig::prelude::*;
use rig::providers::openai::{self, OpenAI};

async fn collect_final(mut stream: StreamingResult) -> Result<PromptResponse> {
    let mut final_response = None;
    while let Some(item) = stream.next().await {
        if let MultiTurnStreamItem::FinalResponse(response) = item? {
            final_response = Some(response);
        }
    }
    final_response.ok_or_else(|| anyhow!("stream finished without a final response"))
}

#[tokio::main]
async fn main() -> Result<()> {
    let agent = AgentBuilder::new(OpenAI::from_env()?.completion(openai::GPT_4O))
        .preamble("You are a helpful assistant with persistent memory.")
        .build();

    let mut history: Vec<Message> = Vec::new();
    for (turn, prompt) in ["My name is Alice.", "What's my name?"]
        .into_iter()
        .enumerate()
    {
        let reply = collect_final(agent.prompt(prompt).history(history.clone()).stream()).await?;
        println!("turn {}: {}", turn + 1, reply.output());
        history.extend(reply.messages);
    }

    Ok(())
}
