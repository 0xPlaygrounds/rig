//! Demonstrates keeping a conversation across prompts.
//!
//! The caller owns the history: each run starts from it, and the messages the
//! run added (the prompt first) are appended to it afterwards. Saving it
//! between processes is `rig-memory`'s job.
//!
//! Requires `OPENAI_API_KEY`.

use anyhow::Result;
use rig::completion::Message;
use rig::prelude::*;
use rig::providers::openai::{self, OpenAI};

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
        let reply = agent.prompt(prompt).history(history.clone()).await?;
        println!("turn {}: {}", turn + 1, reply.output());
        history.extend(reply.messages);
    }

    Ok(())
}
