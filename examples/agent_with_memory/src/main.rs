//! Demonstrates keeping a conversation across prompts.
//!
//! The conversation is kept by `rig::memory`: each run starts from its
//! messages, the messages the run added (the prompt first) are appended to it
//! afterwards, and it is saved. A file store (`DirStore`, feature `fs` of
//! `rig-memory`) keeps it between processes the same way.
//!
//! Requires `OPENAI_API_KEY`.

use anyhow::Result;
use rig::prelude::*;
use rig::providers::openai::{self, OpenAI};

#[tokio::main]
async fn main() -> Result<()> {
    let agent = AgentBuilder::new(OpenAI::from_env()?.completion(openai::GPT_4O))
        .preamble("You are a helpful assistant with persistent memory.")
        .build();

    let store = rig::memory::MemoryStore::default();
    let mut conversation = rig::memory::load(&store, "user-123")?;
    for (turn, prompt) in ["My name is Alice.", "What's my name?"]
        .into_iter()
        .enumerate()
    {
        let reply = agent
            .prompt(prompt)
            .history(conversation.messages().to_vec())
            .await?;
        println!("turn {}: {}", turn + 1, reply.output());
        conversation.extend(reply.messages);
        rig::memory::save(&store, "user-123", &mut conversation)?;
    }

    Ok(())
}
