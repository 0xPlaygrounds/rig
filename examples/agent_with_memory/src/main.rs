//! Demonstrates keeping a conversation across prompts.
//!
//! The conversation is kept by `rig::memory`: each run starts from its
//! messages, the messages the run added (the prompt first) are appended to it
//! afterwards, and it is saved. A file store (`DirStore`, feature `fs` of
//! `rig-memory`) keeps it between processes the same way. With `--stream`
//! each prompt is streamed and the final response carries the same messages.
//!
//! Requires `OPENAI_API_KEY`.

use anyhow::{Result, anyhow};
use futures::StreamExt;
use rig::agent::{MultiTurnStreamItem, PromptResponse, StreamingResult};
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
    let streamed = std::env::args().any(|arg| arg == "--stream");
    let agent = AgentBuilder::new(OpenAI::from_env()?.completion(openai::GPT_4O))
        .preamble("You are a helpful assistant with persistent memory.")
        .build();

    let store = rig::memory::MemoryStore::default();
    let mut conversation = rig::memory::load(&store, "user-123")?;
    for (turn, prompt) in ["My name is Alice.", "What's my name?"]
        .into_iter()
        .enumerate()
    {
        let request = agent
            .prompt(prompt)
            .history(conversation.messages().to_vec());
        let reply = if streamed {
            collect_final(request.stream()).await?
        } else {
            request.await?
        };
        println!("turn {}: {}", turn + 1, reply.output());
        conversation.extend(reply.messages);
        rig::memory::save(&store, "user-123", &mut conversation)?;
    }

    Ok(())
}
