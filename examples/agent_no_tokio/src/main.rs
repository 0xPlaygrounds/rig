//! An agent run driven by `bevy_tasks` instead of tokio.
//!
//! rig-agent has no runtime of its own: [`Agent::run_channel`] hands back a
//! plain future plus a bounded [`RunEvents`] feed, so the future can be spawned
//! on any executor — here Bevy's `AsyncComputeTaskPool` — while a synchronous
//! loop (think: a game frame, an ECS system) drains the events with
//! `try_next` and never blocks on the run. The HTTP transport (`rig-reqwest`)
//! brings its own private tokio runtime for the wire; this crate's manifest
//! depends on neither tokio nor reqwest. The transport is held erased
//! ([`rig::http_client::BoxedHttpClient`]), the way a host runtime keeps one
//! transport for every provider without naming it in its own types.
//!
//! Requires `OPENAI_API_KEY`.

use std::{thread, time::Duration};

use anyhow::Result;
use bevy_tasks::{AsyncComputeTaskPool, TaskPool, futures::check_ready};
use rig::agent::MultiTurnStreamItem;
use rig::client::Provider as _;
use rig::http_client::ReqwestClient;
use rig::prelude::*;
use rig::providers::openai;
use rig::streaming::{Delta, StreamEvent};

const PREAMBLE: &str = "You are a comedian here to entertain the user using humour and jokes.";
const PROMPT: &str = "Entertain me!";
/// How long the "frame" loop sleeps between ticks when nothing is ready.
const FRAME: Duration = Duration::from_millis(16);

fn main() -> Result<()> {
    // A host holds one erased transport for every provider it talks to: the
    // client is `Client<OpenAIResponses>` — `H` defaults to
    // `BoxedHttpClient`, so no transport type reaches this crate's signatures.
    let transport = ReqwestClient::default().boxed();
    let agent = openai::OpenAIResponses::from_env(transport)?
        .agent(openai::GPT_4O)
        .preamble(PREAMBLE)
        .build();

    // Split the run: a future for the pool, an event feed for the frame loop.
    let (run, mut events) = agent.run_channel(PROMPT);
    let pool = AsyncComputeTaskPool::get_or_init(TaskPool::new);
    let mut task = pool.spawn(run);

    // The "game loop": poll the feed without blocking, tick, repeat.
    let response = loop {
        while let Some(event) = events.try_next() {
            match event {
                MultiTurnStreamItem::StreamAssistantItem(StreamEvent::BlockDelta {
                    delta: Delta::Text { text },
                    ..
                }) => {
                    print!("{text}");
                }
                MultiTurnStreamItem::ToolExecutionCommitted { tool_call, .. } => {
                    println!("\n[tool {}]", tool_call.function.name);
                }
                MultiTurnStreamItem::FinalResponse(_) => println!(),
                _ => {}
            }
        }
        if let Some(outcome) = check_ready(&mut task) {
            break outcome?;
        }
        thread::sleep(FRAME);
    };

    println!(
        "done: {} completion call(s), {} total tokens",
        response.completion_calls().len(),
        response.usage().total_tokens
    );
    Ok(())
}
