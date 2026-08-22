//! An agent run driven by `bevy_tasks` instead of tokio.
//!
//! rig-agent has no runtime of its own: [`Agent::run_channel`] hands back a
//! [`RunHandle`], a plain [`RunFuture`] and a bounded [`RunEvents`] feed, so
//! the future can be spawned on any executor — here Bevy's
//! `AsyncComputeTaskPool` — while a synchronous loop (think: a game frame, an
//! ECS system) drains the events with `try_drain` and never blocks on the run.
//! The HTTP transport (`rig-reqwest`) brings its own private tokio runtime for
//! the wire; this crate's manifest depends on neither tokio nor reqwest. The
//! transport is held erased ([`rig::http_client::BoxedHttpClient`]), the way a
//! host runtime keeps one transport for every provider without naming it in
//! its own types.
//!
//! The per-run bundle below — handle, task, feed — is exactly what a Bevy
//! component holding a run looks like; the loop body is the system.
//!
//! Requires `OPENAI_API_KEY`. Set `RIG_EXAMPLE_ABORT_AFTER=<n>` to abort the
//! run through its handle after `n` events and see the cancelled outcome.

use std::{thread, time::Duration};

use anyhow::Result;
use bevy_tasks::{AsyncComputeTaskPool, Task, TaskPool, futures::check_ready};
use rig::agent::{MultiTurnStreamItem, RunEvent, RunEvents, RunFuture, RunHandle};
use rig::client::ProviderFromEnv as _;
use rig::completion::PromptError;
use rig::http_client::ReqwestClient;
use rig::prelude::*;
use rig::providers::openai;
use rig::streaming::StreamedAssistantContent;

const PREAMBLE: &str = "You are a comedian here to entertain the user using humour and jokes.";
const PROMPT: &str = "Entertain me!";
/// How long the "frame" loop sleeps between ticks when nothing is ready.
const FRAME: Duration = Duration::from_millis(16);

/// One in-flight run, as a host keeps it: identity + cancel handle, the task
/// driving it, and the feed its events arrive on. (In Bevy: a component.)
struct ActiveRun {
    handle: RunHandle,
    task: Task<<RunFuture as Future>::Output>,
    events: RunEvents,
}

fn main() -> Result<()> {
    // A host holds one erased transport for every provider it talks to: the
    // client is `Client<OpenAIResponsesExt>` — `H` defaults to
    // `BoxedHttpClient`, so no transport type reaches this crate's signatures.
    let transport = ReqwestClient::default().boxed();
    let agent = openai::OpenAIResponsesExt::from_env_boxed(transport)?
        .agent(openai::GPT_4O)
        .preamble(PREAMBLE)
        .build();

    let abort_after: Option<usize> = std::env::var("RIG_EXAMPLE_ABORT_AFTER")
        .ok()
        .and_then(|n| n.parse().ok());

    // Split the run: a handle to keep, a future for the pool, an event feed
    // for the frame loop. The id is fixed before this returns.
    let (handle, run, events) = agent.run_channel(PROMPT);
    let pool = AsyncComputeTaskPool::get_or_init(TaskPool::new);
    let mut active = ActiveRun {
        handle,
        task: pool.spawn(run),
        events,
    };
    println!("run {}", active.handle.id());

    // The "game loop": drain the feed without blocking, tick, repeat. (In
    // Bevy: one system, querying every `ActiveRun` component.)
    let mut batch: Vec<RunEvent> = Vec::new();
    let mut delivered = 0usize;
    let outcome = loop {
        active.events.try_drain(&mut batch);
        for event in batch.drain(..) {
            debug_assert_eq!(event.run, active.handle.id());
            delivered += 1;
            match event.item {
                MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Text(text)) => {
                    print!("{}", text.text)
                }
                MultiTurnStreamItem::ToolExecutionCommitted { tool_call, .. } => {
                    println!("\n[tool {}]", tool_call.function.name);
                }
                MultiTurnStreamItem::FinalResponse(_) => println!(),
                _ => {}
            }
            if abort_after.is_some_and(|n| delivered >= n) && !active.handle.is_aborted() {
                println!(
                    "\n[aborting run {} after {delivered} events]",
                    active.handle.id()
                );
                active.handle.abort();
            }
        }
        if let Some(outcome) = check_ready(&mut active.task) {
            break outcome;
        }
        thread::sleep(FRAME);
    };

    match outcome {
        Ok(response) => println!(
            "done: {} completion call(s), {} total tokens",
            response.completion_calls().len(),
            response.usage().total_tokens
        ),
        Err(PromptError::PromptCancelled { reason, .. }) => {
            println!("cancelled: {reason}");
        }
        Err(err) => return Err(err.into()),
    }
    Ok(())
}
