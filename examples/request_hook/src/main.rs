//! Demonstrates observing and composing prompt/response lifecycle events with
//! `AgentHook`. Two hooks are stacked via `.add_hook(…).add_hook(…)`:
//!
//! - `LoggingHook` — logs the outgoing prompt text and incoming model response.
//! - `TurnCounterHook` — counts how many completion calls are made and prints
//!   the total when the run finishes (by inspecting `StepEvent::CompletionCall`).
//!
//! Hooks run in registration order; the first to return a non-`Continue` result
//! short-circuits the rest. In this example both hooks always return
//! `Flow::Continue`, so both are consulted on every event.
//!
//! Requires `OPENAI_API_KEY`.

use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::Result;
use rig::agent::{AgentHook, Flow, StepEvent};
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::{CompletionModel, CompletionResponse, Message, Prompt};
use rig::message::UserContent;
use rig::providers::openai;

// ---------------------------------------------------------------------------
// Hook 1: LoggingHook — logs each outgoing prompt and incoming response.
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct LoggingHook<'a> {
    session_id: &'a str,
}

impl<'a, M> AgentHook<M> for LoggingHook<'a>
where
    M: CompletionModel,
{
    async fn on_event(&self, event: StepEvent<'_, M>) -> Flow {
        match event {
            StepEvent::CompletionCall { prompt, .. } => {
                if let Message::User { content } = prompt {
                    let prompt_text = content
                        .iter()
                        .filter_map(|c| match c {
                            UserContent::Text(text) => Some(text.text.clone()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join("\n");
                    if !prompt_text.is_empty() {
                        println!("[{}] sending prompt: {}", self.session_id, prompt_text);
                    }
                }
                Flow::cont()
            }
            StepEvent::CompletionResponse { response, .. } => {
                let response: &CompletionResponse<M::Response> = response;
                println!(
                    "[{}] received response: {:?}",
                    self.session_id, response.choice
                );
                Flow::cont()
            }
            _ => Flow::cont(),
        }
    }
}

// ---------------------------------------------------------------------------
// Hook 2: TurnCounterHook — counts completion calls, prints total at the end.
// ---------------------------------------------------------------------------

#[derive(Default)]
struct TurnCounterHook {
    count: AtomicUsize,
}

impl<M> AgentHook<M> for TurnCounterHook
where
    M: CompletionModel,
{
    async fn on_event(&self, event: StepEvent<'_, M>) -> Flow {
        if let StepEvent::CompletionCall { turn, .. } = event {
            self.count.fetch_add(1, Ordering::Relaxed);
            println!("[turn-counter] starting completion turn #{turn}");
        }
        Flow::cont()
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<()> {
    let agent = openai::Client::from_env()?
        .agent(openai::GPT_4O)
        .preamble("You are a comedian here to entertain the user using humour and jokes.")
        .build();

    // Attach two hooks via `.add_hook(first).add_hook(second)`; they run in
    // registration order and the first non-`Continue` result short-circuits.
    let response = agent
        .prompt("Entertain me!")
        .add_hook(LoggingHook {
            session_id: "demo-session",
        })
        .add_hook(TurnCounterHook::default())
        .await?;

    println!("\nFinal response:\n{response}");

    Ok(())
}
