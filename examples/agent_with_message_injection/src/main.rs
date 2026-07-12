//! Inject a message into an agent run while it is in flight.
//!
//! A long-running agent — a multi-turn tool loop — sometimes needs to hear about
//! something that happened *after* it was prompted: a user follow-up, an
//! external event, a cancellation note. [`AgentRunner::message_injector`] hands
//! back a cloneable [`MessageInjector`] bound to the run; a message pushed
//! through it is delivered as its own user turn immediately before the run's
//! next model call.
//!
//! Delivery is **best-effort and in-flight**: if the run finishes before the
//! next model call, the message is not delivered (and a later `inject` reports
//! `RunFinished`). Injection does not resurrect a completed run — to continue a
//! finished conversation, start a new run with the prior history.
//!
//! Here the agent researches a topic with a (deliberately slow) `research` tool
//! while a background "watcher" task injects a steering note partway through.
//! The "fold input between steps" point mirrors pydantic-ai's `AgentRun.enqueue`,
//! Vercel AI SDK's `prepareStep`, and LangGraph's `Command(resume=…)` — but needs
//! no checkpointer.
//!
//! Requires `OPENAI_API_KEY`. Run with: `cargo run -p agent_with_message_injection`

use std::time::Duration;

use anyhow::Result;
use rig::client::{CompletionClient, ProviderClient};
use rig::providers::openai;
use rig::tool::Tool;
use serde::Deserialize;
use serde_json::json;

// ---------------------------------------------------------------------------
// A deliberately slow tool, so the run lasts long enough to inject into.
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
#[error("tool failed: {0}")]
struct ToolError(String);

#[derive(Deserialize)]
struct ResearchArgs {
    topic: String,
}

struct Research;

impl Tool for Research {
    const NAME: &'static str = "research";
    type Error = ToolError;
    type Args = ResearchArgs;
    type Output = String;

    fn description(&self) -> String {
        "Look up a short finding about a single topic.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "topic": { "type": "string", "description": "The topic to research" }
            },
            "required": ["topic"]
        })
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        println!("   🔎 [research] -> {}", args.topic);
        // Simulate a slow lookup, leaving a window for a message to be injected.
        tokio::time::sleep(Duration::from_millis(900)).await;
        Ok(format!(
            "Finding on '{}': it is widely adopted and well documented.",
            args.topic
        ))
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<()> {
    let agent = openai::Client::from_env()?
        .agent(openai::GPT_4O)
        .preamble(
            "You are a research assistant. Use the `research` tool to investigate each aspect \
             of the user's request, one tool call at a time, then write a short final summary.",
        )
        .tool(Research)
        .build();

    let prompt = "Compare Postgres and SQLite for a small web app. Research them one at a time.";
    println!("User: {prompt}\n");

    // Build a runner instead of calling `agent.prompt(..)` directly, so we can
    // take an injector handle *before* the run starts.
    let mut runner = agent.runner(prompt).max_turns(10);
    let injector = runner.message_injector();

    // Drive the agent on a task; hold the injector here in `main`.
    let run = tokio::spawn(async move { runner.run().await });

    // ...meanwhile, a watcher reacts to an external event and steers the run
    // while it is still mid-tool-loop. (In a real app this might be a user
    // keystroke, a webhook, or a DB poll.)
    tokio::time::sleep(Duration::from_millis(1200)).await;
    println!("\n💬 [watcher] injecting a mid-run steering note...\n");
    match injector.inject(
        "One more requirement from the user: weigh operational cost and ops burden, \
         not just raw performance.",
    ) {
        Ok(()) => {}
        // The run already finished — nothing more to steer.
        Err(err) => println!("   (could not inject: {err})"),
    }

    let response = run.await??;
    println!("\nFinal response:\n{}", response.output);

    Ok(())
}
