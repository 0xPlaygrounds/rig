//! Consume a remote A2A agent from Rig, against a live server.
//!
//! Points at any running A2A-compliant server that publishes a well-known
//! `AgentCard` (set `A2A_AGENT_URL`, default `http://localhost:8080`), and
//! shows the two ways to reach it:
//!
//! - as a **tool** an OpenAI-backed agent may call, threaded into one remote
//!   conversation by [`rig_a2a::conversation_context`];
//! - as a **model** backing a Rig agent directly, no LLM provider involved.
//!
//! For versions that need no credentials and no server, see `a2a_tool.rs` and
//! `a2a_agent.rs`.
//!
//! Requires `OPENAI_API_KEY`.
//!
//! ```sh
//! A2A_AGENT_URL=http://localhost:8080 \
//!   cargo run --example a2a_client -p rig-a2a
//! ```

use rig_a2a::{A2AAgentBuilderExt, A2AClient, A2AThreadInfo, conversation_context};
use rig_agent::agent::{AgentHook, HookContext, ToolResultAction, ToolResultEvent};
use rig_agent::{client::AgentClientExt, completion::Prompt};
use rig_core::client::ProviderClient;
use rig_core::providers::openai;

/// Observe-only hook: the A2A identifiers never reach the model, but the host
/// can log or persist them.
struct LogA2AIds;

impl AgentHook for LogA2AIds {
    async fn on_tool_result(
        &self,
        _context: &HookContext,
        event: ToolResultEvent<'_>,
    ) -> ToolResultAction {
        if let Some(info) = event.tool_context.result::<A2AThreadInfo>() {
            println!(
                "  [host] contextId={:?} state={:?}",
                info.context_id,
                info.state_label()
            );
        }
        ToolResultAction::Keep
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info,rig_a2a=debug")),
        )
        .init();

    let base_url =
        std::env::var("A2A_AGENT_URL").unwrap_or_else(|_| "http://localhost:8080".to_string());

    // Fetch the remote AgentCard. The whole remote agent becomes one Rig tool;
    // its skills are rendered into that tool's description rather than being
    // exposed as separate tools, because A2A carries no skill selector.
    let remote = A2AClient::from_url(&base_url).await?;
    println!(
        "connected to A2A agent {:?} advertising {} skill(s), projected as tool {:?}",
        remote.card().name,
        remote.card().skills.len(),
        remote.tool_name()
    );

    // Direct client usage: one message, explicit threading. Unlike the tool and
    // model surfaces, this API never attaches identifiers on your behalf.
    let outcome = remote
        .message("Introduce yourself in one sentence.")
        .send()
        .await?;
    println!("\ndirect A2A reply: {outcome:?}");

    // Tool usage: an OpenAI-backed agent decides when to call the remote agent.
    // The shared `conversation_context` threads both turns into a single remote
    // conversation; the model never sees a contextId or taskId.
    let openai_client = openai::Client::from_env()?;
    let agent = openai_client
        .agent(openai::GPT_4O_MINI)
        .preamble("Use the remote agent tool to answer, then relay its reply.")
        .a2a_tool(&remote)
        .add_hook(LogA2AIds)
        .build();

    println!("\n-- tool surface --");
    for prompt in [
        "Ask the remote agent to greet me, then relay its greeting.",
        "Now ask it what you just asked, to show it remembers.",
    ] {
        let reply = agent
            .prompt(prompt)
            .tool_context(conversation_context("demo-conversation"))
            .max_turns(4)
            .await?;
        println!("agent reply: {reply}");
    }

    // Model usage: the remote agent backs a Rig agent directly. No LLM provider
    // is involved — every prompt is an A2A `message/send`. Registering local
    // tools or an output schema on this agent would fail the call, because A2A
    // cannot carry them.
    println!("\n-- model surface --");
    let a2a_agent = remote.agent_for_conversation("demo-conversation").build();
    println!(
        "direct agent reply: {}",
        a2a_agent.prompt("Summarize what we discussed.").await?
    );

    Ok(())
}
