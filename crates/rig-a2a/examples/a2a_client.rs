//! Consume a remote A2A agent's skills as tools on a Rig agent.
//!
//! Points at any running A2A-compliant server that publishes a well-known
//! `AgentCard` (set `A2A_AGENT_URL`, default `http://localhost:8080`).
//!
//! Requires `OPENAI_API_KEY`.
//!
//! ```sh
//! A2A_AGENT_URL=http://localhost:8080 \
//!   cargo run --example a2a_client -p rig-a2a
//! ```

use rig_a2a::{A2AAgentBuilderExt, A2AClient};
use rig_core::client::{CompletionClient, ProviderClient};
use rig_core::completion::Prompt;
use rig_core::providers::openai;

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

    // Fetch the remote AgentCard and wrap each declared skill as a Rig tool.
    let remote = A2AClient::from_url(&base_url).await?;
    println!(
        "connected to A2A agent {:?} ({} skill(s))",
        remote.card().name,
        remote.card().skills.len()
    );

    // Direct client usage: send one message. To continue the conversation,
    // echo the returned task's contextId via `.context(...)`.
    let outcome = remote
        .message("Introduce yourself in one sentence.")
        .send()
        .await?;
    println!("direct A2A reply: {outcome:?}");

    // Agent usage: let an OpenAI-backed Rig agent decide when to call the
    // remote agent's skills.
    let openai_client = openai::Client::from_env()?;
    let agent = openai_client
        .agent(openai::GPT_4O_MINI)
        .preamble("Use the available remote agent tools to answer.")
        .a2a_tools(&remote)
        .build();

    let reply = agent
        .prompt("Ask the remote agent to greet me, then relay its greeting.")
        .await?;
    println!("agent reply: {reply}");

    Ok(())
}
