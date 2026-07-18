//! Consume a remote A2A agent from Rig against a live server.
//!
//! Set `A2A_AGENT_URL` to an A2A server that publishes an agent card. The
//! example calls it directly as a Rig agent and composes the same remote as a
//! dynamic sub-agent tool on an OpenAI-backed orchestrator.
//!
//! Requires `OPENAI_API_KEY`.
//!
//! ```sh
//! A2A_AGENT_URL=http://localhost:8080 \
//!   cargo run --example a2a_client -p rig-a2a
//! ```

use rig_a2a::{A2AClient, A2AConversationExt};
use rig_agent::{client::AgentClientExt, completion::Prompt};
use rig_core::client::ProviderClient;
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
    let remote = A2AClient::from_url(&base_url).await?;

    let a2a_agent = remote.agent().build();
    let reply = a2a_agent
        .prompt("Introduce yourself in one sentence.")
        .a2a_conversation("demo-conversation")
        .await?;
    println!("direct agent reply: {reply}");

    // Rig's standard sub-agent conversion supplies the dynamic tool. Bind the
    // conversation before conversion when repeated tool calls must continue
    // the same remote A2A thread.
    let remote_tool = remote
        .agent()
        .a2a_conversation("demo-conversation")
        .build()
        .into_tool();
    let openai_client = openai::Client::from_env()?;
    let orchestrator = openai_client
        .agent(openai::GPT_4O_MINI)
        .preamble("Use the remote agent tool to answer, then relay its reply.")
        .dynamic_tool(remote_tool)
        .build();

    let reply = orchestrator
        .prompt("Ask the remote agent what we discussed.")
        .max_turns(4)
        .await?;
    println!("orchestrator reply: {reply}");

    Ok(())
}
