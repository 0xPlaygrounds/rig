//! Serve an OpenAI-backed Rig agent over A2A at http://localhost:3000.
//!
//! Run with `OPENAI_API_KEY` set:
//!
//! ```sh
//! cargo run -p rig-a2a --example a2a_server
//! ```
//!
//! Any A2A 1.0 client can then read the card at
//! `http://localhost:3000/.well-known/agent-card.json` and send messages,
//! for example over JSON-RPC:
//!
//! ```sh
//! curl -s http://localhost:3000/jsonrpc -H 'content-type: application/json' \
//!   -H 'A2A-Version: 1.0' -d '{"jsonrpc":"2.0","id":1,"method":"SendMessage",
//!   "params":{"message":{"messageId":"m1","role":"ROLE_USER","parts":[{"text":"Hi!"}]}}}'
//! ```

use rig_a2a::A2aServer;
use rig_agent::AgentBuilder;
use rig_core::providers::openai::{self, OpenAI};

const URL: &str = "http://localhost:3000";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let model = OpenAI::from_env()?.completion(openai::GPT_5_4_MINI);
    let agent = AgentBuilder::new(model)
        .name("Rig helper")
        .description("Answers questions concisely.")
        .preamble("You are a concise, friendly assistant.")
        .build();

    let app = A2aServer::new(agent, URL).into_router();
    let listener = tokio::net::TcpListener::bind("localhost:3000").await?;
    println!("Serving the agent card at {URL}/.well-known/agent-card.json");
    rig_a2a::axum::serve(listener, app).await?;
    Ok(())
}
