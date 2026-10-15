//! Connect a Rig agent to a hosted MCP server over streamable HTTP and let
//! it use the server's tools.
//!
//! ```sh
//! cargo run -p rmcp_example --bin remote -- <mcp-url> ["<prompt>"]
//! ```
//!
//! Set `MCP_AUTH_TOKEN` for a server that wants a bearer token. Needs
//! `OPENAI_API_KEY`.

use anyhow::Context;
use rig::{
    prelude::*,
    providers::openai::{self, OpenAI},
    tool::{DynamicTool, rmcp::McpConnection},
};
use rmcp::{
    ServiceExt,
    model::{ClientCapabilities, ClientConfig, Implementation},
    transport::{
        StreamableHttpClientTransport, streamable_http_client::StreamableHttpClientTransportConfig,
    },
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let url = args.next().context("usage: remote <mcp-url> [prompt]")?;
    let prompt = args.next().unwrap_or_else(|| {
        "Which tools do you have? Call the most useful one once and report what it returned."
            .to_owned()
    });

    let mut config = StreamableHttpClientTransportConfig::with_uri(url);
    if let Ok(token) = std::env::var("MCP_AUTH_TOKEN") {
        config = config.auth_header(token);
    }
    let client = ClientConfig::new(
        ClientCapabilities::default(),
        Implementation::new("rig", env!("CARGO_PKG_VERSION")),
    );
    let connection = McpConnection::new(
        client
            .serve(StreamableHttpClientTransport::from_config(config))
            .await?,
    );

    let tools = connection.tools().await?;
    let names = tools.iter().map(|tool| tool.name()).collect::<Vec<_>>();
    println!("Tools: {}", names.join(", "));
    let tools = tools
        .into_iter()
        .map(DynamicTool::try_from)
        .collect::<Result<Vec<_>, _>>()?;

    let agent = AgentBuilder::new(OpenAI::from_env()?.completion(openai::GPT_5_4_MINI))
        .preamble("Answer with the help of the MCP server's tools.")
        .dynamic_tools(tools)
        .build();
    let answer = agent.prompt(prompt).max_turns(5).await?.output();
    println!("{answer}");

    connection.close().await;
    Ok(())
}
