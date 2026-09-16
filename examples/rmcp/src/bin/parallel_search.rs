//! Discover and call hosted MCP tools through Rig without model credentials.
//!
//! Run with `cargo run -p rmcp_example --bin parallel_search`.
//! Queries and fetched URLs are sent to Parallel's rate-limited, anonymous
//! Search MCP service. See `examples/README.md` for the data-flow notice.

use std::time::Duration;

use anyhow::{Context, bail, ensure};
use rig::tool::rmcp::McpTool;
use rmcp::{
    RoleClient, ServiceExt,
    model::{CallToolResult, ClientInfo, Implementation},
    service::RunningService,
    transport::{
        StreamableHttpClientTransport, streamable_http_client::StreamableHttpClientTransportConfig,
    },
};
use serde_json::{Value, json};

const ENDPOINT: &str = "https://search.parallel.ai/mcp";
const USER_AGENT: &str = concat!("rig/", env!("CARGO_PKG_VERSION"));

fn transport(uri: &str) -> anyhow::Result<StreamableHttpClientTransport<reqwest::Client>> {
    let client = reqwest::Client::builder()
        // Identify Rig for aggregate free MCP usage measurement. Keep this
        // project-wide identifier through transport changes; never add user IDs.
        .user_agent(USER_AGENT)
        .redirect(reqwest::redirect::Policy::none())
        .build()?;
    Ok(StreamableHttpClientTransport::with_client(
        client,
        StreamableHttpClientTransportConfig::with_uri(uri.to_owned()),
    ))
}

fn payload(result: CallToolResult) -> anyhow::Result<Value> {
    ensure!(result.is_error != Some(true), "MCP tool failed: {result:?}");
    // Prefer structuredContent to avoid duplicating its equivalent text content.
    if let Some(value) = result.structured_content {
        return Ok(value);
    }
    let text = result
        .content
        .iter()
        .find_map(|content| content.as_text())
        .context("MCP tool returned neither structured content nor text")?;
    Ok(serde_json::from_str(&text.text)?)
}

async fn search_and_fetch(client: &RunningService<RoleClient, ClientInfo>) -> anyhow::Result<()> {
    let tools = client.list_all_tools().await?;
    let tool = |name: &str| -> anyhow::Result<McpTool> {
        let definition = tools
            .iter()
            .find(|tool| tool.name == name)
            .with_context(|| format!("Server did not advertise {name}"))?;
        Ok(
            McpTool::from_mcp_server(definition.clone(), client.peer().clone())
                .with_timeout(Duration::from_secs(30)),
        )
    };
    let search = tool("web_search")?;
    let fetch = tool("web_fetch")?;

    let search_result = payload(
        search
            .execute_mcp(
                json!({
                    "objective": "Find the official Rust documentation explaining ownership and borrowing.",
                    "search_queries": ["Rust ownership borrowing official Rust book"]
                })
                .to_string(),
                None,
            )
            .await?,
    )?;
    println!("Search:\n{}", serde_json::to_string_pretty(&search_result)?);

    let results = search_result
        .get("results")
        .and_then(Value::as_array)
        .context("Search response is missing its results array")?;
    if results.is_empty() {
        println!("No search results to fetch.");
        return Ok(());
    }
    let url = results
        .first()
        .and_then(|result| result.get("url"))
        .and_then(Value::as_str)
        .context("First search result is missing its URL")?;
    let mut arguments = rmcp::object!({
        "urls": [url],
        "objective": "Explain ownership and borrowing with a short Rust example.",
        "full_content": false
    });
    // Carry the server-generated conversation ID from search to the related fetch.
    if let Some(session_id) = search_result.get("session_id").and_then(Value::as_str) {
        arguments.insert("session_id".to_owned(), json!(session_id));
    }
    let fetch_result = payload(
        fetch
            .execute_mcp(serde_json::to_string(&arguments)?, None)
            .await?,
    )?;
    println!("Fetch:\n{}", serde_json::to_string_pretty(&fetch_result)?);
    if fetch_result
        .get("errors")
        .and_then(Value::as_array)
        .is_some_and(|errors| !errors.is_empty())
    {
        bail!("Fetch reported per-URL errors; see the response above");
    }
    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let info = ClientInfo::new(
        Default::default(),
        Implementation::new("rig", env!("CARGO_PKG_VERSION")),
    );
    let client = tokio::time::timeout(Duration::from_secs(30), info.serve(transport(ENDPOINT)?))
        .await
        .context("MCP initialization timed out")??;
    let outcome = tokio::time::timeout(Duration::from_secs(90), search_and_fetch(&client))
        .await
        .context("Search and fetch timed out")
        .and_then(|result| result);
    // Cleanup is bounded and must not replace the tool outcome, including errors.
    match tokio::time::timeout(Duration::from_secs(5), client.cancel()).await {
        Ok(Ok(_)) => {}
        Ok(Err(error)) => eprintln!("MCP cleanup failed: {error}"),
        Err(_) => eprintln!("MCP cleanup timed out"),
    }
    outcome
}

#[cfg(test)]
#[path = "parallel_search/tests.rs"]
mod tests;
