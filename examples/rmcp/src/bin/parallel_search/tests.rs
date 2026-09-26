use std::sync::Arc;

use axum::{
    Json, Router,
    extract::State,
    http::{HeaderMap, StatusCode},
    routing::post,
};
use rmcp::model::ContentBlock;
use tokio::sync::Mutex;

use super::*;

type Requests = Arc<Mutex<Vec<(HeaderMap, Value)>>>;

async fn mock_mcp(
    State(requests): State<Requests>,
    headers: HeaderMap,
    Json(request): Json<Value>,
) -> (StatusCode, Json<Value>) {
    requests.lock().await.push((headers, request.clone()));
    let Some(id) = request.get("id") else {
        return (StatusCode::ACCEPTED, Json(Value::Null));
    };
    let result = match request.get("method").and_then(Value::as_str) {
        Some("initialize") => json!({
            "protocolVersion": request.pointer("/params/protocolVersion"),
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "mock-search", "version": "1.0.0"}
        }),
        Some("tools/list") => json!({
            "tools": [
                {"name": "web_search", "inputSchema": {"type": "object"}},
                {"name": "web_fetch", "inputSchema": {"type": "object"}}
            ]
        }),
        Some("tools/call") => {
            let payload = match request.pointer("/params/name").and_then(Value::as_str) {
                Some("web_search") => json!({
                    "results": [{"url": "https://doc.rust-lang.org/book/ch04-01-what-is-ownership.html", "excerpts": ["Ownership rules"]}],
                    "session_id": "mock-conversation"
                }),
                Some("web_fetch") => json!({
                    "results": [{"url": "https://doc.rust-lang.org/book/ch04-01-what-is-ownership.html", "excerpts": ["Each value has an owner."]}],
                    "errors": []
                }),
                _ => Value::Null,
            };
            json!({"content": [], "structuredContent": payload})
        }
        _ => json!({}),
    };
    (
        StatusCode::OK,
        Json(json!({"jsonrpc": "2.0", "id": id, "result": result})),
    )
}

#[tokio::test]
async fn native_client_identifies_search_and_fetch_without_auth() -> anyhow::Result<()> {
    let requests = Requests::default();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let endpoint = format!("http://{}/mcp", listener.local_addr()?);
    let router = Router::new()
        .route("/mcp", post(mock_mcp))
        .with_state(requests.clone());
    let server = tokio::spawn(async move { axum::serve(listener, router).await });
    let client = ClientInfo::default().serve(transport(&endpoint)?).await?;
    let outcome = tokio::time::timeout(Duration::from_secs(5), search_and_fetch(&client)).await;
    client.cancel().await?;
    server.abort();
    outcome??;

    let captured = requests.lock().await;
    for (headers, _) in captured.iter() {
        ensure!(
            headers.get("user-agent").and_then(|h| h.to_str().ok()) == Some(USER_AGENT),
            "Request did not identify Rig"
        );
        ensure!(
            !headers.contains_key("authorization"),
            "Unexpected authorization header"
        );
        ensure!(
            !headers.contains_key("x-api-key"),
            "Unexpected API key header"
        );
    }
    let calls: Vec<_> = captured
        .iter()
        .filter(|(_, request)| request.get("method").and_then(Value::as_str) == Some("tools/call"))
        .map(|(_, request)| request)
        .collect();
    ensure!(calls.len() == 2, "Expected one search and one fetch");
    ensure!(
        calls.iter().any(
            |request| request.pointer("/params/name").and_then(Value::as_str) == Some("web_search")
        ),
        "Search request was not captured"
    );
    let fetch = calls
        .iter()
        .find(|request| {
            request.pointer("/params/name").and_then(Value::as_str) == Some("web_fetch")
        })
        .context("Fetch request was not captured")?;
    ensure!(
        fetch
            .pointer("/params/arguments/session_id")
            .and_then(Value::as_str)
            == Some("mock-conversation"),
        "Fetch did not reuse the search conversation"
    );
    ensure!(
        fetch
            .pointer("/params/arguments/urls/0")
            .and_then(Value::as_str)
            == Some("https://doc.rust-lang.org/book/ch04-01-what-is-ownership.html"),
        "Fetch did not use the first search result URL"
    );
    Ok(())
}

#[test]
fn tool_errors_are_not_treated_as_search_results() {
    let result = CallToolResult::error(vec![ContentBlock::text("Rate limited")]);
    assert!(payload(result).is_err());
}
