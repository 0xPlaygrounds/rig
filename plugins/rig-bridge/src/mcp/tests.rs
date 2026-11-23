use std::sync::Arc;

use rig_harness::rig_core::completion::ToolDefinition;
use rig_harness::rig_core::message::ToolName;
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use super::{Outcome, Server};

fn tools(names: &[&str]) -> Arc<[Arc<ToolDefinition>]> {
    let schema = json!({ "type": "object", "properties": { "path": { "type": "string" } } });
    let named = names.iter().filter_map(|name| ToolName::new(*name).ok());
    named
        .map(|name| Arc::new(ToolDefinition::new(name, "Reads a file.", schema.clone())))
        .collect()
}

/// One HTTP request to the server: the status and the JSON body.
pub(crate) async fn post(url: &str, token: &str, body: &Value) -> std::io::Result<(u16, Value)> {
    let address = url.trim_start_matches("http://").trim_end_matches("/mcp");
    let mut stream = tokio::net::TcpStream::connect(address).await?;
    let body = body.to_string();
    let request = format!(
        "POST /mcp HTTP/1.1\r\nHost: {address}\r\nAuthorization: Bearer {token}\r\n\
         Content-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(request.as_bytes()).await?;
    let mut response = String::new();
    stream.read_to_string(&mut response).await?;
    let status = response
        .split_whitespace()
        .nth(1)
        .and_then(|status| status.parse().ok())
        .unwrap_or_default();
    let body = response.split_once("\r\n\r\n").map_or("", |(_, body)| body);
    Ok((status, serde_json::from_str(body).unwrap_or(Value::Null)))
}

pub(crate) fn request(id: u64, method: &str, params: Value) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params })
}

/// The names `tools/list` answered with.
fn names(listed: &Value) -> Vec<String> {
    let tools = listed.pointer("/result/tools").and_then(Value::as_array);
    let names = tools
        .into_iter()
        .flatten()
        .filter_map(|tool| tool.get("name")?.as_str());
    names.map(str::to_owned).collect()
}

#[tokio::test]
async fn a_door_lists_its_tools_and_only_to_its_token() {
    let listed = async {
        let server = Server::start().await?;
        let (entry, _calls) = server.open(tools(&["read"]));
        let (other, _other_calls) = server.open(tools(&["write"]));
        let list = request(1, "tools/list", json!({}));
        let (status, listed) = post(entry.url(), entry.token(), &list).await?;
        let (_, others) = post(other.url(), other.token(), &list).await?;
        let (refused, _) = post(entry.url(), "not-a-token", &list).await?;
        let schema = listed
            .pointer("/result/tools/0/inputSchema/properties/path/type")
            .cloned();
        std::io::Result::Ok((status, names(&listed), names(&others), refused, schema))
    };

    assert_eq!(
        listed.await.ok(),
        Some((
            200,
            vec!["read".to_owned()],
            vec!["write".to_owned()],
            401,
            Some(json!("string"))
        )),
    );
}

#[tokio::test]
async fn a_call_stays_open_until_the_session_answers_it() {
    let called = async {
        let server = Server::start().await?;
        let (entry, mut calls) = server.open(tools(&["read"]));
        let call = request(
            7,
            "tools/call",
            json!({ "name": "read", "arguments": { "path": "a.txt" } }),
        );
        let token = entry.token().to_owned();
        let url = entry.url().to_owned();
        let answered = tokio::spawn(async move { post(&url, &token, &call).await });
        let _entry = entry;
        let made = calls.recv().await;
        let made = made.ok_or_else(|| std::io::Error::other("no call reached the session"))?;
        let asked = (made.name.clone(), Value::Object(made.arguments.clone()));
        let open = !answered.is_finished();
        let _ = made.answer.send(Outcome {
            content: vec![json!({ "type": "text", "text": "hello" })],
            is_error: false,
        });
        let (_, response) = answered.await??;
        std::io::Result::Ok((
            asked,
            open,
            response.get("id").cloned(),
            response.get("result").cloned(),
        ))
    };

    assert_eq!(
        called.await.ok(),
        Some((
            ("read".to_owned(), json!({ "path": "a.txt" })),
            true,
            Some(json!(7)),
            Some(json!({ "content": [{ "type": "text", "text": "hello" }], "isError": false })),
        )),
    );
}

#[tokio::test]
async fn a_call_fails_when_its_tool_is_unknown_or_its_session_ended() {
    let failed = async {
        let server = Server::start().await?;
        let (entry, calls) = server.open(tools(&["read"]));
        let call = |name: &str| request(1, "tools/call", json!({ "name": name, "arguments": {} }));
        let (_, unknown) = post(entry.url(), entry.token(), &call("bash")).await?;
        drop(calls);
        let (_, ended) = post(entry.url(), entry.token(), &call("read")).await?;
        let token = entry.token().to_owned();
        let (after, _) = server.open(tools(&[]));
        drop(entry);
        let (closed, _) = post(after.url(), &token, &call("read")).await?;
        let is_error = |response: &Value| response.pointer("/result/isError").cloned();
        std::io::Result::Ok((is_error(&unknown), is_error(&ended), closed))
    };

    assert_eq!(
        failed.await.ok(),
        Some((Some(json!(true)), Some(json!(true)), 401))
    );
}

#[tokio::test]
async fn the_handshake_is_answered_and_a_notification_is_only_accepted() {
    let shook = async {
        let server = Server::start().await?;
        let (entry, _calls) = server.open(tools(&[]));
        let hello = request(0, "initialize", json!({ "protocolVersion": "2025-11-25" }));
        let (_, initialized) = post(entry.url(), entry.token(), &hello).await?;
        let notified = json!({ "jsonrpc": "2.0", "method": "notifications/initialized" });
        let (accepted, _) = post(entry.url(), entry.token(), &notified).await?;
        let (_, unknown) = post(
            entry.url(),
            entry.token(),
            &request(2, "resources/list", json!({})),
        )
        .await?;
        std::io::Result::Ok((
            initialized.pointer("/result/protocolVersion").cloned(),
            initialized.pointer("/result/capabilities/tools").is_some(),
            accepted,
            unknown.pointer("/error/code").cloned(),
        ))
    };

    assert_eq!(
        shook.await.ok(),
        Some((Some(json!("2025-11-25")), true, 202, Some(json!(-32601)))),
    );
}
