//! Owned connections: tools keep theirs open, model names apart from MCP
//! names, call-derived `_meta`, and resources as agent context.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use rig_agent::agent::AgentBuilder;
use rig_agent::test_utils::MockCompletionModel;
use rig_agent::tool::server::ToolServer;
use rig_agent::tool::{DynamicTool, ToolContext};
use rig_core::message::ToolName;
use rmcp::model::*;
use rmcp::service::RequestContext;
use rmcp::{RoleServer, ServerHandler, ServiceExt};
use serde_json::json;
use tokio::task::JoinHandle;

use crate::{McpClientHandler, McpConnection, McpMeta, McpTool};

/// A server whose one tool, `fetch`, answers with the server's label and the
/// tool name it was called by, and whose one resource is a memo.
#[derive(Clone)]
struct LabelledServer {
    label: &'static str,
    calls: Arc<Mutex<Vec<(Option<JsonObject>, RequestMetaObject)>>>,
}

impl LabelledServer {
    fn new(label: &'static str) -> Self {
        Self {
            label,
            calls: Arc::default(),
        }
    }
}

impl ServerHandler for LabelledServer {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(
            ServerCapabilities::builder()
                .enable_tools()
                .enable_resources()
                .build(),
        )
        .with_protocol_version(ProtocolVersion::LATEST)
        .with_server_info(Implementation::new(self.label, "0.1.0"))
    }

    async fn list_tools(
        &self,
        _: Option<PaginatedRequestParams>,
        _: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, ErrorData> {
        Ok(ListToolsResult::with_all_items(vec![Tool::new(
            "fetch",
            "Fetch a thing",
            Arc::new(serde_json::Map::new()),
        )]))
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, ErrorData> {
        if let Ok(mut calls) = self.calls.lock() {
            calls.push((request.arguments.clone(), context.meta.clone()));
        }
        let text = format!("{} {}", self.label, request.name);
        Ok(CallToolResult::success(vec![ContentBlock::text(text)]).into())
    }

    async fn list_resources(
        &self,
        _: Option<PaginatedRequestParams>,
        _: RequestContext<RoleServer>,
    ) -> Result<ListResourcesResult, ErrorData> {
        Ok(ListResourcesResult::with_all_items(vec![Resource::new(
            "memo://insights",
            "insights",
        )]))
    }

    async fn read_resource(
        &self,
        request: ReadResourceRequestParams,
        _: RequestContext<RoleServer>,
    ) -> Result<ReadResourceResponse, ErrorData> {
        let memo = format!("{} memo: ship on Friday", self.label);
        Ok(ReadResourceResult::new(vec![ResourceContents::text(memo, request.uri)]).into())
    }
}

/// Connects to `server`; its task ends once the client side closes.
async fn connect(server: LabelledServer) -> (McpConnection, JoinHandle<()>) {
    let (c2s, sfc) = tokio::io::duplex(8192);
    let (s2c, cfs) = tokio::io::duplex(8192);
    let server_task = tokio::spawn(async move {
        let running = server.serve((sfc, s2c)).await.expect("server start");
        running.waiting().await.expect("server error");
    });
    let client = ClientConfig::default()
        .serve((cfs, c2s))
        .await
        .expect("client connect");
    (McpConnection::new(client), server_task)
}

#[tokio::test]
async fn tools_keep_their_connection_open_until_they_are_dropped() {
    let (connection, server_task) = connect(LabelledServer::new("only")).await;
    let mut tools = connection.tools().await.expect("tools").into_iter();
    let fetch = DynamicTool::try_from(tools.next().expect("one tool")).expect("tool name");
    drop(connection);
    tokio::time::sleep(Duration::from_millis(50)).await;

    let output = rig_agent::tool::RegisteredTool::from_dynamic(fetch.clone())
        .execute("{}".to_owned(), &mut ToolContext::new())
        .await
        .into_result()
        .expect("still connected");
    assert_eq!(output.render(), "only fetch");

    drop(fetch);
    tokio::time::timeout(Duration::from_secs(2), server_task)
        .await
        .expect("the last tool closed the connection")
        .expect("server task");
}

#[tokio::test]
async fn model_names_differ_from_the_mcp_names_sent_to_each_server() {
    let (first, first_task) = connect(LabelledServer::new("first")).await;
    let (second, second_task) = connect(LabelledServer::new("second")).await;
    let alias = |tool: McpTool, name: &str| {
        let tool = tool.with_name(ToolName::new(name).expect("name"));
        assert_eq!(
            (tool.name(), tool.definition().name.as_ref()),
            (name, "fetch")
        );
        DynamicTool::try_from(tool).expect("tool name")
    };
    let handle = ToolServer::new()
        .dynamic_tool(alias(
            first.tools().await.expect("tools").remove(0),
            "first_fetch",
        ))
        .dynamic_tool(alias(
            second.tools().await.expect("tools").remove(0),
            "second_fetch",
        ))
        .run();

    let names = handle
        .tool_defs(None)
        .await
        .expect("definitions")
        .into_iter()
        .map(|definition| definition.name.to_string())
        .collect::<Vec<_>>();
    assert_eq!(names, ["first_fetch", "second_fetch"]);
    for (name, answer) in [
        ("first_fetch", "first fetch"),
        ("second_fetch", "second fetch"),
    ] {
        let result = handle
            .toolset()
            .execute(name, "{}", &mut ToolContext::new())
            .await;
        assert_eq!(result.output().render(), answer);
    }

    first.close().await;
    second.close().await;
    first_task.abort();
    second_task.abort();
}

#[tokio::test]
async fn call_meta_is_derived_from_the_arguments_sent() {
    let server = LabelledServer::new("signed");
    let calls = server.calls.clone();
    let (connection, server_task) = connect(server).await;
    let signed = Arc::new(Mutex::new(Vec::new()));
    let seen = signed.clone();
    let tool = connection
        .tools()
        .await
        .expect("tools")
        .remove(0)
        .with_call_meta(move |call| {
            let arguments = call.arguments.cloned();
            let signature = format!("{}:{}", call.tool, json!(arguments));
            if let Ok(mut seen) = seen.lock() {
                seen.push(arguments);
            }
            let mut meta = RequestMetaObject::new();
            meta.insert("signature".into(), json!(signature));
            Some(meta)
        });
    let handle = ToolServer::new()
        .dynamic_tool(DynamicTool::try_from(tool).expect("tool name"))
        .run();

    let mut run_meta = RequestMetaObject::new();
    run_meta.insert("authorization".into(), json!("Bearer t"));
    run_meta.insert("signature".into(), json!("stale"));
    let mut context = ToolContext::new();
    context.insert(McpMeta(run_meta)).expect("context");
    let result = handle
        .toolset()
        .execute("fetch", r#"{"q":"rig"}"#, &mut context)
        .await;
    assert!(result.is_success());

    let calls = calls.lock().expect("calls").clone();
    let [(arguments, meta)] = calls.as_slice() else {
        panic!("one call, got {calls:?}");
    };
    assert_eq!(meta.get("authorization"), Some(&json!("Bearer t")));
    assert_eq!(meta.get("signature"), Some(&json!(r#"fetch:{"q":"rig"}"#)));
    assert_eq!(
        signed.lock().expect("signed").as_slice(),
        std::slice::from_ref(arguments)
    );

    connection.close().await;
    server_task.abort();
}

#[tokio::test]
async fn managed_tools_derive_call_meta_too() {
    let server = LabelledServer::new("managed");
    let calls = server.calls.clone();
    let (c2s, sfc) = tokio::io::duplex(8192);
    let (s2c, cfs) = tokio::io::duplex(8192);
    let server_task = tokio::spawn(async move {
        let running = server.serve((sfc, s2c)).await.expect("server start");
        running.waiting().await.expect("server error");
    });
    let handle = ToolServer::new().run();
    let connection = McpClientHandler::new(ClientConfig::default(), handle.clone())
        .with_call_meta(|call| {
            let mut meta = RequestMetaObject::new();
            meta.insert("tool".into(), json!(call.tool));
            Some(meta)
        })
        .connect((cfs, c2s))
        .await
        .expect("connect");

    let result = handle
        .toolset()
        .execute("fetch", "{}", &mut ToolContext::new())
        .await;
    assert!(result.is_success());
    let calls = calls.lock().expect("calls").clone();
    assert!(
        matches!(calls.as_slice(), [(_, meta)] if meta.get("tool") == Some(&json!("fetch"))),
        "{calls:?}"
    );

    connection.close().await;
    server_task.abort();
}

#[tokio::test]
async fn resources_are_listed_read_and_given_as_agent_context() {
    let (connection, server_task) = connect(LabelledServer::new("docs")).await;
    let uris = connection
        .resources()
        .await
        .expect("resources")
        .into_iter()
        .map(|resource| resource.uri.clone())
        .collect::<Vec<_>>();
    assert_eq!(uris, ["memo://insights"]);
    let contents = connection.read_resource("memo://insights").await;
    assert!(matches!(
        contents.expect("read").as_slice(),
        [ResourceContents::TextResourceContents { text, .. }] if text == "docs memo: ship on Friday"
    ));

    let model = MockCompletionModel::text("Friday.");
    let agent = AgentBuilder::new(model.clone())
        .dynamic_context_handler(4, connection.resource_context(uris))
        .build();
    agent.prompt("When do we ship?").await.expect("prompt");
    let requests = model.requests();
    let documents = rig_core::test_utils::sent_documents(requests.first().expect("one request"));
    assert!(
        documents.iter().any(|(id, text)| id == "memo://insights"
            && text.contains("docs memo: ship on Friday")),
        "{documents:?}"
    );

    connection.close().await;
    server_task.abort();
}
