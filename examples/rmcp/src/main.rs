//! An example of how you can use `rmcp` with Rig to create an MCP friendly agent.
//!
//! It starts an in-process `rmcp` MCP server, connects an `rmcp` client to it,
//! and wraps the server's tools with the host-owned [`McpToolset`]
//! (`rig::tool::mcp`). Each MCP tool definition becomes a
//! [`PortableDynamicTool`] record whose callback closes over the shared
//! toolset and dispatches the call by name; the records are registered on the
//! agent with `.dynamic_tools(...)`. The toolset is host-owned: call
//! [`McpToolset::refresh`] whenever you want to pick up server-side tool-list
//! changes (the push-based `tools/list_changed` reconciliation of the classic
//! runtime is gone).
use std::sync::Arc;

use rig::{
    prelude::*,
    providers::openai,
    tool::{PortableDynamicTool, ToolExecutionError, mcp::McpToolset},
};
use rmcp::{
    RoleServer, ServerHandler, ServiceExt,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::*,
    schemars,
    service::RequestContext,
    tool, tool_handler, tool_router,
};
use serde_json::json;
use tokio::sync::Mutex;

use hyper_util::{
    rt::{TokioExecutor, TokioIo},
    server::conn::auto::Builder,
    service::TowerToHyperService,
};
use rmcp::transport::streamable_http_server::{
    StreamableHttpService, session::local::LocalSessionManager,
};

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct StructRequest {
    pub a: i32,
    pub b: i32,
}

#[derive(Clone)]
pub struct Counter {
    pub counter: Arc<Mutex<i32>>,
    tool_router: ToolRouter<Counter>,
}

impl Default for Counter {
    fn default() -> Self {
        Self::new()
    }
}

#[tool_router]
impl Counter {
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self {
            counter: Arc::new(Mutex::new(0)),
            tool_router: Self::tool_router(),
        }
    }

    fn _create_resource_text(&self, uri: &str, name: &str) -> Resource {
        Resource::new(uri, name.to_string())
    }

    // #[tool(description = "Increment the counter by 1")]
    // async fn increment(&self) -> Result<CallToolResult, ErrorData> {
    //     let mut counter = self.counter.lock().await;
    //     *counter += 1;
    //     Ok(CallToolResult::success(vec![ContentBlock::text(
    //         counter.to_string(),
    //     )]))
    // }

    // #[tool(description = "Decrement the counter by 1")]
    // async fn decrement(&self) -> Result<CallToolResult, ErrorData> {
    //     let mut counter = self.counter.lock().await;
    //     *counter -= 1;
    //     Ok(CallToolResult::success(vec![ContentBlock::text(
    //         counter.to_string(),
    //     )]))
    // }

    // #[tool(description = "Get the current counter value")]
    // async fn get_value(&self) -> Result<CallToolResult, ErrorData> {
    //     let counter = self.counter.lock().await;
    //     Ok(CallToolResult::success(vec![ContentBlock::text(
    //         counter.to_string(),
    //     )]))
    // }

    // #[tool(description = "Say hello to the client")]
    // fn say_hello(&self) -> Result<CallToolResult, ErrorData> {
    //     Ok(CallToolResult::success(vec![ContentBlock::text("hello")]))
    // }

    // #[tool(description = "Repeat what you say")]
    // fn echo(&self, Parameters(object): Parameters<JsonObject>) -> Result<CallToolResult, ErrorData> {
    //     Ok(CallToolResult::success(vec![ContentBlock::text(
    //         serde_json::Value::Object(object).to_string(),
    //     )]))
    // }

    #[tool(description = "Calculate the sum of two numbers")]
    fn sum(
        &self,
        Parameters(StructRequest { a, b }): Parameters<StructRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        Ok(CallToolResult::success(vec![ContentBlock::text(
            (a + b).to_string(),
        )]))
    }
}
#[tool_handler(router = self.tool_router)]
impl ServerHandler for Counter {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(
            ServerCapabilities::builder()
                .enable_resources()
                .enable_tools()
                .build(),
        )
        .with_protocol_version(ProtocolVersion::LATEST)
        .with_server_info(Implementation::from_build_env())
        .with_instructions("This server provides a counter tool that can increment and decrement values. The counter starts at 0 and can be modified using the 'increment' and 'decrement' tools. Use 'get_value' to check the current count.")
    }

    async fn list_resources(
        &self,
        _request: Option<PaginatedRequestParams>,
        _: RequestContext<RoleServer>,
    ) -> Result<ListResourcesResult, ErrorData> {
        Ok(ListResourcesResult {
            resources: vec![
                self._create_resource_text("str:////Users/to/some/path/", "cwd"),
                self._create_resource_text("memo://insights", "memo-name"),
            ],
            next_cursor: None,
            meta: None,
        })
    }

    async fn read_resource(
        &self,
        ReadResourceRequestParams { uri, .. }: ReadResourceRequestParams,
        _: RequestContext<RoleServer>,
    ) -> Result<ReadResourceResult, ErrorData> {
        match uri.as_str() {
            "str:////Users/to/some/path/" => {
                let cwd = "/Users/to/some/path/";
                Ok(ReadResourceResult::new(vec![ResourceContents::text(
                    cwd, uri,
                )]))
            }
            "memo://insights" => {
                let memo = "Business Intelligence Memo\n\nAnalysis has revealed 5 key insights ...";
                Ok(ReadResourceResult::new(vec![ResourceContents::text(
                    memo, uri,
                )]))
            }
            _ => Err(ErrorData::resource_not_found(
                "resource_not_found",
                Some(json!({
                    "uri": uri
                })),
            )),
        }
    }

    async fn list_resource_templates(
        &self,
        _request: Option<PaginatedRequestParams>,
        _: RequestContext<RoleServer>,
    ) -> Result<ListResourceTemplatesResult, ErrorData> {
        Ok(ListResourceTemplatesResult {
            next_cursor: None,
            resource_templates: Vec::new(),
            meta: None,
        })
    }

    async fn initialize(
        &self,
        _request: InitializeRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<InitializeResult, ErrorData> {
        if let Some(http_request_part) = context.extensions.get::<axum::http::request::Parts>() {
            let initialize_headers = &http_request_part.headers;
            let initialize_uri = &http_request_part.uri;
            tracing::info!(?initialize_headers, %initialize_uri, "initialize from http server");
        }
        Ok(self.get_info())
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let service = TowerToHyperService::new(StreamableHttpService::new(
        || Ok(Counter::new()),
        LocalSessionManager::default().into(),
        Default::default(),
    ));
    let listener = tokio::net::TcpListener::bind("localhost:8080").await?;

    tokio::spawn({
        let service = service.clone();
        async move {
            loop {
                tokio::select! {
                    _ = tokio::signal::ctrl_c() => {
                        println!("Received Ctrl+C, shutting down");
                        break;
                    }
                    accept = listener.accept() => {
                        match accept {
                            Ok((stream, _addr)) => {
                                let io = TokioIo::new(stream);
                                let service = service.clone();

                                tokio::spawn(async move {
                                    if let Err(e) = Builder::new(TokioExecutor::default())
                                        .serve_connection(io, service)
                                        .await
                                    {
                                        eprintln!("Connection error: {e:?}");
                                    }
                                });
                            }
                            Err(e) => {
                                eprintln!("Accept error: {e:?}");
                            }
                        }
                    }
                }
            }
        }
    });

    let client_info = ClientInfo::new(
        ClientCapabilities::default(),
        Implementation::new("rig-core", env!("CARGO_PKG_VERSION")),
    );

    let transport =
        rmcp::transport::StreamableHttpClientTransport::from_uri("http://localhost:8080");

    // Connect the rmcp client, then snapshot the server's tools into a
    // host-owned toolset. Call `toolset.refresh()` (behind a lock) whenever
    // you want to pick up server-side tool-list changes.
    let mcp_service = client_info.serve(transport).await.inspect_err(|e| {
        tracing::error!("MCP client error: {:?}", e);
    })?;

    let server_info = mcp_service.peer_info();
    tracing::info!("Connected to server: {server_info:#?}");

    let toolset = Arc::new(McpToolset::from_sink(mcp_service.peer().clone()).await?);

    // Wrap each MCP tool definition as a dynamic tool record whose callback
    // closes over the shared toolset and dispatches the call by name.
    let mcp_tools: Vec<PortableDynamicTool> = toolset
        .definitions()
        .into_iter()
        .map(|definition| {
            let toolset = Arc::clone(&toolset);
            let tool_name = definition.name.clone();
            PortableDynamicTool::new(
                definition.name,
                definition.description,
                definition.parameters,
                move |args| {
                    let toolset = Arc::clone(&toolset);
                    let tool_name = tool_name.clone();
                    async move {
                        let outcome = toolset
                            .call(&tool_name, &args, None)
                            .await
                            .map_err(ToolExecutionError::from_error)?;
                        if let Some(error) = outcome.result.error() {
                            return Err(error.clone());
                        }
                        Ok(outcome.result.output().clone())
                    }
                },
            )
        })
        .collect();

    let client = openai::Client::from_env()?;
    let agent = client
        .agent(openai::GPT_4O)
        .preamble("You are a helpful assistant who has access to a number of tools from an MCP server designed to be used for incrementing and decrementing a counter.")
        .dynamic_tools(mcp_tools)
        .build();

    let res = agent
        .runner("What is 2+5?")
        .max_turns(2)
        .run()
        .await?
        .output;

    println!("GPT-4o: {res}");

    Ok(())
}
