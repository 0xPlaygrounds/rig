//! An example of how you can use `rmcp` with Rig to create an MCP friendly agent.
use std::sync::Arc;

use rmcp::ServiceExt;

use rig::{
    client::{CompletionClient, ProviderClient},
    completion::Prompt,
    providers::openai,
};
use rmcp::{
    RoleServer, ServerHandler,
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
        RawResource::new(uri, name.to_string()).no_annotation()
    }

    // #[tool(description = "Increment the counter by 1")]
    // async fn increment(&self) -> Result<CallToolResult, ErrorData> {
    //     let mut counter = self.counter.lock().await;
    //     *counter += 1;
    //     Ok(CallToolResult::success(vec![Content::text(
    //         counter.to_string(),
    //     )]))
    // }

    // #[tool(description = "Decrement the counter by 1")]
    // async fn decrement(&self) -> Result<CallToolResult, ErrorData> {
    //     let mut counter = self.counter.lock().await;
    //     *counter -= 1;
    //     Ok(CallToolResult::success(vec![Content::text(
    //         counter.to_string(),
    //     )]))
    // }

    // #[tool(description = "Get the current counter value")]
    // async fn get_value(&self) -> Result<CallToolResult, ErrorData> {
    //     let counter = self.counter.lock().await;
    //     Ok(CallToolResult::success(vec![Content::text(
    //         counter.to_string(),
    //     )]))
    // }

    // #[tool(description = "Say hello to the client")]
    // fn say_hello(&self) -> Result<CallToolResult, ErrorData> {
    //     Ok(CallToolResult::success(vec![Content::text("hello")]))
    // }

    // #[tool(description = "Repeat what you say")]
    // fn echo(&self, Parameters(object): Parameters<JsonObject>) -> Result<CallToolResult, ErrorData> {
    //     Ok(CallToolResult::success(vec![Content::text(
    //         serde_json::Value::Object(object).to_string(),
    //     )]))
    // }

    #[tool(description = "Calculate the sum of two numbers")]
    fn sum(
        &self,
        Parameters(StructRequest { a, b }): Parameters<StructRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        Ok(CallToolResult::success(vec![Content::text(
            (a + b).to_string(),
        )]))
    }
}
#[tool_handler]
impl ServerHandler for Counter {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: ProtocolVersion::V_2024_11_05,
            capabilities: ServerCapabilities::builder()
                .enable_resources()
                .enable_tools()
                .build(),
            server_info: Implementation::from_build_env(),
            instructions: Some("This server provides a counter tool that can increment and decrement values. The counter starts at 0 and can be modified using the 'increment' and 'decrement' tools. Use 'get_value' to check the current count.".to_string()),
        }
    }

    async fn list_resources(
        &self,
        _request: Option<PaginatedRequestParam>,
        _: RequestContext<RoleServer>,
    ) -> Result<ListResourcesResult, ErrorData> {
        Ok(ListResourcesResult {
            resources: vec![
                self._create_resource_text("str:////Users/to/some/path/", "cwd"),
                self._create_resource_text("memo://insights", "memo-name"),
            ],
            next_cursor: None,
        })
    }

    async fn read_resource(
        &self,
        ReadResourceRequestParam { uri }: ReadResourceRequestParam,
        _: RequestContext<RoleServer>,
    ) -> Result<ReadResourceResult, ErrorData> {
        match uri.as_str() {
            "str:////Users/to/some/path/" => {
                let cwd = "/Users/to/some/path/";
                Ok(ReadResourceResult {
                    contents: vec![ResourceContents::text(cwd, uri)],
                })
            }
            "memo://insights" => {
                let memo = "Business Intelligence Memo\n\nAnalysis has revealed 5 key insights ...";
                Ok(ReadResourceResult {
                    contents: vec![ResourceContents::text(memo, uri)],
                })
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
        _request: Option<PaginatedRequestParam>,
        _: RequestContext<RoleServer>,
    ) -> Result<ListResourceTemplatesResult, ErrorData> {
        Ok(ListResourceTemplatesResult {
            next_cursor: None,
            resource_templates: Vec::new(),
        })
    }

    async fn initialize(
        &self,
        _request: InitializeRequestParam,
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

    let transport =
        rmcp::transport::StreamableHttpClientTransport::from_uri("http://localhost:8080");

    let client_info = ClientInfo {
        protocol_version: Default::default(),
        capabilities: ClientCapabilities::default(),
        client_info: Implementation {
            name: "rig-core".to_string(),
            version: "0.13.0".to_string(),
        },
    };

    let client = client_info.serve(transport).await.inspect_err(|e| {
        tracing::error!("client error: {:?}", e);
    })?;

    // Initialize
    let server_info = client.peer_info();
    tracing::info!("Connected to server: {server_info:#?}");

    // List tools
    let tools: Vec<Tool> = client.list_tools(Default::default()).await?.tools;

    // takes the `OPENAI_API_KEY` as an env var on usage
    let openai_client = openai::Client::from_env();
    let agent = openai_client
        .agent("gpt-4o")
        .preamble("You are a helpful assistant who has access to a number of tools from an MCP server designed to be used for incrementing and decrementing a counter.")
        .rmcp_tools(tools, client.peer().to_owned())
        .build();

    let res = agent.prompt("What is 2+5?").multi_turn(2).await.unwrap();

    println!("GPT-4o: {res}");

    Ok(())
}
