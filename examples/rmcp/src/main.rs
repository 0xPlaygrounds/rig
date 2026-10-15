//! Connect a Rig agent to an MCP server: this example serves one over
//! streamable HTTP, then uses its tools and resources.
//!
//! - **Owned connection**: [`McpConnection`] lists the server's tools once.
//!   The tools keep the connection open, and a resource is read again as
//!   context for every model call.
//! - **Managed tools**: [`McpClientHandler`] keeps a tool server in sync
//!   with the server's tool list as `notifications/tools/list_changed` arrives.
//!
//! Needs `OPENAI_API_KEY`. Run with `cargo run -p rmcp_example`.

use rig::{
    prelude::*,
    providers::openai::{self, OpenAI},
    tool::{
        DynamicTool,
        rmcp::{McpClientHandler, McpConnection},
        server::ToolServer,
    },
};
use rmcp::{
    RoleServer, ServerHandler, ServiceExt,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::*,
    schemars,
    service::RequestContext,
    tool, tool_handler, tool_router,
    transport::{
        StreamableHttpClientTransport,
        streamable_http_server::{StreamableHttpService, session::local::LocalSessionManager},
    },
};

const MEMO_URI: &str = "memo://launch";

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct SumRequest {
    pub a: i32,
    pub b: i32,
}

#[derive(Clone)]
pub struct Calculator {
    tool_router: ToolRouter<Calculator>,
}

#[tool_router]
impl Calculator {
    fn new() -> Self {
        Self {
            tool_router: Self::tool_router(),
        }
    }

    #[tool(description = "Calculate the sum of two numbers")]
    fn sum(
        &self,
        Parameters(SumRequest { a, b }): Parameters<SumRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        Ok(CallToolResult::success(vec![ContentBlock::text(
            (a + b).to_string(),
        )]))
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for Calculator {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(
            ServerCapabilities::builder()
                .enable_resources()
                .enable_tools()
                .build(),
        )
        .with_protocol_version(ProtocolVersion::LATEST)
        .with_server_info(Implementation::from_build_env())
    }

    async fn list_resources(
        &self,
        _request: Option<PaginatedRequestParams>,
        _: RequestContext<RoleServer>,
    ) -> Result<ListResourcesResult, ErrorData> {
        Ok(ListResourcesResult::with_all_items(vec![Resource::new(
            MEMO_URI,
            "launch memo",
        )]))
    }

    async fn read_resource(
        &self,
        request: ReadResourceRequestParams,
        _: RequestContext<RoleServer>,
    ) -> Result<ReadResourceResponse, ErrorData> {
        if request.uri != MEMO_URI {
            return Err(ErrorData::resource_not_found(
                "resource_not_found",
                Some(serde_json::json!({ "uri": request.uri })),
            ));
        }
        let memo = "The launch moved to the 14th. Ship 37 crates.";
        Ok(ReadResourceResult::new(vec![ResourceContents::text(memo, request.uri)]).into())
    }
}

/// Serves [`Calculator`] at `http://localhost:8080` until the process ends.
async fn serve() -> anyhow::Result<()> {
    let service = StreamableHttpService::new(
        || Ok(Calculator::new()),
        LocalSessionManager::default().into(),
        Default::default(),
    );
    let router = axum::Router::new().fallback_service(service);
    let listener = tokio::net::TcpListener::bind("localhost:8080").await?;
    tokio::spawn(async move { axum::serve(listener, router).await });
    Ok(())
}

fn client_config() -> ClientConfig {
    ClientConfig::new(
        ClientCapabilities::default(),
        Implementation::new("rig", env!("CARGO_PKG_VERSION")),
    )
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    serve().await?;
    let openai = OpenAI::from_env()?;

    // Owned connection: the tools keep it open, so nothing else must be held.
    let transport = StreamableHttpClientTransport::from_uri("http://localhost:8080");
    let connection = McpConnection::new(client_config().serve(transport).await?);
    let tools = connection
        .tools()
        .await?
        .into_iter()
        .map(DynamicTool::try_from)
        .collect::<Result<Vec<_>, _>>()?;
    let agent = AgentBuilder::new(openai.completion(openai::GPT_5_4_MINI))
        .preamble("Answer with the calculator's tools and the memo you are given.")
        .dynamic_tools(tools)
        .dynamic_context_handler(1, connection.resource_context([MEMO_URI]))
        .build();
    let answer = agent
        .prompt("How many crates ship, plus 5?")
        .max_turns(3)
        .await?
        .output();
    println!("Owned connection: {answer}");
    connection.close().await;

    // Managed tools: the handler re-registers the server's tools whenever
    // they change. The tool server owns them, so keep the connection.
    let tool_server = ToolServer::new().run();
    let transport = StreamableHttpClientTransport::from_uri("http://localhost:8080");
    let connection = McpClientHandler::new(client_config(), tool_server.clone())
        .connect(transport)
        .await?;
    let agent = AgentBuilder::new(openai.completion(openai::GPT_5_4_MINI))
        .preamble("Use the calculator's tools.")
        .tool_server_handle(tool_server)
        .build();
    let answer = agent.prompt("What is 2+5?").max_turns(3).await?.output();
    println!("Managed tools: {answer}");
    connection.close().await;

    Ok(())
}
