//! Parity between handrolled tools and the rmcp adapter (`McpTool`), driven
//! against an in-process MCP server over a duplex transport. Only the Gemini
//! side is cassette-recorded.
//!
//! The keystone is the shared-cassette pair: a native `Tool` agent and an
//! `McpTool` agent replay the *same* cassette, proving the two registration
//! paths produce identical wire requests. After the rmcp migration these
//! cassettes must keep replaying green.
#![cfg(feature = "rmcp")]

use std::sync::Arc;

use rig::client::CompletionClient;
use rig::completion::{Chat, Message};
use rig::providers::gemini;
use rig::tool::ToolDyn;
use rig::tool::rmcp::{McpClientHandler, McpTool};
use rig::tool::server::ToolServer;
use rmcp::handler::client::ClientHandler;
use rmcp::model::Implementation;
use rmcp::model::{
    CallToolRequestParams, CallToolResult, ClientInfo, Content, ErrorData, ListToolsResult,
    PaginatedRequestParams, ProtocolVersion, ServerCapabilities, ServerInfo, Tool,
};
use rmcp::service::RequestContext;
use rmcp::{RoleServer, ServerHandler, ServiceExt};
use serde_json::json;
use tokio::sync::RwLock;

use super::super::agent_run_support::{history_has_assistant_tool_call, tool_result_texts};
use super::super::support::with_gemini_cassette;
use super::super::tools_support::{
    BLUE_CODEWORD, CODEWORD_GUIDANCE, CountingAdd, FORCE_TOOLS_PREAMBLE, RED_PIXEL_PNG_BASE64,
};
use crate::support::assert_mentions_expected_number;

const SHARED_ADD_PROMPT: &str = "Use the add tool to calculate 19 + 23, then report the result.";

/// JSON schema identical to the native `CountingAdd` definition, so the
/// MCP-registered tool serializes to the same wire `ToolDefinition`.
fn add_input_schema() -> serde_json::Map<String, serde_json::Value> {
    let serde_json::Value::Object(map) = json!({
        "type": "object",
        "properties": {
            "x": { "type": "number", "description": "The first operand" },
            "y": { "type": "number", "description": "The second operand" }
        },
        "required": ["x", "y"]
    }) else {
        unreachable!("schema literal is an object")
    };
    map
}

fn subtract_input_schema() -> serde_json::Map<String, serde_json::Value> {
    add_input_schema()
}

fn codeword_input_schema() -> serde_json::Map<String, serde_json::Value> {
    let serde_json::Value::Object(map) = json!({
        "type": "object",
        "properties": {
            "team": { "type": "string", "description": "The team name, lowercase" }
        },
        "required": ["team"]
    }) else {
        unreachable!("schema literal is an object")
    };
    map
}

fn empty_input_schema() -> serde_json::Map<String, serde_json::Value> {
    let serde_json::Value::Object(map) = json!({
        "type": "object",
        "properties": {},
        "required": []
    }) else {
        unreachable!("schema literal is an object")
    };
    map
}

fn add_tool() -> Tool {
    Tool::new("add", "Add x and y together", Arc::new(add_input_schema()))
}

fn subtract_tool() -> Tool {
    Tool::new(
        "subtract",
        "Subtract y from x (i.e. x - y)",
        Arc::new(subtract_input_schema()),
    )
}

fn codeword_tool() -> Tool {
    Tool::new(
        "lookup_codeword",
        "Look up the secret codeword for a team.",
        Arc::new(codeword_input_schema()),
    )
}

fn badge_tool() -> Tool {
    Tool::new(
        "fetch_badge_image",
        "Fetch the attendee badge as an image the assistant must inspect.",
        Arc::new(empty_input_schema()),
    )
}

fn chime_tool() -> Tool {
    Tool::new(
        "play_chime",
        "Play the notification chime.",
        Arc::new(empty_input_schema()),
    )
}

fn operand(arguments: &Option<rmcp::model::JsonObject>, key: &str) -> i64 {
    arguments
        .as_ref()
        .and_then(|arguments| arguments.get(key))
        .and_then(serde_json::Value::as_f64)
        .unwrap_or_else(|| panic!("tool arguments should carry `{key}`: {arguments:?}")) as i64
}

/// In-process MCP server with a swappable tool list and fixed tool behaviors.
#[derive(Clone)]
struct ParityMcpServer {
    tools: Arc<RwLock<Vec<Tool>>>,
}

impl ParityMcpServer {
    fn new(tools: Vec<Tool>) -> Self {
        Self {
            tools: Arc::new(RwLock::new(tools)),
        }
    }

    async fn set_tools(&self, tools: Vec<Tool>) {
        *self.tools.write().await = tools;
    }
}

impl ServerHandler for ParityMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_protocol_version(ProtocolVersion::LATEST)
            .with_server_info(Implementation::new("parity-test-server", "0.1.0"))
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, ErrorData> {
        Ok(ListToolsResult::with_all_items(
            self.tools.read().await.clone(),
        ))
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, ErrorData> {
        match request.name.as_ref() {
            "add" => {
                let result = operand(&request.arguments, "x") + operand(&request.arguments, "y");
                Ok(CallToolResult::success(vec![Content::text(
                    result.to_string(),
                )]))
            }
            "subtract" => {
                let result = operand(&request.arguments, "x") - operand(&request.arguments, "y");
                Ok(CallToolResult::success(vec![Content::text(
                    result.to_string(),
                )]))
            }
            "lookup_codeword" => {
                let team = request
                    .arguments
                    .as_ref()
                    .and_then(|arguments| arguments.get("team"))
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                if team == "blue" {
                    Ok(CallToolResult::success(vec![Content::text(BLUE_CODEWORD)]))
                } else {
                    Ok(CallToolResult::error(vec![Content::text(
                        CODEWORD_GUIDANCE,
                    )]))
                }
            }
            "fetch_badge_image" => Ok(CallToolResult::success(vec![Content::image(
                RED_PIXEL_PNG_BASE64,
                "image/png",
            )])),
            "play_chime" => Ok(CallToolResult::success(vec![rmcp::model::Annotated::new(
                rmcp::model::RawContent::Audio(rmcp::model::RawAudioContent {
                    data: "UklGRgAAAABXQVZF".to_string(),
                    mime_type: "audio/wav".to_string(),
                }),
                None,
            )])),
            other => Err(ErrorData::invalid_params(
                format!("unknown tool {other}"),
                None,
            )),
        }
    }
}

#[derive(Clone)]
struct QuietClient;

impl ClientHandler for QuietClient {}

/// Serve `server` over an in-memory duplex transport and return the client's
/// running service (keep it alive for the duration of the test).
async fn connect_in_process(
    server: ParityMcpServer,
) -> rmcp::service::RunningService<rmcp::RoleClient, QuietClient> {
    let (client_to_server, server_from_client) = tokio::io::duplex(8192);
    let (server_to_client, client_from_server) = tokio::io::duplex(8192);

    tokio::spawn(async move {
        let service = server
            .serve((server_from_client, server_to_client))
            .await
            .expect("in-process MCP server should start");
        let _ = service.waiting().await;
    });

    QuietClient
        .serve((client_from_server, client_to_server))
        .await
        .expect("in-process MCP client should connect")
}

/// Native-tool half of the shared-cassette pair. In record mode both halves
/// record the same scenario (run with `--test-threads=1`); whichever wrote
/// last, both must replay green against it.
#[tokio::test]
async fn native_add_agent_drives_shared_cassette() {
    let add = CountingAdd::default();
    let counter = add.counter.clone();

    // Shares "rmcp_parity/shared_add_single_turn" with the MCP half below.
    with_gemini_cassette("rmcp_parity/shared_add_single_turn", |client| async move {
        let agent = client
            .agent(gemini::completion::GEMINI_2_5_FLASH)
            .preamble(FORCE_TOOLS_PREAMBLE)
            .temperature(0.0)
            .tool(add)
            .default_max_turns(3)
            .build();

        let mut history = Vec::<Message>::new();
        let response = agent
            .chat(SHARED_ADD_PROMPT, &mut history)
            .await
            .expect("native add prompt should succeed");

        assert_eq!(counter.count(), 1, "the native tool should execute once");
        assert_mentions_expected_number(&response, 42);
    })
    .await;
}

/// MCP half of the shared-cassette pair: identical preamble, prompt, and tool
/// schema, registered through `McpTool` instead of a native `Tool` impl. The
/// shared cassette proves the wire requests are identical.
#[tokio::test]
async fn mcp_add_agent_replays_shared_cassette() {
    // Intentionally the same cassette the native half records: identical
    // preamble, prompt, and schema must produce identical wire requests.
    with_gemini_cassette("rmcp_parity/shared_add_single_turn", |client| async move {
        let mcp_client = connect_in_process(ParityMcpServer::new(vec![add_tool()])).await;

        let agent = client
            .agent(gemini::completion::GEMINI_2_5_FLASH)
            .preamble(FORCE_TOOLS_PREAMBLE)
            .temperature(0.0)
            .rmcp_tool(add_tool(), mcp_client.peer().clone())
            .default_max_turns(3)
            .build();

        let mut history = Vec::<Message>::new();
        let response = agent
            .chat(SHARED_ADD_PROMPT, &mut history)
            .await
            .expect("MCP add prompt should succeed");

        let texts: Vec<String> = history.iter().flat_map(tool_result_texts).collect();
        assert_eq!(
            texts,
            vec!["42".to_string()],
            "the MCP tool result should match the native result"
        );
        assert_mentions_expected_number(&response, 42);
    })
    .await;
}

#[tokio::test]
async fn mcp_error_result_reaches_model_and_model_recovers() {
    with_gemini_cassette(
        "rmcp_parity/mcp_error_result_reaches_model_and_model_recovers",
        |client| async move {
            let mcp_client = connect_in_process(ParityMcpServer::new(vec![codeword_tool()])).await;

            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(
                    "You retrieve codewords with the lookup_codeword tool. If a lookup fails, read the error text and follow its guidance, then report the codeword you obtained.",
                )
                .temperature(0.0)
                .rmcp_tool(codeword_tool(), mcp_client.peer().clone())
                .default_max_turns(4)
                .build();

            let mut history = Vec::<Message>::new();
            let response = agent
                .chat("Look up the codeword for the red team.", &mut history)
                .await
                .expect("the run should survive an MCP is_error result");

            let texts: Vec<String> = history.iter().flat_map(tool_result_texts).collect();
            let error_text = texts
                .iter()
                .find(|text| text.contains(CODEWORD_GUIDANCE))
                .unwrap_or_else(|| {
                    panic!("the MCP error text should reach the model: {texts:?}")
                });
            assert!(
                error_text.contains("MCP tool error"),
                "is_error results should keep the McpToolError wrapper the model currently sees: {error_text:?}"
            );
            assert!(
                texts.iter().any(|text| text == BLUE_CODEWORD),
                "the recovered lookup should return the blue codeword: {texts:?}"
            );
            assert!(
                response.contains(BLUE_CODEWORD),
                "final answer should report the recovered codeword, got {response:?}"
            );
        },
    )
    .await;
}

#[tokio::test]
async fn mcp_image_content_flattens_to_data_url_text() {
    with_gemini_cassette(
        "rmcp_parity/mcp_image_content_flattens_to_data_url_text",
        |client| async move {
            let mcp_client = connect_in_process(ParityMcpServer::new(vec![badge_tool()])).await;

            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(
                    "You must call the fetch_badge_image tool, then acknowledge what it returned in one short sentence.",
                )
                .temperature(0.0)
                .rmcp_tool(badge_tool(), mcp_client.peer().clone())
                .default_max_turns(2)
                .build();

            let mut history = Vec::<Message>::new();
            let _response = agent
                .chat("Fetch the attendee badge image.", &mut history)
                .await
                .expect("MCP image tool prompt should succeed");

            // Divergence from native tools (which produce an image part):
            // the rmcp adapter flattens image content into a data-URL string.
            let expected = format!("data:image/png;base64,{RED_PIXEL_PNG_BASE64}");
            let texts: Vec<String> = history.iter().flat_map(tool_result_texts).collect();
            assert_eq!(
                texts,
                vec![expected],
                "MCP image content should flatten to a data-URL text tool result"
            );
        },
    )
    .await;
}

#[tokio::test]
async fn tool_list_changed_swaps_tools_between_turns() {
    let server = ParityMcpServer::new(vec![add_tool()]);

    with_gemini_cassette(
        "rmcp_parity/tool_list_changed_swaps_tools_between_turns",
        |client| async move {
            let tool_server_handle = ToolServer::new().run();

            let (client_to_server, server_from_client) = tokio::io::duplex(8192);
            let (server_to_client, client_from_server) = tokio::io::duplex(8192);

            let server_for_task = server.clone();
            let server_service = tokio::spawn(async move {
                server_for_task
                    .serve((server_from_client, server_to_client))
                    .await
                    .expect("in-process MCP server should start")
            });

            let handler = McpClientHandler::new(ClientInfo::default(), tool_server_handle.clone());
            let _mcp_service = handler
                .connect((client_from_server, client_to_server))
                .await
                .expect("MCP handler should connect and register tools");

            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(FORCE_TOOLS_PREAMBLE)
                .temperature(0.0)
                .tool_server_handle(tool_server_handle.clone())
                .default_max_turns(3)
                .build();

            let mut history = Vec::<Message>::new();
            let first = agent
                .chat("What is 19 + 23?", &mut history)
                .await
                .expect("first prompt should use the initial MCP tool list");
            assert_mentions_expected_number(&first, 42);
            assert!(
                history_has_assistant_tool_call(&history, "add"),
                "the initial MCP tool should be called: {history:?}"
            );

            // Swap the server's tool list and notify the client.
            server.set_tools(vec![subtract_tool()]).await;
            let server_service = server_service.await.expect("server task should be running");
            server_service
                .peer()
                .notify_tool_list_changed()
                .await
                .expect("tool list change notification should send");

            // The handler refreshes asynchronously; wait for the swap to land.
            let mut refreshed = false;
            for _ in 0..50 {
                let defs = tool_server_handle
                    .get_tool_defs(None)
                    .await
                    .expect("definitions should resolve");
                let names: Vec<&str> = defs.iter().map(|def| def.name.as_str()).collect();
                if names == ["subtract"] {
                    refreshed = true;
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            }
            assert!(refreshed, "the tool list should refresh to the swapped set");

            let mut history = Vec::<Message>::new();
            let second = agent
                .chat("What is 50 - 8?", &mut history)
                .await
                .expect("second prompt should use the swapped MCP tool list");
            assert_mentions_expected_number(&second, 42);
            assert!(
                history_has_assistant_tool_call(&history, "subtract"),
                "the swapped-in MCP tool should be called: {history:?}"
            );
        },
    )
    .await;
}

/// No cassette: pins the adapter's hard error on audio content.
#[tokio::test]
async fn mcp_audio_content_is_unsupported() {
    let mcp_client = connect_in_process(ParityMcpServer::new(vec![chime_tool()])).await;
    let tool = McpTool::from_mcp_server(chime_tool(), mcp_client.peer().clone());

    let error = tool
        .call("{}".to_string())
        .await
        .expect_err("audio content should be rejected");

    assert!(
        error.to_string().contains("audio"),
        "the error should name the unsupported audio content: {error}"
    );
}
