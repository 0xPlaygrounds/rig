#![cfg(not(target_family = "wasm"))]

use std::{
    error::Error,
    sync::{Arc, Mutex, MutexGuard},
    time::Duration,
};

use rig_agent::tool::server::ToolServer;
use rig_core::{
    effect::tool_key,
    message::ToolName,
    tool::{DynamicTool, ToolContext, ToolErrorKind, ToolOutput},
};
use rig_rmcp::{
    DEFAULT_MCP_TOOL_TIMEOUT, McpCallToolResult, McpMeta, McpResponseMeta, McpStructuredContent,
    McpTool, Meta,
};
use rmcp::{
    RoleServer, ServerHandler, ServiceExt,
    model::{
        CallToolRequestParams, CallToolResult, ClientInfo, ContentBlock, ErrorData, Implementation,
        JsonObject, ProtocolVersion, ServerCapabilities, ServerInfo, Tool,
    },
    service::{RequestContext, RunningService},
};
use serde_json::json;
use tokio::task::JoinHandle;

type TestResult<T = ()> = Result<T, Box<dyn Error + Send + Sync>>;

fn check(condition: bool, message: &str) -> TestResult {
    if condition {
        Ok(())
    } else {
        Err(std::io::Error::other(message).into())
    }
}

fn required<T>(value: Option<T>, message: &str) -> TestResult<T> {
    value.ok_or_else(|| std::io::Error::other(message).into())
}

fn lock<T>(value: &Mutex<T>) -> MutexGuard<'_, T> {
    value
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[derive(Clone)]
struct Observed {
    name: String,
    arguments: Option<JsonObject>,
    meta: Meta,
}

#[derive(Clone)]
struct NamedServer {
    raw_name: String,
    response: CallToolResult,
    seen: Arc<Mutex<Vec<Observed>>>,
}

impl ServerHandler for NamedServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_protocol_version(ProtocolVersion::LATEST)
            .with_server_info(Implementation::new("alias-test", "0.1.0"))
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, ErrorData> {
        let name = request.name.to_string();
        lock(&self.seen).push(Observed {
            name: name.clone(),
            arguments: request.arguments,
            meta: context.meta,
        });
        if name != self.raw_name {
            return Err(ErrorData::invalid_params(
                "wire name differs from server definition",
                None,
            ));
        }
        Ok(self.response.clone())
    }
}

struct Fixture {
    tool: McpTool,
    seen: Arc<Mutex<Vec<Observed>>>,
    _client: RunningService<rmcp::RoleClient, ClientInfo>,
    server: JoinHandle<TestResult>,
}

impl Drop for Fixture {
    fn drop(&mut self) {
        self.server.abort();
    }
}

async fn fixture(raw_name: &str, response: CallToolResult) -> TestResult<Fixture> {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let server = NamedServer {
        raw_name: raw_name.into(),
        response,
        seen: seen.clone(),
    };
    let (client_to_server, server_from_client) = tokio::io::duplex(8192);
    let (server_to_client, client_from_server) = tokio::io::duplex(8192);
    let server = tokio::spawn(async move {
        let running = server.serve((server_from_client, server_to_client)).await?;
        running.waiting().await?;
        Ok(())
    });
    let client = tokio::time::timeout(
        Duration::from_secs(5),
        ClientInfo::default().serve((client_from_server, client_to_server)),
    )
    .await??;
    let schema: JsonObject = serde_json::from_value(json!({
        "type": "object", "properties": {"path": {"type": "string"}}
    }))?;
    let definition = Tool::new(raw_name.to_owned(), "fixture tool", Arc::new(schema));
    let tool = McpTool::from_mcp_server(definition, client.peer().clone());
    Ok(Fixture {
        tool,
        seen,
        _client: client,
        server,
    })
}

fn success(marker: &str) -> CallToolResult {
    CallToolResult::success(vec![ContentBlock::text(marker)])
}

#[tokio::test]
async fn sanitized_alias_is_advertised_and_looked_up_while_wire_calls_keep_raw_name() -> TestResult
{
    let fixture = fixture("files/read", success("contents")).await?;
    let adapter = fixture
        .tool
        .clone()
        .with_model_name(ToolName::new("files_read")?);
    check(
        adapter.definition().name == "files/read",
        "alias mutated the MCP definition",
    )?;
    check(
        adapter.timeout() == Some(DEFAULT_MCP_TOOL_TIMEOUT),
        "alias changed the default timeout",
    )?;
    let configured = fixture
        .tool
        .clone()
        .with_timeout(Duration::from_secs(17))
        .with_model_name(ToolName::new("configured_read")?)
        .clone();
    check(
        configured.timeout() == Some(Duration::from_secs(17)),
        "alias or clone changed a configured timeout",
    )?;
    let configured: DynamicTool = configured.try_into()?;
    check(
        configured.name().as_str() == "configured_read",
        "clone lost the configured alias",
    )?;
    let cloned = adapter.clone();
    check(
        cloned.timeout() == adapter.timeout(),
        "clone lost the timeout",
    )?;
    let dynamic: DynamicTool = cloned.try_into()?;
    check(
        dynamic.name().as_str() == "files_read",
        "clone lost the alias",
    )?;
    check(
        dynamic.definition().name.as_str() == "files_read",
        "advertised definition lost the alias",
    )?;
    check(
        dynamic.handler().descriptor().key == tool_key("files_read"),
        "handler still uses the raw name",
    )?;
    check(
        dynamic.definition().parameters == adapter.definition().schema_as_json_value(),
        "alias changed argument schema",
    )?;
    let handle = ToolServer::new().dynamic_tool(dynamic).run();
    let names = handle
        .static_tool_defs()
        .into_iter()
        .map(|definition| definition.name.to_string())
        .collect::<Vec<_>>();
    check(
        names == ["files_read"],
        "catalog did not advertise the alias",
    )?;
    check(
        handle.toolset().contains("files_read") && !handle.toolset().contains("files/read"),
        "lookup is not keyed by alias",
    )?;
    let arguments = json!({"path": "report.txt"});
    let result = tokio::time::timeout(
        Duration::from_secs(5),
        handle.execute(
            "files_read",
            &arguments.to_string(),
            &mut ToolContext::new(),
        ),
    )
    .await?;
    check(
        result.output().as_text() == Some("contents") && result.is_success(),
        "aliased execution did not reach the server",
    )?;
    let direct = tokio::time::timeout(
        Duration::from_secs(5),
        adapter.execute_mcp("{}".into(), None),
    )
    .await??;
    check(
        direct == success("contents"),
        "alias changed direct execute_mcp behavior",
    )?;
    let seen = lock(&fixture.seen).clone();
    check(
        seen.len() == 2 && seen.iter().all(|call| call.name == "files/read"),
        "alias leaked into wire calls",
    )?;
    check(
        seen.first()
            .is_some_and(|call| call.arguments.as_ref() == arguments.as_object()),
        "aliased execution changed arguments",
    )?;
    Ok(())
}

#[tokio::test]
async fn aliases_keep_same_named_tools_from_independent_peers_in_one_registry() -> TestResult {
    let first = fixture("fetch", success("first peer")).await?;
    let second = fixture("fetch", success("second peer")).await?;
    let first_tool: DynamicTool = first
        .tool
        .clone()
        .with_model_name(ToolName::new("first_fetch")?)
        .try_into()?;
    let second_tool: DynamicTool = second
        .tool
        .clone()
        .with_model_name(ToolName::new("second_fetch")?)
        .try_into()?;
    let handle = ToolServer::new()
        .dynamic_tools(vec![first_tool, second_tool])
        .run();
    let names = handle
        .static_tool_defs()
        .into_iter()
        .map(|definition| definition.name.to_string())
        .collect::<Vec<_>>();
    check(
        names == ["first_fetch", "second_fetch"],
        "one peer overwrote the other registration",
    )?;
    let mut first_context = ToolContext::new();
    let mut second_context = ToolContext::new();
    let (first_result, second_result) = tokio::time::timeout(Duration::from_secs(5), async {
        tokio::join!(
            handle.execute("first_fetch", "{}", &mut first_context),
            handle.execute("second_fetch", "{}", &mut second_context)
        )
    })
    .await?;
    check(
        first_result.is_success() && first_result.output().as_text() == Some("first peer"),
        "first alias reached the wrong peer",
    )?;
    check(
        second_result.is_success() && second_result.output().as_text() == Some("second peer"),
        "second alias reached the wrong peer",
    )?;
    for fixture in [&first, &second] {
        let seen = lock(&fixture.seen);
        check(
            seen.len() == 1 && seen.first().is_some_and(|call| call.name == "fetch"),
            "peer received an aliased wire name",
        )?;
    }
    check(
        !handle.toolset().contains("fetch"),
        "raw collision name was also registered",
    )?;
    Ok(())
}

#[tokio::test]
async fn unaliased_tools_keep_default_names_and_aliases_do_not_hide_empty_raw_names() -> TestResult
{
    let default_fixture = fixture("fetch", success("default")).await?;
    let dynamic: DynamicTool = default_fixture.tool.clone().try_into()?;
    check(
        dynamic.name().as_str() == "fetch" && dynamic.definition().name.as_str() == "fetch",
        "unaliased tool changed its model name",
    )?;
    check(
        dynamic.handler().descriptor().key == tool_key("fetch"),
        "unaliased handler changed its key",
    )?;
    let output = tokio::time::timeout(Duration::from_secs(5), dynamic.execute(json!({}))).await??;
    check(output.as_text() == Some("default"), "unaliased call failed")?;
    check(
        lock(&default_fixture.seen)
            .first()
            .is_some_and(|call| call.name == "fetch"),
        "unaliased wire name changed",
    )?;
    let empty = fixture("", success("must not execute")).await?;
    let aliased = empty
        .tool
        .clone()
        .with_model_name(ToolName::new("valid_alias")?);
    check(
        DynamicTool::try_from(aliased).is_err(),
        "alias hid an invalid raw MCP tool name",
    )?;
    check(
        lock(&empty.seen).is_empty(),
        "empty-name tool reached the server",
    )?;
    Ok(())
}

#[tokio::test]
async fn alias_preserves_context_metadata_structured_failures_and_raw_results() -> TestResult {
    let structured = json!({"reason": "denied", "peer": "files"});
    let mut response = CallToolResult::structured_error(structured.clone());
    let mut response_meta = Meta::new();
    response_meta
        .0
        .insert("response-id".into(), json!("result-1"));
    response.meta = Some(response_meta.clone());
    let fixture = fixture("files/read", response.clone()).await?;
    let dynamic: DynamicTool = fixture
        .tool
        .clone()
        .with_model_name(ToolName::new("files_read")?)
        .try_into()?;
    let handle = ToolServer::new().dynamic_tool(dynamic).run();
    let mut context = ToolContext::new();
    let mut request_meta = Meta::new();
    request_meta
        .0
        .insert("authorization".into(), json!("host-only-secret"));
    context.insert(McpMeta(request_meta.clone()))?;
    let result = tokio::time::timeout(
        Duration::from_secs(5),
        handle.execute("files_read", "{}", &mut context),
    )
    .await?;
    check(
        result.is_error_kind(ToolErrorKind::Other),
        "alias changed MCP error classification",
    )?;
    check(
        result.output() == &ToolOutput::json(structured.clone()),
        "alias changed structured error presentation",
    )?;
    let error = required(result.error(), "missing structured execution error")?;
    check(
        error.message() == "MCP tool 'files/read' reported an execution error",
        "alias changed server error identity",
    )?;
    check(
        context.require_result::<McpStructuredContent>()?.0 == structured,
        "structured content was not published",
    )?;
    check(
        context.require_result::<McpResponseMeta>()?.0 == response_meta,
        "response metadata was not published",
    )?;
    check(
        context.require_result::<McpCallToolResult>()?.0 == response,
        "raw response was changed",
    )?;
    check(
        context.require::<McpMeta>()?.0 == request_meta,
        "alias mutated inbound context",
    )?;
    let seen = lock(&fixture.seen);
    let call = required(seen.first(), "missing aliased MCP request")?;
    check(
        call.name == "files/read",
        "alias leaked into an erroring wire call",
    )?;
    check(
        call.meta.0.get("authorization") == Some(&json!("host-only-secret")),
        "alias lost request metadata",
    )?;
    check(
        !result.output().render().contains("host-only-secret"),
        "request metadata leaked into model output",
    )?;
    Ok(())
}
