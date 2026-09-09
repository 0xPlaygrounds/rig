#[allow(unused_imports)]
use crate::handler::MAX_CONCURRENT_REFRESHES;
#[allow(unused_imports)]
use crate::native::{McpArgumentError, bounded_best_effort_cancellation};
#[allow(unused_imports)]
use crate::*;
use crate::{McpClientError, McpClientHandler};
use rig_agent::tool::{DynamicTool, ToolOutput, server::ToolServer};
#[allow(unused_imports)]
use rig_agent::tool::{ToolContext, ToolResult};
#[allow(unused_imports)]
use rig_core::message::ImageMediaType;
#[allow(unused_imports)]
use rig_core::tool::ToolExecutionError;
use rmcp::{
    RoleServer, ServerHandler, ServiceExt, handler::client::ClientHandler, model::*,
    service::RequestContext,
};
use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};
use tokio::sync::{Notify, RwLock};

#[derive(Clone)]
struct DynamicToolServer {
    tools: Arc<RwLock<Vec<Tool>>>,
}
impl DynamicToolServer {
    fn new(tools: Vec<Tool>) -> Self {
        Self {
            tools: Arc::new(RwLock::new(tools)),
        }
    }
    async fn set_tools(&self, tools: Vec<Tool>) {
        *self.tools.write().await = tools;
    }
}
impl ServerHandler for DynamicToolServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_protocol_version(ProtocolVersion::LATEST)
            .with_server_info(Implementation::new("test-dynamic-server", "0.1.0"))
    }
    async fn list_tools(
        &self,
        _: Option<PaginatedRequestParams>,
        _: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, ErrorData> {
        Ok(ListToolsResult::with_all_items(
            self.tools.read().await.clone(),
        ))
    }
    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        _: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, ErrorData> {
        Ok(
            CallToolResult::success(vec![ContentBlock::text(format!("called {}", request.name))])
                .into(),
        )
    }
}

#[derive(Clone)]
struct OrderedRefreshServer {
    tools: Arc<RwLock<Vec<Tool>>>,
    list_calls: Arc<AtomicUsize>,
    first_refresh_started: Arc<Notify>,
    release_first_refresh: Arc<Notify>,
    first_refresh_returned: Arc<Notify>,
}

impl OrderedRefreshServer {
    fn new(tools: Vec<Tool>) -> Self {
        Self {
            tools: Arc::new(RwLock::new(tools)),
            list_calls: Arc::new(AtomicUsize::new(0)),
            first_refresh_started: Arc::new(Notify::new()),
            release_first_refresh: Arc::new(Notify::new()),
            first_refresh_returned: Arc::new(Notify::new()),
        }
    }

    async fn set_tools(&self, tools: Vec<Tool>) {
        *self.tools.write().await = tools;
    }
}

impl ServerHandler for OrderedRefreshServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_protocol_version(ProtocolVersion::LATEST)
            .with_server_info(Implementation::new("test-ordered-refresh-server", "0.1.0"))
    }

    async fn list_tools(
        &self,
        _: Option<PaginatedRequestParams>,
        _: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, ErrorData> {
        let call = self.list_calls.fetch_add(1, Ordering::SeqCst);
        let tools = self.tools.read().await.clone();

        // Call zero is connect's initial fetch. Hold the first notification's
        // stale snapshot so a second notification is concurrent with it.
        if call == 1 {
            self.first_refresh_started.notify_one();
            self.release_first_refresh.notified().await;
            self.first_refresh_returned.notify_one();
        }

        Ok(ListToolsResult::with_all_items(tools))
    }
}

#[derive(Clone)]
struct HangingListServer;

impl ServerHandler for HangingListServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_protocol_version(ProtocolVersion::LATEST)
            .with_server_info(Implementation::new("test-hanging-list-server", "0.1.0"))
    }

    async fn list_tools(
        &self,
        _: Option<PaginatedRequestParams>,
        _: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, ErrorData> {
        std::future::pending().await
    }
}

fn make_tool(name: &str, description: &str) -> Tool {
    Tool::new(
        name.to_string(),
        description.to_string(),
        Arc::new(serde_json::Map::new()),
    )
}

fn make_dynamic_tool(name: &str, description: &str) -> DynamicTool {
    DynamicTool::new(
        name,
        description,
        serde_json::json!({"type": "object", "properties": {}}),
        |_context, _args| Box::pin(async { Ok(ToolOutput::text("local")) }),
    )
}

async fn connect<S>(
    server: S,
    handle: rig_agent::tool::server::ToolServerHandle,
) -> (
    rmcp::service::RunningService<
        rmcp::RoleClient,
        McpClientHandler<rig_agent::tool::server::ToolServerHandle>,
    >,
    tokio::task::JoinHandle<rmcp::service::RunningService<rmcp::RoleServer, S>>,
)
where
    S: ServerHandler,
{
    let (c2s, sfc) = tokio::io::duplex(8192);
    let (s2c, cfs) = tokio::io::duplex(8192);
    let server_task =
        tokio::spawn(async move { server.serve((sfc, s2c)).await.expect("server start") });
    let service = McpClientHandler::new(ClientInfo::default(), handle)
        .connect((cfs, c2s))
        .await
        .expect("connect");
    (service, server_task)
}

#[tokio::test]
async fn client_handler_registers_initial_tools() {
    let server = DynamicToolServer::new(vec![
        make_tool("tool_a", "First"),
        make_tool("tool_b", "Second"),
    ]);
    let handle = ToolServer::new().run();
    let (client, task) = connect(server, handle.clone()).await;
    let defs = handle.tool_defs(None).await.unwrap();
    assert_eq!(
        defs.iter().map(|d| d.name.as_str()).collect::<Vec<_>>(),
        vec!["tool_a", "tool_b"]
    );
    client.cancel().await.unwrap();
    task.abort();
}

#[tokio::test]
async fn disconnected_handler_tools_are_retired_on_snapshot() {
    let server = DynamicToolServer::new(vec![make_tool("tool_a", "First")]);
    let handle = ToolServer::new().run();
    let (client, task) = connect(server, handle.clone()).await;
    assert_eq!(handle.tool_defs(None).await.unwrap().len(), 1);

    client.cancel().await.unwrap();

    let defs = handle.tool_defs(None).await.unwrap();
    assert!(
        defs.is_empty(),
        "a disconnected sole owner must not remain provider-visible"
    );
    task.abort();
}

#[tokio::test]
async fn disconnected_handler_tools_are_retired_on_direct_dispatch() {
    let server = DynamicToolServer::new(vec![make_tool("tool_a", "First")]);
    let handle = ToolServer::new().run();
    let (client, task) = connect(server, handle.clone()).await;
    assert_eq!(handle.tool_defs(None).await.unwrap().len(), 1);

    client.cancel().await.unwrap();

    let result = handle
        .execute("tool_a", "{}", &mut rig_agent::tool::ToolContext::new())
        .await;
    assert_eq!(
        result.error().expect("disconnected tool must fail").kind(),
        rig_agent::tool::ToolErrorKind::NotFound
    );
    task.abort();
}

#[tokio::test]
async fn initial_tool_fetch_is_bounded_by_the_refresh_timeout() {
    let (c2s, sfc) = tokio::io::duplex(8192);
    let (s2c, cfs) = tokio::io::duplex(8192);
    let server_task = tokio::spawn(async move {
        HangingListServer
            .serve((sfc, s2c))
            .await
            .expect("server start")
    });
    let refresh_timeout = Duration::from_millis(25);
    let result = McpClientHandler::new(ClientInfo::default(), ToolServer::new().run())
        .with_refresh_timeout(refresh_timeout)
        .connect((cfs, c2s))
        .await;

    assert!(matches!(
        result,
        Err(McpClientError::ToolFetchTimeout(timeout)) if timeout == refresh_timeout
    ));
    server_task.abort();
}

#[tokio::test]
async fn refresh_activity_is_bounded_and_coalesces_excess_notifications() {
    let handler = McpClientHandler::new(ClientInfo::default(), ToolServer::new().run());

    assert!(handler.try_start_refresh().await);
    assert!(handler.try_start_refresh().await);
    assert!(!handler.try_start_refresh().await);
    {
        let activity = handler.refresh_activity.lock().await;
        assert_eq!(activity.active, MAX_CONCURRENT_REFRESHES);
        assert!(activity.dirty);
    }

    assert!(handler.finish_or_restart_refresh().await);
    assert!(!handler.finish_or_restart_refresh().await);
    assert!(!handler.finish_or_restart_refresh().await);
    let activity = handler.refresh_activity.lock().await;
    assert_eq!(activity.active, 0);
    assert!(!activity.dirty);
}

#[tokio::test]
async fn client_handler_refreshes_on_tool_list_changed() {
    let server = DynamicToolServer::new(vec![make_tool("alpha", "Alpha")]);
    let handle = ToolServer::new().run();
    let (c2s, sfc) = tokio::io::duplex(8192);
    let (s2c, cfs) = tokio::io::duplex(8192);
    let copy = server.clone();
    let task = tokio::spawn(async move { copy.serve((sfc, s2c)).await.expect("server start") });
    let client = McpClientHandler::new(ClientInfo::default(), handle.clone())
        .connect((cfs, c2s))
        .await
        .unwrap();
    assert_eq!(handle.tool_defs(None).await.unwrap()[0].name, "alpha");
    server
        .set_tools(vec![make_tool("beta", "Beta"), make_tool("gamma", "Gamma")])
        .await;
    let running = task.await.unwrap();
    running.peer().notify_tool_list_changed().await.unwrap();
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            let defs = handle.tool_defs(None).await.unwrap();
            if defs.len() == 2 {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("refresh");
    let names = handle
        .tool_defs(None)
        .await
        .unwrap()
        .into_iter()
        .map(|d| d.name)
        .collect::<Vec<_>>();
    assert_eq!(names, vec!["beta", "gamma"]);
    client.cancel().await.unwrap();
}

#[tokio::test]
async fn concurrent_refreshes_cannot_roll_back_a_newer_tool_list() {
    let server = OrderedRefreshServer::new(vec![make_tool("stale", "Stale snapshot")]);
    let server_control = server.clone();
    let handle = ToolServer::new().run();
    let (client, server_task) = connect(server, handle.clone()).await;
    let running_server = server_task.await.unwrap();

    running_server
        .peer()
        .notify_tool_list_changed()
        .await
        .unwrap();
    tokio::time::timeout(
        Duration::from_secs(2),
        server_control.first_refresh_started.notified(),
    )
    .await
    .expect("first refresh fetch started");

    assert!(
        client.service().managed_tools.try_write().is_ok(),
        "a hung network fetch must not hold the managed-registry lock"
    );

    server_control
        .set_tools(vec![make_tool("newest", "Newest snapshot")])
        .await;
    running_server
        .peer()
        .notify_tool_list_changed()
        .await
        .unwrap();

    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            let defs = handle.tool_defs(None).await.unwrap();
            if defs.len() == 1 && defs[0].name == "newest" {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("newest refresh committed while the older fetch remained hung");

    // Let the stale response arrive after the newer snapshot committed. Its
    // lower refresh version must be discarded rather than rolling back.
    server_control.release_first_refresh.notify_one();
    tokio::time::timeout(
        Duration::from_secs(2),
        server_control.first_refresh_returned.notified(),
    )
    .await
    .expect("delayed refresh response returned");
    for _ in 0..10 {
        tokio::task::yield_now().await;
    }
    let defs = handle.tool_defs(None).await.unwrap();
    assert_eq!(defs.len(), 1);
    assert_eq!(defs[0].name, "newest");

    assert_eq!(server_control.list_calls.load(Ordering::SeqCst), 3);
    client.cancel().await.unwrap();
}

#[tokio::test]
async fn refresh_rebuilds_owned_tools_in_latest_server_order() {
    let server =
        DynamicToolServer::new(vec![make_tool("alpha", "Alpha"), make_tool("beta", "Beta")]);
    let server_control = server.clone();
    let handle = ToolServer::new().run();
    let (client, server_task) = connect(server, handle.clone()).await;
    server_control
        .set_tools(vec![
            make_tool("beta", "Beta refreshed"),
            make_tool("gamma", "Gamma"),
            make_tool("alpha", "Alpha refreshed"),
        ])
        .await;
    let running_server = server_task.await.unwrap();
    running_server
        .peer()
        .notify_tool_list_changed()
        .await
        .unwrap();

    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            let defs = handle.tool_defs(None).await.unwrap();
            let names = defs
                .iter()
                .map(|definition| definition.name.as_str())
                .collect::<Vec<_>>();
            if names == ["beta", "gamma", "alpha"] && defs[0].description == "Beta refreshed" {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("latest MCP order committed");
    client.cancel().await.unwrap();
}

#[tokio::test]
async fn one_refresh_reclaims_a_name_after_a_peer_owner_disappears() {
    let handle = ToolServer::new().run();
    let first_server = DynamicToolServer::new(vec![make_tool("shared", "First owner")]);
    let first_control = first_server.clone();
    let (first_client, first_server_task) = connect(first_server, handle.clone()).await;
    let first_running_server = first_server_task.await.unwrap();

    let second_server = DynamicToolServer::new(vec![make_tool("shared", "Second owner")]);
    let second_control = second_server.clone();
    let (second_client, second_server_task) = connect(second_server, handle.clone()).await;
    let second_running_server = second_server_task.await.unwrap();
    assert_eq!(
        handle.tool_defs(None).await.unwrap()[0].description,
        "Second owner"
    );

    second_control.set_tools(Vec::new()).await;
    second_running_server
        .peer()
        .notify_tool_list_changed()
        .await
        .unwrap();
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if handle.tool_defs(None).await.unwrap().is_empty() {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("second owner removed its registration");

    // The first handler still has a stale generation token for `shared`.
    // One full-list refresh must reclaim the now-empty slot rather than
    // requiring a second notification to converge.
    first_control
        .set_tools(vec![make_tool("shared", "First owner refreshed")])
        .await;
    first_running_server
        .peer()
        .notify_tool_list_changed()
        .await
        .unwrap();
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            let defs = handle.tool_defs(None).await.unwrap();
            if defs.len() == 1 && defs[0].description == "First owner refreshed" {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("one refresh reclaimed the empty slot");

    second_client.cancel().await.unwrap();
    first_client.cancel().await.unwrap();
}

#[tokio::test]
async fn refresh_does_not_replace_a_newer_local_registration() {
    let server = DynamicToolServer::new(vec![make_tool("alpha", "MCP alpha")]);
    let server_control = server.clone();
    let handle = ToolServer::new().run();
    let (client, server_task) = connect(server, handle.clone()).await;

    handle.add_dynamic_tool(make_dynamic_tool("alpha", "Local alpha"));
    server_control
        .set_tools(vec![make_tool("refresh_complete", "Refresh sentinel")])
        .await;
    let running_server = server_task.await.unwrap();
    running_server
        .peer()
        .notify_tool_list_changed()
        .await
        .unwrap();

    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            let defs = handle.tool_defs(None).await.unwrap();
            if defs
                .iter()
                .any(|definition| definition.name == "refresh_complete")
            {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("MCP refresh completed");

    let defs = handle.tool_defs(None).await.unwrap();
    let alpha = defs
        .iter()
        .find(|definition| definition.name == "alpha")
        .expect("alpha remains registered");
    assert_eq!(alpha.description, "Local alpha");

    let result = handle
        .execute("alpha", "{}", &mut rig_agent::tool::ToolContext::new())
        .await;
    assert_eq!(result.output(), &ToolOutput::text("local"));
    client.cancel().await.unwrap();
}

#[tokio::test]
async fn one_handler_refresh_protects_live_peer_and_reclaims_after_disconnect() {
    let server_a = DynamicToolServer::new(vec![make_tool("alpha", "Handler A")]);
    let server_a_control = server_a.clone();
    let server_b = DynamicToolServer::new(vec![make_tool("alpha", "Handler B")]);
    let handle = ToolServer::new().run();

    let (client_a, server_task_a) = connect(server_a, handle.clone()).await;
    let (client_b, server_task_b) = connect(server_b, handle.clone()).await;

    server_a_control
        .set_tools(vec![
            make_tool("alpha", "Refreshed handler A"),
            make_tool("a_refresh_complete", "Refresh sentinel"),
        ])
        .await;
    let running_server_a = server_task_a.await.unwrap();
    let _running_server_b = server_task_b.await.unwrap();
    running_server_a
        .peer()
        .notify_tool_list_changed()
        .await
        .unwrap();

    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            let defs = handle.tool_defs(None).await.unwrap();
            if defs
                .iter()
                .any(|definition| definition.name == "a_refresh_complete")
            {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("handler A refresh completed");

    let defs = handle.tool_defs(None).await.unwrap();
    let alpha = defs
        .iter()
        .find(|definition| definition.name == "alpha")
        .expect("alpha remains registered");
    assert_eq!(alpha.description, "Handler B");

    // Once B disconnects, its generation must no longer shield the dead
    // registration from A. Otherwise the registry keeps advertising B and
    // execution fails with `Transport closed` indefinitely.
    client_b.cancel().await.unwrap();
    server_a_control
        .set_tools(vec![make_tool("alpha", "Reclaimed handler A")])
        .await;
    running_server_a
        .peer()
        .notify_tool_list_changed()
        .await
        .unwrap();

    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            let defs = handle.tool_defs(None).await.unwrap();
            if defs
                .iter()
                .any(|definition| definition.description == "Reclaimed handler A")
            {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("handler A reclaimed the disconnected peer's registration");

    let result = handle
        .execute("alpha", "{}", &mut rig_agent::tool::ToolContext::new())
        .await;
    assert!(
        result.is_success(),
        "reclaimed tool should execute: {result:?}"
    );

    client_a.cancel().await.unwrap();
}

#[test]
fn client_handler_get_info_delegates() {
    let info = ClientInfo::new(
        ClientCapabilities::default(),
        Implementation::new("test-client", "1.0.0"),
    );
    let handler = McpClientHandler::new(info, ToolServer::new().run());
    let returned = handler.get_info();
    assert_eq!(returned.client_info.name, "test-client");
    assert_eq!(returned.client_info.version, "1.0.0");
}

#[tokio::test]
async fn mcp_tool_preserves_provider_definition() {
    let tool = make_tool("search_docs", "Search the docs");
    let server = DynamicToolServer::new(vec![tool.clone()]);
    let (c2s, sfc) = tokio::io::duplex(8192);
    let (s2c, cfs) = tokio::io::duplex(8192);
    let task = tokio::spawn(async move {
        let running = server.serve((sfc, s2c)).await.unwrap();
        running.waiting().await.unwrap();
    });
    let client = ClientInfo::default().serve((cfs, c2s)).await.unwrap();
    let handle = ToolServer::new()
        .portable_dynamic_tool(McpTool::from_mcp_server(tool, client.peer().clone()).into())
        .run();
    let defs = handle.tool_defs(None).await.unwrap();
    assert_eq!(defs.len(), 1);
    assert_eq!(defs[0].name, "search_docs");
    assert_eq!(defs[0].description, "Search the docs");
    client.cancel().await.unwrap();

    let defs = handle.tool_defs(None).await.unwrap();
    assert!(
        defs.is_empty(),
        "a disconnected directly registered MCP tool must not remain provider-visible"
    );
    task.abort();
}

#[tokio::test]
async fn disconnected_directly_registered_mcp_tool_is_retired_on_dispatch() {
    let tool = make_tool("search_docs", "Search the docs");
    let server = DynamicToolServer::new(vec![tool.clone()]);
    let (c2s, sfc) = tokio::io::duplex(8192);
    let (s2c, cfs) = tokio::io::duplex(8192);
    let task = tokio::spawn(async move {
        let running = server.serve((sfc, s2c)).await.unwrap();
        running.waiting().await.unwrap();
    });
    let client = ClientInfo::default().serve((cfs, c2s)).await.unwrap();
    let handle = ToolServer::new()
        .portable_dynamic_tool(McpTool::from_mcp_server(tool, client.peer().clone()).into())
        .run();

    client.cancel().await.unwrap();

    let result = handle
        .execute(
            "search_docs",
            "{}",
            &mut rig_agent::tool::ToolContext::new(),
        )
        .await;
    assert_eq!(
        result.error().expect("disconnected tool must fail").kind(),
        rig_agent::tool::ToolErrorKind::NotFound
    );
    task.abort();
}

/// Registering MCP tools into an agent through portable tools keeps the
/// configured timeout on each of them, so a hanging call is bounded instead
/// of blocking forever (see issue #1914).
#[tokio::test]
async fn builder_rmcp_tools_thread_timeout_into_registered_tools() {
    use rig_agent::agent::AgentBuilder;
    use rig_agent::test_utils::MockCompletionModel;
    use rig_agent::tool::DynamicTool;
    use rig_agent::tool::{ToolContext, ToolErrorKind};
    use rig_core::tool::PortableDynamicTool;
    use rmcp::model::{
        CallToolRequestParams, CallToolResponse, ClientInfo, ErrorData, Implementation,
        ProtocolVersion, ServerCapabilities, ServerInfo, Tool,
    };
    use rmcp::service::RequestContext;
    use rmcp::{RoleServer, ServerHandler, ServiceExt};

    #[derive(Clone)]
    struct HangingServer;
    impl ServerHandler for HangingServer {
        fn get_info(&self) -> ServerInfo {
            ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
                .with_protocol_version(ProtocolVersion::LATEST)
                .with_server_info(Implementation::new("builder-timeout-test", "0.1.0"))
        }
        async fn call_tool(
            &self,
            _request: CallToolRequestParams,
            _context: RequestContext<RoleServer>,
        ) -> Result<CallToolResponse, ErrorData> {
            std::future::pending::<Result<CallToolResponse, ErrorData>>().await
        }
    }

    fn tool(name: &str) -> Tool {
        Tool::new(
            name.to_string(),
            String::new(),
            Arc::new(serde_json::Map::new()),
        )
    }

    let (c2s, sfc) = tokio::io::duplex(8192);
    let (s2c, cfs) = tokio::io::duplex(8192);
    let server_task = tokio::spawn(async move {
        let running = HangingServer.serve((sfc, s2c)).await.expect("server start");
        running.waiting().await.expect("server error");
    });
    let client = ClientInfo::default()
        .serve((cfs, c2s))
        .await
        .expect("client connect");
    let peer = client.peer().clone();

    // The default the plural builders pass, and a disabled timeout, both
    // reach the built tool verbatim.
    let built = McpTool::from_mcp_server(tool("a"), peer.clone());
    assert_eq!(built.timeout(), Some(DEFAULT_MCP_TOOL_TIMEOUT));
    assert_eq!(built.with_timeout(None).timeout(), None);

    // Every requested tool is registered against the shared client...
    let agent = AgentBuilder::new(MockCompletionModel::text("ok"))
        .dynamic_tools(
            tools_from_server([tool("a"), tool("b")], &peer, DEFAULT_MCP_TOOL_TIMEOUT)
                .into_iter()
                .map(|tool| DynamicTool::from(PortableDynamicTool::from(tool)))
                .collect(),
        )
        .build();
    let definitions = agent.tool_server_handle().tool_defs(None).await.unwrap();
    assert_eq!(
        definitions
            .iter()
            .map(|definition| definition.name.as_str())
            .collect::<Vec<_>>(),
        vec!["a", "b"]
    );

    // ...and the configured timeout actually bounds a hanging call.
    let agent = AgentBuilder::new(MockCompletionModel::text("ok"))
        .dynamic_tools(
            tools_from_server([tool("hang_forever")], &peer, Duration::from_millis(200))
                .into_iter()
                .map(|tool| DynamicTool::from(PortableDynamicTool::from(tool)))
                .collect(),
        )
        .build();
    let timed = tokio::time::timeout(Duration::from_secs(5), async {
        let mut context = ToolContext::new();
        agent
            .tool_server_handle()
            .execute("hang_forever", "{}", &mut context)
            .await
    })
    .await;
    let result = timed.expect("registered tool hung past the safety timeout");
    assert!(result.is_error_kind(ToolErrorKind::Timeout));
    assert!(result.output().render().contains("timed out"));

    drop(client);
    server_task.abort();
}
