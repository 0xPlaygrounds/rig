//! `McpClientHandler`: keeps a [`ManagedToolSink`] in sync with an MCP server's
//! tool list.

use std::collections::HashMap;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use std::time::Duration;

use rmcp::ServiceExt;
use rmcp::model::{ClientRequest, ListToolsRequest, PaginatedRequestParams, ServerResult};
use tokio::sync::{Mutex, RwLock};

use rig_core::tool::{ManagedToolSink, ManagedToolToken, PortableDynamicTool};

use crate::{
    DEFAULT_MCP_REFRESH_TIMEOUT, DEFAULT_MCP_TOOL_TIMEOUT, McpClientError, McpTool,
    send_mcp_request,
};

#[derive(Default)]
pub(crate) struct ManagedToolsState {
    pub(crate) registrations: HashMap<String, ManagedToolToken>,
    pub(crate) committed_refresh: u64,
}

#[derive(Default)]
pub(crate) struct RefreshActivity {
    pub(crate) active: usize,
    pub(crate) dirty: bool,
}

pub(crate) const MAX_CONCURRENT_REFRESHES: usize = 2;

/// An MCP client handler that automatically re-fetches the tool list when the
/// server sends a `notifications/tools/list_changed` notification.
///
/// This handler implements [`rmcp::ClientHandler`] and bridges the MCP
/// notification lifecycle with any [`ManagedToolSink`] — rig-agent's
/// `ToolServerHandle` implements it, so the rig-agent usage is
/// `McpClientHandler::new(client_info, tool_server_handle.clone())`.
/// When the MCP server's available tools change, this handler:
/// 1. Re-fetches the full tool list from the MCP server
/// 2. Replaces or removes registrations still owned by this handler
/// 3. Leaves newer local and peer-handler same-name registrations intact
///
/// # Usage
///
/// Use [`McpClientHandler::connect`] for a streamlined setup that handles
/// connection, initial tool fetch, and registration in one call:
///
/// ```rust,ignore
/// let tool_server_handle = ToolServer::new().run();
/// let handler = McpClientHandler::new(client_info, tool_server_handle.clone());
/// let mcp_service = handler.connect(transport).await?;
/// ```
///
/// The returned `RunningService` keeps the MCP connection alive. When the
/// server updates its tools, the handler automatically syncs with the tool server.
pub struct McpClientHandler<S> {
    client_info: rmcp::model::ClientInfo,
    sink: S,
    /// Per-call timeout applied to every MCP tool this handler registers
    /// (see issue #1914). Defaults to [`DEFAULT_MCP_TOOL_TIMEOUT`].
    timeout: Option<Duration>,
    /// Deadline for initial and list-changed tool-list fetches.
    refresh_timeout: Duration,
    /// Tracks the exact registry generation installed for each tool. Refreshes
    /// only mutate a name while this generation remains current, so a newer
    /// local or peer-handler registration cannot be deleted or overwritten.
    pub(crate) managed_tools: Arc<RwLock<ManagedToolsState>>,
    /// Bounds notification-driven list fetches and coalesces excess signals.
    pub(crate) refresh_activity: Arc<Mutex<RefreshActivity>>,
    /// Monotonic identity assigned when each tool-list fetch begins.
    next_refresh: Arc<AtomicU64>,
}

impl<S> McpClientHandler<S>
where
    S: ManagedToolSink + Send + Sync + 'static,
{
    /// Create a new handler with the given client info and tool sink.
    ///
    /// With rig-agent, pass a clone of the agent's `ToolServerHandle` so tool
    /// updates are reflected in agent requests. Registered tools get
    /// [`DEFAULT_MCP_TOOL_TIMEOUT`]; change it with [`McpClientHandler::with_timeout`].
    pub fn new(client_info: rmcp::model::ClientInfo, sink: S) -> Self {
        Self {
            client_info,
            sink,
            timeout: Some(DEFAULT_MCP_TOOL_TIMEOUT),
            refresh_timeout: DEFAULT_MCP_REFRESH_TIMEOUT,
            managed_tools: Arc::new(RwLock::new(ManagedToolsState::default())),
            refresh_activity: Arc::new(Mutex::new(RefreshActivity::default())),
            next_refresh: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Set (or clear) the per-call timeout applied to every MCP tool this handler
    /// registers. Pass a [`Duration`] to bound calls, or `None` to disable.
    ///
    /// This applies the same setting to every tool managed by the handler.
    pub fn with_timeout(mut self, timeout: impl Into<Option<Duration>>) -> Self {
        self.timeout = timeout.into();
        self
    }

    /// Set the deadline for initial and list-changed tool-list fetches.
    pub fn with_refresh_timeout(mut self, timeout: Duration) -> Self {
        self.refresh_timeout = timeout;
        self
    }

    /// Build the portable adapter with this handler's configured timeout.
    pub(crate) fn build_tool(
        &self,
        tool: rmcp::model::Tool,
        client: rmcp::service::ServerSink,
    ) -> PortableDynamicTool {
        McpTool::from_mcp_server(tool, client)
            .with_timeout(self.timeout)
            .into()
    }

    pub(crate) fn begin_refresh(&self) -> u64 {
        self.next_refresh.fetch_add(1, Ordering::SeqCst) + 1
    }

    pub(crate) async fn fetch_tools(
        &self,
        peer: &rmcp::service::ServerSink,
    ) -> Result<Vec<PortableDynamicTool>, McpClientError> {
        let deadline = tokio::time::Instant::now() + self.refresh_timeout;
        let mut tools = Vec::new();
        let mut cursor = None;

        loop {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            if remaining.is_zero() {
                return Err(McpClientError::ToolFetchTimeout(self.refresh_timeout));
            }
            let mut params = PaginatedRequestParams::default();
            params.cursor = cursor;
            let response = send_mcp_request(
                peer,
                ClientRequest::ListToolsRequest(ListToolsRequest::with_param(params)),
                Some((deadline, self.refresh_timeout)),
            )
            .await
            .map_err(|error| match error {
                rmcp::ServiceError::Timeout { .. } => {
                    McpClientError::ToolFetchTimeout(self.refresh_timeout)
                }
                error => McpClientError::ToolFetchError(error),
            })?;
            let ServerResult::ListToolsResult(page) = response else {
                return Err(McpClientError::ToolFetchError(
                    rmcp::ServiceError::UnexpectedResponse,
                ));
            };
            tools.extend(page.tools);
            cursor = page.next_cursor;
            if cursor.is_none() {
                break;
            }
        }

        Ok(tools
            .into_iter()
            .map(|tool| self.build_tool(tool, peer.clone()))
            .collect())
    }

    pub(crate) async fn try_start_refresh(&self) -> bool {
        let mut activity = self.refresh_activity.lock().await;
        if activity.active >= MAX_CONCURRENT_REFRESHES {
            activity.dirty = true;
            false
        } else {
            activity.active += 1;
            true
        }
    }

    pub(crate) async fn finish_or_restart_refresh(&self) -> bool {
        let mut activity = self.refresh_activity.lock().await;
        if activity.dirty {
            activity.dirty = false;
            true
        } else {
            activity.active -= 1;
            false
        }
    }

    pub(crate) async fn commit_initial(&self, refresh: u64, tools: Vec<PortableDynamicTool>) {
        let mut managed = self.managed_tools.write().await;
        if refresh <= managed.committed_refresh {
            tracing::debug!(refresh, "discarding stale initial MCP tool list");
            return;
        }
        managed.registrations = self.sink.add_managed_tools(tools);
        managed.committed_refresh = refresh;
    }

    pub(crate) async fn commit_refresh(
        &self,
        refresh: u64,
        tools: Vec<PortableDynamicTool>,
    ) -> bool {
        let mut managed = self.managed_tools.write().await;
        if refresh <= managed.committed_refresh {
            tracing::debug!(refresh, "discarding stale MCP tool-list response");
            return false;
        }
        let expected = managed.registrations.clone();
        managed.registrations = self.sink.reconcile_managed_tools(expected, tools);
        managed.committed_refresh = refresh;
        true
    }

    /// Connect to an MCP server, fetch the initial tool list, and register
    /// all tools with the tool server.
    ///
    /// Returns the running MCP service. The connection stays alive as long as the
    /// returned `RunningService` is held. When the server sends
    /// `notifications/tools/list_changed`, this handler automatically re-fetches
    /// and re-registers tools into the sink.
    ///
    /// # Errors
    ///
    /// Returns [`McpClientError`] if the connection or initial tool fetch fails.
    pub async fn connect<T, E, A>(
        self,
        transport: T,
    ) -> Result<rmcp::service::RunningService<rmcp::service::RoleClient, Self>, McpClientError>
    where
        T: rmcp::transport::IntoTransport<rmcp::service::RoleClient, E, A>,
        E: std::error::Error + Send + Sync + 'static,
    {
        let service = ServiceExt::serve(self, transport)
            .await
            .map_err(|e| McpClientError::ConnectionError(e.to_string()))?;

        let handler = service.service();
        let refresh = handler.begin_refresh();
        let tools = handler.fetch_tools(service.peer()).await?;
        handler.commit_initial(refresh, tools).await;

        Ok(service)
    }
}

impl<S> rmcp::handler::client::ClientHandler for McpClientHandler<S>
where
    S: ManagedToolSink + Send + Sync + 'static,
{
    fn get_info(&self) -> rmcp::model::ClientInfo {
        self.client_info.clone()
    }

    async fn on_tool_list_changed(
        &self,
        context: rmcp::service::NotificationContext<rmcp::service::RoleClient>,
    ) {
        if !self.try_start_refresh().await {
            return;
        }

        loop {
            let refresh = self.begin_refresh();
            // Network IO is deliberately outside the ownership lock. Up to two
            // fetches may overlap so a newer snapshot can bypass one stalled
            // request; further notifications coalesce into one follow-up fetch.
            match self.fetch_tools(&context.peer).await {
                Ok(tools) => {
                    if self.commit_refresh(refresh, tools).await {
                        let tool_count = self.managed_tools.read().await.registrations.len();
                        tracing::info!(tool_count, "MCP tool list refreshed successfully");
                    }
                }
                Err(error) => tracing::error!("Failed to re-fetch MCP tool list: {error}"),
            }

            if !self.finish_or_restart_refresh().await {
                break;
            }
        }
    }
}
