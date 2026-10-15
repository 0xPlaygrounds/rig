//! A live MCP client connection shared by the tools and resources taken from it.
//!
//! ```no_run
//! use rig_rmcp::{McpConnection, rmcp};
//! use rmcp::{RoleClient, ServiceExt, transport::IntoTransport};
//!
//! async fn connect<T, E, A>(transport: T) -> Result<(), Box<dyn std::error::Error>>
//! where
//!     T: IntoTransport<RoleClient, E, A>,
//!     E: std::error::Error + Send + Sync + 'static,
//! {
//!     let client = rmcp::model::ClientConfig::default().serve(transport).await?;
//!     let connection = McpConnection::new(client);
//!     let tools = connection.tools().await?;
//!     Ok(())
//! }
//! ```

use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures::FutureExt;
use futures::future::{BoxFuture, Shared};
use rmcp::model::{
    ClientRequest, ListToolsRequest, PaginatedRequestParams, ReadResourceRequestParams, Resource,
    ResourceContents, ServerResult,
};
use rmcp::service::{
    RoleClient, RunningService, RunningServiceCancellationToken, ServerSink, Service,
};

use rig_core::effect::{
    Answered, EffectFamily, EffectKind, FamilyDescriptor, HandlerDescriptor, HandlerKey,
    RetrieveQuery, RetrievedDocuments, family,
};
use rig_core::error::{ErrorKind, ErrorReport};
use rig_core::serve::{Dispatch, Reply, Serve};

use crate::{DEFAULT_MCP_REFRESH_TIMEOUT, McpClientError, McpTool, send_mcp_request};

/// A live MCP client connection.
///
/// Clones share one connection, and every [`McpTool`] built from it holds a
/// clone. The connection closes once the last clone is dropped, so an agent
/// that keeps its tools keeps their server connected.
#[derive(Clone)]
pub struct McpConnection(Arc<Inner>);

struct Inner {
    peer: ServerSink,
    /// Owns the running service; dropping it closes the connection.
    service: Shared<BoxFuture<'static, ()>>,
    /// Taken by the first [`McpConnection::close`].
    cancel: Mutex<Option<RunningServiceCancellationToken>>,
}

impl McpConnection {
    /// Take ownership of a connected rmcp client.
    pub fn new<S: Service<RoleClient>>(service: RunningService<RoleClient, S>) -> Self {
        let peer = service.peer().clone();
        let cancel = Mutex::new(Some(service.cancellation_token()));
        let service = async move {
            // A connection that ends on its own is already logged by rmcp.
            let _ = service.waiting().await;
        }
        .boxed()
        .shared();
        Self(Arc::new(Inner {
            peer,
            service,
            cancel,
        }))
    }

    /// The rmcp handle for requests this type does not wrap.
    pub fn peer(&self) -> &ServerSink {
        &self.0.peer
    }

    /// Whether the connection has ended.
    pub fn is_closed(&self) -> bool {
        self.0.peer.is_transport_closed()
    }

    /// Wait until the server or transport ends the connection.
    pub async fn closed(&self) {
        self.0.service.clone().await;
    }

    /// Close the connection for every clone and tool sharing it, and wait
    /// until it has closed.
    pub async fn close(&self) {
        let cancel = self
            .0
            .cancel
            .lock()
            .ok()
            .and_then(|mut cancel| cancel.take());
        if let Some(cancel) = cancel {
            cancel.cancel();
        }
        self.closed().await;
    }

    /// Every tool the server lists, each keeping this connection open.
    ///
    /// # Errors
    ///
    /// Fails when a request fails or the list takes longer than
    /// [`DEFAULT_MCP_REFRESH_TIMEOUT`].
    pub async fn tools(&self) -> Result<Vec<McpTool>, McpClientError> {
        Ok(list_tools(self.peer(), DEFAULT_MCP_REFRESH_TIMEOUT)
            .await?
            .into_iter()
            .map(|definition| McpTool::new(definition, self))
            .collect())
    }

    /// Every resource the server lists.
    ///
    /// # Errors
    ///
    /// Fails when a request fails.
    pub async fn resources(&self) -> Result<Vec<Resource>, McpClientError> {
        Ok(self.peer().list_all_resources().await?)
    }

    /// The contents of the resource at `uri`.
    ///
    /// # Errors
    ///
    /// Fails when the request fails, including for an unknown resource.
    pub async fn read_resource(
        &self,
        uri: impl Into<String>,
    ) -> Result<Vec<ResourceContents>, McpClientError> {
        let result = self
            .peer()
            .read_resource(ReadResourceRequestParams::new(uri))
            .await?;
        Ok(result.contents)
    }

    /// The resources at `uris` as agent context, read again for every model
    /// call. Pass it to rig-agent's `AgentBuilder::dynamic_context_handler`.
    pub fn resource_context<I>(&self, uris: I) -> McpResourceContext
    where
        I: IntoIterator,
        I::Item: Into<String>,
    {
        McpResourceContext {
            connection: self.clone(),
            uris: uris.into_iter().map(Into::into).collect(),
        }
    }
}

impl<S: Service<RoleClient>> From<RunningService<RoleClient, S>> for McpConnection {
    fn from(service: RunningService<RoleClient, S>) -> Self {
        Self::new(service)
    }
}

/// Lists all of a server's tools, page by page, within `timeout`.
pub(crate) async fn list_tools(
    peer: &ServerSink,
    timeout: Duration,
) -> Result<Vec<rmcp::model::Tool>, McpClientError> {
    let deadline = tokio::time::Instant::now() + timeout;
    let mut tools = Vec::new();
    let mut cursor = None;

    loop {
        let mut params = PaginatedRequestParams::default();
        params.cursor = cursor;
        let response = send_mcp_request(
            peer,
            ClientRequest::ListToolsRequest(ListToolsRequest::with_param(params)),
            Some((deadline, timeout)),
        )
        .await
        .map_err(|error| match error {
            rmcp::ServiceError::Timeout { .. } => McpClientError::Timeout(timeout),
            error => McpClientError::Request(error),
        })?;
        let ServerResult::ListToolsResult(page) = response else {
            return Err(McpClientError::Request(
                rmcp::ServiceError::UnexpectedResponse,
            ));
        };
        tools.extend(page.tools);
        cursor = page.next_cursor;
        if cursor.is_none() {
            return Ok(tools);
        }
    }
}

/// MCP resources as a retrieval handler: each retrieval reads the resources
/// again and answers with their contents, at most the requested number.
/// Build it with [`McpConnection::resource_context`].
pub struct McpResourceContext {
    connection: McpConnection,
    uris: Vec<String>,
}

impl McpResourceContext {
    async fn read(&self, samples: usize) -> Result<Vec<ResourceContents>, ErrorReport> {
        let mut contents = Vec::new();
        for uri in &self.uris {
            if contents.len() >= samples {
                break;
            }
            let read = self
                .connection
                .read_resource(uri.as_str())
                .await
                .map_err(|error| {
                    ErrorReport::new(
                        ErrorKind::Provider,
                        format!("failed to read MCP resource `{uri}`: {error}"),
                    )
                })?;
            contents.extend(read);
        }
        contents.truncate(samples);
        Ok(contents)
    }
}

fn resource_uri(contents: &ResourceContents) -> String {
    match contents {
        ResourceContents::TextResourceContents { uri, .. }
        | ResourceContents::BlobResourceContents { uri, .. } => uri.clone(),
        _ => String::new(),
    }
}

impl Serve for McpResourceContext {
    type Family = family::Retrieve;

    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: HandlerKey::from("retrieve"),
            family: FamilyDescriptor::Retrieve {},
            layers: Vec::new(),
        }
    }

    async fn serve(&self, kind: EffectKind, _dispatch: Dispatch) -> Reply {
        let query = match kind {
            EffectKind::Retrieve { query } => query,
            other => {
                return Reply::Outcome(Err(ErrorReport::new(
                    ErrorKind::HandlerUnavailable,
                    format!(
                        "a {} handler cannot serve a `{}` effect",
                        EffectFamily::Retrieve,
                        other.name()
                    ),
                )));
            }
        };
        let (RetrieveQuery::TopN { req } | RetrieveQuery::TopNIds { req }) = &query;
        let samples = usize::try_from(req.samples()).unwrap_or(usize::MAX);
        let outcome = self.read(samples).await.and_then(|contents| {
            let documents = match query {
                RetrieveQuery::TopN { .. } => RetrievedDocuments::Scored(
                    contents
                        .iter()
                        .map(|contents| {
                            serde_json::to_value(contents)
                                .map(|document| (1.0, resource_uri(contents), document))
                        })
                        .collect::<Result<_, _>>()
                        .map_err(|error| ErrorReport::new(ErrorKind::Json, error.to_string()))?,
                ),
                RetrieveQuery::TopNIds { .. } => RetrievedDocuments::Ids(
                    contents
                        .iter()
                        .map(|contents| (1.0, resource_uri(contents)))
                        .collect(),
                ),
            };
            Ok(Answered::Documents(documents))
        });
        Reply::Outcome(outcome)
    }
}
