//! A2A client that fetches an `AgentCard` and exposes the remote agent.

use std::{sync::Arc, time::Duration};

use a2a::{
    AgentCard, AgentInterface, Message, Part, Role, SendMessageRequest,
    TRANSPORT_PROTOCOL_HTTP_JSON, TRANSPORT_PROTOCOL_JSONRPC,
};
use a2a_client::{A2AClient as InnerClient, A2AClientFactory, Transport};
use a2a_client::{jsonrpc::JsonRpcTransportFactory, rest::RestTransportFactory};

use crate::error::{A2AError, AgentCardError};
use crate::tool::A2ATool;

pub use a2a::SendMessageResponse;

/// Default timeout for agent-card discovery and A2A protocol requests.
pub const DEFAULT_HTTP_TIMEOUT: Duration = Duration::from_secs(300);

/// High-level client for a remote A2A agent.
///
/// Constructed via [`A2AClient::builder`] (or the [`Self::from_url`] /
/// [`Self::from_agent_card`] convenience wrappers for the common cases).
///
/// The builder lets you configure the source ([`A2AClientBuilder::url`] or
/// [`A2AClientBuilder::card`]) and the HTTP client builder used for the
/// agent-card fetch and the default JSON-RPC / REST transports.
pub struct A2AClient {
    card: AgentCard,
    interface: AgentInterface,
    inner: Arc<InnerClient<Box<dyn Transport>>>,
}

/// Builder for [`A2AClient`].
///
/// At minimum, set the source via [`Self::url`] or [`Self::card`] before
/// calling [`Self::build`]. Everything else has a sensible default.
pub struct A2AClientBuilder {
    source: Option<ClientSource>,
    http_client_builder: Option<reqwest::ClientBuilder>,
    timeout: Duration,
    allow_cross_origin_interfaces: bool,
}

enum ClientSource {
    Url(String),
    Card(Box<AgentCard>),
}

impl Default for A2AClientBuilder {
    fn default() -> Self {
        Self {
            source: None,
            http_client_builder: None,
            timeout: DEFAULT_HTTP_TIMEOUT,
            allow_cross_origin_interfaces: false,
        }
    }
}

impl A2AClientBuilder {
    /// Fetch the agent's well-known [`AgentCard`] from `base_url` when
    /// [`Self::build`] runs. Mutually exclusive with [`Self::card`]
    /// (last write wins).
    pub fn url(mut self, base_url: impl Into<String>) -> Self {
        self.source = Some(ClientSource::Url(base_url.into()));
        self
    }

    /// Use the supplied [`AgentCard`] directly instead of fetching it.
    /// Mutually exclusive with [`Self::url`] (last write wins).
    pub fn card(mut self, card: AgentCard) -> Self {
        self.source = Some(ClientSource::Card(Box::new(card)));
        self
    }

    /// Customize the HTTP client used for agent-card discovery and A2A
    /// protocol requests.
    ///
    /// Configure custom root certificates, proxies, or default headers on the
    /// builder before passing it in. Rig applies its timeout and disables HTTP
    /// redirects when [`Self::build`] runs, so a fetched card cannot bypass
    /// same-origin interface validation through a redirect.
    pub fn http_client_builder(mut self, http_client_builder: reqwest::ClientBuilder) -> Self {
        self.http_client_builder = Some(http_client_builder);
        self
    }

    /// Set the timeout for agent-card discovery and each A2A protocol request.
    ///
    /// The default is [`DEFAULT_HTTP_TIMEOUT`].
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Allow a card fetched from [`Self::url`] to select an interface on a
    /// different origin than the well-known card URL.
    ///
    /// By default, the selected interface of a URL-sourced card is same-origin
    /// constrained so a hostile card cannot redirect subsequent A2A calls (and
    /// any configured default HTTP headers) to an internal or unrelated host.
    /// Enable this only for trusted deployments that intentionally publish a
    /// cross-origin interface.
    pub fn allow_cross_origin_interfaces(mut self, allow: bool) -> Self {
        self.allow_cross_origin_interfaces = allow;
        self
    }

    /// Resolve the configured source into an [`A2AClient`].
    ///
    /// Returns [`AgentCardError::MissingSource`] if neither [`Self::url`]
    /// nor [`Self::card`] was called.
    pub async fn build(self) -> Result<A2AClient, A2AError> {
        let source = self
            .source
            .ok_or(A2AError::AgentCard(AgentCardError::MissingSource))?;
        let http_client = self
            .http_client_builder
            .unwrap_or_default()
            .redirect(reqwest::redirect::Policy::none())
            .timeout(self.timeout)
            .build()?;
        let (card, fetched_from) = match source {
            ClientSource::Url(u) => {
                let card = fetch_agent_card(&http_client, &u).await?;
                (card, Some(u))
            }
            ClientSource::Card(c) => (*c, None),
        };

        // A2A requires clients to choose the first interface they support in
        // the card's declared order. Pin the SDK factory to that one interface:
        // its default negotiation otherwise applies client-side protocol
        // preference (JSON-RPC before HTTP+JSON), which can reverse card order.
        let interface = select_interface(&card)?.clone();
        let interface_url = validate_http_interface_url(&interface)?;
        if let Some(base_url) = fetched_from
            && !self.allow_cross_origin_interfaces
        {
            validate_same_origin_interface(&base_url, &interface.url, &interface_url)?;
        }

        // Register the default JSON-RPC / REST transports explicitly, wired to
        // the configured `http_client` (auth headers, TLS roots, proxy), rather
        // than letting the factory auto-fill defaults with fresh `reqwest`
        // clients that would silently drop the caller's HTTP configuration
        // from protocol calls.
        let factory = A2AClientFactory::builder()
            .no_defaults()
            .register(Arc::new(JsonRpcTransportFactory::new(Some(
                http_client.clone(),
            ))))
            .register(Arc::new(RestTransportFactory::new(Some(http_client))))
            .build();
        let mut selected_card = card.clone();
        selected_card.supported_interfaces = vec![interface.clone()];
        let inner = factory.create_from_card(&selected_card).await?;
        Ok(A2AClient {
            card,
            interface,
            inner: Arc::new(inner),
        })
    }
}

impl A2AClient {
    /// Start a new [`A2AClientBuilder`].
    pub fn builder() -> A2AClientBuilder {
        A2AClientBuilder::default()
    }

    /// Convenience for `A2AClient::builder().url(base_url).build().await`.
    /// Uses a default `reqwest::Client`; for custom TLS, proxy, or auth
    /// headers, use the builder directly and chain
    /// [`A2AClientBuilder::http_client_builder`].
    pub async fn from_url(base_url: impl Into<String>) -> Result<Self, A2AError> {
        Self::builder().url(base_url).build().await
    }

    /// Convenience for `A2AClient::builder().card(card).build().await`.
    pub async fn from_agent_card(card: AgentCard) -> Result<Self, A2AError> {
        Self::builder().card(card).build().await
    }

    /// Return the cached `AgentCard` for the remote agent.
    pub fn card(&self) -> &AgentCard {
        &self.card
    }

    /// Return the `AgentInterface` selected for this client.
    ///
    /// A2A clients use the first interface they support in card order. Its
    /// optional `tenant` is copied to requests sent through this wrapper.
    pub fn interface(&self) -> &AgentInterface {
        &self.interface
    }

    /// Produce one Rig [`A2ATool`] per skill declared in the remote
    /// `AgentCard`. If the card declares no skills, a single passthrough
    /// tool is returned that proxies any input to the agent.
    ///
    /// Tool names are provider-safe projections of the agent identity and
    /// skill id, with a short stable hash of the original skill id, so two
    /// distinct skill ids always produce distinct tool names.
    ///
    /// The projection is deterministic in the agent name and skill id, so it
    /// does **not** disambiguate two *different* remote agents that share both
    /// an agent name (e.g. the `"rig-agent"` default) and a skill id: their
    /// tools would collide if merged into one tool set.
    pub fn tools(&self) -> Vec<A2ATool> {
        if self.card.skills.is_empty() {
            return vec![A2ATool::passthrough(
                self.card.name.clone(),
                self.card.description.clone(),
                self.interface.tenant.clone(),
                self.inner.clone(),
            )];
        }

        self.card
            .skills
            .iter()
            .cloned()
            .map(|skill| {
                A2ATool::new(
                    self.card.name.clone(),
                    skill,
                    self.interface.tenant.clone(),
                    self.inner.clone(),
                )
            })
            .collect()
    }

    /// Produce the same tools as [`Self::tools`], converted into Rig
    /// [`DynamicTool`]s ready for registration.
    ///
    /// Use this with [`AgentBuilder::dynamic_tools`] (which is what
    /// [`A2AAgentBuilderExt::a2a_tools`](crate::A2AAgentBuilderExt::a2a_tools)
    /// does), or with [`ToolServerHandle::add_dynamic_tool`] to register the
    /// remote agent's skills on a shared tool server that pre-existing agents
    /// were built from.
    ///
    /// [`AgentBuilder::dynamic_tools`]: rig_core::agent::AgentBuilder::dynamic_tools
    /// [`ToolServerHandle::add_dynamic_tool`]: rig_core::tool::server::ToolServerHandle::add_dynamic_tool
    /// [`DynamicTool`]: rig_core::tool::DynamicTool
    pub fn dynamic_tools(&self) -> Vec<rig_core::tool::DynamicTool> {
        self.tools()
            .into_iter()
            .map(A2ATool::into_dynamic_tool)
            .collect()
    }

    /// Start a request to the remote agent. Chain [`A2ARequest::context`]
    /// and [`A2ARequest::task`] to echo server-generated ids from a previous
    /// response, then call [`A2ARequest::send`].
    ///
    /// Multi-turn conversations echo the `contextId` the server minted on
    /// the first response (per the A2A spec, clients do not generate ids):
    ///
    /// ```no_run
    /// # async fn run(client: rig_a2a::A2AClient) -> Result<(), rig_a2a::A2AError> {
    /// let rig_a2a::SendMessageResponse::Task(task) = client.message("turn 1").send().await? else {
    ///     return Ok(());
    /// };
    /// client.message("turn 2").context(&task.context_id).send().await?;
    /// # Ok(()) }
    /// ```
    pub fn message(&self, text: impl Into<String>) -> A2ARequest<'_> {
        A2ARequest {
            client: self,
            text: text.into(),
            context_id: None,
            task_id: None,
        }
    }

    /// Borrow the inner [`a2a_client::A2AClient`] for advanced use cases not
    /// covered by this wrapper (e.g. streaming, `tasks/list`, push
    /// notification config). Requests made through the inner client are not
    /// automatically populated with the selected interface's tenant; callers
    /// can read it from [`Self::interface`].
    pub fn inner(&self) -> &Arc<InnerClient<Box<dyn Transport>>> {
        &self.inner
    }
}

/// A single A2A request in flight. Returned by [`A2AClient::message`].
///
/// Chain [`Self::context`] / [`Self::task`] to attach a `Message.contextId`
/// / `Message.taskId`, then call [`Self::send`].
pub struct A2ARequest<'a> {
    client: &'a A2AClient,
    text: String,
    context_id: Option<String>,
    task_id: Option<String>,
}

impl<'a> A2ARequest<'a> {
    /// Attach `Message.contextId` to this request. Echo the contextId from a
    /// previous response to continue that conversation; per the A2A spec,
    /// context ids are generated by the server, not the client. Empty ids
    /// are rejected by [`Self::send`].
    pub fn context(mut self, id: impl Into<String>) -> Self {
        self.context_id = Some(id.into());
        self
    }

    /// Attach `Message.taskId` to this request. Echo the id of a task in a
    /// non-terminal state (e.g. `input-required`) to resume it; the A2A spec
    /// does not allow sending messages to tasks in a terminal state. Empty
    /// ids are rejected by [`Self::send`].
    pub fn task(mut self, id: impl Into<String>) -> Self {
        self.task_id = Some(id.into());
        self
    }

    /// Send the message and return the server's `SendMessageResponse`.
    ///
    /// The response is either a [`Task`](a2a::Task) (terminal or
    /// non-terminal — match on `task.status.state` to react to
    /// `InputRequired` / `AuthRequired` / `Failed` / `Canceled` etc.) or a
    /// bare [`Message`]. The caller decides what counts as success.
    pub async fn send(self) -> Result<SendMessageResponse, A2AError> {
        let blank = |id: &Option<String>| id.as_deref().is_some_and(|id| id.trim().is_empty());
        if blank(&self.context_id) {
            return Err(A2AError::InvalidContextId);
        }
        if blank(&self.task_id) {
            return Err(A2AError::InvalidTaskId);
        }
        let mut msg = Message::new(Role::User, vec![Part::text(self.text)]);
        msg.context_id = self.context_id;
        msg.task_id = self.task_id;
        let req = SendMessageRequest {
            message: msg,
            configuration: None,
            metadata: None,
            tenant: self.client.interface.tenant.clone(),
        };
        Ok(self.client.inner.send_message(&req).await?)
    }
}

async fn fetch_agent_card(client: &reqwest::Client, base_url: &str) -> Result<AgentCard, A2AError> {
    const MAX_AGENT_CARD_BYTES: usize = 1024 * 1024;

    // Per RFC 8615 the well-known agent card lives at the origin root. Joining
    // the *absolute* well-known path resolves against the origin (scheme +
    // authority) and discards any path, query, or fragment carried by
    // `base_url`, so a path-bearing or query-bearing base still discovers the
    // card at the root.
    let card_url = url::Url::parse(base_url)?.join(crate::WELL_KNOWN_AGENT_CARD_PATH)?;
    let resp = client.get(card_url).send().await?;
    let status = resp.status();
    let body = read_limited_body(resp, MAX_AGENT_CARD_BYTES).await?;
    if !status.is_success() {
        // Truncate the snippet so we don't blow up error messages on
        // servers that return arbitrarily large HTML 5xx pages.
        let snippet = std::str::from_utf8(&body)
            .ok()
            .map(|s| s.chars().take(512).collect::<String>());
        return Err(A2AError::AgentCard(AgentCardError::FetchFailed {
            status,
            body: snippet,
        }));
    }
    let card = serde_json::from_slice::<AgentCard>(&body)?;
    Ok(card)
}

fn select_interface(card: &AgentCard) -> Result<&AgentInterface, A2AError> {
    card.supported_interfaces
        .iter()
        .find(|interface| {
            matches!(
                interface.protocol_binding.as_str(),
                TRANSPORT_PROTOCOL_JSONRPC | TRANSPORT_PROTOCOL_HTTP_JSON
            ) && protocol_version_matches(&interface.protocol_version)
        })
        .ok_or_else(|| AgentCardError::NoSupportedInterface.into())
}

fn protocol_version_matches(version: &str) -> bool {
    fn major_minor(version: &str) -> Option<(u64, u64)> {
        let mut parts = version.split('.');
        Some((parts.next()?.parse().ok()?, parts.next()?.parse().ok()?))
    }

    let (Some((interface_major, interface_minor)), Some((client_major, client_minor))) =
        (major_minor(version), major_minor(a2a::VERSION))
    else {
        return false;
    };
    interface_major == client_major && interface_minor >= client_minor
}

fn validate_http_interface_url(interface: &AgentInterface) -> Result<url::Url, A2AError> {
    let parsed = url::Url::parse(&interface.url).map_err(|_| {
        A2AError::AgentCard(AgentCardError::InvalidInterfaceUrl {
            interface_url: interface.url.clone(),
        })
    })?;
    if !matches!(parsed.scheme(), "http" | "https") || parsed.host_str().is_none() {
        return Err(A2AError::AgentCard(AgentCardError::InvalidInterfaceUrl {
            interface_url: interface.url.clone(),
        }));
    }
    Ok(parsed)
}

fn validate_same_origin_interface(
    base_url: &str,
    interface_url_text: &str,
    interface_url: &url::Url,
) -> Result<(), A2AError> {
    let base = url::Url::parse(base_url)?.join(crate::WELL_KNOWN_AGENT_CARD_PATH)?;
    let base_origin = origin_string(&base);
    if interface_url.scheme() != base.scheme()
        || interface_url.host_str() != base.host_str()
        || interface_url.port_or_known_default() != base.port_or_known_default()
    {
        return Err(A2AError::AgentCard(AgentCardError::CrossOriginInterface {
            base_origin,
            interface_url: interface_url_text.to_string(),
        }));
    }
    Ok(())
}

fn origin_string(url: &url::Url) -> String {
    url.origin().ascii_serialization()
}

async fn read_limited_body(resp: reqwest::Response, limit: usize) -> Result<Vec<u8>, A2AError> {
    use futures::StreamExt;

    if let Some(len) = resp.content_length()
        && len > limit as u64
    {
        return Err(A2AError::AgentCard(AgentCardError::ResponseTooLarge {
            limit,
        }));
    }

    let mut body = Vec::new();
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        if body.len().saturating_add(chunk.len()) > limit {
            return Err(A2AError::AgentCard(AgentCardError::ResponseTooLarge {
                limit,
            }));
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

#[cfg(test)]
mod tests {
    use super::protocol_version_matches;

    #[test]
    fn protocol_version_matches_supported_major_and_minor() {
        assert!(protocol_version_matches("1.0"));
        assert!(protocol_version_matches("1.0.1"));
        assert!(protocol_version_matches("1.1"));
        assert!(!protocol_version_matches("0.3"));
        assert!(!protocol_version_matches("2.0"));
        assert!(!protocol_version_matches("1"));
        assert!(!protocol_version_matches("invalid"));
    }
}
