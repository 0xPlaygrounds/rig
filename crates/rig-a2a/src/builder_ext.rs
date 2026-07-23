//! Extension trait that adds an [`a2a_tools`](A2AAgentBuilderExt::a2a_tools)
//! to add a remote A2A server's skills as tools to Rig's [`AgentBuilder`].
//!
//! Typical use:
//!
//! ```no_run
//! use rig_a2a::{A2AAgentBuilderExt, A2AClient};
//! use rig_core::client::{CompletionClient, ProviderClient};
//! use rig_core::providers::openai;
//!
//! # async fn run() -> anyhow::Result<()> {
//! let openai_client = openai::Client::from_env()?;
//! let remote = A2AClient::from_url("http://localhost:8080").await?;
//! let agent = openai_client
//!     .agent(openai::GPT_4O_MINI)
//!     .a2a_tools(&remote)
//!     .build();
//! # Ok(()) }
//! ```
//!
//! This produces a static tool layout: every skill on `remote` is bound
//! into the agent's tool set at build time, and the set never changes after
//! `.build()`.

use rig_core::agent::{AgentBuilder, NoToolConfig, WithBuilderTools};
use rig_core::completion::CompletionModel;

use crate::client::A2AClient;

/// Extension trait adding an [`a2a_tools`](Self::a2a_tools) shortcut to
/// [`AgentBuilder`].
pub trait A2AAgentBuilderExt<M>: Sized
where
    M: CompletionModel,
{
    /// Register every skill on `client` as a static tool on this agent.
    ///
    /// Mirrors [`AgentBuilder::rmcp_tools`](rig_core::agent::AgentBuilder)
    /// for the A2A protocol. The tool layout is frozen at build time.
    fn a2a_tools(self, client: &A2AClient) -> AgentBuilder<M, WithBuilderTools>;
}

impl<M> A2AAgentBuilderExt<M> for AgentBuilder<M, NoToolConfig>
where
    M: CompletionModel,
{
    fn a2a_tools(self, client: &A2AClient) -> AgentBuilder<M, WithBuilderTools> {
        self.dynamic_tools(client.dynamic_tools())
    }
}

impl<M> A2AAgentBuilderExt<M> for AgentBuilder<M, WithBuilderTools>
where
    M: CompletionModel,
{
    fn a2a_tools(self, client: &A2AClient) -> AgentBuilder<M, WithBuilderTools> {
        self.dynamic_tools(client.dynamic_tools())
    }
}
