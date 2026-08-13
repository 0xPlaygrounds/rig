//! Extension trait adding [`a2a_tool`](A2AAgentBuilderExt::a2a_tool) to Rig's
//! [`AgentBuilder`], so a remote A2A agent can be attached to a local agent in
//! one call.
//!
//! Typical use:
//!
//! ```no_run
//! use rig_a2a::{A2AAgentBuilderExt, A2AClient};
//! use rig_agent::client::AgentClientExt;
//! use rig_core::client::ProviderClient;
//! use rig_core::providers::openai;
//!
//! # async fn run() -> anyhow::Result<()> {
//! let openai_client = openai::Client::from_env()?;
//! let remote = A2AClient::from_url("http://localhost:8080").await?;
//! let agent = openai_client
//!     .agent(openai::GPT_4O_MINI)
//!     .a2a_tool(&remote)
//!     .build();
//! # Ok(()) }
//! ```
//!
//! The remote agent contributes exactly one tool, named after its card and
//! documented with the skills the card declares. Chain the call once per
//! remote to reach several A2A agents from one Rig agent.

use rig_agent::agent::{AgentBuilder, NoToolConfig, WithBuilderTools};

use crate::client::A2AClient;

/// Extension trait adding an [`a2a_tool`](Self::a2a_tool) shortcut to
/// [`AgentBuilder`].
pub trait A2AAgentBuilderExt: Sized {
    /// Register `client`'s remote agent as a tool on this agent.
    ///
    /// Equivalent to `.dynamic_tool(client.tool())`. The tool set is frozen
    /// when `.build()` runs; to add a remote agent to an already-built agent,
    /// register `client.tool()` on a shared
    /// [`ToolServerHandle`](rig_agent::tool::server::ToolServerHandle)
    /// instead.
    fn a2a_tool(self, client: &A2AClient) -> AgentBuilder<WithBuilderTools>;
}

impl A2AAgentBuilderExt for AgentBuilder<NoToolConfig> {
    fn a2a_tool(self, client: &A2AClient) -> AgentBuilder<WithBuilderTools> {
        self.dynamic_tool(client.tool())
    }
}

impl A2AAgentBuilderExt for AgentBuilder<WithBuilderTools> {
    fn a2a_tool(self, client: &A2AClient) -> AgentBuilder<WithBuilderTools> {
        self.dynamic_tool(client.tool())
    }
}
