//! The agent card and the Axum router that serve an agent over A2A.

use std::sync::Arc;

use a2a::{
    AgentCapabilities, AgentCard, AgentInterface, AgentSkill, TRANSPORT_PROTOCOL_HTTP_JSON,
    TRANSPORT_PROTOCOL_JSONRPC,
};
use a2a_server::agent_card::agent_card_router;
use a2a_server::jsonrpc::jsonrpc_router;
use a2a_server::rest::rest_router;
use a2a_server::{DefaultRequestHandler, InMemoryTaskStore, StaticAgentCard, TaskStore};
use rig_agent::Agent;

use crate::A2aExecutor;

/// The path of the JSON-RPC binding under the server's base URL.
pub const JSONRPC_PATH: &str = "/jsonrpc";
/// The path of the HTTP+JSON binding under the server's base URL.
pub const REST_PATH: &str = "/rest";

/// A Rig agent served over A2A: its agent card, its task store, and the
/// router [`into_router`](Self::into_router) builds from them.
///
/// The router serves the card at `/.well-known/agent-card.json`, the JSON-RPC
/// binding at [`JSONRPC_PATH`] and the HTTP+JSON binding at [`REST_PATH`].
/// Streaming is on; push notifications are off.
pub struct A2aServer<S = InMemoryTaskStore> {
    executor: A2aExecutor,
    card: AgentCard,
    store: S,
}

impl A2aServer {
    /// Serve `agent` at `url`, the base URL clients reach this server at.
    ///
    /// The card takes the agent's name and description, lists both bindings
    /// under `url`, accepts `text/plain` and `application/json` parts, and
    /// declares one `chat` skill. Tasks live in an [`InMemoryTaskStore`].
    pub fn new(agent: Agent, url: impl AsRef<str>) -> Self {
        let card = default_card(&agent, url.as_ref());
        Self {
            executor: A2aExecutor::new(agent),
            card,
            store: InMemoryTaskStore::new(),
        }
    }
}

impl<S: TaskStore> A2aServer<S> {
    /// Change the agent card, for example its skills, version or provider.
    ///
    /// The router serves the card as edited and refuses a part that declares
    /// a media type outside its `default_input_modes`. The routes stay at
    /// [`JSONRPC_PATH`] and [`REST_PATH`] whatever the card's interfaces say.
    #[must_use]
    pub fn edit_card(mut self, edit: impl FnOnce(&mut AgentCard)) -> Self {
        edit(&mut self.card);
        self
    }

    /// Keep tasks in `store` instead of in memory.
    pub fn task_store<T: TaskStore>(self, store: T) -> A2aServer<T> {
        A2aServer {
            executor: self.executor,
            card: self.card,
            store,
        }
    }

    /// The agent card the router serves.
    pub fn card(&self) -> &AgentCard {
        &self.card
    }

    /// The Axum router serving the card and both bindings.
    pub fn into_router(self) -> axum::Router {
        let handler = Arc::new(
            DefaultRequestHandler::new(self.executor, self.store)
                .with_capabilities(self.card.capabilities.clone())
                .with_default_input_modes(self.card.default_input_modes.clone()),
        );
        axum::Router::new()
            .nest(JSONRPC_PATH, jsonrpc_router(Arc::clone(&handler)))
            .nest(REST_PATH, rest_router(handler))
            .merge(agent_card_router(Arc::new(StaticAgentCard::new(self.card))))
    }
}

fn default_card(agent: &Agent, url: &str) -> AgentCard {
    let url = url.trim_end_matches('/');
    let name = agent.name().unwrap_or("Rig agent").to_owned();
    let description = agent
        .description()
        .unwrap_or("An agent built with Rig.")
        .to_owned();
    AgentCard {
        skills: vec![AgentSkill {
            id: "chat".to_owned(),
            name: name.clone(),
            description: description.clone(),
            tags: vec!["chat".to_owned()],
            examples: None,
            input_modes: None,
            output_modes: None,
            security_requirements: None,
        }],
        name,
        description,
        version: "1.0.0".to_owned(),
        supported_interfaces: vec![
            AgentInterface::new(format!("{url}{JSONRPC_PATH}"), TRANSPORT_PROTOCOL_JSONRPC),
            AgentInterface::new(format!("{url}{REST_PATH}"), TRANSPORT_PROTOCOL_HTTP_JSON),
        ],
        capabilities: AgentCapabilities {
            streaming: Some(true),
            push_notifications: Some(false),
            extensions: None,
            extended_agent_card: None,
        },
        default_input_modes: vec!["text/plain".to_owned(), "application/json".to_owned()],
        default_output_modes: vec!["text/plain".to_owned()],
        provider: None,
        documentation_url: None,
        icon_url: None,
        security_schemes: None,
        security_requirements: None,
        signatures: None,
    }
}
