//! A stub A2A agent for the offline examples.
//!
//! Boots the upstream `a2a-server` handler on a loopback port and answers each
//! request with the next reply in a script. This is scaffolding, not the
//! lesson — it stands in for whatever real A2A agent you would point at.

#![allow(dead_code, clippy::expect_used)]

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use a2a::{
    A2AError, AgentCapabilities, AgentCard, AgentInterface, AgentSkill, Artifact, Message, Part,
    Role, StreamResponse, TRANSPORT_PROTOCOL_JSONRPC, Task, TaskState, TaskStatus, new_artifact_id,
};
use a2a_server::task_store::InMemoryTaskStore;
use a2a_server::{
    AgentExecutor, DefaultRequestHandler, ExecutorContext, StaticAgentCard, agent_card,
    jsonrpc::jsonrpc_router,
};
use futures::StreamExt;
use futures::stream::BoxStream;

/// One scripted answer from the stub agent.
#[derive(Clone)]
pub enum Reply {
    Completed(String),
    /// The agent pauses and asks for more input before it can finish.
    InputRequired(String),
    Failed(String),
}

impl Reply {
    pub fn completed(text: &str) -> Self {
        Self::Completed(text.to_string())
    }

    pub fn input_required(prompt: &str) -> Self {
        Self::InputRequired(prompt.to_string())
    }

    pub fn failed(reason: &str) -> Self {
        Self::Failed(reason.to_string())
    }
}

/// A running stub agent. Dropping it shuts the server down.
pub struct StubAgent {
    pub base_url: String,
    handle: tokio::task::JoinHandle<()>,
}

impl Drop for StubAgent {
    fn drop(&mut self) {
        self.handle.abort();
    }
}

/// Boot a stub A2A agent on a random loopback port, answering with `script`.
///
/// The last reply repeats once the script runs out.
pub async fn serve_stub_agent(script: impl IntoIterator<Item = Reply>) -> StubAgent {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind loopback");
    let addr = listener.local_addr().expect("local addr");
    let base_url = format!("http://{addr}");

    let handler = Arc::new(DefaultRequestHandler::new(
        ScriptedAgent {
            replies: Mutex::new(script.into_iter().collect()),
        },
        InMemoryTaskStore::new(),
    ));
    let router = axum::Router::new()
        .merge(agent_card::agent_card_router(Arc::new(
            StaticAgentCard::new(demo_card(&base_url)),
        )))
        .merge(axum::Router::new().nest("/jsonrpc", jsonrpc_router(handler)));

    let handle = tokio::spawn(async move {
        axum::serve(listener, router).await.expect("stub server");
    });
    StubAgent { base_url, handle }
}

struct ScriptedAgent {
    replies: Mutex<VecDeque<Reply>>,
}

impl ScriptedAgent {
    fn next_reply(&self) -> Reply {
        let mut replies = self.replies.lock().expect("script lock");
        if replies.len() > 1 {
            replies.pop_front().expect("non-empty script")
        } else {
            replies.front().cloned().expect("non-empty script")
        }
    }
}

impl AgentExecutor for ScriptedAgent {
    fn execute(
        &self,
        ctx: ExecutorContext,
    ) -> BoxStream<'static, Result<StreamResponse, A2AError>> {
        let (state, status, artifact) = match self.next_reply() {
            Reply::Completed(text) => (TaskState::Completed, None, Some(text)),
            Reply::InputRequired(prompt) => (TaskState::InputRequired, Some(prompt), None),
            Reply::Failed(reason) => (TaskState::Failed, Some(reason), None),
        };
        let task = Task {
            id: ctx.task_id,
            context_id: ctx.context_id,
            status: TaskStatus {
                state,
                message: status.map(|text| Message::new(Role::Agent, vec![Part::text(text)])),
                timestamp: None,
            },
            artifacts: artifact.map(|text| {
                vec![Artifact {
                    artifact_id: new_artifact_id(),
                    name: Some("reply".to_string()),
                    description: None,
                    parts: vec![Part::text(text)],
                    metadata: None,
                    extensions: None,
                }]
            }),
            history: None,
            metadata: None,
        };
        futures::stream::once(async move { Ok(StreamResponse::Task(task)) }).boxed()
    }

    fn cancel(&self, ctx: ExecutorContext) -> BoxStream<'static, Result<StreamResponse, A2AError>> {
        let task_id = ctx.task_id;
        futures::stream::once(async move { Err(A2AError::task_not_cancelable(&task_id)) }).boxed()
    }
}

/// A card declaring several skills, to show that they become one tool.
fn demo_card(base_url: &str) -> AgentCard {
    AgentCard {
        name: "librarian".to_string(),
        description: "Answers questions about a document collection.".to_string(),
        version: "1.0.0".to_string(),
        supported_interfaces: vec![AgentInterface::new(
            format!("{base_url}/jsonrpc"),
            TRANSPORT_PROTOCOL_JSONRPC,
        )],
        capabilities: AgentCapabilities {
            streaming: Some(true),
            push_notifications: Some(false),
            extensions: None,
            extended_agent_card: Some(false),
        },
        default_input_modes: vec!["text/plain".to_string()],
        default_output_modes: vec!["text/plain".to_string()],
        skills: vec![
            skill("summarize", "Summarize", "Summarizes a document."),
            skill("search", "Search", "Finds documents matching a query."),
            skill("cite", "Cite", "Produces a citation for a document."),
        ],
        provider: None,
        documentation_url: None,
        icon_url: None,
        security_schemes: None,
        security_requirements: None,
        signatures: None,
    }
}

fn skill(id: &str, name: &str, description: &str) -> AgentSkill {
    AgentSkill {
        id: id.to_string(),
        name: name.to_string(),
        description: description.to_string(),
        tags: vec!["documents".to_string()],
        examples: Some(vec![format!("{name} the quarterly report")]),
        input_modes: None,
        output_modes: None,
        security_requirements: None,
    }
}
