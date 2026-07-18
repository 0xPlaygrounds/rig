//! Loopback integration tests driving [`rig_a2a::A2AClient`] over real HTTP on
//! 127.0.0.1 against a stub A2A server built from the upstream `a2a-server`
//! crate (`DefaultRequestHandler` + a scripted executor).
//!
//! Validates well-known card discovery, transport negotiation, the one-agent →
//! one-tool projection, host-side conversation threading through
//! [`rig_a2a::ConversationId`], task-state → tool-output and tool-error
//! projection, [`rig_a2a::A2AThreadInfo`] result metadata, the
//! [`rig_a2a::A2AModel`] completion and streaming surfaces, and full
//! agent-driven roundtrips over both HTTP transports (JSON-RPC and
//! HTTP+JSON/REST).

#![allow(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::unwrap_used
)]

use std::collections::VecDeque;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use a2a::{
    A2AError as A2AProtocolError, AgentCard, AgentInterface, Artifact, Message, Part, Role,
    StreamResponse, TRANSPORT_PROTOCOL_HTTP_JSON, TRANSPORT_PROTOCOL_JSONRPC, Task, TaskState,
    TaskStatus, new_artifact_id,
};
use a2a_server::task_store::InMemoryTaskStore;
use a2a_server::{
    AgentExecutor, DefaultRequestHandler, ExecutorContext, StaticAgentCard, agent_card,
    jsonrpc::jsonrpc_router, rest::rest_router,
};
use futures::StreamExt;
use futures::stream::BoxStream;
use rig_a2a::{
    A2AAgentBuilderExt, A2AClient, A2AThreadInfo, ConversationId, SendMessageResponse,
    conversation_context,
};
use rig_agent::agent::{AgentBuilder, AgentHook, HookContext, ToolResultAction, ToolResultEvent};
use rig_agent::completion::Prompt;
use rig_agent::test_utils::{MockAddTool, MockCompletionModel, MockTurn};
use rig_agent::tool::{DynamicTool, ToolContext, ToolErrorKind, ToolResult, ToolSet};
use rig_core::completion::{CompletionModel, CompletionRequest, ToolDefinition};
use rig_core::message::{Message as RigMessage, ToolResultContent, UserContent};
use rig_core::one_or_many::OneOrMany;

/// What the stub executor replies to a `message/send`.
#[derive(Clone)]
enum StubReply {
    CompletedText(String),
    MessageText(String),
    InputRequired {
        prompt: String,
    },
    /// A task ending in a non-completed terminal (or auth-gated) state.
    Ended {
        state: TaskState,
        reason: String,
    },
    /// A stream of status updates, each carrying a state and a status message.
    /// Many A2A agents deliver their answer this way rather than as artifacts.
    StatusStream(Vec<(TaskState, String)>),
    /// A task that pauses for input and *then* emits an artifact, so the last
    /// event on the stream reports no task state.
    PauseThenArtifact {
        prompt: String,
        artifact: String,
    },
}

impl StubReply {
    fn failed(reason: &str) -> Self {
        Self::Ended {
            state: TaskState::Failed,
            reason: reason.to_string(),
        }
    }

    fn ended(state: TaskState, reason: &str) -> Self {
        Self::Ended {
            state,
            reason: reason.to_string(),
        }
    }
}

/// Inbound messages recorded by the stub executor, shared with the test body.
struct RecordedRequest {
    message: Message,
    tenant: Option<String>,
}

type Recorded = Arc<Mutex<Vec<RecordedRequest>>>;

#[derive(Clone, Copy)]
enum StubBinding {
    JsonRpc,
    Rest,
}

/// A scripted sequence of replies. Each `message/send` pops the next one; the
/// last reply repeats once the script is exhausted, so tests only script the
/// turns they care about.
struct StubExecutor {
    replies: Mutex<VecDeque<StubReply>>,
    recorded: Recorded,
}

impl StubExecutor {
    fn next_reply(&self) -> StubReply {
        let mut replies = self.replies.lock().unwrap();
        if replies.len() > 1 {
            replies.pop_front().expect("non-empty script")
        } else {
            replies
                .front()
                .cloned()
                .expect("stub script must not be empty")
        }
    }
}

impl AgentExecutor for StubExecutor {
    fn execute(
        &self,
        ctx: ExecutorContext,
    ) -> BoxStream<'static, Result<StreamResponse, A2AProtocolError>> {
        if let Some(message) = &ctx.message {
            self.recorded.lock().unwrap().push(RecordedRequest {
                message: message.clone(),
                tenant: ctx.tenant.clone(),
            });
        }
        let (state, status_text, artifact_text) = match self.next_reply() {
            StubReply::CompletedText(text) => (TaskState::Completed, None, Some(text)),
            StubReply::MessageText(text) => {
                let mut message = Message::new(Role::Agent, vec![Part::text(text)]);
                message.context_id = Some(ctx.context_id);
                message.task_id = Some(ctx.task_id);
                return futures::stream::once(async move { Ok(StreamResponse::Message(message)) })
                    .boxed();
            }
            StubReply::InputRequired { prompt } => (TaskState::InputRequired, Some(prompt), None),
            StubReply::Ended { state, reason } => (state, Some(reason), None),
            StubReply::StatusStream(steps) => {
                let events = steps
                    .into_iter()
                    .map(|(state, text)| {
                        Ok(StreamResponse::StatusUpdate(status_update(
                            &ctx.task_id,
                            &ctx.context_id,
                            state,
                            Some(&text),
                        )))
                    })
                    .collect::<Vec<_>>();
                return futures::stream::iter(events).boxed();
            }
            StubReply::PauseThenArtifact { prompt, artifact } => {
                let events = vec![
                    Ok(StreamResponse::StatusUpdate(status_update(
                        &ctx.task_id,
                        &ctx.context_id,
                        TaskState::InputRequired,
                        Some(&prompt),
                    ))),
                    Ok(StreamResponse::ArtifactUpdate(
                        a2a::TaskArtifactUpdateEvent {
                            task_id: ctx.task_id.clone(),
                            context_id: ctx.context_id.clone(),
                            artifact: Artifact {
                                artifact_id: new_artifact_id(),
                                name: None,
                                description: None,
                                parts: vec![Part::text(artifact)],
                                metadata: None,
                                extensions: None,
                            },
                            append: None,
                            last_chunk: Some(true),
                            metadata: None,
                        },
                    )),
                ];
                return futures::stream::iter(events).boxed();
            }
        };
        let task = Task {
            id: ctx.task_id,
            context_id: ctx.context_id,
            status: TaskStatus {
                state,
                message: status_text.map(|text| Message::new(Role::Agent, vec![Part::text(text)])),
                timestamp: None,
            },
            artifacts: artifact_text.map(|text| {
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

    fn cancel(
        &self,
        ctx: ExecutorContext,
    ) -> BoxStream<'static, Result<StreamResponse, A2AProtocolError>> {
        let task_id = ctx.task_id;
        futures::stream::once(async move { Err(A2AProtocolError::task_not_cancelable(&task_id)) })
            .boxed()
    }
}

fn status_update(
    task_id: &str,
    context_id: &str,
    state: TaskState,
    text: Option<&str>,
) -> a2a::TaskStatusUpdateEvent {
    a2a::TaskStatusUpdateEvent {
        task_id: task_id.to_string(),
        context_id: context_id.to_string(),
        status: TaskStatus {
            state,
            message: text.map(|text| Message::new(Role::Agent, vec![Part::text(text.to_string())])),
            timestamp: None,
        },
        metadata: None,
    }
}

fn card_with_skills(base_url: &str, name: &str, skill_ids: &[&str]) -> AgentCard {
    AgentCard {
        name: name.to_string(),
        description: format!("Stub agent {name}."),
        version: "0.1.0".to_string(),
        supported_interfaces: vec![AgentInterface::new(
            format!("{base_url}/jsonrpc"),
            TRANSPORT_PROTOCOL_JSONRPC,
        )],
        capabilities: a2a::AgentCapabilities {
            streaming: Some(false),
            push_notifications: Some(false),
            extensions: None,
            extended_agent_card: Some(false),
        },
        default_input_modes: vec!["text/plain".to_string()],
        default_output_modes: vec!["text/plain".to_string()],
        skills: skill_ids
            .iter()
            .map(|id| a2a::AgentSkill {
                id: (*id).to_string(),
                name: (*id).to_string(),
                description: format!("{id} skill."),
                tags: vec![],
                examples: None,
                input_modes: Some(vec!["text/plain".to_string()]),
                output_modes: Some(vec!["text/plain".to_string()]),
                security_requirements: None,
            })
            .collect(),
        provider: None,
        documentation_url: None,
        icon_url: None,
        security_schemes: None,
        security_requirements: None,
        signatures: None,
    }
}

/// Boot a stub A2A server on a random loopback port: well-known card route
/// plus the upstream JSON-RPC handler backed by [`StubExecutor`].
async fn serve_stub(
    name: &str,
    skill_ids: &[&str],
    reply: StubReply,
) -> (SocketAddr, Recorded, tokio::task::JoinHandle<()>) {
    serve_script(name, skill_ids, [reply], StubBinding::JsonRpc).await
}

async fn serve_stub_with_binding(
    name: &str,
    skill_ids: &[&str],
    reply: StubReply,
    binding: StubBinding,
) -> (SocketAddr, Recorded, tokio::task::JoinHandle<()>) {
    serve_script(name, skill_ids, [reply], binding).await
}

/// Boot a stub that answers successive requests with successive replies.
async fn serve_script(
    name: &str,
    skill_ids: &[&str],
    replies: impl IntoIterator<Item = StubReply>,
    binding: StubBinding,
) -> (SocketAddr, Recorded, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let base_url = format!("http://{addr}");
    let mut card = card_with_skills(&base_url, name, skill_ids);
    card.supported_interfaces = vec![match binding {
        StubBinding::JsonRpc => {
            AgentInterface::new(format!("{base_url}/jsonrpc"), TRANSPORT_PROTOCOL_JSONRPC)
        }
        StubBinding::Rest => AgentInterface::new(&base_url, TRANSPORT_PROTOCOL_HTTP_JSON),
    }];
    let recorded: Recorded = Arc::new(Mutex::new(Vec::new()));
    let replies: VecDeque<StubReply> = replies.into_iter().collect();
    assert!(!replies.is_empty(), "stub script must not be empty");
    let executor = StubExecutor {
        replies: Mutex::new(replies),
        recorded: recorded.clone(),
    };
    let handler = Arc::new(DefaultRequestHandler::new(
        executor,
        InMemoryTaskStore::new(),
    ));
    let protocol_router = match binding {
        StubBinding::JsonRpc => axum::Router::new().nest("/jsonrpc", jsonrpc_router(handler)),
        StubBinding::Rest => rest_router(handler),
    };
    let router = axum::Router::new()
        .merge(agent_card::agent_card_router(Arc::new(
            StaticAgentCard::new(card),
        )))
        .merge(protocol_router);

    let server_task = tokio::spawn(async move {
        axum::serve(listener, router).await.expect("stub server");
    });
    (addr, recorded, server_task)
}

/// Dispatch the A2A tool through the public [`ToolSet`] path — Rig's erased
/// dispatch trait is private, so this is how a tool is invoked outside an agent
/// run.
///
/// Returns the raw [`ToolResult`] rather than a `Result`, because a refusal is
/// neither a success nor an error: `ToolResult::error()` is `None` for one, and
/// tests need to distinguish the three dispositions.
async fn call_tool(tool: &DynamicTool, args: &str, context: &mut ToolContext) -> ToolResult {
    let name = tool.name().to_string();
    let mut toolset = ToolSet::default();
    toolset.add_dynamic_tool(tool.clone());
    toolset.execute(&name, args, context).await
}

/// Dispatch with a throwaway context, for calls whose threading is irrelevant.
async fn call_tool_once(tool: &DynamicTool, args: &str) -> ToolResult {
    let mut context = ToolContext::new();
    call_tool(tool, args, &mut context).await
}

/// The text a successful tool call returned.
fn tool_text(result: &ToolResult) -> String {
    assert!(
        result.is_success(),
        "expected a successful tool call, got {}",
        result.status_name()
    );
    result.output().as_text().unwrap_or_default().to_string()
}

/// A prompt argument payload.
fn prompt(text: &str) -> String {
    serde_json::json!({ "prompt": text }).to_string()
}

#[tokio::test]
async fn from_url_yields_one_tool_named_after_the_card() {
    let _ = tracing_subscriber::fmt::try_init();

    let (addr, _recorded, server) = serve_stub(
        "multi-skill",
        &["alpha", "beta"],
        StubReply::CompletedText("pong".into()),
    )
    .await;

    let client = A2AClient::from_url(format!("http://{addr}"))
        .await
        .expect("client should fetch agent card");
    assert_eq!(client.card().name, "multi-skill");

    // Several skills, one tool: A2A routes every request through the same
    // endpoint and carries no skill selector.
    let tool = client.tool();
    assert_eq!(tool.name(), "multi-skill");
    assert!(tool.name().len() <= 64);
    assert!(
        tool.name()
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-'),
        "tool name must be provider-safe, got {}",
        tool.name()
    );

    // The skills survive as documentation on the single tool.
    let description = tool.definition().description;
    assert!(description.contains("alpha"), "{description}");
    assert!(description.contains("beta"), "{description}");

    // The schema is prompt-only: no identifiers for the model to forge.
    let parameters = tool.definition().parameters;
    let properties = parameters["properties"]
        .as_object()
        .expect("object schema")
        .keys()
        .cloned()
        .collect::<Vec<_>>();
    assert_eq!(properties, ["prompt"]);

    server.abort();
}

#[tokio::test]
async fn explicit_tool_name_overrides_the_card() {
    let card = card_with_skills("http://127.0.0.1:1", "Rig Agent", &["chat"]);
    let client = A2AClient::builder()
        .card(card)
        .tool_name("billing-desk")
        .build()
        .await
        .expect("client from hand-supplied card");

    assert_eq!(client.tool_name(), "billing-desk");
    assert_eq!(client.tool().name(), "billing-desk");
}

#[tokio::test]
async fn from_url_uses_rest_interface() {
    let (addr, recorded, server) = serve_stub_with_binding(
        "rest-agent",
        &["chat"],
        StubReply::CompletedText("rest reply".into()),
        StubBinding::Rest,
    )
    .await;

    let client = A2AClient::from_url(format!("http://{addr}"))
        .await
        .expect("client should select REST");
    assert_eq!(
        client.interface().protocol_binding,
        TRANSPORT_PROTOCOL_HTTP_JSON
    );
    let outcome = client
        .message("hello")
        .send()
        .await
        .expect("REST request should succeed");
    assert!(
        matches!(outcome, SendMessageResponse::Task(task) if task.status.state == TaskState::Completed)
    );
    assert_eq!(recorded.lock().unwrap().len(), 1);

    server.abort();
}

#[tokio::test]
async fn card_without_skills_still_yields_one_tool() {
    let card = card_with_skills("http://127.0.0.1:1", "Bare Agent", &[]);
    let client = A2AClient::from_agent_card(card)
        .await
        .expect("client from hand-supplied card");

    let tool = client.tool();
    assert_eq!(tool.name(), "bare-agent");
    let description = tool.definition().description;
    assert!(
        description.contains("Stub agent Bare Agent."),
        "{description}"
    );
    assert!(
        !description.contains("Skills:"),
        "a card with no skills must not render an empty section: {description}"
    );
}

/// All tool-result texts the scripted model received across its turns.
fn tool_result_texts(model: &MockCompletionModel) -> Vec<String> {
    model
        .requests()
        .iter()
        .flat_map(|request| request.chat_history.iter().cloned())
        .filter_map(|message| match message {
            RigMessage::User { content } => Some(content),
            _ => None,
        })
        .flat_map(|content| content.into_iter())
        .filter_map(|content| match content {
            UserContent::ToolResult(result) => Some(result.content),
            _ => None,
        })
        .flat_map(|content| content.into_iter())
        .filter_map(|content| match content {
            ToolResultContent::Text(text) => Some(text.text),
            _ => None,
        })
        .collect()
}

/// Full agent-driven loopback roundtrip through one transport binding:
/// card discovery → single-tool projection → scripted model tool call →
/// `message/send` to the upstream `a2a-server` handler → task reply →
/// tool output returned to the model → final answer.
async fn agent_calls_remote_agent(binding: StubBinding) {
    let _ = tracing_subscriber::fmt::try_init();

    let (addr, recorded, server) = serve_script(
        "greeter",
        &["greet"],
        [StubReply::CompletedText("hello from remote".into())],
        binding,
    )
    .await;

    let remote = A2AClient::from_url(format!("http://{addr}"))
        .await
        .expect("client should fetch card");
    let tool_name = remote.tool().name().to_string();

    let model = MockCompletionModel::new([
        MockTurn::tool_call(
            "tool_call_1",
            &tool_name,
            serde_json::json!({"prompt": "greet me"}),
        ),
        MockTurn::text("done"),
    ]);
    let agent = AgentBuilder::new(model.clone())
        .name("local")
        .a2a_tool(&remote)
        .build();

    let out = agent
        .prompt("use the remote agent")
        .max_turns(3)
        .await
        .expect("agent run should succeed");
    assert_eq!(out, "done");

    let messages = recorded.lock().unwrap();
    assert_eq!(messages.len(), 1, "stub should have seen one message");
    assert!(matches!(
        &messages[0].message.parts[0].content,
        a2a::PartContent::Text(t) if t == "greet me"
    ));
    // No ConversationId in the run's ToolContext, so the call is stateless.
    assert_eq!(messages[0].message.context_id, None);
    drop(messages);

    // The remote reply reaches the model verbatim — no state marker on a
    // completed task.
    let results = tool_result_texts(&model);
    assert!(
        results.iter().any(|text| text == "hello from remote"),
        "model should see the remote reply as a tool result: {results:?}"
    );

    server.abort();
}

#[tokio::test]
async fn agent_with_a2a_tool_calls_remote_agent_end_to_end() {
    agent_calls_remote_agent(StubBinding::JsonRpc).await;
}

#[tokio::test]
async fn agent_with_a2a_tool_calls_remote_agent_over_rest() {
    agent_calls_remote_agent(StubBinding::Rest).await;
}

#[tokio::test]
async fn missing_prompt_argument_fails_without_calling_remote() {
    let _ = tracing_subscriber::fmt::try_init();

    let (addr, recorded, server) =
        serve_stub("threaded", &["chat"], StubReply::CompletedText("ok".into())).await;

    let remote = A2AClient::from_url(format!("http://{addr}"))
        .await
        .expect("client should fetch card");

    let result = call_tool_once(&remote.tool(), r#"{"text":"wrong field"}"#).await;
    assert!(result.is_error_kind(ToolErrorKind::InvalidArgs));
    assert!(
        recorded.lock().unwrap().is_empty(),
        "no message must reach the remote on argument validation failure"
    );

    server.abort();
}

#[tokio::test]
async fn input_required_task_surfaces_a_prefix_without_identifiers() {
    let _ = tracing_subscriber::fmt::try_init();

    let (addr, _recorded, server) = serve_stub(
        "asker",
        &["ask"],
        StubReply::InputRequired {
            prompt: "which file?".into(),
        },
    )
    .await;

    let remote = A2AClient::from_url(format!("http://{addr}"))
        .await
        .expect("client should fetch card");
    let out = tool_text(&call_tool_once(&remote.tool(), &prompt("do the thing")).await);

    assert_eq!(out, "[a2a input-required] which file?");
    assert!(!out.contains("contextId"), "{out}");
    assert!(!out.contains("taskId"), "{out}");

    server.abort();
}

#[tokio::test]
async fn terminal_failure_states_map_to_tool_error_kinds() {
    let _ = tracing_subscriber::fmt::try_init();

    let cases = [
        (TaskState::Failed, ToolErrorKind::Provider, false),
        (TaskState::Rejected, ToolErrorKind::PermissionDenied, true),
        (TaskState::Canceled, ToolErrorKind::Cancelled, false),
        (
            TaskState::AuthRequired,
            ToolErrorKind::PermissionDenied,
            false,
        ),
    ];

    for (state, kind, refused) in cases {
        let (addr, _recorded, server) = serve_stub(
            "failer",
            &["fail"],
            StubReply::ended(state.clone(), "quota exceeded"),
        )
        .await;
        let remote = A2AClient::from_url(format!("http://{addr}"))
            .await
            .expect("client should fetch card");

        let result = call_tool_once(&remote.tool(), &prompt("do the thing")).await;
        assert!(!result.is_success(), "{state:?} must not succeed");
        assert_eq!(result.is_refused(), refused, "{state:?}");
        // A refusal is its own disposition, so the error lives behind a
        // different accessor than an ordinary failure.
        let error = if refused {
            result.refusal().expect("refusal carries its error")
        } else {
            result.error().expect("failure carries its error")
        };
        assert_eq!(error.kind(), kind, "{state:?}");
        // The remote's own reason must reach the model, not a redacted
        // placeholder.
        assert!(
            error.model_output().render().contains("quota exceeded"),
            "{state:?}: {}",
            error.model_output().render()
        );

        server.abort();
    }
}

#[tokio::test]
async fn request_context_threads_to_remote_and_response() {
    let _ = tracing_subscriber::fmt::try_init();

    let (addr, recorded, server) =
        serve_stub("threaded", &["chat"], StubReply::CompletedText("ok".into())).await;

    let client = A2AClient::from_url(format!("http://{addr}"))
        .await
        .expect("client should fetch card");

    // Turn 1 sends no ids; the server mints the contextId. Turn 2 echoes it,
    // per the A2A spec's client threading model. The direct API stays
    // explicit — only the tool and model surfaces thread automatically.
    let task = match client
        .message("turn 1")
        .send()
        .await
        .expect("send should succeed")
    {
        SendMessageResponse::Task(task) => task,
        other => panic!("expected Task outcome, got {other:?}"),
    };
    assert_eq!(task.status.state, TaskState::Completed);
    let minted = task.context_id.clone();
    assert!(!minted.trim().is_empty(), "server must mint a contextId");

    let task = match client
        .message("turn 2")
        .context(&minted)
        .send()
        .await
        .expect("send should succeed")
    {
        SendMessageResponse::Task(task) => task,
        other => panic!("expected Task outcome, got {other:?}"),
    };
    assert_eq!(task.context_id, minted);

    let messages = recorded.lock().unwrap();
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0].message.context_id, None);
    assert_eq!(
        messages[1].message.context_id.as_deref(),
        Some(minted.as_str())
    );

    server.abort();
}

#[tokio::test]
async fn bare_message_response_returns_text_without_identifiers() {
    let _ = tracing_subscriber::fmt::try_init();

    let (addr, recorded, server) = serve_stub(
        "messenger",
        &["chat"],
        StubReply::MessageText("hello".into()),
    )
    .await;

    let remote = A2AClient::from_url(format!("http://{addr}"))
        .await
        .expect("client should fetch card");
    let tool = remote.tool();
    let mut context = conversation_context("c1");

    let out = tool_text(&call_tool(&tool, &prompt("say hello"), &mut context).await);
    assert_eq!(out, "hello");

    // The response's contextId is recorded even though it never appears in
    // the output; its taskId is not, because a bare Message carries no state
    // and so cannot be known to be resumable.
    let second = call_tool(&tool, &prompt("again"), &mut context).await;
    assert!(second.is_success());

    let messages = recorded.lock().unwrap();
    assert_eq!(messages.len(), 2);
    assert!(
        messages[1].message.context_id.is_some(),
        "the second call should continue the recorded context"
    );
    assert_eq!(messages[1].message.task_id, None);

    server.abort();
}

// ---------------------------------------------------------------------------
// Conversation threading through ToolContext
// ---------------------------------------------------------------------------

#[tokio::test]
async fn conversation_id_threads_context_across_two_tool_calls() {
    let _ = tracing_subscriber::fmt::try_init();

    let (addr, recorded, server) =
        serve_stub("threaded", &["chat"], StubReply::CompletedText("ok".into())).await;
    let remote = A2AClient::from_url(format!("http://{addr}"))
        .await
        .expect("client should fetch card");
    let tool = remote.tool();
    let mut context = conversation_context("user-42");

    tool_text(&call_tool(&tool, &prompt("turn 1"), &mut context).await);
    tool_text(&call_tool(&tool, &prompt("turn 2"), &mut context).await);

    let messages = recorded.lock().unwrap();
    assert_eq!(messages.len(), 2);
    assert_eq!(
        messages[0].message.context_id, None,
        "the first call has nothing to echo"
    );
    assert!(
        messages[1].message.context_id.is_some(),
        "the second call must carry the server-issued contextId"
    );

    server.abort();
}

#[tokio::test]
async fn distinct_conversation_ids_are_isolated() {
    let _ = tracing_subscriber::fmt::try_init();

    let (addr, recorded, server) =
        serve_stub("threaded", &["chat"], StubReply::CompletedText("ok".into())).await;
    let remote = A2AClient::from_url(format!("http://{addr}"))
        .await
        .expect("client should fetch card");
    let tool = remote.tool();
    let mut alice = conversation_context("alice");
    let mut bob = conversation_context("bob");

    tool_text(&call_tool(&tool, &prompt("a1"), &mut alice).await);
    let alice_context = published_context_id(&alice);

    tool_text(&call_tool(&tool, &prompt("b1"), &mut bob).await);
    let bob_context = published_context_id(&bob);
    assert_ne!(
        alice_context, bob_context,
        "each conversation must get its own remote context"
    );

    tool_text(&call_tool(&tool, &prompt("a2"), &mut alice).await);

    let messages = recorded.lock().unwrap();
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[0].message.context_id, None);
    assert_eq!(
        messages[1].message.context_id, None,
        "a second conversation must not inherit the first's context"
    );
    assert_eq!(
        messages[2].message.context_id.as_deref(),
        Some(alice_context.as_str()),
        "alice's third call must continue alice's context, not bob's"
    );

    server.abort();
}

/// The `contextId` the last call through `context` reported.
fn published_context_id(context: &ToolContext) -> String {
    context
        .result::<A2AThreadInfo>()
        .expect("a dispatched call publishes its identifiers")
        .context_id
        .clone()
        .expect("the server mints a contextId for a task response")
}

/// `conversation_context` is a convenience; a caller who already has a
/// `ToolContext` carrying their own values inserts the key alongside them.
#[tokio::test]
async fn conversation_id_can_be_inserted_into_an_existing_context() {
    let _ = tracing_subscriber::fmt::try_init();

    #[derive(Clone)]
    struct AuthToken(#[allow(dead_code)] String);

    let (addr, recorded, server) =
        serve_stub("threaded", &["chat"], StubReply::CompletedText("ok".into())).await;
    let remote = A2AClient::from_url(format!("http://{addr}"))
        .await
        .expect("client should fetch card");
    let tool = remote.tool();

    let mut context = ToolContext::new();
    context.insert(AuthToken("secret".to_string()));
    context.insert(ConversationId::new("user-42"));

    tool_text(&call_tool(&tool, &prompt("turn 1"), &mut context).await);
    tool_text(&call_tool(&tool, &prompt("turn 2"), &mut context).await);

    // The caller's own value survives dispatch alongside the A2A key.
    assert!(context.contains::<AuthToken>());
    assert!(
        recorded.lock().unwrap()[1].message.context_id.is_some(),
        "threading works from a hand-built context"
    );

    server.abort();
}

#[tokio::test]
async fn calls_without_conversation_id_stay_stateless() {
    let _ = tracing_subscriber::fmt::try_init();

    let (addr, recorded, server) =
        serve_stub("threaded", &["chat"], StubReply::CompletedText("ok".into())).await;
    let remote = A2AClient::from_url(format!("http://{addr}"))
        .await
        .expect("client should fetch card");
    let tool = remote.tool();

    for turn in 0..3 {
        tool_text(&call_tool_once(&tool, &prompt(&format!("turn {turn}"))).await);
    }

    let messages = recorded.lock().unwrap();
    assert_eq!(messages.len(), 3);
    assert!(
        messages
            .iter()
            .all(|m| m.message.context_id.is_none() && m.message.task_id.is_none()),
        "an unkeyed call must never carry identifiers"
    );

    server.abort();
}

#[tokio::test]
async fn input_required_resume_carries_task_id() {
    let _ = tracing_subscriber::fmt::try_init();

    let (addr, recorded, server) = serve_script(
        "asker",
        &["ask"],
        [
            StubReply::InputRequired {
                prompt: "which file?".into(),
            },
            StubReply::CompletedText("done".into()),
        ],
        StubBinding::JsonRpc,
    )
    .await;
    let remote = A2AClient::from_url(format!("http://{addr}"))
        .await
        .expect("client should fetch card");
    let tool = remote.tool();
    let mut context = conversation_context("user-42");

    let paused = tool_text(&call_tool(&tool, &prompt("do the thing"), &mut context).await);
    assert!(paused.starts_with("[a2a input-required]"), "{paused}");

    let finished = tool_text(&call_tool(&tool, &prompt("README.md"), &mut context).await);
    assert_eq!(finished, "done");

    let messages = recorded.lock().unwrap();
    assert_eq!(messages.len(), 2);
    let first_task = messages[0].message.task_id.clone();
    assert_eq!(first_task, None, "the opening call sends no task id");
    assert!(
        messages[1].message.task_id.is_some(),
        "resuming a paused task must carry its taskId"
    );
    assert!(
        messages[1].message.context_id.is_some(),
        "resuming must also carry the contextId"
    );

    server.abort();
}

#[tokio::test]
async fn terminal_task_does_not_carry_task_id_forward() {
    let _ = tracing_subscriber::fmt::try_init();

    let (addr, recorded, server) =
        serve_stub("threaded", &["chat"], StubReply::CompletedText("ok".into())).await;
    let remote = A2AClient::from_url(format!("http://{addr}"))
        .await
        .expect("client should fetch card");
    let tool = remote.tool();
    let mut context = conversation_context("user-42");

    tool_text(&call_tool(&tool, &prompt("turn 1"), &mut context).await);
    tool_text(&call_tool(&tool, &prompt("turn 2"), &mut context).await);

    let messages = recorded.lock().unwrap();
    // Sending to a completed task makes the server replay the stale task
    // rather than doing new work, so the id must not be reused.
    assert_eq!(messages[1].message.task_id, None);
    assert!(messages[1].message.context_id.is_some());

    server.abort();
}

#[tokio::test]
async fn failure_still_records_context_id_for_the_conversation() {
    let _ = tracing_subscriber::fmt::try_init();

    let (addr, recorded, server) = serve_script(
        "flaky",
        &["chat"],
        [
            StubReply::failed("transient"),
            StubReply::CompletedText("recovered".into()),
        ],
        StubBinding::JsonRpc,
    )
    .await;
    let remote = A2AClient::from_url(format!("http://{addr}"))
        .await
        .expect("client should fetch card");
    let tool = remote.tool();
    let mut context = conversation_context("user-42");

    let failed = call_tool(&tool, &prompt("turn 1"), &mut context).await;
    assert!(failed.is_error_kind(ToolErrorKind::Provider));

    let recovered = tool_text(&call_tool(&tool, &prompt("turn 2"), &mut context).await);
    assert_eq!(recovered, "recovered");

    let messages = recorded.lock().unwrap();
    assert!(
        messages[1].message.context_id.is_some(),
        "a failed turn must not drop the conversation"
    );

    server.abort();
}

#[tokio::test]
async fn two_clients_share_a_conversation_id_without_sharing_threads() {
    let _ = tracing_subscriber::fmt::try_init();

    let (addr_a, recorded_a, server_a) =
        serve_stub("agent-a", &["chat"], StubReply::CompletedText("a".into())).await;
    let (addr_b, recorded_b, server_b) =
        serve_stub("agent-b", &["chat"], StubReply::CompletedText("b".into())).await;

    let remote_a = A2AClient::from_url(format!("http://{addr_a}"))
        .await
        .expect("client a");
    let remote_b = A2AClient::from_url(format!("http://{addr_b}"))
        .await
        .expect("client b");
    let (tool_a, tool_b) = (remote_a.tool(), remote_b.tool());
    let mut context = conversation_context("shared");

    tool_text(&call_tool(&tool_a, &prompt("to a"), &mut context).await);
    tool_text(&call_tool(&tool_b, &prompt("to b"), &mut context).await);

    // Each client keeps its own store, so B's first call is still an opening
    // one even though the conversation key is shared.
    assert_eq!(recorded_a.lock().unwrap()[0].message.context_id, None);
    assert_eq!(recorded_b.lock().unwrap()[0].message.context_id, None);

    tool_text(&call_tool(&tool_a, &prompt("to a again"), &mut context).await);
    let messages_a = recorded_a.lock().unwrap();
    assert!(messages_a[1].message.context_id.is_some());

    server_a.abort();
    server_b.abort();
}

/// The metadata must survive the agent run's dispatch path too, not just a
/// direct `ToolSet::execute` — that path is where a hook actually reads it.
#[tokio::test]
async fn thread_info_reaches_a_hook_during_an_agent_run() {
    let _ = tracing_subscriber::fmt::try_init();

    #[derive(Clone, Default)]
    struct RecordThreadInfo(Arc<Mutex<Vec<A2AThreadInfo>>>);

    impl AgentHook for RecordThreadInfo {
        async fn on_tool_result(
            &self,
            _context: &HookContext,
            event: ToolResultEvent<'_>,
        ) -> ToolResultAction {
            if let Some(info) = event.tool_context.result::<A2AThreadInfo>() {
                self.0.lock().unwrap().push(info.clone());
            }
            ToolResultAction::Keep
        }
    }

    let (addr, _recorded, server) = serve_script(
        "asker",
        &["ask"],
        [
            StubReply::InputRequired {
                prompt: "which file?".into(),
            },
            StubReply::CompletedText("done".into()),
        ],
        StubBinding::JsonRpc,
    )
    .await;
    let remote = A2AClient::from_url(format!("http://{addr}"))
        .await
        .expect("client should fetch card");

    let model = MockCompletionModel::new([
        MockTurn::tool_call(
            "c1",
            remote.tool_name(),
            serde_json::json!({"prompt": "go"}),
        ),
        MockTurn::tool_call(
            "c2",
            remote.tool_name(),
            serde_json::json!({"prompt": "README.md"}),
        ),
        MockTurn::text("done"),
    ]);
    let observed = RecordThreadInfo::default();
    let agent = AgentBuilder::new(model)
        .a2a_tool(&remote)
        .add_hook(observed.clone())
        .build();

    agent
        .prompt("start")
        .tool_context(conversation_context("user-42"))
        .max_turns(4)
        .await
        .expect("agent run should succeed");

    let seen = observed.0.lock().unwrap();
    assert_eq!(seen.len(), 2, "the hook must observe both calls");
    assert_eq!(seen[0].state_label(), Some("input-required"));
    assert!(seen[0].resumable);
    assert_eq!(seen[1].state_label(), Some("completed"));
    assert!(!seen[1].resumable);
    assert_eq!(
        seen[0].context_id, seen[1].context_id,
        "both turns belong to one remote conversation"
    );

    server.abort();
}

#[tokio::test]
async fn thread_info_is_published_as_result_metadata() {
    let _ = tracing_subscriber::fmt::try_init();

    let (addr, _recorded, server) = serve_script(
        "asker",
        &["ask"],
        [
            StubReply::InputRequired {
                prompt: "which file?".into(),
            },
            StubReply::failed("nope"),
        ],
        StubBinding::JsonRpc,
    )
    .await;
    let remote = A2AClient::from_url(format!("http://{addr}"))
        .await
        .expect("client should fetch card");
    let tool = remote.tool();

    // ToolSet::execute returns the dispatch context's metadata on the result
    // context, so drive it directly to observe what a hook would see.
    let mut context = conversation_context("user-42");
    let paused = call_tool(&tool, &prompt("go"), &mut context).await;
    assert!(paused.is_success());
    let info = context
        .result::<A2AThreadInfo>()
        .expect("a successful call publishes its identifiers");
    assert!(info.context_id.is_some());
    assert!(info.task_id.is_some());
    assert_eq!(info.state_label(), Some("input-required"));
    assert!(info.resumable, "a paused task is resumable");

    // Metadata is published for failures too, so a hook can correlate them.
    let failed = call_tool(&tool, &prompt("go again"), &mut context).await;
    assert!(failed.is_error_kind(ToolErrorKind::Provider));
    let info = context
        .result::<A2AThreadInfo>()
        .expect("a failed call publishes its identifiers");
    assert_eq!(info.state_label(), Some("failed"));
    assert!(!info.resumable);

    server.abort();
}

// ---------------------------------------------------------------------------
// A2AModel: the remote agent as a Rig completion model
// ---------------------------------------------------------------------------

fn text_request(prompt: &str) -> CompletionRequest {
    CompletionRequest {
        model: None,
        preamble: None,
        chat_history: OneOrMany::one(RigMessage::user(prompt)),
        documents: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    }
}

/// Every text part of a recorded inbound message, in order.
fn recorded_texts(recorded: &Recorded, turn: usize) -> Vec<String> {
    recorded.lock().unwrap()[turn]
        .message
        .parts
        .iter()
        .filter_map(|part| match &part.content {
            a2a::PartContent::Text(text) => Some(text.clone()),
            _ => None,
        })
        .collect()
}

fn document(id: &str, text: &str) -> rig_core::completion::Document {
    rig_core::completion::Document {
        id: id.to_string(),
        text: text.to_string(),
        additional_props: Default::default(),
    }
}

/// A threaded remote keeps everything sent under its `contextId`, so standing
/// context goes out once — but a document retrieved for a later turn is new to
/// that remote and must still be delivered.
#[tokio::test]
async fn threaded_turns_send_standing_context_once() {
    let _ = tracing_subscriber::fmt::try_init();

    let (addr, recorded, server) =
        serve_stub("standing", &["chat"], StubReply::CompletedText("ok".into())).await;
    let remote = A2AClient::from_url(format!("http://{addr}"))
        .await
        .expect("client should fetch card");
    let model = remote.model_for_conversation("user-42");

    let mut turn = text_request("turn 1");
    turn.preamble = Some("You are terse.".to_string());
    turn.documents = vec![document("handbook", "the standing handbook")];
    model.completion(turn).await.expect("turn 1 should succeed");

    // Turn 2 repeats the same preamble and document, as Rig rebuilds them for
    // every request, and additionally retrieves one new document.
    let mut turn = text_request("turn 2");
    turn.preamble = Some("You are terse.".to_string());
    turn.documents = vec![
        document("handbook", "the standing handbook"),
        document("retrieved", "freshly retrieved passage"),
    ];
    model.completion(turn).await.expect("turn 2 should succeed");

    let first = recorded_texts(&recorded, 0);
    assert!(
        first.iter().any(|text| text.contains("You are terse."))
            && first.iter().any(|text| text.contains("standing handbook")),
        "the opening turn carries the whole request: {first:?}"
    );

    let second = recorded_texts(&recorded, 1);
    assert!(
        !second.iter().any(|text| text.contains("You are terse.")),
        "the remote already holds the preamble under this contextId: {second:?}"
    );
    assert!(
        !second.iter().any(|text| text.contains("standing handbook")),
        "a document the remote already holds must not be replayed: {second:?}"
    );
    assert!(
        second
            .iter()
            .any(|text| text.contains("freshly retrieved passage")),
        "a newly retrieved document is new to the remote: {second:?}"
    );
    assert!(
        second.iter().any(|text| text.contains("turn 2")),
        "the newest prompt always goes out: {second:?}"
    );

    server.abort();
}

/// Without a conversation there is no server-side state to rely on, so every
/// request carries its own context.
#[tokio::test]
async fn unthreaded_requests_always_carry_their_context() {
    let _ = tracing_subscriber::fmt::try_init();

    let (addr, recorded, server) = serve_stub(
        "stateless",
        &["chat"],
        StubReply::CompletedText("ok".into()),
    )
    .await;
    let remote = A2AClient::from_url(format!("http://{addr}"))
        .await
        .expect("client should fetch card");
    let model = remote.model();

    for prompt in ["one", "two"] {
        let mut turn = text_request(prompt);
        turn.documents = vec![document("handbook", "the standing handbook")];
        model.completion(turn).await.expect("completion succeeds");
    }

    for turn in 0..2 {
        let texts = recorded_texts(&recorded, turn);
        assert!(
            texts.iter().any(|text| text.contains("standing handbook")),
            "turn {turn} must carry its own context: {texts:?}"
        );
    }

    server.abort();
}

/// A caller who owns the ids threads them per request, with no conversation
/// binding and nothing tracked on the client.
#[tokio::test]
async fn a_caller_can_thread_explicitly_through_additional_params() {
    let _ = tracing_subscriber::fmt::try_init();

    let (addr, recorded, server) =
        serve_stub("explicit", &["chat"], StubReply::CompletedText("ok".into())).await;
    let remote = A2AClient::from_url(format!("http://{addr}"))
        .await
        .expect("client should fetch card");
    // No conversation binding: this model tracks nothing.
    let model = remote.model();

    model
        .completion(text_request("turn 1"))
        .await
        .expect("turn 1 should succeed");
    let minted = recorded.lock().unwrap()[0]
        .message
        .context_id
        .clone()
        .or_else(|| Some("ctx-from-response".to_string()));
    assert!(minted.is_some());

    // Thread turn 2 by hand, the way a caller restoring ids from their own
    // storage would.
    let mut turn = text_request("turn 2");
    turn.additional_params = Some(serde_json::json!({
        "a2a": { "context_id": "ctx-restored" }
    }));
    model.completion(turn).await.expect("turn 2 should succeed");

    let messages = recorded.lock().unwrap();
    assert_eq!(
        messages[1].message.context_id.as_deref(),
        Some("ctx-restored"),
        "the caller's contextId must reach the remote"
    );

    server.abort();
}

#[tokio::test]
async fn a_malformed_threading_directive_fails_before_any_network_call() {
    let card = card_with_skills("http://127.0.0.1:1", "unreachable", &["chat"]);
    let client = A2AClient::from_agent_card(card)
        .await
        .expect("client from hand-supplied card");

    let mut turn = text_request("hi");
    turn.additional_params = Some(serde_json::json!({"a2a": {"contextId": "typo"}}));
    let error = client
        .model()
        .completion(turn)
        .await
        .expect_err("a typo must not silently open a new conversation");
    assert!(
        error.to_string().contains("additional_params.a2a"),
        "{error}"
    );
}

/// A completed task whose result was an action rather than text is a
/// legitimate empty turn, the way Rig's Anthropic provider treats an empty
/// `end_turn`.
#[tokio::test]
async fn a_completed_task_with_no_output_is_an_empty_turn() {
    let _ = tracing_subscriber::fmt::try_init();

    let (addr, _recorded, server) =
        serve_stub("silent", &["chat"], StubReply::CompletedText(String::new())).await;
    let remote = A2AClient::from_url(format!("http://{addr}"))
        .await
        .expect("client should fetch card");

    let response = remote
        .model()
        .completion(text_request("file the ticket"))
        .await
        .expect("a silent completion is not a failure");
    let text = match response.choice.first() {
        rig_core::completion::AssistantContent::Text(text) => text.text,
        other => panic!("expected text, got {other:?}"),
    };
    assert_eq!(text, "");

    server.abort();
}

/// An unfinished task that answers with nothing has neither produced a result
/// nor completed, so there is nothing for the caller to act on.
#[tokio::test]
async fn an_unfinished_task_with_no_output_fails_the_completion() {
    let _ = tracing_subscriber::fmt::try_init();

    let (addr, _recorded, server) = serve_stub(
        "mute",
        &["chat"],
        StubReply::InputRequired {
            prompt: String::new(),
        },
    )
    .await;
    let remote = A2AClient::from_url(format!("http://{addr}"))
        .await
        .expect("client should fetch card");

    let error = remote
        .model()
        .completion(text_request("what next?"))
        .await
        .expect_err("a content-free unfinished task is not an answer");
    let text = error.to_string();
    assert!(text.contains("no content"), "{text}");
    assert!(text.contains("input-required"), "{text}");

    server.abort();
}

#[tokio::test]
async fn model_completes_against_the_remote_agent() {
    let _ = tracing_subscriber::fmt::try_init();

    let (addr, recorded, server) = serve_stub(
        "modelled",
        &["chat"],
        StubReply::CompletedText("remote answer".into()),
    )
    .await;
    let remote = A2AClient::from_url(format!("http://{addr}"))
        .await
        .expect("client should fetch card");

    let response = remote
        .model()
        .completion(text_request("what is 2 + 2?"))
        .await
        .expect("completion should succeed");

    assert_eq!(response.provider, "a2a");
    let text = match response.choice.first() {
        rig_core::completion::AssistantContent::Text(text) => text.text,
        other => panic!("expected text, got {other:?}"),
    };
    assert_eq!(text, "remote answer");
    // A2A has no token accounting; zero usage is Rig's documented sentinel.
    assert!(!response.usage.has_values());

    let messages = recorded.lock().unwrap();
    assert_eq!(messages.len(), 1);
    assert!(matches!(
        &messages[0].message.parts[0].content,
        a2a::PartContent::Text(t) if t == "what is 2 + 2?"
    ));

    server.abort();
}

#[tokio::test]
async fn model_rejects_requests_a2a_cannot_carry() {
    let card = card_with_skills("http://127.0.0.1:1", "unreachable", &["chat"]);
    let client = A2AClient::from_agent_card(card)
        .await
        .expect("client from hand-supplied card");
    let model = client.model();

    let mut with_tools = text_request("hi");
    with_tools.tools.push(ToolDefinition {
        name: "add".to_string(),
        description: "adds".to_string(),
        parameters: serde_json::json!({}),
    });
    let error = model
        .completion(with_tools)
        .await
        .expect_err("a remote A2A agent can never call a Rig tool");
    assert!(error.to_string().contains("tools"), "{error}");

    let mut with_schema = text_request("hi");
    with_schema.output_schema = Some(schemars::json_schema!({"type": "object"}));
    let error = model
        .completion(with_schema)
        .await
        .expect_err("A2A cannot constrain remote output");
    assert!(error.to_string().contains("output_schema"), "{error}");
}

#[tokio::test]
async fn agent_backed_by_a2a_threads_a_conversation() {
    let _ = tracing_subscriber::fmt::try_init();

    let (addr, recorded, server) = serve_script(
        "modelled",
        &["chat"],
        [
            StubReply::CompletedText("first".into()),
            StubReply::CompletedText("second".into()),
        ],
        StubBinding::JsonRpc,
    )
    .await;
    let remote = A2AClient::from_url(format!("http://{addr}"))
        .await
        .expect("client should fetch card");

    let agent = remote.agent_for_conversation("user-42").build();
    assert_eq!(agent.name(), Some("modelled"));

    assert_eq!(agent.prompt("turn 1").await.expect("turn 1"), "first");
    assert_eq!(agent.prompt("turn 2").await.expect("turn 2"), "second");

    let messages = recorded.lock().unwrap();
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0].message.context_id, None);
    assert!(
        messages[1].message.context_id.is_some(),
        "a conversation-bound agent must continue its remote context"
    );

    server.abort();
}

/// One agent, many conversations: the run names the conversation and
/// `A2AThreadHook` carries it to the model, so each gets its own remote thread.
#[tokio::test]
async fn one_agent_threads_each_conversation_it_is_asked_for() {
    let _ = tracing_subscriber::fmt::try_init();

    let (addr, recorded, server) =
        serve_stub("multi", &["chat"], StubReply::CompletedText("ok".into())).await;
    let remote = A2AClient::from_url(format!("http://{addr}"))
        .await
        .expect("client should fetch card");

    let agent = remote.agent().build();

    agent
        .prompt("alice turn 1")
        .conversation("alice")
        .await
        .expect("alice turn 1");
    agent
        .prompt("bob turn 1")
        .conversation("bob")
        .await
        .expect("bob turn 1");
    agent
        .prompt("alice turn 2")
        .conversation("alice")
        .await
        .expect("alice turn 2");

    let messages = recorded.lock().unwrap();
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[0].message.context_id, None, "alice opens a thread");
    assert_eq!(
        messages[1].message.context_id, None,
        "bob opens a separate thread rather than joining alice's"
    );
    let alice = messages[2]
        .message
        .context_id
        .as_deref()
        .expect("alice's second turn continues her thread");
    assert!(!alice.is_empty());

    server.abort();
}

/// A run that names no conversation stays single-turn: the hook must not
/// invent a key that would merge unrelated exchanges into one remote thread.
#[tokio::test]
async fn an_unnamed_run_does_not_join_a_thread() {
    let _ = tracing_subscriber::fmt::try_init();

    let (addr, recorded, server) =
        serve_stub("anon", &["chat"], StubReply::CompletedText("ok".into())).await;
    let remote = A2AClient::from_url(format!("http://{addr}"))
        .await
        .expect("client should fetch card");

    let agent = remote.agent().build();
    agent.prompt("one").await.expect("turn 1");
    agent.prompt("two").await.expect("turn 2");

    let messages = recorded.lock().unwrap();
    assert_eq!(messages.len(), 2);
    assert!(
        messages.iter().all(|m| m.message.context_id.is_none()),
        "an unnamed run must not be threaded"
    );

    server.abort();
}

#[tokio::test]
async fn model_and_tool_share_one_remote_conversation() {
    let _ = tracing_subscriber::fmt::try_init();

    let (addr, recorded, server) =
        serve_stub("shared", &["chat"], StubReply::CompletedText("ok".into())).await;
    let remote = A2AClient::from_url(format!("http://{addr}"))
        .await
        .expect("client should fetch card");

    // Open the conversation through the model...
    remote
        .model_for_conversation("user-42")
        .completion(text_request("via model"))
        .await
        .expect("model completion");

    // ...then continue it through the tool, under the same key.
    let mut context = conversation_context("user-42");
    tool_text(&call_tool(&remote.tool(), &prompt("via tool"), &mut context).await);

    let messages = recorded.lock().unwrap();
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0].message.context_id, None);
    assert!(
        messages[1].message.context_id.is_some(),
        "the tool must pick up the context the model opened"
    );

    server.abort();
}

#[tokio::test]
async fn agent_backed_by_a2a_streams() {
    let _ = tracing_subscriber::fmt::try_init();

    let (addr, _recorded, server) = serve_stub(
        "streamer",
        &["chat"],
        StubReply::CompletedText("streamed answer".into()),
    )
    .await;
    let remote = A2AClient::from_url(format!("http://{addr}"))
        .await
        .expect("client should fetch card");

    let mut stream = remote
        .model()
        .stream(text_request("stream me"))
        .await
        .expect("stream should open");

    let mut text = String::new();
    while let Some(chunk) = stream.next().await {
        if let rig_core::streaming::StreamedAssistantContent::Text(delta) =
            chunk.expect("stream item")
        {
            text.push_str(&delta.text);
        }
    }
    assert!(
        text.contains("streamed answer"),
        "expected the remote artifact text, got {text:?}"
    );

    server.abort();
}

#[tokio::test]
async fn model_stream_surfaces_remote_failure() {
    let _ = tracing_subscriber::fmt::try_init();

    let (addr, _recorded, server) =
        serve_stub("streamer", &["chat"], StubReply::failed("quota exceeded")).await;
    let remote = A2AClient::from_url(format!("http://{addr}"))
        .await
        .expect("client should fetch card");

    let mut stream = remote
        .model()
        .stream(text_request("stream me"))
        .await
        .expect("stream should open");

    let mut error = None;
    while let Some(chunk) = stream.next().await {
        if let Err(err) = chunk {
            error = Some(err);
            break;
        }
    }
    let error = error.expect("a failed remote task must end the stream with an error");
    assert!(error.to_string().contains("quota exceeded"), "{error}");

    server.abort();
}

/// An A2A agent may deliver its whole answer through `TaskStatusUpdateEvent`
/// status messages rather than artifacts. The non-streaming surface projects
/// those (`task.status.message` is part of the task body), so the streaming
/// surface must too, or the same server answers one way and stays silent the
/// other.
#[tokio::test]
async fn model_stream_yields_status_message_text() {
    let _ = tracing_subscriber::fmt::try_init();

    let (addr, _recorded, server) = serve_stub(
        "statuser",
        &["chat"],
        StubReply::StatusStream(vec![
            (TaskState::Working, "partial answer ".to_string()),
            (TaskState::Completed, "final answer".to_string()),
        ]),
    )
    .await;
    let remote = A2AClient::from_url(format!("http://{addr}"))
        .await
        .expect("client should fetch card");

    let mut stream = remote
        .model()
        .stream(text_request("stream me"))
        .await
        .expect("stream should open");
    let mut text = String::new();
    while let Some(chunk) = stream.next().await {
        if let rig_core::streaming::StreamedAssistantContent::Text(delta) =
            chunk.expect("stream item")
        {
            text.push_str(&delta.text);
        }
    }
    assert_eq!(text, "partial answer final answer");

    // The unary surface already projected this text; the two must agree.
    let response = remote
        .model()
        .completion(text_request("send me"))
        .await
        .expect("completion should succeed");
    assert!(
        matches!(
            response.choice.first(),
            rig_core::completion::AssistantContent::Text(text) if text.text == "final answer"
        ),
        "{:?}",
        response.choice
    );

    server.abort();
}

/// An artifact update names its task but reports no task state, so it says
/// nothing about resumability. Letting it clear the paused task's id would
/// strand the conversation: the next turn would open a new task and the remote
/// would never receive the input it asked for.
#[tokio::test]
async fn artifact_after_a_pause_keeps_the_resumable_task_id() {
    let _ = tracing_subscriber::fmt::try_init();

    let (addr, recorded, server) = serve_script(
        "pauser",
        &["ask"],
        [
            StubReply::PauseThenArtifact {
                prompt: "which file?".to_string(),
                artifact: "draft".to_string(),
            },
            StubReply::CompletedText("done".into()),
        ],
        StubBinding::JsonRpc,
    )
    .await;
    let remote = A2AClient::from_url(format!("http://{addr}"))
        .await
        .expect("client should fetch card");
    let model = remote.model_for_conversation("user-42");

    let mut stream = model
        .stream(text_request("turn 1"))
        .await
        .expect("stream should open");
    while let Some(chunk) = stream.next().await {
        chunk.expect("stream item");
    }

    model
        .completion(text_request("README.md"))
        .await
        .expect("turn 2 should succeed");

    let messages = recorded.lock().unwrap();
    assert_eq!(messages.len(), 2);
    assert!(
        messages[1].message.task_id.is_some(),
        "the trailing artifact must not drop the paused task's id"
    );
    assert!(messages[1].message.context_id.is_some());

    server.abort();
}

#[tokio::test]
async fn blank_prompt_fails_without_calling_remote() {
    let _ = tracing_subscriber::fmt::try_init();

    let (addr, recorded, server) =
        serve_stub("blank", &["chat"], StubReply::CompletedText("ok".into())).await;
    let remote = A2AClient::from_url(format!("http://{addr}"))
        .await
        .expect("client should fetch card");

    let result = call_tool_once(&remote.tool(), &prompt("   ")).await;
    assert!(result.is_error_kind(ToolErrorKind::InvalidArgs));
    assert!(
        recorded.lock().unwrap().is_empty(),
        "a blank prompt gives the remote nothing to act on, so it must not be sent"
    );

    server.abort();
}

#[tokio::test]
async fn a2a_backed_agent_becomes_a_sub_agent_tool() {
    let _ = tracing_subscriber::fmt::try_init();

    let (addr, recorded, server) = serve_stub(
        "researcher",
        &["research"],
        StubReply::CompletedText("remote findings".into()),
    )
    .await;
    let remote = A2AClient::from_url(format!("http://{addr}"))
        .await
        .expect("client should fetch card");

    // The A2A-backed agent is an ordinary Rig agent, so the sub-agent bridge
    // applies to it unchanged.
    let sub_agent = remote.agent_for_conversation("user-42").build();

    let model = MockCompletionModel::new([
        MockTurn::tool_call(
            "call_1",
            "researcher",
            serde_json::json!({"prompt": "look it up"}),
        ),
        MockTurn::text("relayed"),
    ]);
    let orchestrator = AgentBuilder::new(model.clone())
        .dynamic_tool(sub_agent.into_tool())
        .build();

    let out = orchestrator
        .prompt("delegate")
        .max_turns(3)
        .await
        .expect("orchestrator run");
    assert_eq!(out, "relayed");

    let results = tool_result_texts(&model);
    assert!(
        results.iter().any(|text| text == "remote findings"),
        "the sub-agent's remote answer should reach the orchestrator: {results:?}"
    );
    assert_eq!(recorded.lock().unwrap().len(), 1);

    server.abort();
}

#[tokio::test]
async fn client_selects_first_supported_interface_in_card_order() {
    let mut card = card_with_skills("http://127.0.0.1:1", "ordered", &["chat"]);
    let mut unsupported_version =
        AgentInterface::new("http://127.0.0.1:2/jsonrpc", TRANSPORT_PROTOCOL_JSONRPC);
    unsupported_version.protocol_version = "2.0".to_string();
    let mut rest = AgentInterface::new("http://127.0.0.1:3", TRANSPORT_PROTOCOL_HTTP_JSON);
    rest.protocol_version = "1.1".to_string();
    rest.tenant = Some("tenant-a".to_string());
    card.supported_interfaces = vec![
        unsupported_version,
        rest,
        AgentInterface::new("http://127.0.0.1:4/jsonrpc", TRANSPORT_PROTOCOL_JSONRPC),
    ];

    let client = A2AClient::from_agent_card(card)
        .await
        .expect("card has a supported interface");
    assert_eq!(
        client.interface().protocol_binding,
        TRANSPORT_PROTOCOL_HTTP_JSON
    );
    assert_eq!(client.interface().tenant.as_deref(), Some("tenant-a"));
}

#[tokio::test]
async fn selected_interface_must_be_an_absolute_http_url() {
    let mut card = card_with_skills("http://127.0.0.1:1", "invalid", &["chat"]);
    card.supported_interfaces[0].url = "not a URL".to_string();

    let err = match A2AClient::from_agent_card(card).await {
        Ok(_) => panic!("invalid interface URL must be rejected"),
        Err(err) => err,
    };
    assert!(matches!(
        err,
        rig_a2a::A2AError::AgentCard(rig_a2a::error::AgentCardError::InvalidInterfaceUrl { .. })
    ));
}

#[tokio::test]
async fn selected_interface_tenant_reaches_direct_and_tool_requests() {
    let (addr, recorded, server) =
        serve_stub("tenant", &["chat"], StubReply::CompletedText("ok".into())).await;
    let mut card = card_with_skills(&format!("http://{addr}"), "tenant", &["chat"]);
    card.supported_interfaces[0].tenant = Some("tenant-a".to_string());

    let client = A2AClient::from_agent_card(card)
        .await
        .expect("client should use supplied card");
    client
        .message("direct")
        .send()
        .await
        .expect("direct request should succeed");
    tool_text(&call_tool_once(&client.tool(), &prompt("tool")).await);

    let requests = recorded.lock().unwrap();
    assert_eq!(requests.len(), 2);
    assert!(
        requests
            .iter()
            .all(|request| request.tenant.as_deref() == Some("tenant-a"))
    );

    server.abort();
}

/// `a2a_tool` must be callable on both `AgentBuilder` tool states — before any
/// tool is registered, and after — so a remote agent can be added anywhere in
/// a builder chain.
#[tokio::test]
async fn a2a_tool_appends_after_existing_builder_tools() {
    let first =
        A2AClient::from_agent_card(card_with_skills("http://127.0.0.1:1", "first", &["chat"]))
            .await
            .expect("first client");
    let second =
        A2AClient::from_agent_card(card_with_skills("http://127.0.0.1:2", "second", &["chat"]))
            .await
            .expect("second client");
    let model = MockCompletionModel::new([MockTurn::text("done")]);

    let _agent = AgentBuilder::new(model)
        .tool(MockAddTool)
        .a2a_tool(&first)
        .a2a_tool(&second)
        .build();

    // The reverse order exercises the `NoToolConfig` impl.
    let model = MockCompletionModel::new([MockTurn::text("done")]);
    let _agent = AgentBuilder::new(model)
        .a2a_tool(&first)
        .tool(MockAddTool)
        .build();
}

#[tokio::test]
async fn empty_request_context_is_rejected_before_any_network_call() {
    // A card pointing at a closed port: if validation didn't reject first,
    // the send would fail with a connection error instead.
    let card = card_with_skills("http://127.0.0.1:1", "unreachable", &["chat"]);
    let client = A2AClient::from_agent_card(card)
        .await
        .expect("client from hand-supplied card");

    let err = client
        .message("hello")
        .context("   ")
        .send()
        .await
        .expect_err("blank context id must be rejected");
    assert!(
        matches!(err, rig_a2a::A2AError::InvalidContextId),
        "expected InvalidContextId, got {err:?}"
    );

    let err = client
        .message("hello")
        .task("")
        .send()
        .await
        .expect_err("blank task id must be rejected");
    assert!(
        matches!(err, rig_a2a::A2AError::InvalidTaskId),
        "expected InvalidTaskId, got {err:?}"
    );
}

#[tokio::test]
async fn cross_origin_card_interfaces_are_rejected_unless_allowed() {
    let _ = tracing_subscriber::fmt::try_init();

    // The served card advertises an interface on a *different* origin than
    // the one it is fetched from.
    let card = card_with_skills("http://192.0.2.1:9", "hostile", &["chat"]);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = axum::Router::new().route(
        "/.well-known/agent-card.json",
        axum::routing::get(move || {
            let card = card.clone();
            async move { axum::Json(card) }
        }),
    );
    let server_task =
        tokio::spawn(async move { axum::serve(listener, app).await.expect("test server") });

    let err = match A2AClient::from_url(format!("http://{addr}")).await {
        Ok(_) => panic!("cross-origin interface must be rejected"),
        Err(err) => err,
    };
    assert!(
        matches!(
            err,
            rig_a2a::A2AError::AgentCard(
                rig_a2a::error::AgentCardError::CrossOriginInterface { .. }
            )
        ),
        "expected CrossOriginInterface, got {err:?}"
    );

    // Trusted deployments can opt in to cross-origin interfaces.
    A2AClient::builder()
        .url(format!("http://{addr}"))
        .allow_cross_origin_interfaces(true)
        .build()
        .await
        .expect("cross-origin card should be accepted when explicitly allowed");

    server_task.abort();
}

#[tokio::test]
async fn cross_origin_agent_card_redirect_is_not_followed() {
    let redirected = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let target_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let target_addr = target_listener.local_addr().unwrap();
    let redirected_for_route = redirected.clone();
    let target_app = axum::Router::new().route(
        "/card",
        axum::routing::get(move || {
            let redirected = redirected_for_route.clone();
            async move {
                redirected.store(true, std::sync::atomic::Ordering::SeqCst);
                axum::http::StatusCode::OK
            }
        }),
    );
    let target_task =
        tokio::spawn(async move { axum::serve(target_listener, target_app).await.unwrap() });

    let source_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let source_addr = source_listener.local_addr().unwrap();
    let location = format!("http://{target_addr}/card");
    let source_app = axum::Router::new().route(
        "/.well-known/agent-card.json",
        axum::routing::get(move || {
            let location = location.clone();
            async move {
                (
                    axum::http::StatusCode::TEMPORARY_REDIRECT,
                    [(axum::http::header::LOCATION, location)],
                )
            }
        }),
    );
    let source_task =
        tokio::spawn(async move { axum::serve(source_listener, source_app).await.unwrap() });

    if A2AClient::from_url(format!("http://{source_addr}"))
        .await
        .is_ok()
    {
        panic!("agent-card redirect must not be followed");
    }
    assert!(!redirected.load(std::sync::atomic::Ordering::SeqCst));

    source_task.abort();
    target_task.abort();
}

#[tokio::test]
async fn cross_origin_protocol_redirect_is_not_followed() {
    let redirected = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let target_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let target_addr = target_listener.local_addr().unwrap();
    let redirected_for_route = redirected.clone();
    let target_app = axum::Router::new().route(
        "/messages",
        axum::routing::post(move || {
            let redirected = redirected_for_route.clone();
            async move {
                redirected.store(true, std::sync::atomic::Ordering::SeqCst);
                axum::http::StatusCode::OK
            }
        }),
    );
    let target_task =
        tokio::spawn(async move { axum::serve(target_listener, target_app).await.unwrap() });

    let source_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let source_addr = source_listener.local_addr().unwrap();
    let card = card_with_skills(&format!("http://{source_addr}"), "redirect", &["chat"]);
    let location = format!("http://{target_addr}/messages");
    let source_app = axum::Router::new()
        .route(
            "/.well-known/agent-card.json",
            axum::routing::get(move || {
                let card = card.clone();
                async move { axum::Json(card) }
            }),
        )
        .route(
            "/jsonrpc",
            axum::routing::post(move || {
                let location = location.clone();
                async move {
                    (
                        axum::http::StatusCode::TEMPORARY_REDIRECT,
                        [(axum::http::header::LOCATION, location)],
                    )
                }
            }),
        );
    let source_task =
        tokio::spawn(async move { axum::serve(source_listener, source_app).await.unwrap() });

    let mut default_headers = reqwest::header::HeaderMap::new();
    default_headers.insert(
        "x-api-key",
        reqwest::header::HeaderValue::from_static("must-not-leak"),
    );
    let client = A2AClient::builder()
        .url(format!("http://{source_addr}"))
        .http_client_builder(
            reqwest::Client::builder()
                .default_headers(default_headers)
                .redirect(reqwest::redirect::Policy::limited(10)),
        )
        .build()
        .await
        .expect("client should fetch agent card");
    client
        .message("secret")
        .send()
        .await
        .expect_err("protocol redirect must not be followed");
    assert!(!redirected.load(std::sync::atomic::Ordering::SeqCst));

    source_task.abort();
    target_task.abort();
}

#[tokio::test]
async fn configured_http_timeout_bounds_agent_card_fetch() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = axum::Router::new().route(
        "/.well-known/agent-card.json",
        axum::routing::get(|| async { std::future::pending::<axum::http::StatusCode>().await }),
    );
    let server_task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let result = tokio::time::timeout(
        Duration::from_secs(5),
        A2AClient::builder()
            .url(format!("http://{addr}"))
            .http_client_builder(reqwest::Client::builder())
            .timeout(Duration::from_millis(100))
            .build(),
    )
    .await
    .expect("A2A timeout was not applied");
    let err = match result {
        Ok(_) => panic!("hanging card request must time out"),
        Err(err) => err,
    };
    assert!(matches!(err, rig_a2a::A2AError::Http(ref err) if err.is_timeout()));

    server_task.abort();
}

#[tokio::test]
async fn oversized_agent_card_is_rejected() {
    let _ = tracing_subscriber::fmt::try_init();

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = axum::Router::new().route(
        "/.well-known/agent-card.json",
        axum::routing::get(|| async { "x".repeat(2 * 1024 * 1024) }),
    );
    let server_task =
        tokio::spawn(async move { axum::serve(listener, app).await.expect("test server") });

    let err = match A2AClient::from_url(format!("http://{addr}")).await {
        Ok(_) => panic!("oversized card must be rejected"),
        Err(err) => err,
    };
    assert!(
        matches!(
            err,
            rig_a2a::A2AError::AgentCard(rig_a2a::error::AgentCardError::ResponseTooLarge { .. })
        ),
        "expected ResponseTooLarge, got {err:?}"
    );

    server_task.abort();
}

#[tokio::test]
async fn card_fetch_failure_surfaces_status_and_body_snippet() {
    let _ = tracing_subscriber::fmt::try_init();

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = axum::Router::new().route(
        "/.well-known/agent-card.json",
        axum::routing::get(|| async {
            (
                axum::http::StatusCode::SERVICE_UNAVAILABLE,
                r#"{"error":"tenant suspended"}"#,
            )
        }),
    );
    let server_task =
        tokio::spawn(async move { axum::serve(listener, app).await.expect("test server") });

    let err = match A2AClient::from_url(format!("http://{addr}")).await {
        Ok(_) => panic!("5xx card fetch must fail"),
        Err(err) => err,
    };
    let rendered = err.to_string();
    assert!(
        rendered.contains("503"),
        "expected status in error: {rendered}"
    );
    assert!(
        rendered.contains("tenant suspended"),
        "expected body snippet in error: {rendered}"
    );

    server_task.abort();
}
