//! Loopback integration tests driving [`rig_a2a::A2AClient`] over real HTTP on
//! 127.0.0.1 against a stub A2A server built from the upstream `a2a-server`
//! crate (`DefaultRequestHandler` + a scripted executor).
//!
//! Validates well-known card discovery, transport negotiation, A2A completion
//! and streaming, host-side conversation threading, Rig sub-agent composition,
//! and full roundtrips over JSON-RPC and HTTP+JSON.

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
use rig_a2a::{A2AClient, A2AConversationExt, SendMessageResponse};
use rig_agent::agent::AgentBuilder;
use rig_agent::completion::Prompt;
use rig_agent::test_utils::{MockCompletionModel, MockTurn};
use rig_core::completion::{CompletionModel, CompletionRequest, ToolDefinition};
use rig_core::message::{Message as RigMessage, ToolResultContent, UserContent};

/// What the stub executor replies to a `message/send`.
#[derive(Clone)]
enum StubReply {
    CompletedText(String),
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

#[tokio::test]
async fn from_url_builds_an_agent_named_and_described_from_the_card() {
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
    assert_eq!(client.agent_name(), "multi-skill");

    let agent = client.agent().build();
    assert_eq!(agent.name(), Some("multi-skill"));
    let description = agent.description().expect("agent description");
    assert!(description.contains("alpha"), "{description}");
    assert!(description.contains("beta"), "{description}");

    server.abort();
}

#[tokio::test]
async fn explicit_agent_name_overrides_the_card() {
    let card = card_with_skills("http://127.0.0.1:1", "Rig Agent", &["chat"]);
    let client = A2AClient::builder()
        .card(card)
        .agent_name("billing-desk")
        .build()
        .await
        .expect("client from hand-supplied card");

    assert_eq!(client.agent_name(), "billing-desk");
    assert_eq!(client.agent().build().name(), Some("billing-desk"));
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
async fn card_without_skills_still_yields_an_agent() {
    let card = card_with_skills("http://127.0.0.1:1", "Bare Agent", &[]);
    let client = A2AClient::from_agent_card(card)
        .await
        .expect("client from hand-supplied card");

    let agent = client.agent().build();
    assert_eq!(agent.name(), Some("bare-agent"));
    let description = agent.description().expect("agent description");
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

/// Full roundtrip through Rig's standard sub-agent tool conversion.
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
    let agent_name = remote.agent_name().to_string();
    let remote_tool = remote.agent().build().into_tool();

    let model = MockCompletionModel::new([
        MockTurn::tool_call(
            "tool_call_1",
            &agent_name,
            serde_json::json!({"prompt": "greet me"}),
        ),
        MockTurn::text("done"),
    ]);
    let agent = AgentBuilder::new(model.clone())
        .name("local")
        .dynamic_tool(remote_tool)
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
    // The unbound sub-agent is stateless.
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
async fn sub_agent_tool_calls_remote_agent_end_to_end() {
    agent_calls_remote_agent(StubBinding::JsonRpc).await;
}

#[tokio::test]
async fn sub_agent_tool_calls_remote_agent_over_rest() {
    agent_calls_remote_agent(StubBinding::Rest).await;
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
    // per the A2A spec's client threading model. The direct API stays explicit.
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

// ---------------------------------------------------------------------------
// A2AModel: the remote agent as a Rig completion model
// ---------------------------------------------------------------------------

fn text_request(prompt: &str) -> CompletionRequest {
    CompletionRequest {
        model: None,
        preamble: None,
        chat_history: vec![RigMessage::user(prompt)],
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
        Some(rig_core::completion::AssistantContent::Text(text)) => text.text.clone(),
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
        Some(rig_core::completion::AssistantContent::Text(text)) => text.text.clone(),
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

/// One agent, many conversations: each run names both the Rig memory
/// conversation and its remote A2A thread.
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
        .a2a_conversation("alice")
        .await
        .expect("alice turn 1");
    agent
        .prompt("bob turn 1")
        .a2a_conversation("bob")
        .await
        .expect("bob turn 1");
    agent
        .prompt("alice turn 2")
        .a2a_conversation("alice")
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
            Some(rig_core::completion::AssistantContent::Text(text)) if text.text == "final answer"
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

    // Bind the conversation on the ordinary Rig builder before converting the
    // A2A-backed agent through Rig's standard sub-agent bridge.
    let sub_agent = remote.agent().a2a_conversation("user-42").build();

    let model = MockCompletionModel::new([
        MockTurn::tool_call(
            "call_1",
            "researcher",
            serde_json::json!({"prompt": "look it up"}),
        ),
        MockTurn::tool_call(
            "call_2",
            "researcher",
            serde_json::json!({"prompt": "check again"}),
        ),
        MockTurn::text("relayed"),
    ]);
    let orchestrator = AgentBuilder::new(model.clone())
        .dynamic_tool(sub_agent.into_tool())
        .build();

    let out = orchestrator
        .prompt("delegate")
        .max_turns(4)
        .await
        .expect("orchestrator run");
    assert_eq!(out, "relayed");

    let results = tool_result_texts(&model);
    assert!(
        results.iter().any(|text| text == "remote findings"),
        "the sub-agent's remote answer should reach the orchestrator: {results:?}"
    );
    let requests = recorded.lock().unwrap();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].message.context_id, None);
    assert!(
        requests[1].message.context_id.is_some(),
        "the bound sub-agent must continue its remote conversation"
    );

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
async fn selected_interface_tenant_reaches_direct_and_model_requests() {
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
    client
        .model()
        .completion(text_request("model"))
        .await
        .expect("model request should succeed");

    let requests = recorded.lock().unwrap();
    assert_eq!(requests.len(), 2);
    assert!(
        requests
            .iter()
            .all(|request| request.tenant.as_deref() == Some("tenant-a"))
    );

    server.abort();
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
