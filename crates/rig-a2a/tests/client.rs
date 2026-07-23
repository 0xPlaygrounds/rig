//! Loopback integration tests that drive [`rig_a2a::A2AClient`] and
//! [`rig_a2a::A2ATool`] over real HTTP on 127.0.0.1 against a stub A2A
//! server built from the upstream `a2a-server` crate
//! (`DefaultRequestHandler` + a canned-response executor). Validates
//! well-known card discovery, transport negotiation, skill→tool projection,
//! explicit `contextId` threading, task-state → tool-output projection, and
//! full agent-driven roundtrips over both HTTP transports (JSON-RPC and
//! HTTP+JSON/REST).

#![allow(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::unwrap_used
)]

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
use rig_a2a::A2ATool;
use rig_a2a::{A2AAgentBuilderExt, A2AClient, SendMessageResponse};
use rig_core::agent::AgentBuilder;
use rig_core::completion::Prompt;
use rig_core::message::{Message as RigMessage, ToolResultContent, UserContent};
use rig_core::test_utils::{MockAddTool, MockCompletionModel, MockTurn};
use rig_core::tool::{ToolContext, ToolExecutionError, ToolSet};

/// What the stub executor replies to every `message/send`.
#[derive(Clone)]
enum StubReply {
    CompletedText(String),
    MessageText(String),
    InputRequired { prompt: String },
    Failed { reason: String },
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

struct StubExecutor {
    reply: StubReply,
    recorded: Recorded,
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
        let (state, status_text, artifact_text) = match &self.reply {
            StubReply::CompletedText(text) => (TaskState::Completed, None, Some(text.clone())),
            StubReply::MessageText(text) => {
                let mut message = Message::new(Role::Agent, vec![Part::text(text.clone())]);
                message.context_id = Some(ctx.context_id);
                message.task_id = Some(ctx.task_id);
                return futures::stream::once(async move { Ok(StreamResponse::Message(message)) })
                    .boxed();
            }
            StubReply::InputRequired { prompt } => {
                (TaskState::InputRequired, Some(prompt.clone()), None)
            }
            StubReply::Failed { reason } => (TaskState::Failed, Some(reason.clone()), None),
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
    serve_stub_with_binding(name, skill_ids, reply, StubBinding::JsonRpc).await
}

async fn serve_stub_with_binding(
    name: &str,
    skill_ids: &[&str],
    reply: StubReply,
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
    let executor = StubExecutor {
        reply,
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

/// Drive an [`A2ATool`] the way test code once called the (now private)
/// object-safe `ToolDyn::call` directly, going through the public
/// `into_dynamic_tool` + [`ToolSet`] dispatch path instead.
async fn call_tool(tool: &A2ATool, args: &str) -> Result<String, ToolExecutionError> {
    let dynamic = tool.clone().into_dynamic_tool();
    let name = dynamic.name().to_string();
    let mut toolset = ToolSet::default();
    toolset.add_dynamic_tool(dynamic);
    let mut context = ToolContext::new();
    let result = toolset.execute(&name, args, &mut context).await;
    if result.is_success() {
        Ok(result.output().as_text().unwrap_or_default().to_string())
    } else {
        Err(result
            .error()
            .cloned()
            .expect("a failed A2ATool call carries a structured error"))
    }
}

#[tokio::test]
async fn from_url_fetches_card_and_maps_skills_to_distinct_tools() {
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

    let tools = client.tools();
    assert_eq!(tools.len(), 2);
    let names: Vec<String> = tools.iter().map(A2ATool::name).collect();
    assert_ne!(names[0], names[1]);
    for name in &names {
        assert!(name.len() <= 64, "{name}");
        assert!(
            name.chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-'),
            "tool name must be provider-safe, got {name}"
        );
    }

    server.abort();
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
async fn card_without_skills_yields_single_passthrough_tool() {
    let card = card_with_skills("http://127.0.0.1:1", "Bare Agent", &[]);
    let client = A2AClient::from_agent_card(card)
        .await
        .expect("client from hand-supplied card");

    let tools = client.tools();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].name(), "bare-agent");
    assert_eq!(tools[0].description(), "Stub agent Bare Agent.");
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
/// card discovery → skill→tool projection → scripted model tool call →
/// `message/send` to the upstream `a2a-server` handler → task reply →
/// tool output returned to the model → final answer.
async fn agent_calls_remote_skill(binding: StubBinding) {
    let _ = tracing_subscriber::fmt::try_init();

    let (addr, recorded, server) = serve_stub_with_binding(
        "greeter",
        &["greet"],
        StubReply::CompletedText("hello from remote".into()),
        binding,
    )
    .await;

    let remote = A2AClient::from_url(format!("http://{addr}"))
        .await
        .expect("client should fetch card");
    let tool_name = remote.tools()[0].name();

    // Turn 1: the model calls the remote A2A skill; turn 2: it answers with
    // the relayed text. The tool result surfaced to the mock must match what
    // the stub's artifact carried.
    let model = MockCompletionModel::new([
        MockTurn::tool_call(
            "tool_call_1",
            &tool_name,
            serde_json::json!({"text": "greet me"}),
        ),
        MockTurn::text("done"),
    ]);
    let agent = AgentBuilder::new(model.clone())
        .name("local")
        .a2a_tools(&remote)
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
    // No contextId argument was supplied, so the call is stateless.
    assert_eq!(messages[0].message.context_id, None);
    drop(messages);

    // The remote reply made it back into the model's context as the tool
    // result for the scripted call.
    let results = tool_result_texts(&model);
    assert!(
        results
            .iter()
            .any(|text| text.starts_with("hello from remote")),
        "model should see the remote reply as a tool result: {results:?}"
    );

    server.abort();
}

#[tokio::test]
async fn agent_with_a2a_tools_calls_remote_skill_end_to_end() {
    agent_calls_remote_skill(StubBinding::JsonRpc).await;
}

#[tokio::test]
async fn agent_with_a2a_tools_calls_remote_skill_over_rest() {
    agent_calls_remote_skill(StubBinding::Rest).await;
}

#[tokio::test]
async fn explicit_context_id_argument_reaches_remote() {
    let _ = tracing_subscriber::fmt::try_init();

    let (addr, recorded, server) =
        serve_stub("threaded", &["chat"], StubReply::CompletedText("ok".into())).await;

    let remote = A2AClient::from_url(format!("http://{addr}"))
        .await
        .expect("client should fetch card");
    let tools = remote.tools();
    let tool = &tools[0];

    let out = call_tool(
        tool,
        r#"{"text":"continue","contextId":"ctx-42","taskId":"task-7"}"#,
    )
    .await
    .expect("tool call should succeed");
    // The completed output carries the reply plus a contextId marker the
    // model can echo to continue the conversation; the terminal taskId is
    // not surfaced.
    assert_eq!(out, "ok\n[A2A TASK Completed contextId=\"ctx-42\"]");

    let messages = recorded.lock().unwrap();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].message.context_id.as_deref(), Some("ctx-42"));
    assert_eq!(messages[0].message.task_id.as_deref(), Some("task-7"));

    server.abort();
}

#[tokio::test]
async fn blank_context_id_argument_fails_without_calling_remote() {
    let _ = tracing_subscriber::fmt::try_init();

    let (addr, recorded, server) =
        serve_stub("threaded", &["chat"], StubReply::CompletedText("ok".into())).await;

    let remote = A2AClient::from_url(format!("http://{addr}"))
        .await
        .expect("client should fetch card");
    let tools = remote.tools();
    let tool = &tools[0];

    let err = call_tool(tool, r#"{"text":"continue","contextId":"  "}"#)
        .await
        .expect_err("blank contextId must be rejected");
    assert!(
        err.to_string().contains("contextId"),
        "expected contextId validation error, got {err}"
    );
    assert!(
        recorded.lock().unwrap().is_empty(),
        "no message must reach the remote on argument validation failure"
    );

    server.abort();
}

#[tokio::test]
async fn input_required_task_surfaces_marker_with_context_id() {
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
    let tools = remote.tools();
    let out = call_tool(&tools[0], r#"{"text":"do the thing"}"#)
        .await
        .expect("tool call should succeed");

    assert!(
        out.starts_with("[INPUT REQUIRED contextId="),
        "expected InputRequired marker, got {out:?}"
    );
    assert!(
        out.contains(" taskId="),
        "expected taskId in paused-task marker, got {out:?}"
    );
    assert!(
        out.ends_with(" prompt=\"which file?\""),
        "expected remote prompt, got {out:?}"
    );

    server.abort();
}

#[tokio::test]
async fn failed_task_surfaces_marker_with_status() {
    let _ = tracing_subscriber::fmt::try_init();

    let (addr, _recorded, server) = serve_stub(
        "failer",
        &["fail"],
        StubReply::Failed {
            reason: "quota exceeded".into(),
        },
    )
    .await;

    let remote = A2AClient::from_url(format!("http://{addr}"))
        .await
        .expect("client should fetch card");
    let tools = remote.tools();
    let out = call_tool(&tools[0], r#"{"text":"do the thing"}"#)
        .await
        .expect("tool call should succeed");

    assert!(out.contains("Failed"), "{out}");
    assert!(out.contains("quota exceeded"), "{out}");

    server.abort();
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
    // per the A2A spec's client threading model.
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
async fn bare_message_response_surfaces_server_generated_ids() {
    let _ = tracing_subscriber::fmt::try_init();

    let (addr, _recorded, server) = serve_stub(
        "messenger",
        &["chat"],
        StubReply::MessageText("hello".into()),
    )
    .await;

    let remote = A2AClient::from_url(format!("http://{addr}"))
        .await
        .expect("client should fetch card");
    let tools = remote.tools();
    let out = call_tool(&tools[0], r#"{"text":"say hello"}"#)
        .await
        .expect("tool call should succeed");

    assert!(
        out.starts_with("hello\n[A2A MESSAGE contextId=\""),
        "expected response context marker, got {out:?}"
    );
    assert!(
        out.contains(" taskId=\""),
        "expected response task marker, got {out:?}"
    );
    assert!(out.ends_with("\"]"), "expected closed marker, got {out:?}");

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
    let tools = client.tools();
    call_tool(&tools[0], r#"{"text":"tool"}"#)
        .await
        .expect("tool request should succeed");

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
async fn a2a_tools_append_after_existing_builder_tools() {
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
        .a2a_tools(&first)
        .a2a_tools(&second)
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
