//! An agent served on a loopback port, driven by the official A2A client.
#![cfg(not(target_family = "wasm"))]

use std::sync::Arc;

use a2a_client::agent_card::AgentCardResolver;
use a2a_client::{A2AClient, A2AClientFactory, Transport};
use anyhow::{Context, ensure};
use futures::StreamExt;
use rig_a2a::A2aServer;
use rig_a2a::a2a::{
    CancelTaskRequest, GetTaskRequest, Message, Part, Role, SendMessageRequest,
    SendMessageResponse, StreamResponse, TRANSPORT_PROTOCOL_HTTP_JSON, TRANSPORT_PROTOCOL_JSONRPC,
    Task, TaskState, error_code,
};
use rig_agent::test_utils::MockControlledTool;
use rig_agent::{Agent, AgentBuilder};
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use tokio::sync::Notify;

type Client = A2AClient<Box<dyn Transport>>;

/// Serve `agent` on a free loopback port and return its base URL.
async fn serve(agent: Agent) -> anyhow::Result<String> {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let url = format!("http://{}", listener.local_addr()?);
    let app = A2aServer::new(agent, &url).into_router();
    tokio::spawn(async move { rig_a2a::axum::serve(listener, app).await });
    Ok(url)
}

/// A client of the agent at `url` over `binding`, chosen from its card.
async fn connect(url: &str, binding: &str) -> anyhow::Result<Client> {
    let card = AgentCardResolver::new(None).resolve(url).await?;
    ensure!(card.name == "echo", "the card names the agent");
    let factory = A2AClientFactory::builder()
        .preferred_bindings(vec![binding.to_owned()])
        .build();
    Ok(factory.create_from_card(&card).await?)
}

fn request(text: &str, context_id: Option<&str>) -> SendMessageRequest {
    let mut message = Message::new(Role::User, vec![Part::text(text)]);
    message.context_id = context_id.map(str::to_owned);
    SendMessageRequest {
        message,
        configuration: None,
        metadata: None,
        tenant: None,
    }
}

fn answer(task: &Task) -> Option<String> {
    let artifact = task.artifacts.as_ref()?.first()?;
    Some(artifact.parts.iter().filter_map(Part::as_text).collect())
}

fn reply(chunks: &[&str]) -> Vec<MockStreamEvent> {
    chunks
        .iter()
        .map(|chunk| MockStreamEvent::text(*chunk))
        .chain([MockStreamEvent::final_response_with_default_usage()])
        .collect()
}

#[tokio::test]
async fn a_context_streams_and_remembers_over_both_bindings() -> anyhow::Result<()> {
    let model = MockCompletionModel::from_stream_turns([
        reply(&["Hel", "lo"]),
        reply(&["Again"]),
        reply(&["Third"]),
    ]);
    let agent = AgentBuilder::new(model.clone()).name("echo").build();
    let url = serve(agent).await?;
    let jsonrpc = connect(&url, TRANSPORT_PROTOCOL_JSONRPC).await?;

    let mut events = Vec::new();
    let mut stream = jsonrpc.send_streaming_message(&request("Hi", None)).await?;
    while let Some(event) = stream.next().await {
        events.push(event?);
    }
    let Some(StreamResponse::Task(started)) = events.first() else {
        anyhow::bail!("the stream starts with the task: {events:?}");
    };
    ensure!(started.status.state == TaskState::Working);
    let chunks: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            StreamResponse::ArtifactUpdate(update) => Some((
                update
                    .artifact
                    .parts
                    .iter()
                    .filter_map(Part::as_text)
                    .collect::<String>(),
                update.append.unwrap_or(false),
                update.last_chunk.unwrap_or(false),
            )),
            _ => None,
        })
        .collect();
    ensure!(
        chunks
            == [
                ("Hel".to_owned(), false, false),
                ("lo".to_owned(), true, false),
                ("Hello".to_owned(), false, true),
            ],
        "deltas stream, then the answer replaces them: {chunks:?}"
    );
    let Some(StreamResponse::StatusUpdate(done)) = events.last() else {
        anyhow::bail!("the stream ends with a status: {events:?}");
    };
    ensure!(done.status.state == TaskState::Completed);

    let task = jsonrpc
        .get_task(&GetTaskRequest {
            id: started.id.clone(),
            history_length: None,
            tenant: None,
        })
        .await?;
    ensure!(task.status.state == TaskState::Completed);
    ensure!(answer(&task).as_deref() == Some("Hello"), "{task:?}");

    let SendMessageResponse::Task(second) = jsonrpc
        .send_message(&request("And again?", Some(&started.context_id)))
        .await?
    else {
        anyhow::bail!("a message starts a task");
    };
    ensure!(second.context_id == started.context_id && second.id != started.id);
    ensure!(answer(&second).as_deref() == Some("Again"), "{second:?}");

    // The same context over the HTTP+JSON binding, from a second client.
    let rest = connect(&url, TRANSPORT_PROTOCOL_HTTP_JSON).await?;
    let SendMessageResponse::Task(third) = rest
        .send_message(&request("Once more", Some(&started.context_id)))
        .await?
    else {
        anyhow::bail!("a message starts a task");
    };
    ensure!(answer(&third).as_deref() == Some("Third"), "{third:?}");

    let requests = model.requests();
    let history = requests.last().context("the model was called")?;
    let texts: Vec<String> = history
        .chat_history
        .iter()
        .map(|m| format!("{m:?}"))
        .collect();
    for said in ["Hi", "Hello", "And again?", "Again", "Once more"] {
        ensure!(
            texts.iter().any(|text| text.contains(said)),
            "the context's history reaches the model: {said} missing from {texts:?}"
        );
    }
    Ok(())
}

#[tokio::test]
async fn cancel_ends_a_running_task() -> anyhow::Result<()> {
    let started = Arc::new(Notify::new());
    let tool = MockControlledTool::new(Arc::clone(&started), Arc::new(Notify::new()));
    let model = MockCompletionModel::from_stream_turns([vec![
        MockStreamEvent::tool_call("call-1", "controlled", serde_json::json!({})),
        MockStreamEvent::final_response_with_default_usage(),
    ]]);
    let agent = AgentBuilder::new(model)
        .name("echo")
        .tool(tool)
        .default_max_turns(2)
        .build();
    let client = connect(&serve(agent).await?, TRANSPORT_PROTOCOL_JSONRPC).await?;

    let mut stream = client
        .send_streaming_message(&request("Wait", None))
        .await?;
    let Some(Ok(StreamResponse::Task(task))) = stream.next().await else {
        anyhow::bail!("the stream starts with the task");
    };
    started.notified().await;

    let canceled = client
        .cancel_task(&CancelTaskRequest {
            id: task.id.clone(),
            metadata: None,
            tenant: None,
        })
        .await?;
    ensure!(canceled.status.state == TaskState::Canceled, "{canceled:?}");
    let mut last = None;
    while let Some(event) = stream.next().await {
        last = Some(event?);
    }
    let Some(StreamResponse::StatusUpdate(update)) = last else {
        anyhow::bail!("the stream ends with the cancellation: {last:?}");
    };
    ensure!(update.status.state == TaskState::Canceled);
    Ok(())
}

#[tokio::test]
async fn parts_the_agent_cannot_read_are_refused() -> anyhow::Result<()> {
    let agent = AgentBuilder::new(MockCompletionModel::from_stream_turns(Vec::<
        Vec<MockStreamEvent>,
    >::new()))
    .name("echo")
    .build();
    let client = connect(&serve(agent).await?, TRANSPORT_PROTOCOL_JSONRPC).await?;

    let mut undeclared = request("", None);
    undeclared.message.parts =
        vec![Part::raw(b"%PDF-".to_vec()).with_media_type("application/pdf")];
    let error = client
        .send_message(&undeclared)
        .await
        .err()
        .context("a media type outside the card's input modes is refused")?;
    ensure!(
        error.code == error_code::CONTENT_TYPE_NOT_SUPPORTED,
        "{error:?}"
    );

    let mut untyped = request("", None);
    untyped.message.parts = vec![Part::url("https://example.com/file")];
    let SendMessageResponse::Task(task) = client.send_message(&untyped).await? else {
        anyhow::bail!("a message starts a task");
    };
    ensure!(task.status.state == TaskState::Rejected, "{task:?}");
    Ok(())
}
