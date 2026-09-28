//! Migrated from `examples/openai_websocket_mode.rs`.

use anyhow::Result;
use futures::StreamExt;
use rig::driver::Model;
use rig::message::AssistantContent;
use rig::providers::openai;
use rig::streaming::{Item, StreamEvent};
use rig_test_support::cassette_models::OpenAiModels;

use crate::support::{
    Adder, STREAMING_TOOLS_PREAMBLE, STREAMING_TOOLS_PROMPT, Subtract, TOOLS_PREAMBLE,
    TOOLS_PROMPT, assert_mentions_expected_number, assert_nonempty_response,
    collect_stream_final_response,
};
use rig::completion::CompletionRequest;
use rig::http_client::{self, NoBody, Request};
use rig::wasm_compat::{WasmBoxedFuture, WasmCompatSend};
use rig::ws_client::{
    BoxedWebSocketConnection, CloseFrame, ConnectOptions, Frame, WebSocketClientExt,
    WebSocketConnection,
};
use std::sync::{Arc, Mutex};

/// Install rustls' process-wide crypto provider before a websocket connects.
/// This test graph enables both `ring` and `aws-lc-rs`, so the tungstenite
/// connector cannot pick one; `aws-lc-rs` is the one reqwest uses. Installing
/// twice is a no-op.
pub(crate) fn install_tls_provider() {
    let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
}

fn extract_text(choice: &[AssistantContent]) -> String {
    choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.clone()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

/// Warm up, then send only the new input each turn: the chaining opt-in
/// carries the conversation on the connection.
#[tokio::test]
#[ignore = "requires OPENAI_API_KEY and --features websocket"]
async fn websocket_chained_roundtrip() -> Result<()> {
    install_tls_provider();
    let client = OpenAiModels::from_env().expect("config should build from env");
    let model = client.responses(openai::GPT_4O_MINI);
    let socket = model.responses_websocket().chaining().connect().await?;

    let warmup = Model::new(socket.wire.clone().warmup(), socket.transport.clone());
    let warmup_request =
        CompletionRequest::new("You will answer a follow-up question about websocket mode.")
            .preamble("Be precise and concise.");
    let warmed = warmup.call(warmup_request).await?;
    anyhow::ensure!(
        warmed
            .response_id
            .as_deref()
            .is_some_and(|id| !id.is_empty()),
        "warmup should return a response id"
    );

    let request = CompletionRequest::new("Explain the benefit of websocket mode in one sentence.");
    let mut stream = socket.stream(request)?;
    let mut streamed_text = String::new();
    while let Some(item) = stream.next().await {
        if let Item::Event(StreamEvent::Text { text, .. }) = item? {
            streamed_text.push_str(&text);
        }
    }
    stream.finish().await?;
    assert_nonempty_response(&streamed_text);

    let chained_request =
        CompletionRequest::new("Now restate that as three very short bullet points.");
    let response = socket.call(chained_request).await?;
    let text = extract_text(&response.choice);
    assert_nonempty_response(&text);

    socket.transport.close().await?;
    Ok(())
}

/// The bundled backend, keeping every text message the provider sent.
#[derive(Clone, Default)]
struct Recording {
    inner: rig::rig_tungstenite::TungsteniteClient,
    received: Arc<Mutex<Vec<String>>>,
}

impl Recording {
    fn received(&self) -> Vec<serde_json::Value> {
        self.received
            .lock()
            .expect("recording")
            .iter()
            .map(|text| serde_json::from_str(text).expect("every message is JSON"))
            .collect()
    }
}

struct Tap(BoxedWebSocketConnection, Arc<Mutex<Vec<String>>>);

impl WebSocketConnection for Tap {
    fn send(&mut self, frame: Frame) -> WasmBoxedFuture<'_, http_client::Result<()>> {
        self.0.send(frame)
    }

    fn recv(&mut self) -> WasmBoxedFuture<'_, http_client::Result<Option<Frame>>> {
        Box::pin(async move {
            let received = self.0.recv().await;
            if let Ok(Some(Frame::Text(text))) = &received {
                self.1.lock().expect("recording").push(text.clone());
            }
            received
        })
    }

    fn close(&mut self, frame: Option<CloseFrame>) -> WasmBoxedFuture<'_, http_client::Result<()>> {
        self.0.close(frame)
    }
}

impl WebSocketClientExt for Recording {
    fn connect(
        &self,
        request: Request<NoBody>,
        options: ConnectOptions,
    ) -> impl Future<Output = http_client::Result<BoxedWebSocketConnection>> + WasmCompatSend {
        let (inner, received) = (self.inner, self.received.clone());
        async move {
            let connection = inner.connect(request, options).await?;
            Ok(Box::new(Tap(connection, received)) as BoxedWebSocketConnection)
        }
    }
}

/// What the transport assumes of the live protocol: every event names its
/// type, each turn ends at a `response.completed` carrying a
/// `sequence_number`, and a trailing `response.done` for the same response,
/// when the provider sends one, is not read as the next turn's end.
#[tokio::test]
#[ignore = "requires OPENAI_API_KEY and --features websocket"]
async fn websocket_live_events_have_the_shape_the_transport_reads() -> Result<()> {
    install_tls_provider();
    let client = OpenAiModels::from_env().expect("config should build from env");
    let backend = Recording::default();
    let socket = client
        .responses(openai::GPT_4O_MINI)
        .responses_websocket()
        .connect_with(&backend)
        .await?;

    let first = socket.call("Say hello in three words.").await?;
    let second = socket.call("Say goodbye in three words.").await?;
    assert_nonempty_response(&extract_text(&first.choice));
    assert_nonempty_response(&extract_text(&second.choice));
    anyhow::ensure!(
        first.response_id != second.response_id,
        "each turn is its own response"
    );

    let events = backend.received();
    anyhow::ensure!(
        events.iter().all(|event| event["type"].is_string()),
        "every event names its type: {events:?}"
    );
    let terminals: Vec<_> = events
        .iter()
        .filter(|event| event["type"] == "response.completed")
        .collect();
    anyhow::ensure!(terminals.len() == 2, "one terminal per turn: {terminals:?}");
    anyhow::ensure!(
        terminals
            .iter()
            .all(|event| event["sequence_number"].is_u64()),
        "every terminal carries its sequence number: {terminals:?}"
    );
    socket.transport.close().await?;
    Ok(())
}

/// An agent's tool loop runs over one websocket connection, each turn
/// carrying the whole conversation.
#[tokio::test]
#[ignore = "requires OPENAI_API_KEY and --features websocket"]
async fn websocket_agent_tool_loop() -> Result<()> {
    install_tls_provider();
    let client = OpenAiModels::from_env().expect("config should build from env");
    let socket = client
        .responses(openai::GPT_4O_MINI)
        .responses_websocket()
        .connect()
        .await?;
    let agent = rig::AgentBuilder::new(socket.clone())
        .preamble(TOOLS_PREAMBLE)
        .tool(Adder)
        .tool(Subtract)
        .build();

    let response = agent.prompt(TOOLS_PROMPT).max_turns(3).await?.output;
    assert_mentions_expected_number(&response, -3);
    socket.transport.close().await?;
    Ok(())
}

/// The same tool loop, streamed.
#[tokio::test]
#[ignore = "requires OPENAI_API_KEY and --features websocket"]
async fn websocket_agent_streaming_tool_loop() -> Result<()> {
    install_tls_provider();
    let client = OpenAiModels::from_env().expect("config should build from env");
    let socket = client
        .responses(openai::GPT_4O_MINI)
        .responses_websocket()
        .connect()
        .await?;
    let agent = rig::AgentBuilder::new(socket.clone())
        .preamble(STREAMING_TOOLS_PREAMBLE)
        .tool(Adder)
        .tool(Subtract)
        .build();

    let mut stream = agent.prompt(STREAMING_TOOLS_PROMPT).max_turns(3).stream();
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("streaming tool prompt should succeed");
    assert_nonempty_response(&response);
    socket.transport.close().await?;
    Ok(())
}

/// A turn dropped mid-stream is read to its end before the next turn, which
/// answers its own prompt; two turns sent at once both complete.
#[tokio::test]
#[ignore = "requires OPENAI_API_KEY and --features websocket"]
async fn websocket_dropped_and_queued_turns() -> Result<()> {
    install_tls_provider();
    let client = OpenAiModels::from_env().expect("config should build from env");
    let socket = client
        .responses(openai::GPT_4O_MINI)
        .responses_websocket()
        .connect()
        .await?;

    let mut dropped = socket.stream("Count from 1 to 40, one number per line.")?;
    let mut seen = false;
    while let Some(item) = dropped.next().await {
        if let Item::Event(StreamEvent::Text { .. }) = item? {
            seen = true;
            break;
        }
    }
    anyhow::ensure!(seen, "the dropped turn streamed before it was dropped");
    drop(dropped);

    let after = socket
        .call("Reply with exactly the word: pineapple")
        .await?;
    let text = extract_text(&after.choice).to_lowercase();
    anyhow::ensure!(
        text.contains("pineapple") && !text.contains("40"),
        "the turn after a dropped one answers its own prompt: {text}"
    );

    let (one, two) = tokio::join!(
        socket.call("Reply with exactly the word: apple"),
        socket.call("Reply with exactly the word: banana"),
    );
    let (one, two) = (one?, two?);
    anyhow::ensure!(extract_text(&one.choice).to_lowercase().contains("apple"));
    anyhow::ensure!(extract_text(&two.choice).to_lowercase().contains("banana"));
    socket.transport.close().await?;
    Ok(())
}
