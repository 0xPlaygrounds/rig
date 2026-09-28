//! The Responses websocket transport against recorded provider turns: a turn
//! streams its events, chains when asked, runs an agent's tool loop, and a
//! dropped or queued turn stays in step. Recorded with the bundled
//! tungstenite backend against `wss://api.openai.com/v1/responses`; replay
//! needs no network. Every turn sends `store: false`, so a recording leaves
//! no response stored on the account.

use std::sync::{Arc, Mutex};

use futures::StreamExt;
use rig::completion::CompletionRequest;
use rig::driver::Model;
use rig::http_client::{self, NoBody, Request};
use rig::message::AssistantContent;
use rig::providers::openai;
use rig::streaming::{Item, StreamEvent};
use rig::wasm_compat::{WasmBoxedFuture, WasmCompatSend};
use rig::ws_client::{
    BoxedWebSocketConnection, CloseFrame, ConnectOptions, Frame, WebSocketClientExt,
    WebSocketConnection,
};

use super::super::support::with_openai_websocket_turn_cassette;
use crate::support::{
    Adder, STREAMING_TOOLS_PREAMBLE, STREAMING_TOOLS_PROMPT, Subtract, TOOLS_PREAMBLE,
    TOOLS_PROMPT, assert_mentions_expected_number, assert_nonempty_response,
    collect_stream_final_response,
};

/// `prompt`, stored nowhere.
fn ask(prompt: &str) -> CompletionRequest {
    unstored(CompletionRequest::new(prompt))
}

fn unstored(mut request: CompletionRequest) -> CompletionRequest {
    request.additional_params = Some(serde_json::json!({ "store": false }));
    request
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

/// A streamed turn yields its text as it arrives, and a warmup then two
/// chained turns carry the conversation with only the new input.
#[tokio::test]
async fn chained_turns_after_a_warmup() {
    with_openai_websocket_turn_cassette(
        "websocket_turns/chained_turns_after_a_warmup",
        |client, backend| async move {
            let socket = client
                .openai
                .responses(openai::GPT_4O_MINI)
                .responses_websocket()
                .chaining()
                .connect_with(&backend)
                .await
                .expect("connects");

            let warmup = Model::new(socket.wire.clone().warmup(), socket.transport.clone());
            let warmed = warmup
                .call(unstored(
                    CompletionRequest::new(
                        "You will answer a follow-up question about websocket mode.",
                    )
                    .preamble("Be precise and concise."),
                ))
                .await
                .expect("warmup");
            assert!(
                warmed
                    .response_id
                    .as_deref()
                    .is_some_and(|id| !id.is_empty()),
                "a warmup returns a response id"
            );

            let mut stream = socket
                .stream(ask(
                    "Explain the benefit of websocket mode in one sentence.",
                ))
                .expect("stream opens");
            let mut streamed = String::new();
            let mut fragments = 0;
            while let Some(item) = stream.next().await {
                if let Item::Event(StreamEvent::Text { text, .. }) = item.expect("item") {
                    streamed.push_str(&text);
                    fragments += 1;
                }
            }
            let first = stream.finish().await.expect("streamed turn");
            assert_nonempty_response(&streamed);
            assert!(fragments > 1, "the text arrived in fragments");
            assert_eq!(extract_text(&first.choice), streamed);

            let chained = socket
                .call(ask("Now restate that as three very short bullet points."))
                .await
                .expect("chained turn");
            assert_nonempty_response(&extract_text(&chained.choice));
            assert_eq!(
                socket.transport.last_response_id().await,
                chained.response_id
            );
            socket.transport.close().await.expect("close");
        },
    )
    .await;
}

/// An agent's tool loop runs over one connection, each turn carrying the
/// whole conversation and no previous response.
#[tokio::test]
async fn agent_tool_loop() {
    with_openai_websocket_turn_cassette(
        "websocket_turns/agent_tool_loop",
        |client, backend| async move {
            let socket = client
                .openai
                .responses(openai::GPT_4O_MINI)
                .responses_websocket()
                .connect_with(&backend)
                .await
                .expect("connects");
            let agent = rig::AgentBuilder::new(socket.clone())
                .preamble(TOOLS_PREAMBLE)
                .additional_params(serde_json::json!({ "store": false }))
                .tool(Adder)
                .tool(Subtract)
                .build();

            let response = agent
                .prompt(TOOLS_PROMPT)
                .max_turns(3)
                .await
                .expect("the tool loop completes")
                .output;
            assert_mentions_expected_number(&response, -3);
            socket.transport.close().await.expect("close");
        },
    )
    .await;
}

/// The same tool loop, streamed.
#[tokio::test]
async fn agent_streaming_tool_loop() {
    with_openai_websocket_turn_cassette(
        "websocket_turns/agent_streaming_tool_loop",
        |client, backend| async move {
            let socket = client
                .openai
                .responses(openai::GPT_4O_MINI)
                .responses_websocket()
                .connect_with(&backend)
                .await
                .expect("connects");
            let agent = rig::AgentBuilder::new(socket.clone())
                .preamble(STREAMING_TOOLS_PREAMBLE)
                .additional_params(serde_json::json!({ "store": false }))
                .tool(Adder)
                .tool(Subtract)
                .build();

            let mut stream = agent.prompt(STREAMING_TOOLS_PROMPT).max_turns(3).stream();
            let response = collect_stream_final_response(&mut stream)
                .await
                .expect("the streamed tool loop completes");
            assert_nonempty_response(&response);
            socket.transport.close().await.expect("close");
        },
    )
    .await;
}

/// A turn dropped mid-stream is read to its end before the next turn, which
/// answers its own prompt; two turns sent at once both complete, one after
/// the other.
#[tokio::test]
async fn dropped_and_queued_turns() {
    with_openai_websocket_turn_cassette(
        "websocket_turns/dropped_and_queued_turns",
        |client, backend| async move {
            let socket = client
                .openai
                .responses(openai::GPT_4O_MINI)
                .responses_websocket()
                .connect_with(&backend)
                .await
                .expect("connects");

            let mut dropped = socket
                .stream(ask("Count from 1 to 40, one number per line."))
                .expect("stream opens");
            let mut seen = false;
            while let Some(item) = dropped.next().await {
                if let Item::Event(StreamEvent::Text { .. }) = item.expect("item") {
                    seen = true;
                    break;
                }
            }
            assert!(seen, "the dropped turn streamed before it was dropped");
            drop(dropped);

            let after = socket
                .call(ask("Reply with exactly the word: pineapple"))
                .await
                .expect("the turn after a dropped one");
            let text = extract_text(&after.choice).to_lowercase();
            assert!(
                text.contains("pineapple") && !text.contains("40"),
                "the turn after a dropped one answers its own prompt: {text}"
            );

            let (one, two) = tokio::join!(
                socket.call(ask("Reply with exactly the word: apple")),
                socket.call(ask("Reply with exactly the word: banana")),
            );
            let (one, two) = (one.expect("first queued"), two.expect("second queued"));
            assert!(extract_text(&one.choice).to_lowercase().contains("apple"));
            assert!(extract_text(&two.choice).to_lowercase().contains("banana"));
            assert_ne!(one.response_id, two.response_id);
            socket.transport.close().await.expect("close");
        },
    )
    .await;
}

/// A backend that keeps every text message the provider sent.
#[derive(Clone)]
struct Tapped<B> {
    inner: B,
    received: Arc<Mutex<Vec<String>>>,
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
                self.1.lock().expect("tap").push(text.clone());
            }
            received
        })
    }

    fn close(&mut self, frame: Option<CloseFrame>) -> WasmBoxedFuture<'_, http_client::Result<()>> {
        self.0.close(frame)
    }
}

impl<B: WebSocketClientExt> WebSocketClientExt for Tapped<B> {
    fn connect(
        &self,
        request: Request<NoBody>,
        options: ConnectOptions,
    ) -> impl Future<Output = http_client::Result<BoxedWebSocketConnection>> + WasmCompatSend {
        let (inner, received) = (self.inner.clone(), self.received.clone());
        async move {
            let connection = inner.connect(request, options).await?;
            Ok(Box::new(Tap(connection, received)) as BoxedWebSocketConnection)
        }
    }
}

/// What the transport assumes of the protocol, checked against the
/// provider's own events: every event names its type, and each turn ends at
/// a `response.completed` carrying its `sequence_number`.
#[tokio::test]
async fn events_have_the_shape_the_transport_reads() {
    with_openai_websocket_turn_cassette(
        "websocket_turns/events_have_the_shape_the_transport_reads",
        |client, backend| async move {
            let backend = Tapped {
                inner: backend,
                received: Arc::default(),
            };
            let socket = client
                .openai
                .responses(openai::GPT_4O_MINI)
                .responses_websocket()
                .connect_with(&backend)
                .await
                .expect("connects");

            let first = socket
                .call(ask("Say hello in three words."))
                .await
                .expect("first");
            let second = socket
                .call(ask("Say goodbye in three words."))
                .await
                .expect("second");
            assert_nonempty_response(&extract_text(&first.choice));
            assert_nonempty_response(&extract_text(&second.choice));
            assert_ne!(first.response_id, second.response_id);

            let events: Vec<serde_json::Value> = backend
                .received
                .lock()
                .expect("tap")
                .iter()
                .map(|text| serde_json::from_str(text).expect("every message is JSON"))
                .collect();
            assert!(
                events.iter().all(|event| event["type"].is_string()),
                "every event names its type: {events:?}"
            );
            let terminals: Vec<_> = events
                .iter()
                .filter(|event| event["type"] == "response.completed")
                .collect();
            assert_eq!(terminals.len(), 2, "one terminal per turn");
            assert!(
                terminals
                    .iter()
                    .all(|event| event["sequence_number"].is_u64()),
                "every terminal carries its sequence number: {terminals:?}"
            );
            socket.transport.close().await.expect("close");
        },
    )
    .await;
}

/// GPT-5.5 streams a turn over the websocket.
#[tokio::test]
async fn gpt_5_5_streams_a_turn() {
    with_openai_websocket_turn_cassette(
        "websocket_turns/gpt_5_5_streams_a_turn",
        |client, backend| async move {
            let socket = client
                .openai
                .responses(openai::GPT_5_5)
                .responses_websocket()
                .connect_with(&backend)
                .await
                .expect("connects");
            let mut stream = socket
                .stream(ask(
                    "Explain one benefit of websocket mode in one sentence.",
                ))
                .expect("stream opens");
            let mut streamed = String::new();
            while let Some(item) = stream.next().await {
                if let Item::Event(StreamEvent::Text { text, .. }) = item.expect("item") {
                    streamed.push_str(&text);
                }
            }
            stream.finish().await.expect("finishes");
            assert_nonempty_response(&streamed);
            socket.transport.close().await.expect("close");
        },
    )
    .await;
}
