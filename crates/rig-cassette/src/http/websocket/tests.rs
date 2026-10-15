//! A WebSocket turn recorded through a scripted live backend replays from
//! its fixture, a changed message is refused, and a recording that stores a
//! response is refused as an HTTP one is.

use super::super::{CassetteError, CassetteMode};
use super::*;
use serde_json::json;

const UPSTREAM: &str = "https://api.openai.com/v1";

/// A live backend whose provider answers every message with `replies`.
#[derive(Clone)]
struct Scripted {
    replies: Vec<String>,
}

struct ScriptedConnection {
    replies: Vec<String>,
    inbound: VecDeque<String>,
}

impl WebSocketClientExt for Scripted {
    fn connect(
        &self,
        _request: Request<NoBody>,
        _options: ConnectOptions,
    ) -> impl Future<Output = http_client::Result<BoxedWebSocketConnection>> + WasmCompatSend {
        let replies = self.replies.clone();
        async move {
            Ok(Box::new(ScriptedConnection {
                replies,
                inbound: VecDeque::new(),
            }) as BoxedWebSocketConnection)
        }
    }
}

impl WebSocketConnection for ScriptedConnection {
    fn send(&mut self, _frame: Frame) -> WasmBoxedFuture<'_, http_client::Result<()>> {
        self.inbound.extend(self.replies.iter().cloned());
        Box::pin(std::future::ready(Ok(())))
    }

    fn recv(&mut self) -> WasmBoxedFuture<'_, http_client::Result<Option<Frame>>> {
        Box::pin(std::future::ready(Ok(self
            .inbound
            .pop_front()
            .map(Frame::Text))))
    }

    fn close(
        &mut self,
        _frame: Option<CloseFrame>,
    ) -> WasmBoxedFuture<'_, http_client::Result<()>> {
        Box::pin(std::future::ready(Ok(())))
    }
}

struct Scratch(assert_fs::TempDir);

impl Scratch {
    fn new() -> Self {
        Self(assert_fs::TempDir::new().expect("scratch directory"))
    }

    async fn start(&self, mode: CassetteMode) -> WebSocketCassette {
        WebSocketCassette::over(
            ProviderCassette::start_with_attempts(
                RecordVia::Direct,
                "openai",
                CassetteSpec::new("websocket/turn"),
                UPSTREAM,
                mode,
                self.0.path().join("turn.yaml"),
                self.0.path().join("attempts"),
            )
            .await,
        )
    }
}

fn replies() -> Vec<String> {
    vec![
        json!({ "type": "response.created", "response": { "id": "resp_1" } }).to_string(),
        json!({ "type": "response.completed", "response": { "id": "resp_1", "status": "completed" } })
            .to_string(),
    ]
}

fn create(model: &str, store: bool) -> String {
    json!({ "type": "response.create", "model": model, "store": store, "input": "hi" }).to_string()
}

/// Connect through `cassette`, send `message` and read the turn to its end.
async fn turn(cassette: &WebSocketCassette, message: String) -> http_client::Result<Vec<Frame>> {
    let backend = cassette.backend(Scripted { replies: replies() });
    let request = Request::builder()
        .method("GET")
        .uri("wss://api.openai.com/v1/responses")
        .header("authorization", "Bearer sk-live")
        .body(NoBody)
        .expect("request");
    let mut connection = backend.connect(request, ConnectOptions::new()).await?;
    connection.send(Frame::Text(message)).await?;
    let mut received = Vec::new();
    while let Some(frame) = connection.recv().await? {
        received.push(frame);
    }
    Ok(received)
}

fn json_of(frames: &[Frame]) -> Vec<serde_json::Value> {
    frames
        .iter()
        .map(|frame| match frame {
            Frame::Text(text) => serde_json::from_str(text).expect("JSON message"),
            other => panic!("a text frame, got {other:?}"),
        })
        .collect()
}

#[tokio::test]
async fn a_recorded_turn_replays_and_a_changed_message_is_refused() {
    let scratch = Scratch::new();
    let recording = scratch.start(CassetteMode::Record).await;
    let recorded = turn(&recording, create("gpt-5.4", false))
        .await
        .expect("records");
    recording.finish().await;
    let fixture = std::fs::read_to_string(scratch.0.path().join("turn.yaml")).expect("fixture");
    assert!(fixture.contains("status: 101"), "{fixture}");
    assert!(!fixture.contains("sk-live"), "{fixture}");

    let replay = scratch.start(CassetteMode::Replay).await;
    let replayed = turn(&replay, create("gpt-5.4", false))
        .await
        .expect("replays");
    // The recording keeps each message's JSON, not its key order.
    assert_eq!(json_of(&replayed), json_of(&recorded));
    replay.finish().await;

    let replay = scratch.start(CassetteMode::Replay).await;
    turn(&replay, create("another-model", false))
        .await
        .expect_err("a changed message is refused");
    let refused = replay.cassette.try_finish().await;
    assert!(
        matches!(refused, Err(CassetteError::ReplayMismatch { .. })),
        "{refused:?}"
    );
}

#[tokio::test]
async fn a_recorded_turn_that_stores_a_response_is_refused() {
    let scratch = Scratch::new();
    let recording = scratch.start(CassetteMode::Record).await;
    turn(&recording, create("gpt-5.4", true))
        .await
        .expect("records");
    recording.record_turns().await;
    let refused = recording.cassette.try_finish().await;
    assert!(
        matches!(refused, Err(CassetteError::RecordingRefused { .. })),
        "{refused:?}"
    );
}

#[test]
fn messages_round_trip_through_their_sse_body() {
    let sent = ["{\"a\":1}".to_owned(), "line one\nline two".to_owned()];
    let body: String = sent.iter().map(|message| sse(message)).collect();
    assert_eq!(messages(&body), sent);
}
