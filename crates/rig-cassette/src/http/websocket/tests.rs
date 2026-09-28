use super::*;
use futures::FutureExt;
use serde_json::json;
use std::panic::AssertUnwindSafe;

/// A live backend stand-in: each sent message releases the next turn's
/// events, and the connection ends when they run out.
#[derive(Clone, Default)]
struct Scripted(Arc<StdMutex<(VecDeque<Vec<String>>, VecDeque<String>)>>);

impl Scripted {
    fn turns(turns: Vec<Vec<String>>) -> Self {
        Self(Arc::new(StdMutex::new((turns.into(), VecDeque::new()))))
    }
}

struct ScriptedConnection(Scripted);

impl WebSocketClientExt for Scripted {
    fn connect(
        &self,
        _request: Request<NoBody>,
        _options: ConnectOptions,
    ) -> impl Future<Output = http_client::Result<BoxedWebSocketConnection>> + WasmCompatSend {
        let script = self.clone();
        async move { Ok(Box::new(ScriptedConnection(script)) as BoxedWebSocketConnection) }
    }
}

impl WebSocketConnection for ScriptedConnection {
    fn send(&mut self, _frame: Frame) -> WasmBoxedFuture<'_, http_client::Result<()>> {
        let mut state = (self.0).0.lock().expect("script");
        if let Some(turn) = state.0.pop_front() {
            state.1.extend(turn);
        }
        Box::pin(std::future::ready(Ok(())))
    }

    fn recv(&mut self) -> WasmBoxedFuture<'_, http_client::Result<Option<Frame>>> {
        let next = (self.0).0.lock().expect("script").1.pop_front();
        Box::pin(std::future::ready(Ok(next.map(Frame::Text))))
    }

    fn close(
        &mut self,
        _frame: Option<CloseFrame>,
    ) -> WasmBoxedFuture<'_, http_client::Result<()>> {
        Box::pin(std::future::ready(Ok(())))
    }
}

fn handshake() -> Request<NoBody> {
    Request::builder()
        .uri("wss://api.openai.com/v1/responses")
        .header("authorization", "Bearer sk-live-secret")
        .body(NoBody)
        .expect("request")
}

fn create(text: &str) -> String {
    json!({ "type": "response.create", "model": "gpt-4o-mini", "input": text }).to_string()
}

fn event(delta: &str) -> String {
    json!({ "type": "response.output_text.delta", "delta": delta }).to_string()
}

async fn session(mode: CassetteMode, root: &Path) -> (WebSocketCassette, PathBuf) {
    let path = root.join("openai/websocket/turns.yaml");
    let cassette = WebSocketCassette::start_at(
        "openai",
        CassetteSpec::new("websocket/turns"),
        "https://api.openai.com/v1",
        mode,
        path.clone(),
        root.join("attempts"),
    )
    .await;
    (cassette, path)
}

/// The next message, as JSON: a recording keeps each event as canonical,
/// key-sorted JSON.
async fn text(connection: &mut BoxedWebSocketConnection) -> Option<serde_json::Value> {
    match connection.recv().await.expect("recv") {
        Some(Frame::Text(text)) => Some(serde_json::from_str(&text).expect("JSON event")),
        _ => None,
    }
}

fn parsed(text: String) -> Option<serde_json::Value> {
    serde_json::from_str(&text).ok()
}

/// Record two turns through a live backend, then play them back without it.
async fn record_two_turns(root: &Path) -> PathBuf {
    let (recording, path) = session(CassetteMode::Record, root).await;
    assert!(recording.records());
    let live = Scripted::turns(vec![vec![event("a"), event("b")], vec![event("c")]]);
    let mut connection = recording
        .backend(live)
        .connect(handshake(), ConnectOptions::new())
        .await
        .expect("connects");
    connection
        .send(Frame::Text(create("one")))
        .await
        .expect("send");
    assert_eq!(text(&mut connection).await, parsed(event("a")));
    assert_eq!(text(&mut connection).await, parsed(event("b")));
    connection
        .send(Frame::Text(create("two")))
        .await
        .expect("send");
    assert_eq!(text(&mut connection).await, parsed(event("c")));
    recording.finish().await;
    path
}

#[tokio::test]
async fn a_recording_replays_its_turns_in_order_without_the_live_backend() {
    let root = assert_fs::TempDir::new().expect("temp dir");
    let path = record_two_turns(root.path()).await;

    let fixture = std::fs::read_to_string(&path).expect("fixture written");
    assert_eq!(fixture.matches("status: 101").count(), 2, "{fixture}");
    assert!(fixture.contains("path: /v1/responses"), "{fixture}");
    assert!(!fixture.contains("sk-live-secret"), "{fixture}");

    let (replay, _) = session(CassetteMode::Replay, root.path()).await;
    assert!(!replay.records());
    // Replay never touches the backend it is handed.
    let mut connection = replay
        .backend(Scripted::default())
        .connect(handshake(), ConnectOptions::new())
        .await
        .expect("connects");
    connection
        .send(Frame::Text(create("one")))
        .await
        .expect("the first turn matches");
    assert_eq!(text(&mut connection).await, parsed(event("a")));
    assert_eq!(text(&mut connection).await, parsed(event("b")));
    // Past the recording, the peer is gone.
    assert_eq!(text(&mut connection).await, None);
    connection
        .send(Frame::Text(create("two")))
        .await
        .expect("the second turn matches");
    assert_eq!(text(&mut connection).await, parsed(event("c")));
    replay.finish().await;
}

#[tokio::test]
async fn a_message_the_recording_does_not_hold_fails_the_replay() {
    let root = assert_fs::TempDir::new().expect("temp dir");
    record_two_turns(root.path()).await;

    let (replay, _) = session(CassetteMode::Replay, root.path()).await;
    let mut connection = replay
        .backend(Scripted::default())
        .connect(handshake(), ConnectOptions::new())
        .await
        .expect("connects");
    connection
        .send(Frame::Text(create("something else")))
        .await
        .expect_err("the send is refused");
    let failure = AssertUnwindSafe(replay.finish())
        .catch_unwind()
        .await
        .expect_err("the replay fails");
    let message = failure
        .downcast_ref::<String>()
        .cloned()
        .unwrap_or_default();
    assert!(message.contains("unexpected message"), "{message}");
}

#[tokio::test]
async fn an_unplayed_turn_fails_the_replay() {
    let root = assert_fs::TempDir::new().expect("temp dir");
    record_two_turns(root.path()).await;

    let (replay, _) = session(CassetteMode::Replay, root.path()).await;
    let mut connection = replay
        .backend(Scripted::default())
        .connect(handshake(), ConnectOptions::new())
        .await
        .expect("connects");
    connection
        .send(Frame::Text(create("one")))
        .await
        .expect("send");
    let failure = AssertUnwindSafe(replay.finish())
        .catch_unwind()
        .await
        .expect_err("the replay fails");
    let message = failure
        .downcast_ref::<String>()
        .cloned()
        .unwrap_or_default();
    assert!(message.contains("unplayed"), "{message}");
}

#[tokio::test]
async fn a_connection_to_another_path_fails_the_replay() {
    let root = assert_fs::TempDir::new().expect("temp dir");
    record_two_turns(root.path()).await;

    let (replay, _) = session(CassetteMode::Replay, root.path()).await;
    let elsewhere = Request::builder()
        .uri("wss://api.openai.com/v1/realtime")
        .body(NoBody)
        .expect("request");
    let mut connection = replay
        .backend(Scripted::default())
        .connect(elsewhere, ConnectOptions::new())
        .await
        .expect("connects");
    for (turn, events) in [("one", 2), ("two", 1)] {
        connection
            .send(Frame::Text(create(turn)))
            .await
            .expect("send");
        for _ in 0..events {
            text(&mut connection).await;
        }
    }
    let failure = AssertUnwindSafe(replay.finish())
        .catch_unwind()
        .await
        .expect_err("the replay fails");
    let message = failure
        .downcast_ref::<String>()
        .cloned()
        .unwrap_or_default();
    assert!(message.contains("/v1/realtime"), "{message}");
}

#[tokio::test]
async fn a_turn_sent_before_the_previous_one_was_read_fails_the_replay() {
    let root = assert_fs::TempDir::new().expect("temp dir");
    record_two_turns(root.path()).await;

    let (replay, _) = session(CassetteMode::Replay, root.path()).await;
    let mut connection = replay
        .backend(Scripted::default())
        .connect(handshake(), ConnectOptions::new())
        .await
        .expect("connects");
    connection
        .send(Frame::Text(create("one")))
        .await
        .expect("send");
    connection
        .send(Frame::Text(create("two")))
        .await
        .expect_err("the first turn was not read");
    let failure = AssertUnwindSafe(replay.finish())
        .catch_unwind()
        .await
        .expect_err("the replay fails");
    let message = failure
        .downcast_ref::<String>()
        .cloned()
        .unwrap_or_default();
    assert!(message.contains("unread"), "{message}");
}

#[tokio::test]
async fn a_replay_dropped_without_finish_panics() {
    let root = assert_fs::TempDir::new().expect("temp dir");
    record_two_turns(root.path()).await;

    let (replay, _) = session(CassetteMode::Replay, root.path()).await;
    let dropped = std::panic::catch_unwind(AssertUnwindSafe(|| drop(replay)));
    assert!(
        dropped.is_err(),
        "the drop guard reports an unchecked replay"
    );
}

/// A turn that stores its response (no `store: false`) is created state the
/// recorder cannot see deleted: the recording is refused and kept aside.
#[tokio::test]
async fn a_recording_that_stores_a_response_is_refused() {
    let root = assert_fs::TempDir::new().expect("temp dir");
    let (recording, path) = session(CassetteMode::Record, root.path()).await;
    let created = json!({ "type": "response.created", "response": { "id": "resp_kept" } });
    let live = Scripted::turns(vec![vec![created.to_string()]]);
    let mut connection = recording
        .backend(live)
        .connect(handshake(), ConnectOptions::new())
        .await
        .expect("connects");
    connection
        .send(Frame::Text(create("stored")))
        .await
        .expect("send");
    text(&mut connection).await;
    let failure = AssertUnwindSafe(recording.finish())
        .catch_unwind()
        .await
        .expect_err("the recording is refused");
    let message = failure
        .downcast_ref::<String>()
        .cloned()
        .unwrap_or_default();
    assert!(message.contains("resp_kept"), "{message}");
    assert!(!path.exists(), "the fixture was not written");

    // The same turn with `store: false` creates nothing and is written.
    let root = assert_fs::TempDir::new().expect("temp dir");
    let (recording, path) = session(CassetteMode::Record, root.path()).await;
    let live = Scripted::turns(vec![vec![created.to_string()]]);
    let mut connection = recording
        .backend(live)
        .connect(handshake(), ConnectOptions::new())
        .await
        .expect("connects");
    let unstored = json!({ "type": "response.create", "store": false, "input": "x" });
    connection
        .send(Frame::Text(unstored.to_string()))
        .await
        .expect("send");
    text(&mut connection).await;
    recording.finish().await;
    assert!(path.exists());
}
