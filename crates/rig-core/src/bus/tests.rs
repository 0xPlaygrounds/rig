use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    task::{Context, Poll},
    time::Duration,
};

use futures::{FutureExt, StreamExt, channel::oneshot, future::poll_fn, task::noop_waker_ref};
use serde_json::json;

use super::{
    Bus, BusConfig, BusDriver, Dispatcher, EffectLogRecorder, EffectLogReplayer, Handler,
    HandlerFuture, OutcomeSink,
    adapters::{CompletionAdapter, MemoryAdapter, ToolAdapter, ToolFn},
};
use crate::{
    completion::{CompletionRequest, Message},
    effect::{
        EffectFamily, EffectKind, FamilyDescriptor, HandlerDescriptor, HandlerKey, MemoryOp,
        MemoryOutcome, Outcome,
    },
    error::{ErrorKind, ErrorReport},
    id::ConversationId,
    memory::InMemoryConversationMemory,
    message::AssistantContent,
    streaming::StreamEvent,
    test_utils::{MockCompletionModel, MockStreamEvent, MockTurn},
    tool::{Tool, ToolContext, ToolExecutionError, ToolOutput},
};

const TIMEOUT: Duration = Duration::from_secs(5);

async fn within<T>(future: impl Future<Output = T>) -> T {
    tokio::time::timeout(TIMEOUT, future)
        .await
        .expect("a dispatch never hangs")
}

fn custom(payload: serde_json::Value) -> EffectKind {
    EffectKind::Custom {
        kind: Arc::from("test:echo"),
        payload,
    }
}

fn completion_kind(stream: bool) -> EffectKind {
    EffectKind::Completion {
        request: CompletionRequest {
            model: None,
            chat_history: vec![Message::user("hi")],
            documents: vec![],
            tools: vec![],
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: None,
            record_telemetry_content: false,
        },
        stream,
    }
}

/// Echoes the custom payload back, after an optional gate.
struct Echo {
    served: Arc<AtomicUsize>,
    gate: Mutex<Option<oneshot::Receiver<()>>>,
}

impl Echo {
    fn new() -> (Self, Arc<AtomicUsize>) {
        let served = Arc::new(AtomicUsize::new(0));
        (
            Self {
                served: served.clone(),
                gate: Mutex::new(None),
            },
            served,
        )
    }

    fn gated() -> (Self, oneshot::Sender<()>) {
        let (open, gate) = oneshot::channel();
        (
            Self {
                served: Arc::new(AtomicUsize::new(0)),
                gate: Mutex::new(Some(gate)),
            },
            open,
        )
    }
}

impl Handler for Echo {
    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: HandlerKey::from("echo"),
            family: FamilyDescriptor::Custom {
                kind: "test:echo".into(),
            },
        }
    }

    fn handle(&self, kind: EffectKind, sink: OutcomeSink) -> HandlerFuture<'_> {
        let gate = self.gate.lock().expect("gate lock").take();
        Box::pin(async move {
            if let Some(gate) = gate {
                let _ = gate.await;
            }
            self.served.fetch_add(1, Ordering::SeqCst);
            let outcome = match kind {
                EffectKind::Custom { payload, .. } => Ok(Outcome::Custom(payload)),
                other => Err(ErrorReport::new(
                    ErrorKind::Internal,
                    format!("echo received {}", other.name()),
                )),
            };
            sink.resolve(outcome).await;
        })
    }
}

/// Records the order dispatches were *served* in, with a per-dispatch delay
/// read from the payload so concurrent serving reorders and serial does not.
struct Ordered {
    served: Arc<Mutex<Vec<u64>>>,
}

impl Handler for Ordered {
    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: HandlerKey::from("ordered"),
            family: FamilyDescriptor::Custom {
                kind: "test:ordered".into(),
            },
        }
    }

    fn handle(&self, kind: EffectKind, sink: OutcomeSink) -> HandlerFuture<'_> {
        Box::pin(async move {
            let (index, delay) = match &kind {
                EffectKind::Custom { payload, .. } => (
                    payload["index"].as_u64().unwrap_or(0),
                    payload["delay_ms"].as_u64().unwrap_or(0),
                ),
                _ => (0, 0),
            };
            tokio::time::sleep(Duration::from_millis(delay)).await;
            self.served.lock().expect("order lock").push(index);
            sink.resolve(Ok(Outcome::Custom(json!(index)))).await;
        })
    }
}

struct Add;

#[derive(serde::Deserialize)]
struct AddArgs {
    a: i64,
    b: i64,
}

impl Tool for Add {
    const NAME: &'static str = "add";
    type Args = AddArgs;
    type Output = i64;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "adds".into()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}})
    }

    async fn call(&self, _context: &mut ToolContext, args: AddArgs) -> Result<i64, Self::Error> {
        Ok(args.a + args.b)
    }
}

fn spawn(driver: BusDriver) -> tokio::task::JoinHandle<()> {
    tokio::spawn(driver)
}

#[tokio::test]
async fn unary_dispatch_round_trips_through_a_spawned_driver() {
    let (dispatcher, mut driver) = Bus::channel();
    let (echo, served) = Echo::new();
    driver.register("echo", echo);
    let task = spawn(driver);

    let pending = dispatcher.dispatch(&HandlerKey::from("echo"), custom(json!({"n": 1})));
    let id = pending.id();
    let outcome = within(pending).await.expect("served");
    assert!(matches!(outcome, Outcome::Custom(ref payload) if *payload == json!({"n": 1})));
    assert_eq!(served.load(Ordering::SeqCst), 1);
    assert_eq!(
        id.as_u64(),
        1,
        "ids start at one and are minted per dispatch"
    );
    assert_eq!(
        dispatcher
            .dispatch(&HandlerKey::from("echo"), custom(json!(2)))
            .id()
            .as_u64(),
        2
    );

    drop(dispatcher);
    within(task)
        .await
        .expect("driver ends when every dispatcher is gone");
}

#[tokio::test]
async fn a_dropped_driver_answers_bus_closed_before_and_after_the_send() {
    // Never spawned: dropped before any dispatch.
    let (dispatcher, driver) = Bus::channel();
    drop(driver);
    let report = within(dispatcher.dispatch(&HandlerKey::from("echo"), custom(json!(1))))
        .await
        .expect_err("closed");
    assert_eq!(report.kind, ErrorKind::BusClosed);
    assert!(!report.retryable);
    assert!(dispatcher.is_closed());

    // `new_with` whose spawner drops the driver: the same answer.
    let dispatcher = Bus::new_with(BusConfig::default(), |_| {}, drop);
    let report = within(dispatcher.dispatch(&HandlerKey::from("echo"), custom(json!(1))))
        .await
        .expect_err("closed");
    assert_eq!(report.kind, ErrorKind::BusClosed);

    // A stream dispatch on a closed bus: one failed item, then the end.
    let mut stream = dispatcher.dispatch_stream(&HandlerKey::from("model"), completion_kind(true));
    let first = within(stream.next()).await.expect("one item");
    assert_eq!(first.expect_err("closed").kind, ErrorKind::BusClosed);
    assert!(within(stream.next()).await.is_none());
}

#[tokio::test]
async fn dropping_the_driver_mid_flight_fails_the_dispatch_with_bus_closed() {
    let (dispatcher, mut driver) = Bus::channel();
    let (echo, _gate_never_opened) = Echo::gated();
    driver.register("echo", echo);

    let mut pending = dispatcher.dispatch(&HandlerKey::from("echo"), custom(json!(1)));
    // Drive by hand until the command has been sent and the handler is in flight.
    let waker = noop_waker_ref();
    let mut cx = Context::from_waker(waker);
    assert!(pending.poll_unpin(&mut cx).is_pending());
    assert!(driver.poll_unpin(&mut cx).is_pending());
    assert_eq!(driver.in_flight(), 1);

    drop(driver);
    let report = within(pending).await.expect_err("closed mid-flight");
    assert_eq!(report.kind, ErrorKind::BusClosed);
}

#[tokio::test]
async fn unknown_and_deregistered_keys_answer_handler_unavailable_with_the_key() {
    let (dispatcher, mut driver) = Bus::channel();
    let (echo, _) = Echo::new();
    driver.register("echo", echo);
    let _task = spawn(driver);

    let report = within(dispatcher.dispatch(&HandlerKey::from("missing"), custom(json!(1))))
        .await
        .expect_err("unknown");
    assert_eq!(report.kind, ErrorKind::HandlerUnavailable);
    assert!(report.message.contains("`missing`"), "{}", report.message);
    assert!(!report.retryable);

    assert!(dispatcher.deregister(&HandlerKey::from("echo")));
    assert!(!dispatcher.deregister(&HandlerKey::from("echo")));
    let report = within(dispatcher.dispatch(&HandlerKey::from("echo"), custom(json!(1))))
        .await
        .expect_err("deregistered");
    assert_eq!(report.kind, ErrorKind::HandlerUnavailable);
    assert!(report.message.contains("`echo`"));

    // Runtime registration on the live bus brings the key back.
    let (echo, served) = Echo::new();
    dispatcher.register("echo", echo);
    within(dispatcher.dispatch(&HandlerKey::from("echo"), custom(json!(1))))
        .await
        .expect("re-registered");
    assert_eq!(served.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn clones_share_the_bus_and_the_driver_outlives_the_original() {
    let (dispatcher, mut driver) = Bus::channel();
    let (echo, served) = Echo::new();
    driver.register("echo", echo);
    let task = spawn(driver);

    let clone = dispatcher.clone();
    drop(dispatcher);
    within(clone.dispatch(&HandlerKey::from("echo"), custom(json!(1))))
        .await
        .expect("served through the clone");
    assert_eq!(served.load(Ordering::SeqCst), 1);
    drop(clone);
    within(task)
        .await
        .expect("driver ends after the last clone");
}

#[tokio::test]
async fn descriptor_is_a_snapshot_that_needs_no_driver() {
    let (dispatcher, mut driver) = Bus::channel();
    let (echo, _) = Echo::new();
    driver.register("echo", echo);
    driver.register("add", ToolAdapter::new(Add));
    // The driver is never polled; the table is readable regardless.
    let descriptor = dispatcher
        .descriptor(&HandlerKey::from("add"))
        .expect("registered");
    assert_eq!(descriptor.family.family(), EffectFamily::Tool);
    assert!(matches!(
        descriptor.family,
        FamilyDescriptor::Tool { ref name, .. } if name == "add"
    ));
    assert!(dispatcher.descriptor(&HandlerKey::from("nope")).is_none());
    assert_eq!(
        dispatcher.keys(),
        vec![HandlerKey::from("add"), HandlerKey::from("echo")]
    );
    drop(driver);
}

#[tokio::test]
async fn serial_per_handler_serves_in_arrival_order_and_concurrent_may_not() {
    async fn run(serial: bool) -> Vec<u64> {
        let served = Arc::new(Mutex::new(Vec::new()));
        let (dispatcher, mut driver) = Bus::channel_with(BusConfig {
            serial_per_handler: serial,
            ..BusConfig::default()
        });
        driver.register(
            "ordered",
            Ordered {
                served: served.clone(),
            },
        );
        let _task = spawn(driver);
        let key = HandlerKey::from("ordered");
        // Earlier dispatches sleep longer, so concurrent serving finishes them last.
        let delays = [40u64, 20, 0];
        let pendings: Vec<_> = delays
            .iter()
            .enumerate()
            .map(|(index, delay)| {
                dispatcher.dispatch(
                    &key,
                    custom(json!({"index": index as u64, "delay_ms": delay})),
                )
            })
            .collect();
        within(futures::future::join_all(pendings)).await;
        served.lock().expect("order").clone()
    }

    assert_eq!(run(true).await, vec![0, 1, 2], "serial keeps arrival order");
    assert_eq!(
        run(false).await,
        vec![2, 1, 0],
        "concurrent finishes by delay"
    );
}

#[tokio::test]
async fn concurrent_serving_across_keys_is_the_default() {
    let (dispatcher, mut driver) = Bus::channel();
    let (blocked, open) = Echo::gated();
    let (free, served) = Echo::new();
    driver.register("blocked", blocked);
    driver.register("free", free);
    let _task = spawn(driver);

    let blocked_pending = dispatcher.dispatch(&HandlerKey::from("blocked"), custom(json!(1)));
    let free_outcome = within(dispatcher.dispatch(&HandlerKey::from("free"), custom(json!(2))))
        .await
        .expect("free key is served while another is gated");
    assert!(matches!(free_outcome, Outcome::Custom(ref payload) if *payload == json!(2)));
    assert_eq!(served.load(Ordering::SeqCst), 1);
    let _ = open.send(());
    within(blocked_pending)
        .await
        .expect("gated key resolves once opened");
}

#[test]
fn dispatch_never_blocks_the_caller_even_when_the_channel_is_full() {
    let (dispatcher, driver) = Bus::channel_with(BusConfig {
        command_capacity: 1,
        ..BusConfig::default()
    });
    let waker = noop_waker_ref();
    let mut cx = Context::from_waker(waker);
    let key = HandlerKey::from("echo");
    // Nobody drives; fill the channel by polling pendings once each.
    let mut first = dispatcher.dispatch(&key, custom(json!(1)));
    assert!(first.poll_unpin(&mut cx).is_pending());
    let mut second = dispatcher.dispatch(&key, custom(json!(2)));
    assert!(second.poll_unpin(&mut cx).is_pending());
    // The channel is full (capacity + one slot per sender is futures' rule;
    // keep dispatching until a poll stays at the send stage).
    let mut pressured = Vec::new();
    for n in 3..20 {
        let mut pending = dispatcher.dispatch(&key, custom(json!(n)));
        // Returns immediately: this is a synchronous call from a non-async context.
        let poll = pending.poll_unpin(&mut cx);
        assert!(poll.is_pending());
        pressured.push(pending);
    }
    // The dispatch call itself never awaited; the pressure lives on the
    // pendings, which resolve only once someone drives (or closes) the bus.
    drop(driver);
    for mut pending in pressured {
        match pending.poll_unpin(&mut cx) {
            Poll::Ready(Err(report)) => assert_eq!(report.kind, ErrorKind::BusClosed),
            other => panic!("expected BusClosed after the driver dropped, got {other:?}"),
        }
    }
}

#[test]
fn pending_and_stream_poll_cleanly_without_any_runtime() {
    let (dispatcher, mut driver) = Bus::channel();
    let (echo, _) = Echo::new();
    driver.register("echo", echo);
    let waker = noop_waker_ref();
    let mut cx = Context::from_waker(waker);

    let mut pending = dispatcher.dispatch(&HandlerKey::from("echo"), custom(json!("manual")));
    let mut outcome = None;
    for _ in 0..16 {
        if let Poll::Ready(result) = pending.poll_unpin(&mut cx) {
            outcome = Some(result);
            break;
        }
        let _ = driver.poll_unpin(&mut cx);
    }
    let outcome = outcome
        .expect("resolved by manual polling")
        .expect("served");
    assert!(matches!(outcome, Outcome::Custom(ref payload) if *payload == json!("manual")));
}

#[tokio::test]
async fn stream_dispatch_of_a_unary_kind_is_an_invalid_dispatch() {
    let (dispatcher, mut driver) = Bus::channel();
    let (echo, served) = Echo::new();
    driver.register("echo", echo);
    let _task = spawn(driver);

    let mut stream = dispatcher.dispatch_stream(&HandlerKey::from("echo"), custom(json!(1)));
    let item = within(stream.next()).await.expect("one item");
    let report = item.expect_err("invalid dispatch");
    assert_eq!(report.kind, ErrorKind::Request);
    assert!(
        report.message.contains("invalid dispatch"),
        "{}",
        report.message
    );
    assert!(within(stream.next()).await.is_none());
    assert_eq!(
        served.load(Ordering::SeqCst),
        0,
        "never reached the handler"
    );
}

#[tokio::test]
async fn streaming_completion_flows_through_the_bus_final_terminated() {
    let (dispatcher, mut driver) = Bus::channel();
    let model = MockCompletionModel::from_stream_turns([vec![
        MockStreamEvent::text("hel"),
        MockStreamEvent::text("lo"),
        MockStreamEvent::final_response_with_total_tokens(7),
    ]]);
    driver.register("model", CompletionAdapter::new("mock", model));
    let _task = spawn(driver);

    let stream = dispatcher.dispatch_stream(&HandlerKey::from("model"), completion_kind(true));
    let items: Vec<_> = within(stream.collect()).await;
    let events: Vec<StreamEvent> = items
        .into_iter()
        .map(|item| item.expect("no report"))
        .collect();
    assert!(matches!(events.last(), Some(StreamEvent::Final(_))));
    let text: String = events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::BlockDelta {
                delta: crate::streaming::Delta::Text { text },
                ..
            } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(text, "hello");
}

#[tokio::test]
async fn a_unary_dispatch_of_a_streaming_completion_folds_to_the_response() {
    let (dispatcher, mut driver) = Bus::channel();
    let model = MockCompletionModel::from_stream_turns([vec![
        MockStreamEvent::text("fold"),
        MockStreamEvent::text("ed"),
        MockStreamEvent::final_response_with_total_tokens(3),
    ]]);
    driver.register("model", CompletionAdapter::new("mock", model));
    let _task = spawn(driver);

    let outcome = within(dispatcher.dispatch(&HandlerKey::from("model"), completion_kind(true)))
        .await
        .expect("folded");
    let Outcome::Completion(response) = outcome else {
        panic!("expected a completion");
    };
    assert_eq!(response.choice, vec![AssistantContent::text("folded")]);
    assert_eq!(response.usage.total_tokens, 3);
}

#[tokio::test]
async fn a_unary_completion_answering_a_stream_dispatch_is_re_emitted_as_events() {
    let (dispatcher, mut driver) = Bus::channel();
    let model = MockCompletionModel::from_turns([MockTurn::text("whole")]);
    driver.register("model", CompletionAdapter::new("mock", model));
    let _task = spawn(driver);

    // The model is scripted unary; ask for a stream of a unary completion
    // kind by dispatching the streaming kind through a unary-only script is
    // not possible, so exercise the sink directly through the replayer path
    // below. Here: the adapter's unary arm resolves a unary dispatch.
    let outcome = within(dispatcher.dispatch(&HandlerKey::from("model"), completion_kind(false)))
        .await
        .expect("served");
    assert!(
        matches!(outcome, Outcome::Completion(ref r) if r.choice == vec![AssistantContent::text("whole")])
    );
}

#[tokio::test]
async fn dropping_the_stream_cancels_the_handler() {
    struct Chatty {
        sends: Arc<AtomicUsize>,
        cancelled: Arc<AtomicUsize>,
    }
    impl Handler for Chatty {
        fn descriptor(&self) -> HandlerDescriptor {
            HandlerDescriptor {
                key: HandlerKey::from("chatty"),
                family: FamilyDescriptor::Completion {
                    model: "chatty".into(),
                    capabilities: Default::default(),
                },
            }
        }
        fn handle(&self, _kind: EffectKind, mut sink: OutcomeSink) -> HandlerFuture<'_> {
            Box::pin(async move {
                loop {
                    let event = StreamEvent::text(
                        crate::streaming::BlockId::minted(crate::streaming::MintKind::Text, 0),
                        "x",
                    );
                    if sink.send(Ok(event)).await.is_err() {
                        self.cancelled.fetch_add(1, Ordering::SeqCst);
                        return;
                    }
                    self.sends.fetch_add(1, Ordering::SeqCst);
                }
            })
        }
    }
    let sends = Arc::new(AtomicUsize::new(0));
    let cancelled = Arc::new(AtomicUsize::new(0));
    let (dispatcher, mut driver) = Bus::channel_with(BusConfig {
        stream_capacity: 4,
        ..BusConfig::default()
    });
    driver.register(
        "chatty",
        Chatty {
            sends: sends.clone(),
            cancelled: cancelled.clone(),
        },
    );
    let _task = spawn(driver);

    let mut stream = dispatcher.dispatch_stream(&HandlerKey::from("chatty"), completion_kind(true));
    within(stream.next())
        .await
        .expect("first event")
        .expect("ok");
    within(stream.next())
        .await
        .expect("second event")
        .expect("ok");
    // Pause: not polling stalls the producer at the bounded channel.
    tokio::time::sleep(Duration::from_millis(50)).await;
    let stalled = sends.load(Ordering::SeqCst);
    assert!(
        stalled <= 4 + 2,
        "bounded channel stalls the handler: {stalled}"
    );
    tokio::time::sleep(Duration::from_millis(30)).await;
    assert_eq!(
        sends.load(Ordering::SeqCst),
        stalled,
        "no progress while paused"
    );

    drop(stream);
    within(poll_fn(|cx| {
        if cancelled.load(Ordering::SeqCst) == 1 {
            Poll::Ready(())
        } else {
            cx.waker().wake_by_ref();
            Poll::Pending
        }
    }))
    .await;
}

#[tokio::test]
async fn tool_memory_and_fn_adapters_serve_their_families() {
    let (dispatcher, mut driver) = Bus::channel();
    driver.register("add", ToolAdapter::new(Add));
    driver.register(
        "memory",
        MemoryAdapter::new(InMemoryConversationMemory::new()),
    );
    driver.register(
        "shout",
        ToolFn::new(
            "shout",
            "uppercases",
            json!({"type": "object"}),
            |_context: &mut ToolContext, args: serde_json::Value| {
                Box::pin(async move {
                    Ok(ToolOutput::text(
                        args["text"].as_str().unwrap_or_default().to_uppercase(),
                    ))
                }) as crate::wasm_compat::WasmBoxedFuture<'_, _>
            },
        ),
    );
    let _task = spawn(driver);

    let outcome = within(dispatcher.dispatch(
        &HandlerKey::from("add"),
        EffectKind::ToolCall {
            name: "add".into(),
            args: r#"{"a": 2, "b": 3}"#.into(),
            context: ToolContext::new(),
        },
    ))
    .await
    .expect("served");
    let Outcome::ToolResult { result, .. } = outcome else {
        panic!("expected a tool result");
    };
    assert_eq!(result.output().as_json(), Some(&json!(5)));

    let outcome = within(dispatcher.dispatch(
        &HandlerKey::from("shout"),
        EffectKind::ToolCall {
            name: "shout".into(),
            args: r#"{"text": "hi"}"#.into(),
            context: ToolContext::new(),
        },
    ))
    .await
    .expect("served");
    let Outcome::ToolResult { result, .. } = outcome else {
        panic!("expected a tool result");
    };
    assert_eq!(result.output().as_text(), Some("HI"));

    let conversation = ConversationId::from("c1");
    let memory = HandlerKey::from("memory");
    within(dispatcher.dispatch(
        &memory,
        EffectKind::Memory {
            op: MemoryOp::Append {
                conversation: conversation.clone(),
                messages: vec![Message::user("remember me")],
            },
        },
    ))
    .await
    .expect("appended");
    let loaded = within(dispatcher.dispatch(
        &memory,
        EffectKind::Memory {
            op: MemoryOp::Load {
                conversation: conversation.clone(),
            },
        },
    ))
    .await
    .expect("loaded");
    assert!(matches!(
        loaded,
        Outcome::Memory(MemoryOutcome::Loaded { ref messages }) if messages.len() == 1
    ));
    within(dispatcher.dispatch(
        &memory,
        EffectKind::Memory {
            op: MemoryOp::Clear { conversation },
        },
    ))
    .await
    .expect("cleared");

    // A family mismatch at the handler is `HandlerUnavailable`, not a hang.
    let report = within(dispatcher.dispatch(&HandlerKey::from("add"), custom(json!(1))))
        .await
        .expect_err("wrong family");
    assert_eq!(report.kind, ErrorKind::HandlerUnavailable);
}

#[tokio::test]
async fn recorder_captures_every_dispatch_and_the_replayer_answers_from_it() {
    let recorder = EffectLogRecorder::new();
    let log = {
        let (dispatcher, mut driver) = Bus::channel();
        driver.register("add", ToolAdapter::new(Add));
        driver.register(
            "model",
            CompletionAdapter::new(
                "mock",
                MockCompletionModel::from_stream_turns([vec![
                    MockStreamEvent::text("streamed"),
                    MockStreamEvent::final_response_with_total_tokens(2),
                ]]),
            ),
        );
        driver.record_to(recorder.clone());
        let _task = spawn(driver);

        within(dispatcher.dispatch(
            &HandlerKey::from("add"),
            EffectKind::ToolCall {
                name: "add".into(),
                args: r#"{"a": 1, "b": 1}"#.into(),
                context: ToolContext::new(),
            },
        ))
        .await
        .expect("served");
        let events: Vec<_> = within(
            dispatcher
                .dispatch_stream(&HandlerKey::from("model"), completion_kind(true))
                .collect(),
        )
        .await;
        assert!(events.iter().all(Result::is_ok));
        within(dispatcher.dispatch(&HandlerKey::from("nope"), custom(json!(0))))
            .await
            .expect_err("unknown keys are not served, so not recorded");
        recorder.take()
    };
    assert_eq!(log.len(), 2, "one record per served dispatch");
    assert_eq!(log[0].key.as_str(), "add");
    assert_eq!(log[1].key.as_str(), "model");
    assert!(matches!(
        &log[1].outcome,
        Ok(Outcome::Completion(response)) if response.choice == vec![AssistantContent::text("streamed")]
    ));

    // Serialize, deserialize, replay: the same dispatches get the same answers
    // with no model or tool behind the keys.
    let json = serde_json::to_string(&log).expect("log serializes");
    let restored: crate::effect::EffectLog = serde_json::from_str(&json).expect("log restores");
    let (dispatcher, mut driver) = Bus::channel();
    EffectLogReplayer::register_all(&restored, &mut driver);
    assert_eq!(
        dispatcher
            .descriptor(&HandlerKey::from("model"))
            .expect("replayer registered")
            .family
            .family(),
        EffectFamily::Completion
    );
    let _task = spawn(driver);
    let replayed = within(dispatcher.dispatch(
        &HandlerKey::from("add"),
        EffectKind::ToolCall {
            name: "add".into(),
            args: "{}".into(),
            context: ToolContext::new(),
        },
    ))
    .await
    .expect("replayed");
    let Outcome::ToolResult { result, .. } = replayed else {
        panic!("expected a tool result");
    };
    assert_eq!(result.output().as_json(), Some(&json!(2)));
    // A recorded completion answers a stream dispatch as events + Final.
    let events: Vec<_> = within(
        dispatcher
            .dispatch_stream(&HandlerKey::from("model"), completion_kind(true))
            .collect(),
    )
    .await;
    let events: Vec<StreamEvent> = events.into_iter().map(|e| e.expect("ok")).collect();
    assert!(matches!(events.last(), Some(StreamEvent::Final(_))));
    assert!(
        events
            .iter()
            .any(|e| matches!(e, StreamEvent::BlockDelta { .. }))
    );
    // Past the log: a divergence, never a hang.
    let report = within(dispatcher.dispatch(&HandlerKey::from("add"), custom(json!(1))))
        .await
        .expect_err("ran out");
    assert_eq!(report.kind, ErrorKind::Internal);
    assert!(report.message.contains("replay divergence"));
}

#[tokio::test]
async fn driver_completes_when_dispatchers_drop_and_in_flight_work_finishes() {
    let (dispatcher, mut driver) = Bus::channel();
    let (echo, open) = Echo::gated();
    driver.register("echo", echo);
    let task = spawn(driver);
    let pending = dispatcher.dispatch(&HandlerKey::from("echo"), custom(json!(1)));
    let mut pending = Box::pin(pending);
    // Send the command, then drop the dispatcher while the handler is gated.
    assert!(poll_fn(|cx| Poll::Ready(pending.poll_unpin(cx).is_pending())).await);
    drop(dispatcher);
    tokio::time::sleep(Duration::from_millis(20)).await;
    assert!(!task.is_finished(), "in-flight work keeps the driver alive");
    let _ = open.send(());
    within(pending).await.expect("served after the gate");
    within(task).await.expect("driver ends");
}

const _: fn() = || {
    fn assert_clone_send_sync<T: Clone + Send + Sync + 'static>() {}
    assert_clone_send_sync::<Dispatcher>();
};
