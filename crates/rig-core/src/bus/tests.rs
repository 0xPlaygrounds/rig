use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    task::{Context, Poll},
    time::Duration,
};

use futures::{FutureExt, StreamExt, channel::oneshot, future::poll_fn, task::noop_waker_ref};
use serde_json::json;

use super::{
    Bus, BusConfig, BusDriver, Dispatcher, EffectLogRecorder, EffectLogReplayer, Handler,
    HandlerFuture, Key, ModelHandle, OutcomeSink, Registrar,
    adapters::{CompletionAdapter, MemoryAdapter, ToolAdapter, ToolFn},
};
use crate::effect::CustomEffect;
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
    let (dispatcher, _registrar, mut driver) = Bus::channel();
    let (echo, served) = Echo::new();
    driver.register("echo", echo).expect("register");
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
    let (dispatcher, _registrar, driver) = Bus::channel();
    drop(driver);
    let report = within(dispatcher.dispatch(&HandlerKey::from("echo"), custom(json!(1))))
        .await
        .expect_err("closed");
    assert_eq!(report.kind, ErrorKind::BusClosed);
    assert!(!report.retryable);
    assert!(dispatcher.is_closed());

    // `new_with` whose spawner drops the driver: the same answer.
    let (dispatcher, _registrar) = Bus::new_with(BusConfig::default(), |_| {}, drop);
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
    let (dispatcher, _registrar, mut driver) = Bus::channel();
    let (echo, _gate_never_opened) = Echo::gated();
    driver.register("echo", echo).expect("register");

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
    let (dispatcher, registrar, mut driver) = Bus::channel();
    let (echo, _) = Echo::new();
    driver.register("echo", echo).expect("register");
    let _task = spawn(driver);

    let report = within(dispatcher.dispatch(&HandlerKey::from("missing"), custom(json!(1))))
        .await
        .expect_err("unknown");
    assert_eq!(report.kind, ErrorKind::HandlerUnavailable);
    assert!(report.message.contains("`missing`"), "{}", report.message);
    assert!(!report.retryable);

    assert!(registrar.deregister(&HandlerKey::from("echo")));
    assert!(!registrar.deregister(&HandlerKey::from("echo")));
    let report = within(dispatcher.dispatch(&HandlerKey::from("echo"), custom(json!(1))))
        .await
        .expect_err("deregistered");
    assert_eq!(report.kind, ErrorKind::HandlerUnavailable);
    assert!(report.message.contains("`echo`"));

    // Runtime registration on the live bus brings the key back.
    let (echo, served) = Echo::new();
    registrar.register("echo", echo).expect("register");
    within(dispatcher.dispatch(&HandlerKey::from("echo"), custom(json!(1))))
        .await
        .expect("re-registered");
    assert_eq!(served.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn clones_share_the_bus_and_the_driver_outlives_the_original() {
    let (dispatcher, _registrar, mut driver) = Bus::channel();
    let (echo, served) = Echo::new();
    driver.register("echo", echo).expect("register");
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
    let (dispatcher, _registrar, mut driver) = Bus::channel();
    let (echo, _) = Echo::new();
    driver.register("echo", echo).expect("register");
    driver
        .register("add", ToolAdapter::new(Add))
        .expect("register");
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
        let (dispatcher, _registrar, mut driver) = Bus::channel_with(BusConfig {
            serial_per_handler: serial,
            ..BusConfig::default()
        });
        driver
            .register(
                "ordered",
                Ordered {
                    served: served.clone(),
                },
            )
            .expect("register");
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
    let (dispatcher, _registrar, mut driver) = Bus::channel();
    let (blocked, open) = Echo::gated();
    let (free, served) = Echo::new();
    driver.register("blocked", blocked).expect("register");
    driver.register("free", free).expect("register");
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
    let (dispatcher, _registrar, driver) = Bus::channel_with(BusConfig {
        command_capacity: 1,
        ..BusConfig::default()
    });
    let waker = noop_waker_ref();
    let mut cx = Context::from_waker(waker);
    let key = HandlerKey::from("echo");
    // Nobody drives; the first poll buffers the first command and the bound
    // is reached. Every later dispatch parks at its send stage: the call
    // returns immediately (this is a synchronous, non-async context) and
    // the pressure lives on the pending.
    let mut first = dispatcher.dispatch(&key, custom(json!(1)));
    assert!(first.poll_unpin(&mut cx).is_pending());
    assert_eq!(dispatcher.buffered(), 1);
    let mut pressured = Vec::new();
    for n in 2..20 {
        let mut pending = dispatcher.dispatch(&key, custom(json!(n)));
        assert!(pending.poll_unpin(&mut cx).is_pending());
        pressured.push(pending);
    }
    assert_eq!(
        dispatcher.buffered(),
        1,
        "the bound is bus-wide: eighteen parked dispatches buffered nothing"
    );
    // The dispatch call itself never awaited; the pendings resolve only once
    // someone drives (or closes) the bus.
    drop(driver);
    for mut pending in pressured {
        match pending.poll_unpin(&mut cx) {
            Poll::Ready(Err(report)) => assert_eq!(report.kind, ErrorKind::BusClosed),
            other => panic!("expected BusClosed after the driver dropped, got {other:?}"),
        }
    }
}

#[test]
fn a_dispatch_parked_on_the_bound_is_sent_once_the_driver_drains() {
    let (dispatcher, _registrar, mut driver) = Bus::channel_with(BusConfig {
        command_capacity: 1,
        ..BusConfig::default()
    });
    let (echo, served) = Echo::new();
    driver.register("echo", echo).expect("register");
    let waker = noop_waker_ref();
    let mut cx = Context::from_waker(waker);
    let key = HandlerKey::from("echo");
    let mut first = dispatcher.dispatch(&key, custom(json!(1)));
    let mut second = dispatcher.dispatch(&key, custom(json!(2)));
    assert!(first.poll_unpin(&mut cx).is_pending());
    assert!(second.poll_unpin(&mut cx).is_pending());
    assert_eq!(
        dispatcher.buffered(),
        1,
        "the second is parked, not buffered"
    );
    // One driver poll drains the first; the parked second is woken and its
    // next poll sends.
    let _ = driver.poll_unpin(&mut cx);
    assert_eq!(served.load(Ordering::SeqCst), 1);
    assert_eq!(dispatcher.buffered(), 0);
    assert!(second.poll_unpin(&mut cx).is_pending());
    assert_eq!(
        dispatcher.buffered(),
        1,
        "the parked dispatch sent after the drain"
    );
    let _ = driver.poll_unpin(&mut cx);
    assert_eq!(served.load(Ordering::SeqCst), 2);
    assert!(first.poll_unpin(&mut cx).is_ready());
    assert!(second.poll_unpin(&mut cx).is_ready());
}

#[test]
fn a_buffered_dispatch_answers_bus_closed_when_the_driver_drops_before_taking_it() {
    // The buffer lives on the shared half, not in the driver: dropping the
    // driver must fail what it never took, or the pending waits forever.
    let (dispatcher, _registrar, driver) = Bus::channel();
    let waker = noop_waker_ref();
    let mut cx = Context::from_waker(waker);
    let mut pending = dispatcher.dispatch(&HandlerKey::from("echo"), custom(json!(1)));
    assert!(pending.poll_unpin(&mut cx).is_pending());
    assert_eq!(dispatcher.buffered(), 1);
    drop(driver);
    match pending.poll_unpin(&mut cx) {
        Poll::Ready(Err(report)) => assert_eq!(report.kind, ErrorKind::BusClosed),
        other => panic!("expected BusClosed, got {other:?}"),
    }
    assert_eq!(dispatcher.buffered(), 0);
}

#[test]
fn a_dispatch_dropped_while_parked_on_the_bound_sends_nothing() {
    let (dispatcher, _registrar, mut driver) = Bus::channel_with(BusConfig {
        command_capacity: 1,
        ..BusConfig::default()
    });
    let (echo, served) = Echo::new();
    driver.register("echo", echo).expect("register");
    let waker = noop_waker_ref();
    let mut cx = Context::from_waker(waker);
    let key = HandlerKey::from("echo");
    let mut first = dispatcher.dispatch(&key, custom(json!(1)));
    let mut parked = dispatcher.dispatch(&key, custom(json!(2)));
    assert!(first.poll_unpin(&mut cx).is_pending());
    assert!(parked.poll_unpin(&mut cx).is_pending());
    drop(parked);
    for _ in 0..4 {
        let _ = driver.poll_unpin(&mut cx);
    }
    assert!(first.poll_unpin(&mut cx).is_ready());
    assert_eq!(
        served.load(Ordering::SeqCst),
        1,
        "the handler never saw the dispatch dropped before it was sent"
    );
}

#[test]
fn deregistering_a_serial_key_drains_its_queue_with_handler_unavailable() {
    let (dispatcher, registrar, mut driver) = Bus::channel_with(BusConfig {
        serial_per_handler: true,
        ..BusConfig::default()
    });
    let (blocked, open) = Echo::gated();
    let key = HandlerKey::from("echo");
    driver.register("echo", blocked).expect("register");
    let waker = noop_waker_ref();
    let mut cx = Context::from_waker(waker);
    // A goes in flight (gated); B and C queue behind it.
    let mut a = dispatcher.dispatch(&key, custom(json!("a")));
    let mut b = dispatcher.dispatch(&key, custom(json!("b")));
    let mut c = dispatcher.dispatch(&key, custom(json!("c")));
    for pending in [&mut a, &mut b, &mut c] {
        assert!(pending.poll_unpin(&mut cx).is_pending());
    }
    let _ = driver.poll_unpin(&mut cx);
    assert_eq!(driver.in_flight(), 1);
    // The handler goes away with a non-empty queue: nothing waits on A.
    assert!(registrar.deregister(&key));
    let _ = driver.poll_unpin(&mut cx);
    for (pending, name) in [(&mut b, "b"), (&mut c, "c")] {
        match pending.poll_unpin(&mut cx) {
            Poll::Ready(Err(report)) => {
                assert_eq!(report.kind, ErrorKind::HandlerUnavailable, "{name}");
                assert!(report.message.contains("echo"), "{name}: {report:?}");
            }
            other => panic!("{name} should have drained, got {other:?}"),
        }
    }
    // A still completes on its own, and a re-registered key serves again.
    let _ = open.send(());
    let _ = driver.poll_unpin(&mut cx);
    assert!(a.poll_unpin(&mut cx).is_ready());
    let (echo, served) = Echo::new();
    registrar.register("echo", echo).expect("register");
    let mut d = dispatcher.dispatch(&key, custom(json!("d")));
    assert!(d.poll_unpin(&mut cx).is_pending());
    let _ = driver.poll_unpin(&mut cx);
    assert!(d.poll_unpin(&mut cx).is_ready());
    assert_eq!(served.load(Ordering::SeqCst), 1);
}

/// A handler that dispatches to its own key from inside its poll — once;
/// the nested serve answers plainly.
struct SelfCaller {
    dispatcher: Dispatcher,
    key: HandlerKey,
    nested: AtomicBool,
}

impl SelfCaller {
    fn new(dispatcher: Dispatcher, key: HandlerKey) -> Self {
        Self {
            dispatcher,
            key,
            nested: AtomicBool::new(false),
        }
    }
}

impl Handler for SelfCaller {
    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: self.key.clone(),
            family: FamilyDescriptor::Custom {
                kind: "test:self-caller".into(),
            },
        }
    }

    fn handle(&self, _kind: EffectKind, sink: OutcomeSink) -> HandlerFuture<'_> {
        if self.nested.swap(true, Ordering::SeqCst) {
            return Box::pin(async move {
                sink.resolve(Ok(Outcome::Custom(json!("plain")))).await;
            });
        }
        let dispatcher = self.dispatcher.clone();
        let key = self.key.clone();
        Box::pin(async move {
            let mut nested = dispatcher.dispatch(&key, custom(json!("nested")));
            // The first poll of the nested dispatch runs inside this
            // handler's poll: the bus must answer it, not queue it.
            let first = poll_fn(|cx| Poll::Ready(nested.poll_unpin(cx))).await;
            let outcome = match first {
                Poll::Ready(Err(report)) => Ok(Outcome::Custom(json!({
                    "kind": format!("{:?}", report.kind),
                    "message": report.message,
                }))),
                other => Err(ErrorReport::new(
                    ErrorKind::Internal,
                    format!("the nested dispatch was not refused: {other:?}"),
                )),
            };
            sink.resolve(outcome).await;
        })
    }
}

#[test]
fn a_reentrant_dispatch_to_the_in_flight_key_under_serial_serving_is_refused() {
    let (dispatcher, _registrar, mut driver) = Bus::channel_with(BusConfig {
        serial_per_handler: true,
        ..BusConfig::default()
    });
    let key = HandlerKey::from("self");
    driver
        .register("self", SelfCaller::new(dispatcher.clone(), key.clone()))
        .expect("register");
    let waker = noop_waker_ref();
    let mut cx = Context::from_waker(waker);
    let mut outer = dispatcher.dispatch(&key, custom(json!("outer")));
    let mut outcome = None;
    for _ in 0..8 {
        if let Poll::Ready(result) = outer.poll_unpin(&mut cx) {
            outcome = Some(result);
            break;
        }
        let _ = driver.poll_unpin(&mut cx);
    }
    let outcome = outcome
        .expect("resolved, never queued behind itself")
        .expect("served");
    let Outcome::Custom(payload) = outcome else {
        panic!("custom outcome expected, got {outcome:?}");
    };
    assert_eq!(payload["kind"], json!("Request"));
    assert!(
        payload["message"]
            .as_str()
            .is_some_and(|message| message.contains("re-entrant") && message.contains("self")),
        "{payload}"
    );
}

#[test]
fn concurrent_serving_lets_a_handler_dispatch_to_its_own_key() {
    // Without serial serving a nested dispatch to the same key is served
    // alongside — the refusal is a serial-mode rule only.
    let (dispatcher, _registrar, mut driver) = Bus::channel();
    let key = HandlerKey::from("self");
    driver
        .register("self", SelfCaller::new(dispatcher.clone(), key.clone()))
        .expect("register");
    let waker = noop_waker_ref();
    let mut cx = Context::from_waker(waker);
    let mut outer = dispatcher.dispatch(&key, custom(json!("outer")));
    let mut outcome = None;
    for _ in 0..8 {
        if let Poll::Ready(result) = outer.poll_unpin(&mut cx) {
            outcome = Some(result);
            break;
        }
        let _ = driver.poll_unpin(&mut cx);
    }
    // The nested dispatch was accepted (its first poll returned Pending), so
    // the handler reports it as "not refused" — an Internal error here.
    let report = outcome
        .expect("resolved")
        .expect_err("the nested dispatch was accepted, not refused");
    assert_eq!(report.kind, ErrorKind::Internal);
}

#[test]
fn pending_and_stream_poll_cleanly_without_any_runtime() {
    let (dispatcher, _registrar, mut driver) = Bus::channel();
    let (echo, _) = Echo::new();
    driver.register("echo", echo).expect("register");
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
    let (dispatcher, _registrar, mut driver) = Bus::channel();
    let (echo, served) = Echo::new();
    driver.register("echo", echo).expect("register");
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
    let (dispatcher, _registrar, mut driver) = Bus::channel();
    let model = MockCompletionModel::from_stream_turns([vec![
        MockStreamEvent::text("hel"),
        MockStreamEvent::text("lo"),
        MockStreamEvent::final_response_with_total_tokens(7),
    ]]);
    driver
        .register("model", CompletionAdapter::new("mock", model))
        .expect("register");
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
    let (dispatcher, _registrar, mut driver) = Bus::channel();
    let model = MockCompletionModel::from_stream_turns([vec![
        MockStreamEvent::text("fold"),
        MockStreamEvent::text("ed"),
        MockStreamEvent::final_response_with_total_tokens(3),
    ]]);
    driver
        .register("model", CompletionAdapter::new("mock", model))
        .expect("register");
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
async fn a_unary_dispatch_of_a_unary_script_resolves_the_completion() {
    let (dispatcher, _registrar, mut driver) = Bus::channel();
    let model = MockCompletionModel::from_turns([MockTurn::text("whole")]);
    driver
        .register("model", CompletionAdapter::new("mock", model))
        .expect("register");
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
    let (dispatcher, _registrar, mut driver) = Bus::channel_with(BusConfig {
        stream_capacity: 4,
        ..BusConfig::default()
    });
    driver
        .register(
            "chatty",
            Chatty {
                sends: sends.clone(),
                cancelled: cancelled.clone(),
            },
        )
        .expect("register");
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
    let (dispatcher, _registrar, mut driver) = Bus::channel();
    driver
        .register("add", ToolAdapter::new(Add))
        .expect("register");
    driver
        .register(
            "memory",
            MemoryAdapter::new(InMemoryConversationMemory::new()),
        )
        .expect("register");
    driver
        .register(
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
        )
        .expect("register");
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
        let (dispatcher, _registrar, mut driver) = Bus::channel();
        driver
            .register("add", ToolAdapter::new(Add))
            .expect("register");
        driver
            .register(
                "model",
                CompletionAdapter::new(
                    "mock",
                    MockCompletionModel::from_stream_turns([vec![
                        MockStreamEvent::text("streamed"),
                        MockStreamEvent::final_response_with_total_tokens(2),
                    ]]),
                ),
            )
            .expect("register");
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
    let (dispatcher, _registrar, mut driver) = Bus::channel();
    EffectLogReplayer::register_all(&restored, &mut driver).expect("fresh keys");
    assert_eq!(
        dispatcher
            .descriptor(&HandlerKey::from("model"))
            .expect("replayer registered")
            .family
            .family(),
        EffectFamily::Completion
    );
    let _task = spawn(driver);
    // The same dispatch — name and arguments — gets the recorded answer; a
    // different payload is a divergence, never a guess (pinned separately).
    let replayed = within(dispatcher.dispatch(
        &HandlerKey::from("add"),
        EffectKind::ToolCall {
            name: "add".into(),
            args: r#"{"a": 1, "b": 1}"#.into(),
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
    let (dispatcher, _registrar, mut driver) = Bus::channel();
    let (echo, open) = Echo::gated();
    driver.register("echo", echo).expect("register");
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

/// One bounded-channel hop plus one oneshot per unary dispatch, measured
/// with a mock handler over 10k dispatches on a single-threaded executor:
/// the median must stay under 50 µs. A miss is a finding about the channel
/// shape, not a reason to loosen the number (the PR description records the
/// measured value).
#[test]
fn unary_dispatch_median_is_under_fifty_microseconds() {
    let (dispatcher, _registrar, mut driver) = Bus::channel();
    let (echo, _) = Echo::new();
    driver.register("echo", echo).expect("register");
    let key = HandlerKey::from("echo");
    let waker = noop_waker_ref();
    let mut cx = Context::from_waker(waker);

    // Warm up the allocator and the channel.
    for _ in 0..100 {
        let mut pending = dispatcher.dispatch(&key, custom(json!(0)));
        loop {
            if pending.poll_unpin(&mut cx).is_ready() {
                break;
            }
            let _ = driver.poll_unpin(&mut cx);
        }
    }

    let mut samples = Vec::with_capacity(10_000);
    for n in 0..10_000u64 {
        let started = std::time::Instant::now();
        let mut pending = dispatcher.dispatch(&key, custom(json!(n)));
        loop {
            if pending.poll_unpin(&mut cx).is_ready() {
                break;
            }
            let _ = driver.poll_unpin(&mut cx);
        }
        samples.push(started.elapsed());
    }
    samples.sort();
    let median = samples[samples.len() / 2];
    eprintln!(
        "unary dispatch median: {median:?} (p90 {:?})",
        samples[samples.len() * 9 / 10]
    );
    assert!(
        median < Duration::from_micros(50),
        "unary dispatch median {median:?} exceeds 50 µs"
    );
}

// ---------------------------------------------------------------------------
// T10: the log is in dispatch order, replay checks the payload, a key keeps
// its family, a cut-short stream is reported and recorded, the tap agrees
// with the consumer, and cancellation is pinned for unary dispatches too.
// ---------------------------------------------------------------------------

/// A handler whose future never answers; its drop is the observable
/// cancellation.
struct Hanging {
    dropped: Arc<AtomicBool>,
}

struct DropFlag(Arc<AtomicBool>);

impl Drop for DropFlag {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

impl Handler for Hanging {
    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: HandlerKey::from("hanging"),
            family: FamilyDescriptor::Custom {
                kind: "test:hanging".into(),
            },
        }
    }

    fn handle(&self, _kind: EffectKind, sink: OutcomeSink) -> HandlerFuture<'_> {
        let flag = DropFlag(self.dropped.clone());
        Box::pin(async move {
            let _flag = flag;
            let _sink = sink;
            futures::future::pending::<()>().await;
        })
    }
}

#[test]
fn dropping_a_pending_cancels_its_unary_handler() {
    let (dispatcher, _registrar, mut driver) = Bus::channel();
    let dropped = Arc::new(AtomicBool::new(false));
    driver
        .register(
            "hanging",
            Hanging {
                dropped: dropped.clone(),
            },
        )
        .expect("register");
    let waker = noop_waker_ref();
    let mut cx = Context::from_waker(waker);
    let mut pending = dispatcher.dispatch(&HandlerKey::from("hanging"), custom(json!(1)));
    assert!(pending.poll_unpin(&mut cx).is_pending());
    let _ = driver.poll_unpin(&mut cx);
    assert_eq!(driver.in_flight(), 1);
    assert!(!dropped.load(Ordering::SeqCst));
    drop(pending);
    let _ = driver.poll_unpin(&mut cx);
    assert!(
        dropped.load(Ordering::SeqCst),
        "dropping the Pending dropped the handler future on the driver's next poll"
    );
    assert_eq!(driver.in_flight(), 0);
}

#[test]
fn an_effect_stream_polls_cleanly_without_any_runtime() {
    let (dispatcher, _registrar, mut driver) = Bus::channel();
    let model = MockCompletionModel::from_stream_turns([vec![
        MockStreamEvent::text("manual "),
        MockStreamEvent::text("stream"),
        MockStreamEvent::final_response_with_total_tokens(2),
    ]]);
    driver
        .register("model", CompletionAdapter::new("mock", model))
        .expect("register");
    let waker = noop_waker_ref();
    let mut cx = Context::from_waker(waker);
    let mut stream = dispatcher.dispatch_stream(&HandlerKey::from("model"), completion_kind(true));
    let mut items = Vec::new();
    let mut ended = false;
    for _ in 0..64 {
        match stream.poll_next_unpin(&mut cx) {
            Poll::Ready(Some(item)) => items.push(item),
            Poll::Ready(None) => {
                ended = true;
                break;
            }
            Poll::Pending => {
                let _ = driver.poll_unpin(&mut cx);
            }
        }
    }
    assert!(ended, "the stream ended under manual polling: {items:?}");
    assert!(
        matches!(items.last(), Some(Ok(StreamEvent::Final(_)))),
        "the last item is the terminal: {items:?}"
    );
    assert!(items.iter().all(Result::is_ok), "{items:?}");
}

#[tokio::test]
async fn concurrent_dispatches_to_one_key_record_and_replay_in_dispatch_order() {
    let recorder = EffectLogRecorder::new();
    let (dispatcher, _registrar, mut driver) = Bus::channel();
    driver
        .register(
            "ordered",
            Ordered {
                served: Arc::new(Mutex::new(Vec::new())),
            },
        )
        .expect("register");
    driver.record_to(recorder.clone());
    let _task = spawn(driver);
    let key = HandlerKey::from("ordered");
    // The first dispatch is slow, the second fast: they *resolve* in the
    // order 2, 1 but were dispatched 1, 2.
    let slow = dispatcher.dispatch(&key, custom(json!({"index": 1, "delay_ms": 60})));
    let fast = dispatcher.dispatch(&key, custom(json!({"index": 2, "delay_ms": 0})));
    let (slow, fast) = within(futures::future::join(slow, fast)).await;
    assert!(matches!(slow, Ok(Outcome::Custom(ref v)) if *v == json!(1)));
    assert!(matches!(fast, Ok(Outcome::Custom(ref v)) if *v == json!(2)));
    let log = recorder.take();
    let indexes: Vec<u64> = log
        .iter()
        .map(|record| match &record.kind {
            EffectKind::Custom { payload, .. } => payload["index"].as_u64().unwrap_or(0),
            _ => 0,
        })
        .collect();
    assert_eq!(
        indexes,
        vec![1, 2],
        "the log is in dispatch order, not resolution order"
    );
    assert!(log[0].id < log[1].id);

    // Replay hands each dispatch its own recorded outcome.
    let (replay_dispatcher, _registrar, mut replay_driver) = Bus::channel();
    EffectLogReplayer::register_all(&log, &mut replay_driver).expect("fresh keys");
    let _replay = spawn(replay_driver);
    let first =
        within(replay_dispatcher.dispatch(&key, custom(json!({"index": 1, "delay_ms": 60}))))
            .await
            .expect("replayed");
    let second =
        within(replay_dispatcher.dispatch(&key, custom(json!({"index": 2, "delay_ms": 0}))))
            .await
            .expect("replayed");
    assert!(
        matches!(first, Outcome::Custom(ref v) if *v == json!(1)),
        "{first:?}"
    );
    assert!(
        matches!(second, Outcome::Custom(ref v) if *v == json!(2)),
        "{second:?}"
    );
}

#[tokio::test]
async fn replaying_a_changed_payload_is_a_divergence_not_a_guess() {
    let recorder = EffectLogRecorder::new();
    let (dispatcher, _registrar, mut driver) = Bus::channel();
    let (echo, _) = Echo::new();
    driver.register("echo", echo).expect("register");
    driver.record_to(recorder.clone());
    let _task = spawn(driver);
    let key = HandlerKey::from("echo");
    within(dispatcher.dispatch(&key, custom(json!("recorded"))))
        .await
        .expect("served");
    let log = recorder.take();

    let (replay_dispatcher, _registrar, mut replay_driver) = Bus::channel();
    EffectLogReplayer::register_all(&log, &mut replay_driver).expect("fresh keys");
    let _replay = spawn(replay_driver);
    let report = within(replay_dispatcher.dispatch(&key, custom(json!("different"))))
        .await
        .expect_err("a different payload does not replay the recorded answer");
    assert_eq!(report.kind, ErrorKind::Internal);
    assert!(report.message.contains("replay divergence"), "{report:?}");
    assert!(report.message.contains("differs"), "{report:?}");
}

#[test]
fn register_refuses_a_family_change_under_a_live_key() {
    let (dispatcher, registrar, mut driver) = Bus::channel();
    let (echo, served) = Echo::new();
    driver.register("k", echo).expect("register");
    let refused = registrar
        .register(
            "k",
            CompletionAdapter::new("mock", MockCompletionModel::text("x")),
        )
        .expect_err("a Completion handler cannot replace a Custom one");
    assert_eq!(refused.kind, ErrorKind::HandlerUnavailable);
    assert!(refused.message.contains("k"), "{refused:?}");
    // The original handler still serves, and a same-family replacement is fine.
    let waker = noop_waker_ref();
    let mut cx = Context::from_waker(waker);
    let mut pending = dispatcher.dispatch(&HandlerKey::from("k"), custom(json!(1)));
    assert!(pending.poll_unpin(&mut cx).is_pending());
    let _ = driver.poll_unpin(&mut cx);
    assert!(pending.poll_unpin(&mut cx).is_ready());
    assert_eq!(served.load(Ordering::SeqCst), 1);
    let (replacement, _) = Echo::new();
    registrar.register("k", replacement).expect("same family");
}

/// Sends one text event and drops its sink before the terminal.
struct CutShort;

impl Handler for CutShort {
    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: HandlerKey::from("cut"),
            family: FamilyDescriptor::Completion {
                model: crate::completion::ModelRef::new("cut"),
                capabilities: crate::completion::ProviderCapabilities::default(),
            },
        }
    }

    fn handle(&self, _kind: EffectKind, mut sink: OutcomeSink) -> HandlerFuture<'_> {
        Box::pin(async move {
            let _ = sink
                .send(Ok(StreamEvent::BlockDelta {
                    id: crate::streaming::BlockId::minted(crate::streaming::MintKind::Text, 0),
                    delta: crate::streaming::Delta::Text {
                        text: "partial".into(),
                    },
                }))
                .await;
            // Dropped here, before any `Final`.
        })
    }
}

#[tokio::test]
async fn a_stream_that_ends_without_final_is_reported_and_recorded() {
    let recorder = EffectLogRecorder::new();
    let (dispatcher, _registrar, mut driver) = Bus::channel();
    driver.register("cut", CutShort).expect("register");
    driver.record_to(recorder.clone());
    let _task = spawn(driver);
    let mut stream = dispatcher.dispatch_stream(&HandlerKey::from("cut"), completion_kind(true));
    let mut items = Vec::new();
    while let Some(item) = within(stream.next()).await {
        items.push(item);
    }
    let last = items.last().expect("the truncation is an item");
    let report = last.as_ref().expect_err("the truncation is an error item");
    assert_eq!(report.kind, ErrorKind::Response, "{report:?}");
    assert!(report.message.contains("before its terminal"), "{report:?}");
    // Give the driver a moment to observe the dropped sink, then the log
    // holds the same failure the consumer saw.
    within(async {
        loop {
            let log = recorder.log();
            if !log.is_empty() {
                break log;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await;
    let log = recorder.take();
    assert_eq!(log.len(), 1);
    let recorded = log[0]
        .outcome
        .as_ref()
        .expect_err("recorded as the failure it was");
    assert_eq!(recorded.kind, ErrorKind::Response);
}

/// Answers a stream dispatch with a non-completion outcome.
struct WrongFamilyAnswer;

impl Handler for WrongFamilyAnswer {
    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: HandlerKey::from("wrong"),
            family: FamilyDescriptor::Completion {
                model: crate::completion::ModelRef::new("wrong"),
                capabilities: crate::completion::ProviderCapabilities::default(),
            },
        }
    }

    fn handle(&self, _kind: EffectKind, sink: OutcomeSink) -> HandlerFuture<'_> {
        Box::pin(async move {
            sink.resolve(Ok(Outcome::Custom(json!("not a completion"))))
                .await;
        })
    }
}

#[tokio::test]
async fn the_tap_records_what_the_consumer_receives() {
    let recorder = EffectLogRecorder::new();
    let (dispatcher, _registrar, mut driver) = Bus::channel();
    driver
        .register("wrong", WrongFamilyAnswer)
        .expect("register");
    driver.record_to(recorder.clone());
    let _task = spawn(driver);
    let mut stream = dispatcher.dispatch_stream(&HandlerKey::from("wrong"), completion_kind(true));
    let mut items = Vec::new();
    while let Some(item) = within(stream.next()).await {
        items.push(item);
    }
    assert_eq!(items.len(), 1);
    let report = items[0].as_ref().expect_err("delivered as an error");
    assert_eq!(report.kind, ErrorKind::Internal);
    let log = recorder.take();
    let recorded = log[0]
        .outcome
        .as_ref()
        .expect_err("recorded as the same error");
    assert_eq!(recorded.kind, ErrorKind::Internal);
    assert_eq!(recorded.message, report.message);
}

// ---- the registrar: descriptors now, handlers on the driver's next poll ----

#[tokio::test]
async fn a_dispatch_right_after_a_runtime_registration_is_served_by_the_new_handler() {
    // No driver poll between the registration and the dispatch: the driver's
    // first poll installs the handler before it serves the command.
    let (dispatcher, registrar, driver) = Bus::channel();
    let (echo, served) = Echo::new();
    registrar.register("echo", echo).expect("fresh key");
    let pending = dispatcher.dispatch(&HandlerKey::from("echo"), custom(json!(1)));
    let _task = spawn(driver);
    within(pending).await.expect("served by the new handler");
    assert_eq!(served.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn a_typed_view_binds_synchronously_after_a_runtime_registration() {
    let (dispatcher, registrar, _driver) = Bus::channel();
    registrar
        .register(
            "model",
            CompletionAdapter::new("mock", MockCompletionModel::text("hi")),
        )
        .expect("fresh key");
    // Nobody has polled the driver: the descriptor is there, the bind works.
    let model: ModelHandle = dispatcher
        .handle(&HandlerKey::from("model"))
        .expect("bound from the descriptor table");
    assert_eq!(model.model_ref().as_str(), "mock");
    assert_eq!(
        dispatcher
            .descriptor(&HandlerKey::from("model"))
            .expect("published")
            .family
            .family(),
        EffectFamily::Completion
    );
    // A family change under the live key is refused here, synchronously.
    let (echo, _) = Echo::new();
    let report = registrar
        .register("model", echo)
        .expect_err("a custom handler cannot replace a completion");
    assert_eq!(report.kind, ErrorKind::HandlerUnavailable);
    assert_eq!(
        dispatcher
            .descriptor(&HandlerKey::from("model"))
            .expect("still published")
            .family
            .family(),
        EffectFamily::Completion
    );
}

#[tokio::test]
async fn a_dispatch_after_a_deregistration_is_unavailable_before_the_driver_served_it() {
    let (dispatcher, registrar, mut driver) = Bus::channel();
    let (echo, served) = Echo::new();
    driver.register("echo", echo).expect("register");
    assert!(registrar.deregister(&HandlerKey::from("echo")));
    assert!(dispatcher.descriptor(&HandlerKey::from("echo")).is_none());
    // The driver has not served the removal; the dispatch still fails —
    // the removal is applied before the command is served.
    let pending = dispatcher.dispatch(&HandlerKey::from("echo"), custom(json!(1)));
    let _task = spawn(driver);
    let report = within(pending).await.expect_err("deregistered");
    assert_eq!(report.kind, ErrorKind::HandlerUnavailable);
    assert_eq!(served.load(Ordering::SeqCst), 0);
}

/// A handler whose drop registers another handler on the same bus.
struct RegistersOnDrop {
    registrar: Registrar,
}

impl Handler for RegistersOnDrop {
    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: HandlerKey::from("echo"),
            family: FamilyDescriptor::Custom {
                kind: "test:echo".into(),
            },
        }
    }

    fn handle(&self, _kind: EffectKind, sink: OutcomeSink) -> HandlerFuture<'_> {
        Box::pin(async move {
            sink.resolve(Ok(Outcome::Custom(json!("never")))).await;
        })
    }
}

impl Drop for RegistersOnDrop {
    fn drop(&mut self) {
        let (echo, _) = Echo::new();
        self.registrar
            .register("from-drop", echo)
            .expect("a fresh key from inside a drop");
    }
}

#[tokio::test]
async fn a_displaced_handler_is_dropped_outside_every_lock() {
    let (dispatcher, registrar, mut driver) = Bus::channel();
    driver
        .register(
            "echo",
            RegistersOnDrop {
                registrar: registrar.clone(),
            },
        )
        .expect("register");
    let (echo, served) = Echo::new();
    registrar.register("echo", echo).expect("same family");
    let _task = spawn(driver);
    // The replacement is applied on the driver's poll; the displaced
    // handler's drop registers `from-drop` through the registrar without
    // deadlocking on the mailbox or the descriptor table.
    within(dispatcher.dispatch(&HandlerKey::from("echo"), custom(json!(1))))
        .await
        .expect("served by the replacement");
    assert_eq!(served.load(Ordering::SeqCst), 1);
    assert!(
        dispatcher
            .descriptor(&HandlerKey::from("from-drop"))
            .is_some(),
        "the displaced handler's drop ran and registered"
    );
}

/// A handler that reports its drop.
struct DropCounter(Arc<AtomicUsize>);

impl Handler for DropCounter {
    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: HandlerKey::from("flag"),
            family: FamilyDescriptor::Custom {
                kind: "test:flag".into(),
            },
        }
    }

    fn handle(&self, _kind: EffectKind, sink: OutcomeSink) -> HandlerFuture<'_> {
        Box::pin(async move {
            sink.resolve(Ok(Outcome::Custom(json!(null)))).await;
        })
    }
}

impl Drop for DropCounter {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[tokio::test]
async fn a_dropped_driver_drops_the_registrations_it_never_installed() {
    let (dispatcher, registrar, driver) = Bus::channel();
    let dropped = Arc::new(AtomicUsize::new(0));
    registrar
        .register("flag", DropCounter(dropped.clone()))
        .expect("fresh key");
    assert_eq!(dropped.load(Ordering::SeqCst), 0);
    drop(driver);
    assert_eq!(
        dropped.load(Ordering::SeqCst),
        1,
        "the handler posted and never installed went with the driver"
    );
    // A registration on the closed bus still publishes its descriptor; the
    // dispatch answers closed, not unavailable.
    registrar
        .register("late", DropCounter(dropped.clone()))
        .expect("the descriptor table outlives the driver");
    assert!(dispatcher.descriptor(&HandlerKey::from("late")).is_some());
    assert!(registrar.is_closed());
    let report = within(dispatcher.dispatch(&HandlerKey::from("late"), custom(json!(1))))
        .await
        .expect_err("closed");
    assert_eq!(report.kind, ErrorKind::BusClosed);
}

// ---- typed families, typed keys, custom effects ----

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
struct AskUser {
    prompt: String,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
struct Reply {
    text: String,
}

impl crate::effect::CustomEffect for AskUser {
    const KIND: &'static str = "test:ask_user";
    type Answer = Reply;
}

/// Answers `AskUser` with the prompt echoed, or with a payload that is not
/// a `Reply` when asked to misbehave.
struct AskUserHandler {
    misbehave: bool,
}

impl Handler for AskUserHandler {
    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: HandlerKey::from("ask"),
            family: FamilyDescriptor::Custom {
                kind: AskUser::KIND.into(),
            },
        }
    }

    fn handle(&self, kind: EffectKind, sink: OutcomeSink) -> HandlerFuture<'_> {
        let misbehave = self.misbehave;
        Box::pin(async move {
            let EffectKind::Custom { payload, .. } = kind else {
                sink.resolve(Err(ErrorReport::new(ErrorKind::Internal, "not custom")))
                    .await;
                return;
            };
            let answer = if misbehave {
                json!({"nope": 1})
            } else {
                let ask: AskUser = serde_json::from_value(payload).expect("an AskUser");
                json!({"text": format!("you asked: {}", ask.prompt)})
            };
            sink.resolve(Ok(Outcome::Custom(answer))).await;
        })
    }
}

#[tokio::test]
async fn a_typed_key_binds_with_an_existence_check_and_a_handle_dispatches_its_family() {
    let (dispatcher, _registrar, mut driver) = Bus::channel();
    let key: Key<crate::effect::family::Completion> = driver
        .register_typed(
            "model",
            CompletionAdapter::new(
                "mock",
                MockCompletionModel::from_turns([MockTurn::text("typed"), MockTurn::text("typed")]),
            ),
        )
        .expect("a completion adapter proves a completion key");
    assert_eq!(key.as_str(), "model");
    assert_eq!(format!("{key}"), "model");
    assert_eq!(
        serde_json::to_value(&key).expect("serializes"),
        json!("model"),
        "on the wire a typed key is the bare string"
    );
    let back: Key<crate::effect::family::Completion> =
        serde_json::from_value(json!("model")).expect("deserializes");
    assert_eq!(back, key);
    let _task = spawn(driver);

    let model = dispatcher.bind(&key).expect("bound by existence");
    let response = within(model.dispatch(completion_request_value()))
        .await
        .expect("the family's own answer");
    assert_eq!(response.choice, vec![AssistantContent::text("typed")]);
    let response = within(model.complete(completion_request_value()))
        .await
        .expect("the convenience is the same dispatch");
    assert_eq!(response.choice, vec![AssistantContent::text("typed")]);

    // A key asserted for the wrong family fails at bind, not silently.
    let lie: Key<crate::effect::family::Tool> = Key::new_unchecked(HandlerKey::from("model"));
    let report = dispatcher
        .bind(&lie)
        .expect_err("a completion is not a tool");
    assert_eq!(report.kind, ErrorKind::HandlerUnavailable);
}

#[tokio::test]
async fn register_typed_refuses_a_handler_of_another_family() {
    let (_dispatcher, registrar, _driver) = Bus::channel();
    let report = registrar
        .register_typed::<crate::effect::family::Tool>(
            "model",
            CompletionAdapter::new("mock", MockCompletionModel::text("x")),
        )
        .expect_err("a completion adapter cannot prove a tool key");
    assert_eq!(report.kind, ErrorKind::HandlerUnavailable);
    assert!(
        report.message.contains("Key<tool_call>") && report.message.contains("completion"),
        "{}",
        report.message
    );
    assert!(
        registrar.descriptor(&HandlerKey::from("model")).is_none(),
        "nothing was published"
    );
}

#[tokio::test]
async fn a_custom_effect_round_trips_through_a_typed_handle() {
    let (dispatcher, registrar, driver) = Bus::channel();
    let key = registrar
        .register_typed::<crate::effect::family::Custom<AskUser>>(
            "ask",
            AskUserHandler { misbehave: false },
        )
        .expect("a custom handler proves its kind");
    registrar
        .register("ask-badly", AskUserHandler { misbehave: true })
        .expect("fresh key");
    let _task = spawn(driver);

    let ask = dispatcher.bind(&key).expect("bound");
    let reply = within(ask.custom(AskUser {
        prompt: "name?".into(),
    }))
    .await
    .expect("the declared answer");
    assert_eq!(
        reply,
        Reply {
            text: "you asked: name?".into()
        }
    );

    // `Dispatcher::custom` binds an explicit key against the declared kind.
    let ask = dispatcher
        .custom::<AskUser>(&HandlerKey::from("ask-badly"))
        .expect("the kind matches");
    let report = within(ask.custom(AskUser { prompt: "?".into() }))
        .await
        .expect_err("not a Reply");
    assert_eq!(report.kind, ErrorKind::Internal);
    assert!(report.message.contains(AskUser::KIND), "{}", report.message);

    // A different kind under the key is refused at bind.
    #[derive(serde::Serialize, serde::Deserialize)]
    struct Other;
    impl crate::effect::CustomEffect for Other {
        const KIND: &'static str = "test:other";
        type Answer = ();
    }
    let report = dispatcher
        .custom::<Other>(&HandlerKey::from("ask"))
        .expect_err("another kind");
    assert_eq!(report.kind, ErrorKind::HandlerUnavailable);
    assert!(report.message.contains("test:other"), "{}", report.message);
}

fn completion_request_value() -> crate::completion::CompletionRequest {
    match completion_kind(false) {
        EffectKind::Completion { request, .. } => request,
        other => panic!("a completion kind, got {}", other.name()),
    }
}
