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

use super::{Bus, BusConfig, BusDriver, Dispatcher, ModelHandle, Registrar};
use rig_core::effect::{CustomEffect, Key};
use rig_core::serve::{
    OutcomeSink, Serve,
    adapters::{CompletionAdapter, MemoryAdapter, RerankAdapter, ToolAdapter, ToolFn},
};
use rig_core::{
    completion::{CompletionRequest, Message},
    effect::{
        EffectFamily, EffectKind, FamilyDescriptor, HandlerDescriptor, HandlerKey, MemoryOp,
        MemoryOutcome, Outcome,
    },
    error::{ErrorKind, ErrorReport},
    id::ConversationId,
    memory::InMemoryConversationMemory,
    message::AssistantContent,
    rerank::{RerankError, RerankModel, RerankResponse, RerankResult},
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

impl Serve for Echo {
    type Family = rig_core::effect::family::Dynamic;

    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: HandlerKey::from("echo"),
            family: FamilyDescriptor::Custom {
                kind: "test:echo".into(),
            },
        }
    }

    async fn serve(&self, kind: EffectKind, sink: OutcomeSink) {
        let gate = self.gate.lock().expect("gate lock").take();
        {
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
        }
    }
}

/// Records the order dispatches were *served* in, with a per-dispatch delay
/// read from the payload so concurrent serving reorders and serial does not.
struct Ordered {
    served: Arc<Mutex<Vec<u64>>>,
}

impl Serve for Ordered {
    type Family = rig_core::effect::family::Dynamic;

    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: HandlerKey::from("ordered"),
            family: FamilyDescriptor::Custom {
                kind: "test:ordered".into(),
            },
        }
    }

    async fn serve(&self, kind: EffectKind, sink: OutcomeSink) {
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

impl Serve for SelfCaller {
    type Family = rig_core::effect::family::Dynamic;

    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: self.key.clone(),
            family: FamilyDescriptor::Custom {
                kind: "test:self-caller".into(),
            },
        }
    }

    async fn serve(&self, _kind: EffectKind, sink: OutcomeSink) {
        if self.nested.swap(true, Ordering::SeqCst) {
            sink.resolve(Ok(Outcome::Custom(json!("plain")))).await;
            return;
        }
        let dispatcher = self.dispatcher.clone();
        let key = self.key.clone();
        {
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
        }
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
                delta: rig_core::streaming::Delta::Text { text },
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
    impl Serve for Chatty {
        type Family = rig_core::effect::family::Dynamic;

        fn descriptor(&self) -> HandlerDescriptor {
            HandlerDescriptor {
                key: HandlerKey::from("chatty"),
                family: FamilyDescriptor::Completion {
                    model: "chatty".into(),
                    capabilities: Default::default(),
                },
            }
        }
        async fn serve(&self, _kind: EffectKind, sink: OutcomeSink) {
            let mut out = sink.writer();
            loop {
                if out.text("x").await.is_err() {
                    self.cancelled.fetch_add(1, Ordering::SeqCst);
                    return;
                }
                self.sends.fetch_add(1, Ordering::SeqCst);
            }
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
                    }) as rig_core::wasm_compat::WasmBoxedFuture<'_, _>
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

impl Serve for Hanging {
    type Family = rig_core::effect::family::Dynamic;

    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: HandlerKey::from("hanging"),
            family: FamilyDescriptor::Custom {
                kind: "test:hanging".into(),
            },
        }
    }

    async fn serve(&self, _kind: EffectKind, sink: OutcomeSink) {
        let _flag = DropFlag(self.dropped.clone());
        let _sink = sink;
        futures::future::pending::<()>().await;
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

impl Serve for RegistersOnDrop {
    type Family = rig_core::effect::family::Dynamic;

    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: HandlerKey::from("echo"),
            family: FamilyDescriptor::Custom {
                kind: "test:echo".into(),
            },
        }
    }

    async fn serve(&self, _kind: EffectKind, sink: OutcomeSink) {
        sink.resolve(Ok(Outcome::Custom(json!("never")))).await;
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

impl Serve for DropCounter {
    type Family = rig_core::effect::family::Dynamic;

    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: HandlerKey::from("flag"),
            family: FamilyDescriptor::Custom {
                kind: "test:flag".into(),
            },
        }
    }

    async fn serve(&self, _kind: EffectKind, sink: OutcomeSink) {
        sink.resolve(Ok(Outcome::Custom(json!(null)))).await;
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

impl rig_core::effect::CustomEffect for AskUser {
    const KIND: &'static str = "test:ask_user";
    type Answer = Reply;
}

/// Answers `AskUser` with the prompt echoed, or with a payload that is not
/// a `Reply` when asked to misbehave.
struct AskUserHandler {
    misbehave: bool,
}

impl Serve for AskUserHandler {
    type Family = rig_core::effect::family::Custom<AskUser>;

    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: HandlerKey::from("ask"),
            family: FamilyDescriptor::Custom {
                kind: AskUser::KIND.into(),
            },
        }
    }

    async fn serve(&self, kind: EffectKind, sink: OutcomeSink) {
        let misbehave = self.misbehave;
        {
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
        }
    }
}

#[tokio::test]
async fn a_typed_key_binds_with_an_existence_check_and_a_handle_dispatches_its_family() {
    let (dispatcher, _registrar, mut driver) = Bus::channel();
    let key: Key<rig_core::effect::family::Completion> = driver
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
    let back: Key<rig_core::effect::family::Completion> =
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
    let lie: Key<rig_core::effect::family::Tool> = Key::new_unchecked(HandlerKey::from("model"));
    let report = dispatcher
        .bind(&lie)
        .expect_err("a completion is not a tool");
    assert_eq!(report.kind, ErrorKind::HandlerUnavailable);
}

#[tokio::test]
async fn register_typed_refuses_a_handler_of_another_family() {
    let (_dispatcher, registrar, _driver) = Bus::channel();
    let report = registrar
        .register_typed::<rig_core::effect::family::Tool>(
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
        .register_typed::<rig_core::effect::family::Custom<AskUser>>(
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
    impl rig_core::effect::CustomEffect for Other {
        const KIND: &'static str = "test:other";
        type Answer = ();
    }
    let report = dispatcher
        .custom::<Other>(&HandlerKey::from("ask"))
        .expect_err("another kind");
    assert_eq!(report.kind, ErrorKind::HandlerUnavailable);
    assert!(report.message.contains("test:other"), "{}", report.message);
}

fn completion_request_value() -> rig_core::completion::CompletionRequest {
    match completion_kind(false) {
        EffectKind::Completion { request, .. } => request,
        other => panic!("a completion kind, got {}", other.name()),
    }
}

#[test]
fn a_command_offered_after_the_close_is_refused_under_the_queue_lock() {
    // The race this pins: a dispatch's first poll saw the bus open, the
    // driver dropped (emptying the buffer), and the dispatch's send then
    // lands. Deciding the close under the queue lock means the send is
    // refused rather than buffered for a driver that will never drain it.
    let (dispatcher, _registrar, driver) = Bus::channel();
    let shared = Arc::clone(&dispatcher.shared);
    drop(driver);
    let cx = std::task::Context::from_waker(noop_waker_ref());
    let (reply, _receiver) = oneshot::channel();
    let (_guard, cancel) = oneshot::channel();
    let offered = shared.enqueue(
        Box::new(super::dispatcher::Command {
            id: rig_core::effect::EffectId::from_raw(9),
            key: HandlerKey::from("echo"),
            kind: custom(json!(1)),
            reply: super::dispatcher::Reply::Unary(reply),
            span: tracing::Span::none(),
            cancel,
        }),
        &Arc::new(futures::task::AtomicWaker::new()),
        &cx,
    );
    assert!(matches!(offered, super::dispatcher::Enqueue::Closed));
    assert_eq!(shared.buffered(), 0, "nothing is buffered after the close");
}

#[tokio::test]
async fn a_registration_posted_just_before_a_dispatch_serves_it_whichever_poll_each_lands_in() {
    // The driver takes the queue first and the mailbox second, so a
    // registration made before a dispatch is applied before the dispatch is
    // served even when the two land around one driver poll.
    let (dispatcher, registrar, driver) = Bus::channel();
    let _task = spawn(driver);
    for round in 0..50 {
        let key = HandlerKey::from(format!("late-{round}"));
        let (echo, served) = Echo::new();
        registrar.register(key.clone(), echo).expect("fresh key");
        within(dispatcher.dispatch(&key, custom(json!(round))))
            .await
            .expect("served by the handler registered just before it");
        assert_eq!(served.load(Ordering::SeqCst), 1);
    }
}

// ---- the log carries its header; streams can be recorded verbatim ----

// ---- the stream writer mints; a handler names no block id ----

#[tokio::test]
async fn a_stream_written_through_the_writer_is_well_formed() {
    struct Writes;

    impl Serve for Writes {
        type Family = rig_core::effect::family::Dynamic;

        fn descriptor(&self) -> HandlerDescriptor {
            HandlerDescriptor {
                key: HandlerKey::from("writer"),
                family: FamilyDescriptor::Completion {
                    model: "writer".into(),
                    capabilities: Default::default(),
                },
            }
        }

        async fn serve(&self, _kind: EffectKind, sink: OutcomeSink) {
            let mut out = sink.writer();
            let _ = out.reasoning("thinking").await;
            let _ = out.text("hel").await;
            let _ = out.text("lo").await;
            let _ = out.tool_call("add", json!({"x": 1})).await;
            let _ = out.text("after").await;
            let _ = out
                .finish(rig_core::streaming::StreamFinal::new(
                    "writer",
                    rig_core::completion::Usage::new(),
                ))
                .await;
        }
    }

    let (dispatcher, _registrar, mut driver) = Bus::channel();
    driver.register("writer", Writes).expect("register");
    let _task = spawn(driver);
    let mut stream = dispatcher.dispatch_stream(&HandlerKey::from("writer"), completion_kind(true));
    let mut items = Vec::new();
    while let Some(item) = within(stream.next()).await {
        items.push(item);
    }
    // The conformance laws hold for a stream nobody minted ids for: every
    // delta's block was started, every started block ends before the
    // terminal, distinct blocks carry distinct ids.
    let events: Vec<StreamEvent> = items
        .into_iter()
        .map(|item| item.expect("a clean stream"))
        .collect();
    let starts: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::BlockStart { id, .. } => Some(id.clone()),
            _ => None,
        })
        .collect();
    let ends: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::BlockEnd { id, .. } => Some(id.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(
        starts.len(),
        4,
        "reasoning, text, tool call, text: {starts:?}"
    );
    let distinct: std::collections::BTreeSet<String> =
        starts.iter().map(ToString::to_string).collect();
    assert_eq!(distinct.len(), 4, "distinct minted ids: {starts:?}");
    for id in &starts {
        assert!(ends.contains(id), "block {id} ends before the terminal");
    }
    assert!(matches!(events.last(), Some(StreamEvent::Final(_))));
    let mut accumulator = rig_core::streaming::BlockAccumulator::new();
    for event in &events {
        accumulator
            .apply(event)
            .expect("the accumulator accepts every event");
    }
    let choice = accumulator.finish();
    assert_eq!(
        choice.len(),
        4,
        "reasoning, text, tool call, text as content: {choice:?}"
    );
}

/// A rerank model that counts its clones.
struct CloneCountingRerank {
    clones: Arc<AtomicUsize>,
}

impl Clone for CloneCountingRerank {
    fn clone(&self) -> Self {
        self.clones.fetch_add(1, Ordering::SeqCst);
        Self {
            clones: Arc::clone(&self.clones),
        }
    }
}

impl RerankModel for CloneCountingRerank {
    fn max_documents(&self) -> usize {
        7
    }

    async fn rerank(
        &self,
        _query: &str,
        documents: Vec<String>,
    ) -> Result<RerankResponse, RerankError> {
        Ok(RerankResponse::new(
            documents
                .into_iter()
                .enumerate()
                .map(|(index, document)| RerankResult {
                    index,
                    document: Some(document),
                    relevance_score: 1.0,
                })
                .collect(),
            "probe",
        ))
    }
}

/// The adapter owns the model by value: no dispatch, through however many
/// clones of the handle, ever clones it; `max_documents` and the label ride
/// on the descriptor.
#[tokio::test]
async fn a_rerank_adapter_never_clones_the_model_and_publishes_its_batch_size() {
    let clones = Arc::new(AtomicUsize::new(0));
    let (dispatcher, _registrar, mut driver) = Bus::channel();
    driver
        .register(
            "rerank:probe",
            RerankAdapter::new(
                "probe",
                CloneCountingRerank {
                    clones: Arc::clone(&clones),
                },
            ),
        )
        .expect("register");
    let task = spawn(driver);

    let handle: super::RerankHandle = dispatcher
        .handle(&HandlerKey::from("rerank:probe"))
        .expect("a rerank handler");
    for _ in 0..3 {
        let response = within(handle.rerank("q", vec!["a".to_owned(), "b".to_owned()]))
            .await
            .expect("rerank");
        assert_eq!(response.results.len(), 2);
        assert_eq!(response.provider, "probe");
        let via_clone = within(handle.clone().rerank("q", vec!["c".to_owned()]))
            .await
            .expect("rerank via clone");
        assert_eq!(via_clone.results[0].document.as_deref(), Some("c"));
    }
    assert_eq!(clones.load(Ordering::SeqCst), 0);
    assert_eq!(handle.max_documents(), Some(7));
    assert_eq!(handle.model_label(), "probe");
    assert_eq!(
        handle.descriptor().family,
        FamilyDescriptor::Rerank {
            model: "probe".to_owned(),
            max_documents: 7,
        }
    );

    drop((handle, dispatcher));
    within(task).await.expect("driver task");
}

struct NonCloneRerank;

impl RerankModel for NonCloneRerank {
    fn max_documents(&self) -> usize {
        1
    }

    async fn rerank(
        &self,
        _query: &str,
        _documents: Vec<String>,
    ) -> Result<RerankResponse, RerankError> {
        Err(RerankError::ResponseError("probe".to_owned()))
    }
}

/// A model that is not `Clone` registers, and its error crosses the bus as
/// a classified report.
#[tokio::test]
async fn a_non_clone_rerank_model_registers_and_its_error_is_a_report() {
    let (dispatcher, _registrar, mut driver) = Bus::channel();
    driver
        .register("rerank:once", RerankAdapter::new("once", NonCloneRerank))
        .expect("register");
    let task = spawn(driver);
    let handle: super::RerankHandle = dispatcher
        .handle(&HandlerKey::from("rerank:once"))
        .expect("a rerank handler");
    let report = within(handle.rerank("q", vec![]))
        .await
        .expect_err("the model fails");
    assert_eq!(report.kind, ErrorKind::Response);
    assert!(report.message.contains("probe"), "{}", report.message);
    drop((handle, dispatcher));
    within(task).await.expect("driver task");
}

/// Counts what the driver tells a recorder: `begin` per served dispatch,
/// `resolve` per outcome.
#[derive(Clone, Default)]
struct Counting {
    begun: Arc<AtomicUsize>,
    resolved: Arc<AtomicUsize>,
}

impl super::Recorder for Counting {
    fn handlers(&self, _handlers: Vec<HandlerDescriptor>) {}
    fn begin(&self, _id: rig_core::effect::EffectId, _key: HandlerKey, _kind: EffectKind) {
        self.begun.fetch_add(1, Ordering::SeqCst);
    }
    fn keep_events(&self) -> bool {
        false
    }
    fn event(&self, _id: rig_core::effect::EffectId, _event: &StreamEvent) {}
    fn resolve(&self, _id: rig_core::effect::EffectId, _outcome: Result<Outcome, ErrorReport>) {
        self.resolved.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn a_pending_dropped_before_the_driver_polls_never_reaches_its_handler() {
    // The Bevy shape: a system dispatches (one poll sends the command) and
    // the entity is despawned in the same frame, before the driver task
    // runs. The handler must not get a poll — one poll of a provider call
    // is an HTTP request — and the recorder must not open a record.
    let (dispatcher, _registrar, mut driver) = Bus::channel();
    let (echo, served) = Echo::new();
    driver.register("echo", echo).expect("register");
    let recorder = Counting::default();
    driver.record_to(recorder.clone());
    let waker = noop_waker_ref();
    let mut cx = Context::from_waker(waker);
    let mut pending = dispatcher.dispatch(&HandlerKey::from("echo"), custom(json!(1)));
    assert!(pending.poll_unpin(&mut cx).is_pending());
    assert_eq!(dispatcher.buffered(), 1, "sent, not yet taken");
    drop(pending);
    for _ in 0..4 {
        let _ = driver.poll_unpin(&mut cx);
    }
    assert_eq!(served.load(Ordering::SeqCst), 0, "the handler was polled");
    assert_eq!(
        recorder.begun.load(Ordering::SeqCst),
        0,
        "a record was opened"
    );
    assert_eq!(driver.in_flight(), 0);
    assert_eq!(dispatcher.buffered(), 0);

    // The same for a stream dispatch.
    let mut stream = dispatcher.dispatch_stream(&HandlerKey::from("echo"), completion_kind(true));
    assert!(stream.poll_next_unpin(&mut cx).is_pending());
    assert_eq!(dispatcher.buffered(), 1);
    drop(stream);
    for _ in 0..4 {
        let _ = driver.poll_unpin(&mut cx);
    }
    assert_eq!(served.load(Ordering::SeqCst), 0);
    assert_eq!(recorder.begun.load(Ordering::SeqCst), 0);
    assert_eq!(driver.in_flight(), 0);

    // And a live dispatch on the same driver is still served and recorded.
    let mut live = dispatcher.dispatch(&HandlerKey::from("echo"), custom(json!(2)));
    assert!(live.poll_unpin(&mut cx).is_pending());
    for _ in 0..4 {
        let _ = driver.poll_unpin(&mut cx);
    }
    assert!(live.poll_unpin(&mut cx).is_ready());
    assert_eq!(served.load(Ordering::SeqCst), 1);
    assert_eq!(recorder.begun.load(Ordering::SeqCst), 1);
    assert_eq!(recorder.resolved.load(Ordering::SeqCst), 1);
}

#[test]
fn a_serial_key_is_not_occupied_by_a_dispatch_cancelled_before_it_was_served() {
    let (dispatcher, _registrar, mut driver) = Bus::channel_with(BusConfig {
        serial_per_handler: true,
        ..BusConfig::default()
    });
    let (echo, served) = Echo::new();
    driver.register("echo", echo).expect("register");
    let waker = noop_waker_ref();
    let mut cx = Context::from_waker(waker);
    let key = HandlerKey::from("echo");
    let mut cancelled = dispatcher.dispatch(&key, custom(json!(1)));
    let mut kept = dispatcher.dispatch(&key, custom(json!(2)));
    assert!(cancelled.poll_unpin(&mut cx).is_pending());
    assert!(kept.poll_unpin(&mut cx).is_pending());
    drop(cancelled);
    for _ in 0..4 {
        let _ = driver.poll_unpin(&mut cx);
    }
    assert!(
        kept.poll_unpin(&mut cx).is_ready(),
        "the cancelled dispatch left the key busy"
    );
    assert_eq!(served.load(Ordering::SeqCst), 1);
}

#[test]
fn a_probe_resolves_a_dispatch_without_an_executor() {
    let (dispatcher, _registrar, mut driver) = Bus::channel();
    let (echo, served) = Echo::new();
    driver.register("echo", echo).expect("register");
    let waker = noop_waker_ref();
    let mut cx = Context::from_waker(waker);
    let mut pending = dispatcher.dispatch(&HandlerKey::from("echo"), custom(json!(7)));
    assert!(pending.poll_outcome().is_none(), "the first probe sends");
    assert_eq!(dispatcher.buffered(), 1);
    let _ = driver.poll_unpin(&mut cx);
    let outcome = pending.poll_outcome().expect("served").expect("ok");
    assert!(matches!(outcome, Outcome::Custom(ref payload) if *payload == json!(7)));
    assert_eq!(served.load(Ordering::SeqCst), 1);

    let mut stream = dispatcher.dispatch_stream(&HandlerKey::from("model"), completion_kind(true));
    // Unknown key: the probe sends, the driver answers, the probe drains.
    assert!(stream.poll_item().is_none());
    let _ = driver.poll_unpin(&mut cx);
    let first = stream.poll_item().expect("ready").expect("an item");
    assert_eq!(
        first.expect_err("unavailable").kind,
        ErrorKind::HandlerUnavailable
    );
    assert!(stream.poll_item().expect("ready").is_none(), "then the end");
}

#[test]
fn ten_thousand_probes_on_a_full_bus_keep_one_waker() {
    // A frame-ticked host probes a parked dispatch once per frame; the bus
    // keeps one slot per parked value, not one waker per probe.
    let (dispatcher, _registrar, mut driver) = Bus::channel_with(BusConfig {
        command_capacity: 1,
        ..BusConfig::default()
    });
    let (echo, served) = Echo::new();
    driver.register("echo", echo).expect("register");
    let key = HandlerKey::from("echo");
    let mut first = dispatcher.dispatch(&key, custom(json!(1)));
    let mut parked = dispatcher.dispatch(&key, custom(json!(2)));
    assert!(first.poll_outcome().is_none());
    for _ in 0..10_000 {
        assert!(parked.poll_outcome().is_none());
        // A real waker per poll, as `block_on(poll_once)` mints, is one
        // slot too.
        let waker = futures::task::waker(Arc::new(CountingWake));
        let mut cx = Context::from_waker(&waker);
        assert!(parked.poll_unpin(&mut cx).is_pending());
    }
    assert_eq!(dispatcher.shared.parked_senders(), 1);
    let waker = noop_waker_ref();
    let mut cx = Context::from_waker(waker);
    for _ in 0..4 {
        let _ = driver.poll_unpin(&mut cx);
    }
    assert!(first.poll_outcome().is_some());
    // The drain freed the buffer and woke the parked value: its next probe
    // sends, and one more driver poll serves it.
    assert!(parked.poll_outcome().is_none(), "sent on this probe");
    assert_eq!(dispatcher.buffered(), 1);
    for _ in 0..4 {
        let _ = driver.poll_unpin(&mut cx);
    }
    assert!(parked.poll_outcome().is_some(), "served");
    assert_eq!(served.load(Ordering::SeqCst), 2);
    assert_eq!(dispatcher.shared.parked_senders(), 0);
}

struct CountingWake;

impl futures::task::ArcWake for CountingWake {
    fn wake_by_ref(_arc_self: &Arc<Self>) {}
}

#[test]
fn a_parked_value_dropped_before_the_drain_leaves_no_slot_to_wake() {
    let (dispatcher, _registrar, mut driver) = Bus::channel_with(BusConfig {
        command_capacity: 1,
        ..BusConfig::default()
    });
    let (echo, _served) = Echo::new();
    driver.register("echo", echo).expect("register");
    let key = HandlerKey::from("echo");
    let mut first = dispatcher.dispatch(&key, custom(json!(1)));
    let mut parked = dispatcher.dispatch(&key, custom(json!(2)));
    assert!(first.poll_outcome().is_none());
    assert!(parked.poll_outcome().is_none());
    assert_eq!(dispatcher.shared.parked_senders(), 1);
    drop(parked);
    let waker = noop_waker_ref();
    let mut cx = Context::from_waker(waker);
    let _ = driver.poll_unpin(&mut cx);
    assert_eq!(
        dispatcher.shared.parked_senders(),
        0,
        "the dead slot was dropped by the drain"
    );
}

/// Detaches every sink it is given into a mailbox and returns at once —
/// the shape of a tool answered by a Bevy system.
struct Detaching {
    mailbox: Arc<Mutex<Vec<rig_core::serve::DetachedSink>>>,
}

impl Serve for Detaching {
    type Family = rig_core::effect::family::Dynamic;

    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: HandlerKey::from("world"),
            family: FamilyDescriptor::Custom {
                kind: "test:world".into(),
            },
        }
    }

    async fn serve(&self, _kind: EffectKind, sink: OutcomeSink) {
        self.mailbox.lock().expect("mailbox").push(sink.detach());
    }
}

#[test]
fn a_detached_sink_keeps_its_serial_slot_until_answered() {
    let (dispatcher, _registrar, mut driver) = Bus::channel_with(BusConfig {
        serial_per_handler: true,
        ..BusConfig::default()
    });
    let mailbox = Arc::new(Mutex::new(Vec::new()));
    driver
        .register(
            "world",
            Detaching {
                mailbox: mailbox.clone(),
            },
        )
        .expect("register");
    let key = HandlerKey::from("world");
    let mut first = dispatcher.dispatch(&key, custom(json!(1)));
    let mut second = dispatcher.dispatch(&key, custom(json!(2)));
    assert!(first.poll_outcome().is_none());
    assert!(second.poll_outcome().is_none());
    let waker = noop_waker_ref();
    let mut cx = Context::from_waker(waker);
    for _ in 0..4 {
        let _ = driver.poll_unpin(&mut cx);
    }
    // The handler returned at once, but the dispatch is still in flight:
    // the key is busy and the second command waits behind it.
    assert_eq!(mailbox.lock().expect("mailbox").len(), 1);
    assert_eq!(driver.in_flight(), 1, "keyed on the sink, not the future");
    assert!(second.poll_outcome().is_none());

    let sink = mailbox.lock().expect("mailbox").remove(0);
    assert!(!sink.is_closed());
    let mut resolving = sink.resolve(Ok(Outcome::Custom(json!("answered"))));
    assert!(resolving.poll_unpin(&mut cx).is_ready());
    let outcome = first.poll_outcome().expect("answered").expect("ok");
    assert!(matches!(outcome, Outcome::Custom(ref v) if *v == json!("answered")));
    for _ in 0..4 {
        let _ = driver.poll_unpin(&mut cx);
    }
    // The slot was released by the answer: the second is now with the world.
    assert_eq!(mailbox.lock().expect("mailbox").len(), 1);
    assert_eq!(driver.in_flight(), 1);
    let sink = mailbox.lock().expect("mailbox").remove(0);
    let mut resolving = sink.resolve(Ok(Outcome::Custom(json!("second"))));
    assert!(resolving.poll_unpin(&mut cx).is_ready());
    assert!(second.poll_outcome().is_some());
    for _ in 0..4 {
        let _ = driver.poll_unpin(&mut cx);
    }
    assert_eq!(driver.in_flight(), 0);
}

#[test]
fn dropping_the_pending_closes_a_detached_sink() {
    let (dispatcher, _registrar, mut driver) = Bus::channel();
    let mailbox = Arc::new(Mutex::new(Vec::new()));
    driver
        .register(
            "world",
            Detaching {
                mailbox: mailbox.clone(),
            },
        )
        .expect("register");
    let mut pending = dispatcher.dispatch(&HandlerKey::from("world"), custom(json!(1)));
    assert!(pending.poll_outcome().is_none());
    let waker = noop_waker_ref();
    let mut cx = Context::from_waker(waker);
    let _ = driver.poll_unpin(&mut cx);
    assert_eq!(mailbox.lock().expect("mailbox").len(), 1);
    assert!(!mailbox.lock().expect("mailbox")[0].is_closed());
    drop(pending);
    let _ = driver.poll_unpin(&mut cx);
    // The resolver sees the cancel; the dispatch stays in flight until the
    // resolver lets the sink go.
    assert!(mailbox.lock().expect("mailbox")[0].is_closed());
    assert_eq!(driver.in_flight(), 1);
    mailbox.lock().expect("mailbox").clear();
    let _ = driver.poll_unpin(&mut cx);
    assert_eq!(driver.in_flight(), 0);
}

#[test]
fn a_reopened_bus_serves_handles_bound_before_the_drop() {
    let (dispatcher, _registrar, mut driver) = Bus::channel();
    let (echo, served) = Echo::new();
    driver.register("echo", echo).expect("register");
    // The host's bound view: a key it keeps (a typed `Handle` is the same
    // thing over a key; `ModelHandle` is proven on the fixture side).
    let handle = HandlerKey::from("echo");
    let waker = noop_waker_ref();
    let mut cx = Context::from_waker(waker);
    // Served once, then the driver goes.
    let mut before = dispatcher.dispatch(&handle, custom(json!(1)));
    assert!(before.poll_outcome().is_none());
    let _ = driver.poll_unpin(&mut cx);
    assert!(before.poll_outcome().is_some());
    let mut minted_before_drop = dispatcher.dispatch(&handle, custom(json!(2)));
    drop(driver);
    assert!(dispatcher.is_closed());
    assert!(
        dispatcher.descriptor(&HandlerKey::from("echo")).is_none(),
        "the handlers died with the driver; the table says so"
    );
    let report = dispatcher
        .dispatch(&handle, custom(json!(3)))
        .poll_outcome()
        .expect("closed")
        .expect_err("closed");
    assert_eq!(report.kind, ErrorKind::BusClosed);

    let (registrar, mut driver) = Bus::reopen(&dispatcher).expect("no driver alive");
    assert!(!dispatcher.is_closed());
    // What was minted against the dead driver stays dead.
    let report = minted_before_drop
        .poll_outcome()
        .expect("decided")
        .expect_err("closed");
    assert_eq!(report.kind, ErrorKind::BusClosed);
    // Re-register (through the new registrar) and the same handle works.
    let (echo, served_again) = Echo::new();
    registrar.register("echo", echo).expect("fresh table");
    let mut after = dispatcher.dispatch(&handle, custom(json!(4)));
    assert!(after.poll_outcome().is_none());
    for _ in 0..4 {
        let _ = driver.poll_unpin(&mut cx);
    }
    let outcome = after.poll_outcome().expect("served").expect("ok");
    assert!(matches!(outcome, Outcome::Custom(ref v) if *v == json!(4)));
    assert_eq!(served.load(Ordering::SeqCst), 1);
    assert_eq!(served_again.load(Ordering::SeqCst), 1);
    assert_eq!(
        dispatcher.descriptor(&handle).expect("re-registered").key,
        handle
    );
}

#[test]
fn reopen_while_a_driver_is_alive_is_refused() {
    let (dispatcher, _registrar, driver) = Bus::channel();
    let report = Bus::reopen(&dispatcher).expect_err("a driver is alive");
    assert_eq!(report.kind, ErrorKind::Request);
    assert!(report.retryable, "retry once the driver has been dropped");
    drop(driver);
    let (_registrar, driver) = Bus::reopen(&dispatcher).expect("now free");
    assert!(Bus::reopen(&dispatcher).is_err(), "the new driver holds it");
    drop(driver);
    assert!(Bus::reopen(&dispatcher).is_ok());
}

#[test]
fn pendings_and_streams_created_while_closed_stay_closed_after_reopen() {
    let (dispatcher, _registrar, driver) = Bus::channel();
    drop(driver);
    let mut pending = dispatcher.dispatch(&HandlerKey::from("echo"), custom(json!(1)));
    let mut stream = dispatcher.dispatch_stream(&HandlerKey::from("echo"), completion_kind(true));
    let (_registrar, mut driver) = Bus::reopen(&dispatcher).expect("free");
    let (echo, served) = Echo::new();
    driver.register("echo", echo).expect("register");
    let report = pending
        .poll_outcome()
        .expect("decided")
        .expect_err("closed");
    assert_eq!(report.kind, ErrorKind::BusClosed);
    let item = stream.poll_item().expect("decided").expect("one item");
    assert_eq!(item.expect_err("closed").kind, ErrorKind::BusClosed);
    assert!(stream.poll_item().expect("decided").is_none());
    let waker = noop_waker_ref();
    let mut cx = Context::from_waker(waker);
    let _ = driver.poll_unpin(&mut cx);
    assert_eq!(served.load(Ordering::SeqCst), 0, "nothing was resurrected");
    assert_eq!(dispatcher.buffered(), 0);
}

#[test]
fn descriptors_is_one_snapshot_and_a_bus_id_tells_buses_apart() {
    let (dispatcher, registrar, mut driver) = Bus::channel();
    let (echo, _) = Echo::new();
    driver.register("echo", echo).expect("register");
    driver
        .register("add", ToolAdapter::new(Add))
        .expect("register");
    let snapshot = dispatcher.descriptors();
    assert_eq!(
        snapshot.iter().map(|d| d.key.as_str()).collect::<Vec<_>>(),
        ["add", "echo"],
        "key order, one lock"
    );
    assert_eq!(snapshot[0].family.family(), EffectFamily::Tool);
    // A registration after the snapshot does not tear it.
    let (echo, _) = Echo::new();
    registrar.register("echo2", echo).expect("register");
    assert_eq!(snapshot.len(), 2);
    assert_eq!(dispatcher.descriptors().len(), 3);

    let (other, _r, _d) = Bus::channel();
    assert_ne!(dispatcher.id(), other.id(), "two buses, two ids");
    assert_eq!(dispatcher.id(), dispatcher.clone().id(), "one bus, one id");
    assert_eq!(dispatcher.id(), registrar_bus_id(&registrar, &dispatcher));
    assert_ne!(dispatcher.id().as_u64(), 0);
    assert!(dispatcher.id().to_string().starts_with("bus#"));
}

fn registrar_bus_id(_registrar: &Registrar, dispatcher: &Dispatcher) -> super::BusId {
    // A registrar has no id of its own: the dispatcher's is the bus's.
    dispatcher.id()
}

#[test]
fn a_rebind_before_registration_fails_at_first_dispatch_not_at_bind() {
    let (dispatcher, _registrar, mut driver) = Bus::channel();
    // What a scene stored: the descriptor, no handler yet.
    let stored = HandlerDescriptor {
        key: HandlerKey::from("model"),
        family: FamilyDescriptor::Completion {
            model: "gpt".into(),
            capabilities: Default::default(),
        },
    };
    let handle: ModelHandle = super::Handle::rebind(dispatcher.clone(), stored.clone());
    assert_eq!(handle.key(), &HandlerKey::from("model"));
    assert_eq!(
        handle.descriptor(),
        stored,
        "the snapshot until the table says otherwise"
    );
    let waker = noop_waker_ref();
    let mut cx = Context::from_waker(waker);
    let mut pending = handle.complete(completion_request_value());
    assert!(pending.poll_unpin(&mut cx).is_pending());
    let _ = driver.poll_unpin(&mut cx);
    let report = match pending.poll_unpin(&mut cx) {
        Poll::Ready(Err(report)) => report,
        other => panic!("expected HandlerUnavailable, got {other:?}"),
    };
    assert_eq!(report.kind, ErrorKind::HandlerUnavailable);
    // Registered later, the same handle works.
    driver
        .register(
            "model",
            CompletionAdapter::new(
                "gpt",
                MockCompletionModel::from_turns([MockTurn::text("hi")]),
            ),
        )
        .expect("register");
    let mut pending = handle.complete(completion_request_value());
    assert!(pending.poll_unpin(&mut cx).is_pending());
    let _ = driver.poll_unpin(&mut cx);
    assert!(pending.poll_unpin(&mut cx).is_ready());
}

#[test]
#[should_panic(expected = "serves the tool_call family, not completion")]
fn a_rebind_of_the_wrong_family_panics_at_the_hosts_line() {
    let (dispatcher, _registrar, _driver) = Bus::channel();
    let stored = rig_core::serve::Serve::descriptor(&ToolAdapter::new(Add));
    let _: ModelHandle = super::Handle::rebind(dispatcher, stored);
}

#[test]
fn a_pending_whose_dispatcher_died_before_its_first_poll_is_bus_closed_while_a_stream_is_in_flight()
{
    let (dispatcher, _registrar, mut driver) = Bus::channel();
    let (echo, _open_never) = Echo::gated();
    driver.register("echo", echo).expect("register");
    let waker = noop_waker_ref();
    let mut cx = Context::from_waker(waker);
    // A long dispatch in flight (the gate never opens).
    let mut held = dispatcher.dispatch(&HandlerKey::from("echo"), custom(json!(1)));
    assert!(held.poll_outcome().is_none());
    let _ = driver.poll_unpin(&mut cx);
    assert_eq!(driver.in_flight(), 1);
    // A dispatch minted but not yet polled when its dispatcher goes.
    let mut late = dispatcher.dispatch(&HandlerKey::from("echo"), custom(json!(2)));
    drop(dispatcher);
    // The driver notices the last dispatcher went with nothing buffered.
    let _ = driver.poll_unpin(&mut cx);
    assert!(
        driver.poll_unpin(&mut cx).is_pending(),
        "the held dispatch keeps the driver alive"
    );
    // The late send is refused at once — not held until the stream ends.
    let report = late
        .poll_outcome()
        .expect("decided now")
        .expect_err("closed");
    assert_eq!(report.kind, ErrorKind::BusClosed);
    assert!(
        held.poll_outcome().is_none(),
        "the in-flight one is untouched"
    );
}
