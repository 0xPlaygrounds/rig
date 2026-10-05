use rig_core::error::ProviderError;
use rig_core::serve::Dispatch;
use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    task::{Context, Poll},
    time::Duration,
};

use futures::{FutureExt, StreamExt, channel::oneshot, task::noop_waker_ref};
use serde_json::json;

use super::{Bus, BusDriver, Dispatcher, ModelHandle, Pending, Registrar, ServingPolicy};
use rig_core::effect::{CustomEffect, Key};
use rig_core::serve::{
    Reply as CoreReply, Serve,
    adapters::{MemoryAdapter, ModelAdapter, ToolAdapter, ToolFn},
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
    rerank::{RerankResponse, RerankResult},
    streaming::{Item, Relayed, StreamEvent},
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
        request: CompletionRequest::new("hi"),
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
            layers: Vec::new(),
        }
    }

    async fn serve(&self, kind: EffectKind, _dispatch: Dispatch) -> rig_core::serve::Reply {
        let gate = self.gate.lock().expect("gate lock").take();
        {
            if let Some(gate) = gate {
                let _ = gate.await;
            }
            self.served.fetch_add(1, Ordering::SeqCst);
            let outcome = match kind {
                EffectKind::Custom { payload, .. } => Ok(Outcome::Custom { payload }),
                other => Err(ErrorReport::new(
                    ErrorKind::Internal,
                    format!("echo received {}", other.name()),
                )),
            };
            rig_core::serve::Reply::Outcome(outcome)
        }
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
    assert!(matches!(outcome, Outcome::Custom { ref payload } if *payload == json!({"n": 1})));
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
    let (dispatcher, _registrar) = Bus::new_with(ServingPolicy::default(), |_| {}, drop);
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
async fn concurrent_serving_across_keys_is_the_default() {
    let (dispatcher, _registrar, mut driver) = Bus::channel();
    let (blocked, open) = Echo::gated();
    let (free, served) = Echo::new();
    driver.register("blocked", blocked).expect("register");
    driver.register("free", free).expect("register");
    let mut blocked_pending = dispatcher.dispatch(&HandlerKey::from("blocked"), custom(json!(1)));
    let mut cx = Context::from_waker(noop_waker_ref());
    assert!(blocked_pending.poll_unpin(&mut cx).is_pending());
    assert!(driver.poll_unpin(&mut cx).is_pending());
    assert_eq!(
        driver.in_flight(),
        1,
        "the gated key must actually be serving"
    );
    let _task = spawn(driver);
    let free_outcome = within(dispatcher.dispatch(&HandlerKey::from("free"), custom(json!(2))))
        .await
        .expect("free key is served while another is gated");
    assert!(matches!(free_outcome, Outcome::Custom { ref payload } if *payload == json!(2)));
    assert_eq!(served.load(Ordering::SeqCst), 1);
    let _ = open.send(());
    within(blocked_pending)
        .await
        .expect("gated key resolves once opened");
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
fn deregistering_a_serial_key_drains_its_queue_with_handler_unavailable() {
    let (dispatcher, registrar, mut driver) = Bus::channel_with(ServingPolicy {
        serial_per_handler: true,
        ..ServingPolicy::default()
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

const _: fn() = || {
    fn assert_clone_send_sync<T: Clone + Send + Sync + 'static>() {}
    assert_clone_send_sync::<Dispatcher>();
};

// ---------------------------------------------------------------------------
// T10: the log is in dispatch order, replay checks the payload, a key keeps
// its family, a cut-short stream is reported and recorded, the tap agrees
// with the consumer, and cancellation is pinned for unary dispatches too.
// ---------------------------------------------------------------------------

#[test]
fn register_refuses_a_family_change_under_a_live_key() {
    let (dispatcher, registrar, mut driver) = Bus::channel();
    let (echo, served) = Echo::new();
    driver.register("k", echo).expect("register");
    let refused = registrar
        .register(
            "k",
            ModelAdapter::new("mock", MockCompletionModel::text("x")),
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

#[test]
fn direct_deregistration_drops_an_earlier_mailbox_registration_immediately() {
    let (_dispatcher, registrar, mut driver) = Bus::channel();
    let dropped = Arc::new(AtomicUsize::new(0));
    registrar
        .register("flag", DropCounter(dropped.clone()))
        .expect("queued");
    let key = HandlerKey::from("flag");
    assert!(driver.deregister(&key));
    assert!(registrar.descriptor(&key).is_none());
    assert_eq!(dropped.load(Ordering::SeqCst), 1);
    let mut cx = Context::from_waker(noop_waker_ref());
    let _ = driver.poll_unpin(&mut cx);
    assert!(
        !driver.deregister(&key),
        "removed handler cannot be resurrected"
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
            layers: Vec::new(),
        }
    }

    async fn serve(&self, _kind: EffectKind, _dispatch: Dispatch) -> rig_core::serve::Reply {
        rig_core::serve::Reply::Outcome(Ok(Outcome::Custom {
            payload: json!(null),
        }))
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
            layers: Vec::new(),
        }
    }

    async fn serve(&self, kind: EffectKind, _dispatch: Dispatch) -> rig_core::serve::Reply {
        let misbehave = self.misbehave;
        {
            let EffectKind::Custom { payload, .. } = kind else {
                return rig_core::serve::Reply::Outcome(Err(ErrorReport::new(
                    ErrorKind::Internal,
                    "not custom",
                )));
            };
            let answer = if misbehave {
                json!({"nope": 1})
            } else {
                let ask: AskUser = serde_json::from_value(payload).expect("an AskUser");
                json!({"text": format!("you asked: {}", ask.prompt)})
            };
            rig_core::serve::Reply::Outcome(Ok(Outcome::Custom { payload: answer }))
        }
    }
}

#[tokio::test]
async fn a_typed_key_binds_with_an_existence_check_and_a_handle_dispatches_its_family() {
    let (dispatcher, _registrar, mut driver) = Bus::channel();
    let key: Key<rig_core::effect::family::Completion> = driver
        .register_typed(
            "model",
            ModelAdapter::new(
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
    let response = within(model.dispatch(CompletionRequest::new("hi")))
        .await
        .expect("the family's own answer");
    assert_eq!(response.choice, vec![AssistantContent::text("typed")]);
    let response = within(model.call(CompletionRequest::new("hi")))
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
            ModelAdapter::new("mock", MockCompletionModel::text("x")),
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
    let reply = within(ask.dispatch(AskUser {
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
    let report = within(ask.dispatch(AskUser { prompt: "?".into() }))
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
        super::dispatcher::Command {
            lineage: super::dispatcher::Lineage::new(rig_core::effect::EffectId::from_raw(9), None),
            id: rig_core::effect::EffectId::from_raw(9),
            key: HandlerKey::from("echo"),
            kind: custom(json!(1)),
            parent: None,
            scope: None,
            context: None,
            adapter_context: None,
            published: None,
            reply: super::dispatcher::Reply::Unary(reply),
            span: tracing::Span::none(),
            cancel,
        },
        &Arc::new(futures::task::AtomicWaker::new()),
        &cx,
    );
    assert!(matches!(offered, super::dispatcher::Enqueue::Closed));
    assert_eq!(shared.buffered(), 0, "nothing is buffered after the close");
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
                layers: Vec::new(),
            }
        }

        async fn serve(&self, _kind: EffectKind, _dispatch: Dispatch) -> CoreReply {
            CoreReply::written(
                rig_core::message::Origin::new("writer", "writer", "writer"),
                |mut out| async move {
                    let _ = out.reasoning("thinking").await;
                    let _ = out.text("hel").await;
                    let _ = out.text("lo").await;
                    let _ = out.tool_call("add", json!({"x": 1})).await;
                    let _ = out.text("after").await;
                    let _ = out
                        .finish(rig_core::operation::Finish {
                            usage: rig_core::completion::Usage::default(),
                            ..rig_core::operation::Finish::default()
                        })
                        .await;
                },
            )
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
    // The stream a writer makes is in order: every event's part started,
    // every started part ends before the response, and the response holds
    // every part.
    let mut items: Vec<Relayed> = items
        .into_iter()
        .map(|item| item.expect("a clean stream"))
        .collect();
    let Some(Relayed::Done(response)) = items.pop() else {
        panic!("the stream ends with the response: {items:?}");
    };
    let transcript = rig_core::streaming::Transcript::parse(
        serde_json::to_value(
            items
                .into_iter()
                .filter_map(|item| match item {
                    Relayed::Item(item) => Some(item),
                    Relayed::Origin(_) => None,
                    Relayed::Done(_) => panic!("one response, last"),
                })
                .collect::<Vec<_>>(),
        )
        .expect("items serialize"),
    )
    .expect("the writer's items are in order and every part ends");
    let starts = transcript
        .events()
        .filter(|event| matches!(event, StreamEvent::Start { .. }))
        .count();
    assert_eq!(starts, 4, "reasoning, text, tool call, text");
    assert_eq!(
        response.choice.len(),
        4,
        "reasoning, text, tool call, text as content: {:?}",
        response.choice
    );
}

/// A rerank runtime: every document keeps its place at score 1.0, or the
/// call fails with `failure`.
#[derive(Clone)]
struct ProbeRerank {
    max_documents: usize,
    failure: Option<&'static str>,
}

impl ProbeRerank {
    fn model(self) -> rig_core::Model<rig_core::driver::Local<rig_core::operation::Rerank>, Self> {
        rig_core::Model::new(
            rig_core::driver::Local::new("probe")
                .with_capabilities(rig_core::wire::Capabilities::rerank(self.max_documents)),
            self,
        )
    }
}

impl rig_core::driver::Transport<rig_core::driver::Local<rig_core::operation::Rerank>>
    for ProbeRerank
{
    fn send(
        &self,
        request: rig_core::operation::RerankRequest,
        _exchange: rig_core::driver::Exchange,
    ) -> rig_core::driver::Opening<rig_core::driver::Step<rig_core::operation::Rerank>> {
        let failure = self.failure;
        rig_core::driver::Opening::ready(match failure {
            Some(message) => {
                rig_core::driver::Opened::failed(ProviderError::Response(message.to_owned()))
            }
            None => rig_core::driver::Opened::new(futures::stream::iter([Ok(
                rig_core::driver::Step::End(RerankResponse {
                    provider: "probe".into(),
                    ..RerankResponse::new(
                        request
                            .documents
                            .into_iter()
                            .enumerate()
                            .map(|(index, document)| RerankResult {
                                index,
                                document: Some(document),
                                relevance_score: 1.0,
                            })
                            .collect(),
                    )
                }),
            )])),
        })
    }
}

/// Every clone of the handle dispatches to the one registered model;
/// `max_documents` and the label ride on the descriptor.
#[tokio::test]
async fn a_rerank_adapter_serves_every_handle_clone_and_publishes_its_batch_size() {
    let (dispatcher, _registrar, mut driver) = Bus::channel();
    let probe = ProbeRerank {
        max_documents: 7,
        failure: None,
    };
    driver
        .register("rerank:probe", ModelAdapter::new("probe", probe.model()))
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
    assert_eq!(handle.max_documents(), Some(7));
    assert_eq!(handle.label(), "probe");
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

/// A rerank model's error crosses the bus as a classified report.
#[tokio::test]
async fn a_rerank_model_error_crosses_the_bus_as_a_report() {
    let (dispatcher, _registrar, mut driver) = Bus::channel();
    let failing = ProbeRerank {
        max_documents: 1,
        failure: Some("probe"),
    };
    driver
        .register("rerank:once", ModelAdapter::new("once", failing.model()))
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
    discarded: Arc<AtomicUsize>,
    observation: Option<rig_core::observe::AdapterContext>,
    observation_sink: Option<Arc<rig_core::observe::ObservationLog>>,
    contexts: Arc<
        Mutex<
            std::collections::BTreeMap<
                rig_core::effect::EffectId,
                rig_core::observe::AdapterContext,
            >,
        >,
    >,
}

impl rig_core::serve::Recorder for Counting {
    fn origin(&self, _id: EffectId, _origin: &rig_core::message::Origin) {}
    fn adapter_context(
        &self,
        id: rig_core::effect::EffectId,
    ) -> Option<rig_core::observe::AdapterContext> {
        self.contexts
            .lock()
            .unwrap()
            .get(&id)
            .cloned()
            .or_else(|| self.observation.clone())
    }
    fn tool_output(&self, _: rig_core::effect::EffectId, _: rig_core::tool::ToolResultContext) {}
    fn handlers(&self, _handlers: Vec<HandlerDescriptor>) {}
    fn begin(
        &self,
        id: rig_core::effect::EffectId,
        key: HandlerKey,
        kind: EffectKind,
        origin: rig_core::serve::Origin,
    ) {
        self.begun.fetch_add(1, Ordering::SeqCst);
        if let Some(sink) = &self.observation_sink {
            let subject = rig_core::observe::Subject {
                scope: origin.scope.map(|scope| scope.to_string()),
                effect: Some(id),
                parent: origin.parent,
                key: Some(key),
                family: Some(kind.family()),
                ..Default::default()
            };
            self.contexts.lock().unwrap().insert(
                id,
                rig_core::observe::AdapterContext::new(
                    sink.clone(),
                    subject,
                    format!("dispatch/{id:?}"),
                ),
            );
        }
    }
    fn discard(&self, _id: rig_core::effect::EffectId) {
        self.discarded.fetch_add(1, Ordering::SeqCst);
    }
    fn patch(&self, _id: rig_core::effect::EffectId, _kind: EffectKind) {}
    fn keep_events(&self) -> bool {
        false
    }
    fn event(&self, _id: rig_core::effect::EffectId, _event: &Item<StreamEvent>) {}
    fn resolve(&self, _id: rig_core::effect::EffectId, _outcome: Result<Outcome, ErrorReport>) {
        self.resolved.fetch_add(1, Ordering::SeqCst);
    }
}

#[tokio::test]
async fn recorder_context_reaches_model_handles_without_overwriting_callers() {
    use rig_core::observe::{AdapterContext, ObservationLog, Subject};
    for streamed in [false, true] {
        for explicit in [false, true] {
            let model = if streamed {
                MockCompletionModel::from_stream_turns([vec![
                    MockStreamEvent::text("ok"),
                    MockStreamEvent::final_response_with_total_tokens(1),
                ]])
            } else {
                MockCompletionModel::text("ok")
            };
            let (dispatcher, _registrar, mut driver) = Bus::channel();
            driver
                .register("model", ModelAdapter::new("mock", model.clone()))
                .unwrap();
            let sink = Arc::new(ObservationLog::default());
            let recorder = Counting {
                observation: Some(AdapterContext::new(
                    sink.clone(),
                    Subject::scoped("host"),
                    "recorder",
                )),
                ..Counting::default()
            };
            driver.record_to(recorder.clone());
            let task = tokio::spawn(driver);
            let handle: ModelHandle = dispatcher.handle(&HandlerKey::from("model")).unwrap();
            let request = CompletionRequest::new("hi");
            let context =
                explicit.then(|| AdapterContext::new(sink, Subject::scoped("caller"), "caller"));
            if streamed {
                let mut stream = match context {
                    Some(context) => handle.stream_observed(request, context),
                    None => handle.stream(request),
                };
                while let Some(event) = within(stream.next()).await {
                    event.unwrap();
                }
            } else {
                let completion = match context {
                    Some(context) => handle.call_observed(request, context),
                    None => handle.call(request),
                };
                within(completion).await.unwrap();
            }
            let requests = model.requests();
            assert_eq!(requests.len(), 1);
            assert_eq!(
                model.contexts()[0].as_ref().unwrap().operation(),
                if explicit { "caller" } else { "recorder" }
            );
            assert_eq!(recorder.begun.load(Ordering::SeqCst), 1);
            assert_eq!(recorder.resolved.load(Ordering::SeqCst), 1);
            task.abort();
        }
    }
}

/// One poll with a no-op waker: the outcome if the dispatch has resolved.
/// What a ticking host used to get from `Pending::poll_outcome`; the bus
/// no longer offers it (a world holds no future to probe), and these tests
/// keep it as a spelling of "poll once, no executor".
fn probe(pending: &mut Pending) -> Option<Result<Outcome, ErrorReport>> {
    let mut cx = Context::from_waker(noop_waker_ref());
    match pending.poll_unpin(&mut cx) {
        Poll::Ready(outcome) => Some(outcome),
        Poll::Pending => None,
    }
}

#[test]
fn ten_thousand_probes_on_a_full_bus_keep_one_waker() {
    // A frame-ticked host probes a parked dispatch once per frame; the bus
    // keeps one slot per parked value, not one waker per probe.
    let (dispatcher, _registrar, mut driver) = Bus::channel_with(ServingPolicy {
        command_capacity: 1,
        ..ServingPolicy::default()
    });
    let (echo, served) = Echo::new();
    driver.register("echo", echo).expect("register");
    let key = HandlerKey::from("echo");
    let mut first = dispatcher.dispatch(&key, custom(json!(1)));
    let mut parked = dispatcher.dispatch(&key, custom(json!(2)));
    assert!(probe(&mut first).is_none());
    for _ in 0..10_000 {
        assert!(probe(&mut parked).is_none());
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
    assert!(probe(&mut first).is_some());
    // The drain freed the buffer and woke the parked value: its next probe
    // sends, and one more driver poll serves it.
    assert!(probe(&mut parked).is_none(), "sent on this probe");
    assert_eq!(dispatcher.buffered(), 1);
    for _ in 0..4 {
        let _ = driver.poll_unpin(&mut cx);
    }
    assert!(probe(&mut parked).is_some(), "served");
    assert_eq!(served.load(Ordering::SeqCst), 2);
    assert_eq!(dispatcher.shared.parked_senders(), 0);
}

struct CountingWake;

impl futures::task::ArcWake for CountingWake {
    fn wake_by_ref(_arc_self: &Arc<Self>) {}
}

#[test]
fn a_parked_value_dropped_before_the_drain_leaves_no_slot_to_wake() {
    let (dispatcher, _registrar, mut driver) = Bus::channel_with(ServingPolicy {
        command_capacity: 1,
        ..ServingPolicy::default()
    });
    let (echo, _served) = Echo::new();
    driver.register("echo", echo).expect("register");
    let key = HandlerKey::from("echo");
    let mut first = dispatcher.dispatch(&key, custom(json!(1)));
    let mut parked = dispatcher.dispatch(&key, custom(json!(2)));
    assert!(probe(&mut first).is_none());
    assert!(probe(&mut parked).is_none());
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
fn a_pending_whose_dispatcher_died_before_its_first_poll_is_bus_closed_while_a_stream_is_in_flight()
{
    let (dispatcher, _registrar, mut driver) = Bus::channel();
    let (echo, _open_never) = Echo::gated();
    driver.register("echo", echo).expect("register");
    let waker = noop_waker_ref();
    let mut cx = Context::from_waker(waker);
    // A long dispatch in flight (the gate never opens).
    let mut held = dispatcher.dispatch(&HandlerKey::from("echo"), custom(json!(1)));
    assert!(probe(&mut held).is_none());
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
    let report = probe(&mut late).expect("decided now").expect_err("closed");
    assert_eq!(report.kind, ErrorKind::BusClosed);
    assert!(probe(&mut held).is_none(), "the in-flight one is untouched");
}

// ---------------------------------------------------------------------------
// Causal dispatch: a command carries its parent; re-entrancy is a chain, a
// cancel reaches the chain.

/// Dispatches to `child` through its dispatch context (the way back onto
/// the bus), from the calling thread or from a spawned one, and reports the
/// nested dispatch's first poll as its own outcome. The child `Pending` is
/// parked in `held` when a slot is given, so a test can watch a child whose
/// parent handler is gone.
struct Parent {
    key: HandlerKey,
    child: HandlerKey,
    from_another_thread: bool,
    held: Option<Arc<Mutex<Vec<super::Pending>>>>,
    /// Await the child's answer and report it (else report the child's
    /// first poll and let it go).
    await_child: bool,
}

impl Serve for Parent {
    type Family = rig_core::effect::family::Dynamic;

    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: self.key.clone(),
            family: FamilyDescriptor::Custom {
                kind: "test:parent".into(),
            },
            layers: Vec::new(),
        }
    }

    async fn serve(&self, _kind: EffectKind, dispatch: Dispatch) -> rig_core::serve::Reply {
        let dispatcher =
            super::DispatchScope::dispatcher(&dispatch).expect("served by a bus driver");
        assert_eq!(dispatcher.parent(), Some(dispatch.id()));
        let child = self.child.clone();
        let first_poll = move |dispatcher: Dispatcher| {
            let mut nested = dispatcher.dispatch(&child, custom(json!("nested")));
            let mut cx = Context::from_waker(noop_waker_ref());
            let first = nested.poll_unpin(&mut cx);
            (first, nested)
        };
        let (first, nested) = if self.from_another_thread {
            std::thread::spawn(move || first_poll(dispatcher))
                .join()
                .expect("the nested thread")
        } else {
            first_poll(dispatcher)
        };
        if let Some(held) = &self.held {
            held.lock().expect("held").push(nested);
            // The parent stays in flight until its consumer goes.
            return std::future::pending().await;
        }
        if self.await_child {
            let outcome = match first {
                Poll::Ready(result) => result,
                Poll::Pending => nested.await,
            };
            let outcome = match outcome {
                Ok(outcome) => Ok(outcome),
                Err(report) => Ok(Outcome::Custom {
                    payload: json!({
                        "kind": format!("{:?}", report.kind),
                        "message": report.message,
                    }),
                }),
            };
            return rig_core::serve::Reply::Outcome(outcome);
        }
        let outcome = match first {
            Poll::Ready(Err(report)) => Ok(Outcome::Custom {
                payload: json!({
                    "kind": format!("{:?}", report.kind),
                    "message": report.message,
                }),
            }),
            Poll::Ready(Ok(outcome)) => Ok(outcome),
            Poll::Pending => Ok(Outcome::Custom {
                payload: json!("accepted"),
            }),
        };
        rig_core::serve::Reply::Outcome(outcome)
    }
}

/// Poll `pending` and the driver by turns until the dispatch resolves;
/// `None` when sixteen rounds were not enough.
fn drive_to_outcome(
    driver: &mut BusDriver,
    pending: &mut super::Pending,
) -> Option<Result<Outcome, ErrorReport>> {
    let mut cx = Context::from_waker(noop_waker_ref());
    for _ in 0..16 {
        if let Poll::Ready(result) = pending.poll_unpin(&mut cx) {
            return Some(result);
        }
        let _ = driver.poll_unpin(&mut cx);
    }
    None
}

#[test]
fn a_dispatch_made_through_dispatch_context_carries_its_parent() {
    let (dispatcher, _registrar, mut driver) = Bus::channel();
    let (echo, served) = Echo::new();
    driver.register("echo", echo).expect("register");
    let held = Arc::new(Mutex::new(Vec::new()));
    driver
        .register(
            "parent",
            Parent {
                key: HandlerKey::from("parent"),
                child: HandlerKey::from("echo"),
                from_another_thread: false,
                held: Some(held.clone()),
                await_child: false,
            },
        )
        .expect("register");
    let mut cx = Context::from_waker(noop_waker_ref());
    let mut outer = dispatcher.dispatch(&HandlerKey::from("parent"), custom(json!("outer")));
    assert_eq!(outer.parent(), None, "a consumer's dispatch has no parent");
    assert!(outer.poll_unpin(&mut cx).is_pending());
    let _ = driver.poll_unpin(&mut cx);
    let mut child = held
        .lock()
        .expect("held")
        .pop()
        .expect("the child was parked");
    assert_eq!(
        child.parent(),
        Some(outer.id()),
        "the nested dispatch names the dispatch it was made from"
    );
    // The child is served while its parent is in flight.
    let outcome = drive_to_outcome(&mut driver, &mut child)
        .expect("resolved")
        .expect("served");
    assert!(
        matches!(&outcome, Outcome::Custom { payload } if *payload == json!("nested")),
        "{outcome:?}"
    );
    assert_eq!(served.load(Ordering::SeqCst), 1);
    assert_eq!(driver.in_flight(), 1, "the parent");
    drop(outer);
    let _ = driver.poll_unpin(&mut cx);
    assert_eq!(driver.in_flight(), 0);
    let stream = dispatcher.dispatch_stream(&HandlerKey::from("echo"), custom(json!(1)));
    assert_eq!(stream.parent(), None);
}

// ---------------------------------------------------------------------------
// Layers on the bus: a suspending layer keeps its serial slot and observes
// cancellation; the world side's channel closing is the layer's failure.

/// An approval gate: `before` sends the dispatch to the "world" and waits
/// for its decision on a oneshot.
struct Approval {
    asks: std::sync::mpsc::Sender<(EffectId, oneshot::Sender<rig_core::serve::Decision>)>,
}

impl rig_core::serve::Intercept for Approval {
    fn name(&self) -> String {
        "approval".to_owned()
    }

    async fn before(&self, id: EffectId, _kind: &EffectKind) -> rig_core::serve::Decision {
        let (decide, decided) = oneshot::channel();
        self.asks.send((id, decide)).expect("the world listens");
        match decided.await {
            Ok(decision) => decision,
            Err(oneshot::Canceled) => rig_core::serve::Decision::Deny(ErrorReport::new(
                ErrorKind::Internal,
                "layer `approval`: the world closed the answer channel without deciding",
            )),
        }
    }

    async fn after(
        &self,
        _id: EffectId,
        _kind: &EffectKind,
        _outcome: &Result<Outcome, ErrorReport>,
    ) -> rig_core::serve::Verdict {
        rig_core::serve::Verdict::Keep
    }
}

use rig_core::effect::EffectId;

#[test]
fn a_suspending_layer_whose_world_closes_the_channel_resolves_internal_naming_the_layer() {
    let (dispatcher, _registrar, mut driver) = Bus::channel();
    let (asks, world) = std::sync::mpsc::channel();
    let (echo, served) = Echo::new();
    driver
        .register_erased(
            HandlerKey::from("echo"),
            rig_core::serve::ErasedHandler::new(echo).layered(Approval { asks }),
        )
        .expect("register");
    let mut cx = Context::from_waker(noop_waker_ref());
    let mut pending = dispatcher.dispatch(&HandlerKey::from("echo"), custom(json!(1)));
    assert!(pending.poll_unpin(&mut cx).is_pending());
    let _ = driver.poll_unpin(&mut cx);
    let (_, decide) = world.try_recv().expect("asked");
    drop(decide);
    let report = drive_to_outcome(&mut driver, &mut pending)
        .expect("resolved")
        .expect_err("the world never decided");
    assert_eq!(report.kind, ErrorKind::Internal);
    assert!(
        report.message.contains("layer `approval`"),
        "{}",
        report.message
    );
    assert_eq!(served.load(Ordering::SeqCst), 0);
}

#[test]
fn cancellation_reaches_grandchild_after_middle_dispatch_completes() {
    use super::dispatcher::Shared;
    use rig_core::effect::EffectId;
    let shared = Shared::new(ServingPolicy::default());
    let grandparent = EffectId::from_raw(1);
    let middle = EffectId::from_raw(2);
    let child = EffectId::from_raw(3);
    let _grandparent = shared
        .begin_in_flight(grandparent, HandlerKey::from("grandparent"), None)
        .ok()
        .expect("grandparent active");
    let _middle = shared
        .begin_in_flight(middle, HandlerKey::from("middle"), Some(grandparent))
        .ok()
        .expect("middle active");
    let child_cancel = shared
        .begin_in_flight(child, HandlerKey::from("child"), Some(middle))
        .ok()
        .expect("child active");
    assert!(!shared.end_in_flight(middle));
    shared.cancel_descendants(grandparent);
    assert!(
        child_cancel.is_set(),
        "completed intermediate dispatch must not sever cancellation ancestry"
    );
}

struct CaptureLineage {
    captured: Arc<Mutex<Option<Dispatcher>>>,
    complete: bool,
    detached: Option<Arc<Mutex<Vec<rig_core::serve::Resolver>>>>,
}
impl Serve for CaptureLineage {
    type Family = rig_core::effect::family::Dynamic;
    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: HandlerKey::from("capture"),
            family: FamilyDescriptor::Custom {
                kind: "capture".into(),
            },
            layers: Vec::new(),
        }
    }
    async fn serve(&self, _: EffectKind, dispatch: Dispatch) -> CoreReply {
        *self.captured.lock().unwrap() = Some(super::DispatchScope::dispatcher(&dispatch).unwrap());
        if let Some(detached) = &self.detached {
            let (resolver, answer) = rig_core::serve::deferred();
            detached.lock().unwrap().push(resolver);
            return CoreReply::Outcome(answer.await);
        }
        if !self.complete {
            futures::future::pending::<()>().await;
        }
        CoreReply::Outcome(Ok(Outcome::Custom {
            payload: json!("complete"),
        }))
    }
}

fn completed_middle_chain(policy: ServingPolicy) -> (Dispatcher, BusDriver, Pending, Dispatcher) {
    completed_middle_chain_with_resolver(policy, None)
}

fn completed_middle_chain_with_resolver(
    policy: ServingPolicy,
    detached: Option<Arc<Mutex<Vec<rig_core::serve::Resolver>>>>,
) -> (Dispatcher, BusDriver, Pending, Dispatcher) {
    let (root, _registrar, mut driver) = Bus::channel_with(policy);
    let outer_capture = Arc::new(Mutex::new(None));
    let middle_capture = Arc::new(Mutex::new(None));
    driver
        .register(
            "outer",
            CaptureLineage {
                captured: outer_capture.clone(),
                complete: false,
                detached,
            },
        )
        .unwrap();
    driver
        .register(
            "middle",
            CaptureLineage {
                captured: middle_capture.clone(),
                complete: true,
                detached: None,
            },
        )
        .unwrap();
    let mut outer = root.dispatch(&HandlerKey::from("outer"), custom(json!(null)));
    let mut cx = Context::from_waker(noop_waker_ref());
    assert!(outer.poll_unpin(&mut cx).is_pending());
    let _ = driver.poll_unpin(&mut cx);
    let parent = outer_capture.lock().unwrap().take().unwrap();
    let mut middle = parent.dispatch(&HandlerKey::from("middle"), custom(json!(null)));
    drive_to_outcome(&mut driver, &mut middle).unwrap().unwrap();
    let retained = middle_capture.lock().unwrap().take().unwrap();
    assert_eq!(retained.parent(), Some(middle.id()));
    assert_eq!(
        driver.in_flight(),
        1,
        "middle completed, outer remains active"
    );
    (root, driver, outer, retained)
}

#[test]
fn retained_lineage_cancels_unpolled_buffered_active_and_serial_queued_children() {
    for stage in 0..4 {
        let (root, mut driver, outer, retained) = completed_middle_chain(ServingPolicy {
            serial_per_handler: true,
            ..ServingPolicy::default()
        });
        let (echo, _gate) = Echo::gated();
        driver.register("echo", echo).unwrap();
        let mut cx = Context::from_waker(noop_waker_ref());
        let _busy = if stage == 3 {
            let mut busy = root.dispatch(&HandlerKey::from("echo"), custom(json!("busy")));
            assert!(busy.poll_unpin(&mut cx).is_pending());
            let _ = driver.poll_unpin(&mut cx);
            Some(busy)
        } else {
            None
        };
        let mut child = retained.dispatch(&HandlerKey::from("echo"), custom(json!("child")));
        if stage > 0 {
            assert!(child.poll_unpin(&mut cx).is_pending());
        }
        if stage > 1 {
            let _ = driver.poll_unpin(&mut cx);
        }
        if stage == 3 {
            assert_eq!(driver.queued(), 1);
        }
        drop(outer);
        for _ in 0..4 {
            let _ = driver.poll_unpin(&mut cx);
        }
        let report = drive_to_outcome(&mut driver, &mut child)
            .unwrap()
            .unwrap_err();
        assert_eq!(report.kind, ErrorKind::Cancelled, "stage {stage}");
        assert_eq!(driver.queued(), 0);
        let mut later = retained
            .clone()
            .dispatch(&HandlerKey::from("echo"), custom(json!("later")));
        assert!(
            matches!(later.poll_unpin(&mut cx), Poll::Ready(Err(ref e)) if e.kind == ErrorKind::Cancelled)
        );
    }
}

#[test]
fn retained_lineage_cancels_parked_unary_and_stream_sends() {
    let (root, mut driver, outer, retained) = completed_middle_chain(ServingPolicy {
        command_capacity: 1,
        ..ServingPolicy::default()
    });
    let mut cx = Context::from_waker(noop_waker_ref());
    let mut filler = root.dispatch(&HandlerKey::from("missing"), custom(json!(null)));
    assert!(filler.poll_unpin(&mut cx).is_pending());
    let mut unary = retained.dispatch(&HandlerKey::from("missing"), custom(json!(null)));
    let mut stream = retained.dispatch_stream(
        &HandlerKey::from("missing"),
        EffectKind::Completion {
            request: CompletionRequest::new("hi"),
            stream: true,
        },
    );
    assert!(unary.poll_unpin(&mut cx).is_pending());
    assert!(stream.poll_next_unpin(&mut cx).is_pending());
    assert_eq!(root.shared.parked_senders(), 2);
    drop(outer);
    for _ in 0..4 {
        let _ = driver.poll_unpin(&mut cx);
    }
    assert!(
        matches!(unary.poll_unpin(&mut cx), Poll::Ready(Err(ref e)) if e.kind == ErrorKind::Cancelled)
    );
    assert!(
        matches!(stream.poll_next_unpin(&mut cx), Poll::Ready(Some(Err(ref e))) if e.kind == ErrorKind::Cancelled)
    );
    assert!(stream.poll_next_unpin(&mut cx).is_ready());
}

/// Two memories and two indexes on one bus: each labelled adapter mints its
/// own key (`memory:<label>`, `retrieve:<label>`), so a host serves several
/// backends beside an agent's own bare `memory`/`retrieve` and dispatches
/// to each by key.
#[tokio::test]
async fn labelled_memory_and_retrieve_adapters_serve_side_by_side() {
    use rig_core::effect::{MemoryOp, MemoryOutcome, memory_key, retrieve_key};
    use rig_core::serve::adapters::{MemoryAdapter, RetrieveAdapter};
    use rig_core::test_utils::CountingMemory;
    let (dispatcher, _registrar, mut driver) = Bus::channel();
    let tenant_a = CountingMemory::default();
    let tenant_b = CountingMemory::default();
    driver
        .register("memory:a", MemoryAdapter::labelled("a", tenant_a.clone()))
        .expect("first memory");
    driver
        .register("memory:b", MemoryAdapter::labelled("b", tenant_b.clone()))
        .expect("second memory");
    driver
        .register(
            "retrieve:docs",
            RetrieveAdapter::labelled("docs", crate::test_utils::MockToolIndex::new(["t1"])),
        )
        .expect("first index");
    driver
        .register(
            "retrieve:code",
            RetrieveAdapter::labelled("code", crate::test_utils::MockToolIndex::new(["t2"])),
        )
        .expect("second index");
    assert_eq!(memory_key("a"), HandlerKey::from("memory:a"));
    assert_eq!(retrieve_key("code"), HandlerKey::from("retrieve:code"));
    let _task = spawn(driver);

    let conversation = rig_core::id::ConversationId::from("c1");
    within(dispatcher.dispatch(
        &memory_key("a"),
        EffectKind::Memory {
            op: MemoryOp::Append {
                conversation: conversation.clone(),
                messages: vec![Message::user("only in a")],
            },
        },
    ))
    .await
    .expect("appended to a");
    for (label, expected) in [("a", 1), ("b", 0)] {
        let loaded = within(dispatcher.dispatch(
            &memory_key(label),
            EffectKind::Memory {
                op: MemoryOp::Load {
                    conversation: conversation.clone(),
                },
            },
        ))
        .await
        .expect("loaded");
        assert!(
            matches!(loaded, Outcome::Memory(MemoryOutcome::Loaded { ref messages }) if messages.len() == expected),
            "memory:{label} holds its own conversation: {loaded:?}"
        );
    }
    assert_eq!(tenant_a.load_count(), 1);
    assert_eq!(tenant_b.load_count(), 1);
}
