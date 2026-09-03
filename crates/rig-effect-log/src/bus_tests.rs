//! Record and replay over a live bus: the recorder captures what the
//! driver serves, the replayer answers it back. Moved here from the bus's
//! own tests with the log crate; the helpers are copies of the bus's.

use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use futures::{StreamExt, channel::oneshot};

use serde_json::json;

use rig_bus::{Bus, BusDriver};

use super::{EFFECT_LOG_FORMAT, EffectLog, EffectLogRecorder, EffectLogReplayer};

use rig_core::serve::{
    OutcomeSink, Serve,
    adapters::{CompletionAdapter, ToolAdapter},
};

use rig_core::{
    completion::{CompletionRequest, Message},
    effect::{EffectFamily, EffectKind, FamilyDescriptor, HandlerDescriptor, HandlerKey, Outcome},
    error::{ErrorKind, ErrorReport},
    message::AssistantContent,
    streaming::StreamEvent,
    test_utils::{MockCompletionModel, MockStreamEvent},
    tool::{Tool, ToolContext, ToolExecutionError},
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

fn spawn(driver: BusDriver) -> tokio::task::JoinHandle<()> {
    tokio::spawn(driver)
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

#[derive(serde::Deserialize)]
struct AddArgs {
    a: i64,
    b: i64,
}

struct Add;

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

/// Sends one text event and drops its sink before the terminal.
struct CutShort;

impl Serve for CutShort {
    type Family = rig_core::effect::family::Dynamic;

    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: HandlerKey::from("cut"),
            family: FamilyDescriptor::Completion {
                model: rig_core::completion::ModelRef::new("cut"),
                capabilities: rig_core::completion::ProviderCapabilities::default(),
            },
        }
    }

    async fn serve(&self, _kind: EffectKind, sink: OutcomeSink) {
        let mut out = sink.writer();
        let _ = out.text("partial").await;
        // Dropped here, before any `Final`.
    }
}

/// Answers a stream dispatch with a non-completion outcome.
struct WrongFamilyAnswer;

impl Serve for WrongFamilyAnswer {
    type Family = rig_core::effect::family::Dynamic;

    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: HandlerKey::from("wrong"),
            family: FamilyDescriptor::Completion {
                model: rig_core::completion::ModelRef::new("wrong"),
                capabilities: rig_core::completion::ProviderCapabilities::default(),
            },
        }
    }

    async fn serve(&self, _kind: EffectKind, sink: OutcomeSink) {
        sink.resolve(Ok(Outcome::Custom(json!("not a completion"))))
            .await;
    }
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
    let restored: EffectLog = serde_json::from_str(&json).expect("log restores");
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
    assert_eq!(report.kind, ErrorKind::Divergence);
    assert!(report.message.contains("replay divergence"));
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
    assert_eq!(report.kind, ErrorKind::Divergence);
    assert!(report.message.contains("replay divergence"), "{report:?}");
    assert!(report.message.contains("differs"), "{report:?}");
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

#[tokio::test]
async fn a_log_carries_its_header_and_a_replay_checks_it() {
    let (dispatcher, _registrar, mut driver) = Bus::channel();
    let (echo, _) = Echo::new();
    driver.register("echo", echo).expect("register");
    let recorder = EffectLogRecorder::new();
    driver.record_to(recorder.clone());
    let _task = spawn(driver);
    within(dispatcher.dispatch(&HandlerKey::from("echo"), custom(json!({"n": 1}))))
        .await
        .expect("served");
    let log = recorder.take();
    assert_eq!(log.header.format, EFFECT_LOG_FORMAT);
    assert_eq!(log.header.run_spec, None, "a bare-bus record names no spec");
    assert_eq!(
        log.header
            .handlers
            .iter()
            .map(|descriptor| descriptor.key.clone())
            .collect::<Vec<_>>(),
        [HandlerKey::from("echo")],
        "the handlers registered when recording began"
    );
    assert_eq!(
        log.header.signature.get(&HandlerKey::from("echo")),
        Some(&EffectFamily::Custom),
        "the effect signature, read off the trace"
    );
    let json = serde_json::to_value(&log).expect("serializes");
    assert!(json.get("header").is_some() && json.get("records").is_some());
    let back: EffectLog = serde_json::from_value(json).expect("restores");
    assert_eq!(back.header, log.header);
    assert_eq!(back.len(), 1);

    // A future format is refused with the version in the message.
    let mut future = back.clone();
    future.header.format = EFFECT_LOG_FORMAT + 1;
    let (_dispatcher, _registrar, mut driver) = Bus::channel();
    let report = EffectLogReplayer::register_all(&future, &mut driver).expect_err("refused");
    assert!(
        report
            .message
            .contains(&format!("format {}", EFFECT_LOG_FORMAT + 1)),
        "{}",
        report.message
    );

    // A signature that names a family the records do not answer is refused
    // at registration, not at the first dispatch.
    let mut lying = back.clone();
    lying
        .header
        .signature
        .insert(HandlerKey::from("echo"), EffectFamily::Completion);
    let (_dispatcher, _registrar, mut driver) = Bus::channel();
    let report = EffectLogReplayer::register_all(&lying, &mut driver).expect_err("refused");
    assert!(
        report
            .message
            .contains("signature says `echo` serves completion"),
        "{}",
        report.message
    );
    assert!(
        driver.registrar().keys().is_empty(),
        "nothing was registered"
    );
}

#[tokio::test]
async fn a_stream_recorded_verbatim_replays_its_own_events() {
    let events = || {
        vec![
            MockStreamEvent::text("hel"),
            MockStreamEvent::text("lo"),
            MockStreamEvent::final_response_with_default_usage(),
        ]
    };
    let record = |recorder: EffectLogRecorder| async move {
        let (dispatcher, _registrar, mut driver) = Bus::channel();
        driver
            .register(
                "model",
                CompletionAdapter::new("mock", MockCompletionModel::from_stream_turns([events()])),
            )
            .expect("register");
        driver.record_to(recorder.clone());
        let _task = spawn(driver);
        let mut stream =
            dispatcher.dispatch_stream(&HandlerKey::from("model"), completion_kind(true));
        let mut seen = 0;
        while let Some(item) = within(stream.next()).await {
            item.expect("clean");
            seen += 1;
        }
        (seen, recorder.take())
    };

    // The default: the fold, no events.
    let (_, folded) = record(EffectLogRecorder::new()).await;
    assert!(folded[0].events.is_none());
    assert!(matches!(folded[0].outcome, Ok(Outcome::Completion(_))));

    // Keeping events: the record holds them verbatim.
    let (seen, kept) = record(EffectLogRecorder::keeping_stream_events()).await;
    let recorded = kept[0].events.as_ref().expect("events kept");
    assert_eq!(recorded.len(), seen, "every event the consumer received");
    assert!(
        matches!(kept[0].outcome, Ok(Outcome::Completion(_))),
        "and still the fold"
    );

    // A replay of the kept record re-emits the events, delta boundaries and
    // all; a replay of the folded record re-emits the fold.
    let replay = |log: EffectLog| async move {
        let (dispatcher, _registrar, mut driver) = Bus::channel();
        EffectLogReplayer::register_all(&log, &mut driver).expect("fresh keys");
        let _task = spawn(driver);
        let mut stream =
            dispatcher.dispatch_stream(&HandlerKey::from("model"), completion_kind(true));
        let mut items = Vec::new();
        while let Some(item) = within(stream.next()).await {
            items.push(item.expect("clean"));
        }
        items
    };
    let recorded = recorded.clone();
    let from_kept = replay(kept).await;
    assert_eq!(from_kept, recorded, "the recorded events, verbatim");
    let from_fold = replay(folded).await;
    assert_ne!(
        from_fold, recorded,
        "the fold re-emits its own boundaries, not the original deltas"
    );
}

/// Holds its dispatch open until dropped.
struct Held;

impl Serve for Held {
    type Family = rig_core::effect::family::Dynamic;

    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: HandlerKey::from("held"),
            family: FamilyDescriptor::Custom {
                kind: "test:held".into(),
            },
        }
    }

    async fn serve(&self, _kind: EffectKind, sink: OutcomeSink) {
        let _sink = sink;
        futures::future::pending::<()>().await;
    }
}

#[tokio::test]
async fn a_dispatch_cancelled_in_flight_is_recorded_as_cancelled_and_replays_as_such() {
    use futures::FutureExt;
    let recorder = EffectLogRecorder::new();
    let (dispatcher, _registrar, mut driver) = Bus::channel();
    driver.register("held", Held).expect("register");
    driver.record_to(recorder.clone());
    let waker = futures::task::noop_waker_ref();
    let mut cx = std::task::Context::from_waker(waker);
    let kind = EffectKind::Custom {
        kind: Arc::from("test:held"),
        payload: json!({"n": 1}),
    };
    // Unary: in flight, then the consumer drops its `Pending`.
    let mut pending = dispatcher.dispatch(&HandlerKey::from("held"), kind.clone());
    assert!(pending.poll_unpin(&mut cx).is_pending());
    let _ = driver.poll_unpin(&mut cx);
    assert_eq!(driver.in_flight(), 1);
    drop(pending);
    let _ = driver.poll_unpin(&mut cx);
    assert_eq!(driver.in_flight(), 0);
    // Stream: the same, through an `EffectStream`.
    let mut stream = dispatcher.dispatch_stream(&HandlerKey::from("held"), completion_kind(true));
    assert!(stream.poll_next_unpin(&mut cx).is_pending());
    let _ = driver.poll_unpin(&mut cx);
    assert_eq!(driver.in_flight(), 1);
    drop(stream);
    let _ = driver.poll_unpin(&mut cx);
    assert_eq!(driver.in_flight(), 0);

    let log = recorder.take();
    assert_eq!(log.len(), 2);
    for record in log.iter() {
        let report = record
            .outcome
            .as_ref()
            .expect_err("a cancelled dispatch is recorded as a failure");
        assert_eq!(report.kind, ErrorKind::Cancelled, "{report:?}");
        assert!(!report.retryable);
    }

    // A replay answers the cancel as the cancel it was, not as a handler
    // or provider failure.
    let (dispatcher, _registrar, mut driver) = Bus::channel();
    EffectLogReplayer::register_all(&log, &mut driver).expect("fresh keys");
    let _task = spawn(driver);
    let report = within(dispatcher.dispatch(&HandlerKey::from("held"), kind))
        .await
        .expect_err("replayed as the recorded failure");
    assert_eq!(report.kind, ErrorKind::Cancelled, "{report:?}");
}

/// A tool call's dispatch context is part of the effect: the same name and
/// arguments under a different context is a divergence, named by path.
#[tokio::test]
async fn a_tool_call_under_a_different_context_is_a_divergence() {
    use super::LogHeader;
    use rig_core::{
        effect::{EffectId, EffectRecord},
        tool::{ContextValue, ToolOutput, ToolResult},
    };
    #[derive(serde::Serialize, serde::Deserialize)]
    struct Tag(String);
    impl ContextValue for Tag {
        const KEY: &'static str = "tag";
    }
    let key = HandlerKey::from("tool:echo");
    let mut recorded_context = ToolContext::new();
    recorded_context
        .insert(Tag("recorded".into()))
        .expect("context value");
    let log: EffectLog = EffectLog {
        header: LogHeader::default(),
        records: vec![EffectRecord {
            id: EffectId::from_raw(1),
            key: key.clone(),
            kind: EffectKind::ToolCall {
                name: "echo".into(),
                args: "{}".into(),
                context: recorded_context,
            },
            outcome: Ok(Outcome::ToolResult {
                result: ToolResult::success(ToolOutput::text("ok")),
                context: ToolContext::new(),
            }),
            events: None,
        }],
    };
    let (dispatcher, _registrar, mut driver) = Bus::channel();
    EffectLogReplayer::register_all(&log, &mut driver).expect("fresh keys");
    let _task = spawn(driver);
    let mut other_context = ToolContext::new();
    other_context
        .insert(Tag("arrived".into()))
        .expect("context value");
    let report = within(dispatcher.dispatch(
        &key,
        EffectKind::ToolCall {
            name: "echo".into(),
            args: "{}".into(),
            context: other_context,
        },
    ))
    .await
    .expect_err("a different context is a different effect");
    assert_eq!(report.kind, ErrorKind::Divergence);
    assert!(report.message.contains("context"), "{report:?}");
}
