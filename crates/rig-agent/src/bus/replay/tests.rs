//! Record and replay over a live bus: the recorder captures what the
//! driver serves, the replayer answers it back. Written in rig-effect-log
//! when the bus was its own crate; moved here with `register_all`.

use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use futures::{StreamExt, channel::oneshot};

use serde_json::json;

use crate::bus::{Bus, BusDriver};

use rig_effect_log::{EffectLog, EffectLogRecorder};

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

#[tokio::test]
async fn kept_stream_replay_preserves_every_error_item_and_its_position() {
    struct Items(Vec<Result<StreamEvent, ErrorReport>>);
    impl Serve for Items {
        type Family = rig_core::effect::family::Completion;
        fn descriptor(&self) -> HandlerDescriptor {
            HandlerDescriptor {
                key: "model".into(),
                family: FamilyDescriptor::Completion {
                    model: rig_core::completion::ModelRef::new("errors"),
                    capabilities: rig_core::completion::ProviderCapabilities::default(),
                },
                layers: vec![],
            }
        }
        async fn serve(&self, _: EffectKind, mut sink: OutcomeSink) {
            for item in &self.0 {
                sink.send(item.clone()).await.unwrap();
            }
        }
    }
    for error_first in [false, true] {
        let error = Err(ErrorReport::new(ErrorKind::Response, "first error"));
        let terminal = Ok(StreamEvent::Final(rig_core::streaming::StreamFinal::new(
            "test",
            rig_core::completion::Usage::new(),
        )));
        let mut items = if error_first {
            vec![error, terminal]
        } else {
            vec![terminal, error]
        };
        items.push(Err(ErrorReport::new(ErrorKind::Provider, "late error")));
        let key = HandlerKey::from("model");
        let (dispatcher, _, mut driver) = Bus::channel();
        let recorder = EffectLogRecorder::keeping_stream_events();
        driver.register(key.clone(), Items(items)).unwrap();
        driver.record_to(recorder.clone());
        let _live = spawn(driver);
        let live = within(
            dispatcher
                .dispatch_stream(&key, completion_kind(true))
                .collect::<Vec<_>>(),
        )
        .await;
        let log: EffectLog =
            serde_json::from_str(&serde_json::to_string(&recorder.log()).unwrap()).unwrap();
        assert_eq!(log.header.stream_errors.values().next().unwrap().len(), 2);
        let (dispatcher, _, mut driver) = Bus::channel();
        super::register_all(&log, &mut driver).unwrap();
        let _replay = spawn(driver);
        let replay = within(
            dispatcher
                .dispatch_stream(&key, completion_kind(true))
                .collect::<Vec<_>>(),
        )
        .await;
        assert_eq!(
            serde_json::to_value(replay).unwrap(),
            serde_json::to_value(live).unwrap()
        );
    }
}

#[tokio::test]
async fn replayed_model_handle_retains_live_capabilities_and_model_identity() {
    struct Composing;
    impl Serve for Composing {
        type Family = rig_core::effect::family::Completion;
        fn descriptor(&self) -> HandlerDescriptor {
            HandlerDescriptor {
                key: HandlerKey::from("model"),
                family: FamilyDescriptor::Completion {
                    model: rig_core::completion::ModelRef::new("composing"),
                    capabilities: rig_core::completion::ProviderCapabilities::new()
                        .with_native_output_tool_composition(true),
                },
                layers: vec![],
            }
        }
        async fn serve(&self, _kind: EffectKind, sink: OutcomeSink) {
            sink.resolve(Ok(Outcome::Completion(
                rig_core::completion::CompletionResponse::new(
                    vec![AssistantContent::text("ok")],
                    rig_core::completion::Usage::new(),
                    "composing",
                ),
            )))
            .await;
        }
    }
    let (dispatcher, _, mut driver) = Bus::channel();
    driver.register("model", Composing).unwrap();
    let recorder = EffectLogRecorder::new();
    driver.record_to(recorder.clone());
    let live_model: crate::bus::ModelHandle =
        dispatcher.handle(&HandlerKey::from("model")).unwrap();
    let _live = spawn(driver);
    within(dispatcher.dispatch(&HandlerKey::from("model"), completion_kind(false)))
        .await
        .unwrap();
    let log: EffectLog =
        serde_json::from_str(&serde_json::to_string(&recorder.log()).unwrap()).unwrap();
    let (dispatcher, _, mut driver) = Bus::channel();
    super::register_all(&log, &mut driver).unwrap();
    let replay_model: crate::bus::ModelHandle =
        dispatcher.handle(&HandlerKey::from("model")).unwrap();
    assert_eq!(replay_model.capabilities(), live_model.capabilities());
    assert_eq!(replay_model.model_ref(), live_model.model_ref());
    assert!(
        replay_model
            .capabilities()
            .composes_native_output_with_tools
    );
    let _replay = spawn(driver);
    assert!(
        within(dispatcher.dispatch(&HandlerKey::from("model"), completion_kind(false)))
            .await
            .is_ok()
    );
}

/// Raw tool dispatches also get a publication slot; recording must not
/// consume it before the pending caller can read it.
#[tokio::test]
async fn raw_tool_dispatch_records_and_replays_published_output() {
    use rig_core::tool::{ContextValue, PublishedContext, ToolOutput, ToolResult};
    #[derive(Debug, PartialEq, serde::Serialize, serde::Deserialize)]
    struct Artifact(String);
    impl ContextValue for Artifact {
        const KEY: &'static str = "artifact.v1";
    }
    struct Publisher;
    impl Serve for Publisher {
        type Family = rig_core::effect::family::Tool;
        fn descriptor(&self) -> HandlerDescriptor {
            HandlerDescriptor {
                key: HandlerKey::from("tool:publish"),
                family: FamilyDescriptor::Tool {
                    name: "publish".into(),
                    description: "artifact".into(),
                    parameters: json!({}),
                    embedding: None,
                },
                layers: vec![],
            }
        }
        async fn serve(&self, _: EffectKind, sink: OutcomeSink) {
            let mut context = ToolContext::new();
            context
                .insert_result(Artifact("result-123".into()))
                .expect("encodes");
            sink.scope::<PublishedContext>()
                .expect("raw dispatch has a slot")
                .publish(context);
            sink.resolve(Ok(Outcome::ToolResult {
                result: ToolResult::success(ToolOutput::text("ok")),
            }))
            .await;
        }
    }
    let key = HandlerKey::from("tool:publish");
    let kind = EffectKind::ToolCall {
        name: "publish".into(),
        args: "{}".into(),
    };
    let recorder = EffectLogRecorder::new();
    let (dispatcher, _registrar, mut driver) = Bus::channel();
    driver.register(key.clone(), Publisher).expect("fresh");
    driver.record_to(recorder.clone());
    let _task = spawn(driver);
    let pending = dispatcher.dispatch(&key, kind.clone());
    let published = pending.published_context().expect("slot");
    let original = within(pending).await.expect("answer");
    assert_eq!(
        published
            .take()
            .expect("caller retains output")
            .result::<Artifact>()
            .expect("decodes"),
        Some(Artifact("result-123".into()))
    );
    let json = serde_json::to_string(&recorder.log()).expect("serializes");
    let log = serde_json::from_str(&json).expect("deserializes");
    let (dispatcher, _registrar, mut driver) = Bus::channel();
    super::register_all(&log, &mut driver).expect("replayers");
    let _task = spawn(driver);
    let pending = dispatcher.dispatch(&key, kind);
    let published = pending.published_context().expect("slot");
    let replayed = within(pending).await.expect("answer");
    assert_eq!(
        serde_json::to_value(original).expect("serializes"),
        serde_json::to_value(replayed).expect("serializes")
    );
    assert_eq!(
        published
            .take()
            .expect("replayed output")
            .result::<Artifact>()
            .expect("decodes"),
        Some(Artifact("result-123".into()))
    );
}

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
            layers: Vec::new(),
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
                EffectKind::Custom { payload, .. } => Ok(Outcome::Custom { payload }),
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
            layers: Vec::new(),
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
        sink.resolve(Ok(Outcome::Custom {
            payload: json!(index),
        }))
        .await;
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
            layers: Vec::new(),
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
            layers: Vec::new(),
        }
    }

    async fn serve(&self, _kind: EffectKind, sink: OutcomeSink) {
        sink.resolve(Ok(Outcome::Custom {
            payload: json!("not a completion"),
        }))
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
    super::register_all(&restored, &mut driver).expect("fresh keys");
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
    assert!(matches!(slow, Ok(Outcome::Custom { payload: ref v }) if *v == json!(1)));
    assert!(matches!(fast, Ok(Outcome::Custom { payload: ref v }) if *v == json!(2)));
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
    super::register_all(&log, &mut replay_driver).expect("fresh keys");
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
        matches!(first, Outcome::Custom { payload: ref v } if *v == json!(1)),
        "{first:?}"
    );
    assert!(
        matches!(second, Outcome::Custom { payload: ref v } if *v == json!(2)),
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
    super::register_all(&log, &mut replay_driver).expect("fresh keys");
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
    assert!(json["header"].get("format").is_none());
    let back: EffectLog = serde_json::from_value(json).expect("restores");
    assert_eq!(back.header, log.header);
    assert_eq!(back.len(), 1);

    // A signature that names a family the records do not answer is refused
    // at registration, not at the first dispatch.
    let mut lying = back.clone();
    lying
        .header
        .signature
        .insert(HandlerKey::from("echo"), EffectFamily::Completion);
    let (_dispatcher, _registrar, mut driver) = Bus::channel();
    let report = super::register_all(&lying, &mut driver).expect_err("refused");
    assert!(
        report
            .message
            .contains("conflicting families for `echo`: completion and custom"),
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
        super::register_all(&log, &mut driver).expect("fresh keys");
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
struct Held(FamilyDescriptor);

impl Serve for Held {
    type Family = rig_core::effect::family::Dynamic;

    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: HandlerKey::from("held"),
            family: self.0.clone(),
            layers: Vec::new(),
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
    driver
        .register(
            "held",
            Held(FamilyDescriptor::Custom {
                kind: "test:held".into(),
            }),
        )
        .expect("register");
    driver
        .register(
            "held-stream",
            Held(FamilyDescriptor::Completion {
                model: rig_core::completion::ModelRef::new("held"),
                capabilities: rig_core::completion::ProviderCapabilities::default(),
            }),
        )
        .expect("register stream");
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
    // Stream: the same, through a key described as a completion handler.
    let mut stream =
        dispatcher.dispatch_stream(&HandlerKey::from("held-stream"), completion_kind(true));
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
    super::register_all(&log, &mut driver).expect("fresh keys");
    let _task = spawn(driver);
    let report = within(dispatcher.dispatch(&HandlerKey::from("held"), kind))
        .await
        .expect_err("replayed as the recorded failure");
    assert_eq!(report.kind, ErrorKind::Cancelled, "{report:?}");
    let mut stream =
        dispatcher.dispatch_stream(&HandlerKey::from("held-stream"), completion_kind(true));
    let report = within(stream.next())
        .await
        .expect("terminal item")
        .expect_err("recorded stream cancellation");
    assert_eq!(report.kind, ErrorKind::Cancelled);
}

/// A tool call's dispatch context is not part of the effect (format 5: it
/// travels beside the sink, never on the wire): the same name and
/// arguments under a different context is the same record, and the
/// replayer answers it.
#[tokio::test]
async fn a_tool_call_under_a_different_context_is_the_same_record() {
    use rig_core::{
        effect::{EffectId, EffectRecord},
        tool::{ContextValue, ToolOutput, ToolResult},
    };
    use rig_effect_log::LogHeader;
    #[derive(serde::Serialize, serde::Deserialize)]
    struct Tag(String);
    impl ContextValue for Tag {
        const KEY: &'static str = "tag";
    }
    let key = HandlerKey::from("tool:echo");
    let log: EffectLog = EffectLog {
        header: LogHeader::default(),
        records: vec![EffectRecord {
            tool_output: None,
            parent: None,
            scope: None,
            id: EffectId::from_raw(1),
            key: key.clone(),
            kind: EffectKind::ToolCall {
                name: "echo".into(),
                args: "{}".into(),
            },
            outcome: Ok(Outcome::ToolResult {
                result: ToolResult::success(ToolOutput::text("ok")),
            }),
            events: None,
        }],
    };
    let (dispatcher, _registrar, mut driver) = Bus::channel();
    super::register_all(&log, &mut driver).expect("fresh keys");
    let _task = spawn(driver);
    let mut other_context = ToolContext::new();
    other_context
        .insert(Tag("arrived".into()))
        .expect("context value");
    let outcome = within(dispatcher.dispatch_tool_with_id(
        dispatcher.mint_id(),
        &key,
        EffectKind::ToolCall {
            name: "echo".into(),
            args: "{}".into(),
        },
        other_context,
    ))
    .await
    .expect("the context is not the effect: the record answers");
    match outcome {
        Outcome::ToolResult { result } => assert_eq!(result.output().as_text(), Some("ok")),
        other => panic!("a tool result: {other:?}"),
    }
}

/// A stream recorded with its events that ended in an error — the
/// consumer's cancel, the provider's refusal — replays its events and then
/// the error, which is the record's outcome. (It used to end after the
/// events, and the consumer saw "the stream ended before its terminal
/// record" instead of the cancel or the refusal it recorded.)
#[tokio::test]
async fn a_streamed_error_record_replays_its_events_and_then_its_error() {
    // A stream recorded verbatim, then cut: its first event kept, its
    // outcome the cancel a dropped consumer records.
    let recorder = EffectLogRecorder::keeping_stream_events();
    let (dispatcher, _registrar, mut driver) = Bus::channel();
    driver
        .register(
            "model",
            CompletionAdapter::new(
                "mock",
                MockCompletionModel::from_stream_turns([vec![
                    MockStreamEvent::text("hel"),
                    MockStreamEvent::text("lo"),
                    MockStreamEvent::final_response_with_default_usage(),
                ]]),
            ),
        )
        .expect("register");
    driver.record_to(recorder.clone());
    let _task = spawn(driver);
    let mut stream = dispatcher.dispatch_stream(&HandlerKey::from("model"), completion_kind(true));
    while let Some(item) = within(stream.next()).await {
        item.expect("clean");
    }
    drop(stream);
    let mut log = recorder.take();
    let events = log.records[0].events.as_mut().expect("events kept");
    events.truncate(1);
    let first = events[0].clone();
    log.records[0].outcome = Err(ErrorReport::new(ErrorKind::Cancelled, "dropped mid-stream"));

    let (dispatcher, _registrar, mut driver) = Bus::channel();
    super::register_all(&log, &mut driver).expect("fresh keys");
    let _task = spawn(driver);
    let mut stream = dispatcher.dispatch_stream(&HandlerKey::from("model"), completion_kind(true));
    let replayed = within(stream.next()).await.expect("the recorded event");
    assert_eq!(replayed.expect("the event"), first);
    let then = within(stream.next()).await.expect("then the error");
    let report = then.expect_err("the record's outcome");
    assert_eq!(report.kind, ErrorKind::Cancelled, "{report:?}");
    assert!(
        within(stream.next()).await.is_none(),
        "and nothing after it"
    );
}

/// A custom effect whose `Serialize` fails.
#[derive(Debug, serde::Deserialize)]
struct Unserializable;

impl serde::Serialize for Unserializable {
    fn serialize<S: serde::Serializer>(&self, _serializer: S) -> Result<S::Ok, S::Error> {
        Err(serde::ser::Error::custom("no wire form"))
    }
}

impl rig_core::effect::CustomEffect for Unserializable {
    const KIND: &'static str = "test:echo";
    type Answer = serde_json::Value;
}

/// A custom effect that does not serialize never reaches a handler: the
/// typed dispatch resolves `Request` with the serde message before any
/// send, the handler's counter stays at zero, and the log holds no record.
#[tokio::test]
async fn a_custom_effect_that_does_not_serialize_never_reaches_a_handler() {
    let (dispatcher, _registrar, mut driver) = Bus::channel();
    let (echo, served) = Echo::new();
    driver.register("echo", echo).expect("a fresh key");
    let recorder = EffectLogRecorder::new();
    driver.record_to(recorder.clone());
    let driver = spawn(driver);
    let key: rig_core::effect::Key<rig_core::effect::family::Custom<Unserializable>> =
        rig_core::effect::Key::new_unchecked(HandlerKey::from("echo"));
    let handle = dispatcher.bind(&key).expect("bound by family");
    let report = within(handle.dispatch(Unserializable))
        .await
        .expect_err("no wire form");
    assert_eq!(report.kind, ErrorKind::Request);
    assert!(report.message.contains("no wire form"), "{report:?}");
    assert_eq!(served.load(Ordering::SeqCst), 0, "the handler never served");
    drop((handle, dispatcher));
    within(driver).await.expect("the driver finishes");
    let log = recorder.take();
    assert!(log.records.is_empty(), "no record: {:?}", log.records);
}

/// Dispatches to `echo` through its sink's dispatcher and answers with the
/// child's outcome.
struct Nesting;

impl Serve for Nesting {
    type Family = rig_core::effect::family::Dynamic;

    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: HandlerKey::from("nesting"),
            family: FamilyDescriptor::Custom {
                kind: "test:nesting".into(),
            },
            layers: Vec::new(),
        }
    }

    async fn serve(&self, _kind: EffectKind, sink: OutcomeSink) {
        let dispatcher =
            crate::bus::SinkDispatch::dispatcher(&sink).expect("served by a bus driver");
        let child = dispatcher
            .dispatch(&HandlerKey::from("echo"), custom(json!({"who": "child"})))
            .await;
        sink.resolve(child).await;
    }
}

#[tokio::test]
async fn a_record_names_the_dispatch_it_was_made_from() {
    let recorder = EffectLogRecorder::new();
    let (dispatcher, _registrar, mut driver) = Bus::channel();
    let (echo, _) = Echo::new();
    driver.register("echo", echo).expect("register");
    driver.register("nesting", Nesting).expect("register");
    driver.record_to(recorder.clone());
    let _task = spawn(driver);
    let outer = dispatcher.dispatch(&HandlerKey::from("nesting"), custom(json!("outer")));
    let outer_id = outer.id();
    let outcome = within(outer).await.expect("served");
    assert!(matches!(&outcome, Outcome::Custom { payload } if *payload == json!({"who": "child"})));
    // A consumer's own dispatch, for the contrast.
    within(dispatcher.dispatch(&HandlerKey::from("echo"), custom(json!({"who": "own"}))))
        .await
        .expect("served");
    let log = recorder.take();
    assert_eq!(log.len(), 3, "the parent, its child, the consumer's own");
    assert_eq!(log[0].key.as_str(), "nesting");
    assert_eq!(log[0].parent, None, "a consumer's dispatch has no parent");
    assert_eq!(log[1].key.as_str(), "echo");
    assert_eq!(
        log[1].parent,
        Some(outer_id),
        "the child names the dispatch it was made from"
    );
    assert_eq!(log[2].parent, None);
    let json: serde_json::Value =
        serde_json::from_str(&serde_json::to_string(&log).expect("serializes")).expect("json");
    assert_eq!(
        json["records"][1]["parent"], json["records"][0]["id"],
        "the parent travels on the wire by id"
    );
    assert!(json["records"][0]["parent"].is_null());
}

#[tokio::test]
async fn a_record_names_the_scope_of_the_program_that_made_it() {
    // A scoped dispatcher stamps every dispatch made through it, and a
    // handler's nested dispatch inherits the scope of the dispatch it
    // serves; a plain dispatcher stamps nothing.
    let recorder = EffectLogRecorder::new();
    let (dispatcher, _registrar, mut driver) = Bus::channel();
    let (echo, _) = Echo::new();
    driver.register("echo", echo).expect("register");
    driver.register("nesting", Nesting).expect("register");
    driver.record_to(recorder.clone());
    let _task = spawn(driver);
    let run = dispatcher.scoped("run-1");
    assert_eq!(run.scope().map(|scope| &**scope), Some("run-1"));
    assert_eq!(dispatcher.scope(), None, "the original is unscoped");
    within(run.dispatch(
        &HandlerKey::from("nesting"),
        custom(json!({"who": "outer"})),
    ))
    .await
    .expect("served");
    within(dispatcher.dispatch(&HandlerKey::from("echo"), custom(json!({"who": "own"}))))
        .await
        .expect("served");
    let log = recorder.take();
    assert_eq!(log.len(), 3);
    assert_eq!(log[0].scope.as_deref(), Some("run-1"));
    assert_eq!(
        log[1].scope.as_deref(),
        Some("run-1"),
        "the nested dispatch inherits the scope of the one it descends from"
    );
    assert_eq!(log[1].parent, Some(log[0].id));
    assert_eq!(log[2].scope, None, "an unscoped dispatcher stamps nothing");
    let json: serde_json::Value =
        serde_json::from_str(&serde_json::to_string(&log).expect("serializes")).expect("json");
    assert_eq!(json["records"][0]["scope"], json!("run-1"));
    assert!(json["records"][2]["scope"].is_null(), "absent when none");
    let restored: EffectLog = serde_json::from_value(json).expect("restores");
    assert_eq!(restored[1].scope.as_deref(), Some("run-1"));
}
