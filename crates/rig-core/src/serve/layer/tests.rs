use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};

use futures::{StreamExt, channel::mpsc, channel::oneshot};
use serde_json::json;

use super::*;
use crate::{
    completion::Usage,
    effect::{FamilyDescriptor, HandlerKey},
    streaming::StreamFinal,
};

fn custom(payload: serde_json::Value) -> EffectKind {
    EffectKind::Custom {
        kind: Arc::from("test"),
        payload,
    }
}

/// Answers a custom effect with its payload; counts its calls.
struct Echo {
    served: Arc<AtomicUsize>,
}

impl Serve for Echo {
    type Family = family::Dynamic;

    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: HandlerKey::from("echo"),
            family: FamilyDescriptor::Custom {
                kind: "test".into(),
            },
            layers: Vec::new(),
        }
    }

    async fn serve(&self, kind: EffectKind, sink: OutcomeSink) {
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

/// Streams two text deltas and a terminal.
struct Streamer;

impl Serve for Streamer {
    type Family = family::Dynamic;

    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: HandlerKey::from("streamer"),
            family: FamilyDescriptor::Custom {
                kind: "test".into(),
            },
            layers: Vec::new(),
        }
    }

    async fn serve(&self, _kind: EffectKind, sink: OutcomeSink) {
        let mut writer = sink.writer();
        writer.text("hel").await.expect("open");
        writer.text("lo").await.expect("open");
        writer
            .finish(StreamFinal::new("test", Usage::default()))
            .await
            .expect("open");
    }
}

type Before = Box<dyn Fn(&EffectKind) -> Decision + Send + Sync>;
type After = Box<dyn Fn(&Result<Outcome, ErrorReport>) -> Verdict + Send + Sync>;

/// Records what it saw, in order, and decides as configured.
struct Policy {
    name: &'static str,
    seen: Arc<Mutex<Vec<String>>>,
    before: Before,
    after: After,
}

impl Policy {
    fn observing(name: &'static str, seen: &Arc<Mutex<Vec<String>>>) -> Self {
        Self {
            name,
            seen: Arc::clone(seen),
            before: Box::new(|_| Decision::Proceed),
            after: Box::new(|_| Verdict::Keep),
        }
    }
}

impl Intercept for Policy {
    fn name(&self) -> String {
        self.name.to_owned()
    }

    async fn before(&self, _id: EffectId, kind: &EffectKind) -> Decision {
        self.seen
            .lock()
            .expect("seen")
            .push(format!("{}.before", self.name));
        (self.before)(kind)
    }

    async fn after(
        &self,
        _id: EffectId,
        _kind: &EffectKind,
        outcome: &Result<Outcome, ErrorReport>,
    ) -> Verdict {
        self.seen
            .lock()
            .expect("seen")
            .push(format!("{}.after", self.name));
        (self.after)(outcome)
    }
}

/// A recorder's view: outcomes, events and discards, as the driver's tap
/// would deliver them.
#[derive(Default)]
struct Tapped {
    outcomes: Mutex<Vec<Result<Outcome, ErrorReport>>>,
    events: Mutex<Vec<StreamEvent>>,
    discarded: AtomicUsize,
    patched: Mutex<Vec<EffectKind>>,
}

impl super::super::Observe for Arc<Tapped> {
    fn outcome(&mut self, outcome: &Result<Outcome, ErrorReport>) {
        self.outcomes
            .lock()
            .expect("outcomes")
            .push(outcome.clone());
    }

    fn keep_events(&self) -> bool {
        true
    }

    fn event(&mut self, event: &StreamEvent) {
        self.events.lock().expect("events").push(event.clone());
    }

    fn discard(&mut self) {
        self.discarded.fetch_add(1, Ordering::SeqCst);
    }

    fn patch(&mut self, kind: &EffectKind) {
        self.patched.lock().expect("patched").push(kind.clone());
    }
}

fn tapped(sink: OutcomeSink, tapped: &Arc<Tapped>) -> OutcomeSink {
    sink.with_observer(Box::new(Arc::clone(tapped)))
}

fn echo() -> (ErasedHandler, Arc<AtomicUsize>) {
    let served = Arc::new(AtomicUsize::new(0));
    (
        ErasedHandler::new(Echo {
            served: Arc::clone(&served),
        }),
        served,
    )
}

/// Dispatch `kind` unary through `handler` with a recording tap; the
/// consumer's outcome and the tap's view.
async fn unary(
    handler: &ErasedHandler,
    kind: EffectKind,
) -> (Result<Outcome, ErrorReport>, Arc<Tapped>) {
    let tap = Arc::new(Tapped::default());
    let (reply, receiver) = oneshot::channel();
    let sink = tapped(OutcomeSink::unary(EffectId::from_raw(7), reply), &tap);
    handler.handle(kind, sink).await;
    let outcome = receiver.await.expect("answered");
    (outcome, tap)
}

#[tokio::test]
async fn layers_nest_outermost_first() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let (handler, served) = echo();
    let layered = handler
        .layered(Policy::observing("a", &seen))
        .layered(Policy::observing("b", &seen));
    assert_eq!(layered.descriptor().layers, ["b", "a"], "outermost first");
    assert_eq!(layered.descriptor().key.as_str(), "echo", "the inner's key");
    let (outcome, tap) = unary(&layered, custom(json!(1))).await;
    assert!(matches!(outcome, Ok(Outcome::Custom(payload)) if payload == json!(1)));
    assert_eq!(served.load(Ordering::SeqCst), 1);
    assert_eq!(
        *seen.lock().expect("seen"),
        ["b.before", "a.before", "a.after", "b.after"],
        "the handler-stack order: b sees the dispatch first, a's after runs first"
    );
    // The record is the handler's answer, once.
    let outcomes = tap.outcomes.lock().expect("outcomes");
    assert_eq!(outcomes.len(), 1);
    assert!(matches!(&outcomes[0], Ok(Outcome::Custom(payload)) if *payload == json!(1)));
    assert_eq!(tap.discarded.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn a_patch_of_the_same_family_is_what_the_handler_serves() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let (handler, _) = echo();
    let mut policy = Policy::observing("patcher", &seen);
    policy.before = Box::new(|_| Decision::Patch(custom(json!("patched"))));
    let layered = handler.layered(policy);
    let (outcome, tap) = unary(&layered, custom(json!("original"))).await;
    assert!(matches!(outcome, Ok(Outcome::Custom(payload)) if payload == json!("patched")));
    let outcomes = tap.outcomes.lock().expect("outcomes");
    assert!(
        matches!(&outcomes[0], Ok(Outcome::Custom(payload)) if *payload == json!("patched")),
        "the record holds what was served"
    );
    let patched = tap.patched.lock().expect("patched");
    assert_eq!(
        patched.len(),
        1,
        "the tap saw the patch: the record's request is what was served"
    );
    assert!(
        matches!(&patched[0], EffectKind::Custom { payload, .. } if *payload == json!("patched"))
    );
}

#[tokio::test]
async fn a_patch_of_another_family_is_internal_and_never_a_dispatch() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let (handler, served) = echo();
    let mut policy = Policy::observing("wrong", &seen);
    policy.before = Box::new(|_| {
        Decision::Patch(EffectKind::Memory {
            op: crate::effect::MemoryOp::Clear {
                conversation: crate::id::ConversationId::from("c"),
            },
        })
    });
    let layered = handler.layered(policy);
    let (outcome, tap) = unary(&layered, custom(json!(1))).await;
    let report = outcome.expect_err("internal");
    assert_eq!(report.kind, ErrorKind::Internal);
    assert!(
        report.message.contains("layer `wrong`"),
        "{}",
        report.message
    );
    assert!(report.message.contains("custom") && report.message.contains("memory"));
    assert_eq!(served.load(Ordering::SeqCst), 0, "no dispatch");
    assert!(
        tap.outcomes.lock().expect("outcomes").is_empty(),
        "no record"
    );
    assert_eq!(tap.discarded.load(Ordering::SeqCst), 1);
    assert_eq!(
        *seen.lock().expect("seen"),
        ["wrong.before"],
        "no after: nothing was answered"
    );
}

#[tokio::test]
async fn a_denial_never_reaches_the_handler_and_leaves_no_record() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let (handler, served) = echo();
    let mut policy = Policy::observing("gate", &seen);
    policy.before = Box::new(|_| Decision::deny("not today"));
    let layered = handler.layered(policy);
    let (outcome, tap) = unary(&layered, custom(json!(1))).await;
    let report = outcome.expect_err("denied");
    assert_eq!(report.kind, ErrorKind::Denied);
    assert!(!report.retryable);
    assert_eq!(report.message, "not today");
    assert_eq!(served.load(Ordering::SeqCst), 0);
    assert!(
        tap.outcomes.lock().expect("outcomes").is_empty(),
        "no record"
    );
    assert_eq!(
        tap.discarded.load(Ordering::SeqCst),
        1,
        "the slot is forgotten"
    );
    // A denial with a kind of its own meaning travels as given.
    let (handler, _) = echo();
    let mut policy = Policy::observing("stop", &seen);
    policy.before =
        Box::new(|_| Decision::Deny(ErrorReport::new(ErrorKind::Cancelled, "the program stops")));
    let (outcome, _) = unary(&handler.layered(policy), custom(json!(1))).await;
    assert_eq!(outcome.expect_err("cancelled").kind, ErrorKind::Cancelled);
}

#[tokio::test]
async fn a_replacement_on_the_way_out_leaves_the_handlers_answer_in_the_record() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let (handler, _) = echo();
    let mut policy = Policy::observing("replacer", &seen);
    policy.after = Box::new(|_| Verdict::Replace(Ok(Outcome::Custom(json!("replaced")))));
    let layered = handler.layered(policy);
    let (outcome, tap) = unary(&layered, custom(json!("real"))).await;
    assert!(
        matches!(&outcome, Ok(Outcome::Custom(payload)) if *payload == json!("replaced")),
        "the consumer receives the replacement"
    );
    let outcomes = tap.outcomes.lock().expect("outcomes");
    assert_eq!(outcomes.len(), 1, "one record");
    assert!(
        matches!(&outcomes[0], Ok(Outcome::Custom(payload)) if *payload == json!("real")),
        "the record holds the handler's answer"
    );
}

#[tokio::test]
async fn a_layer_over_a_streaming_handler_sees_the_folded_outcome_in_after() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let folded = Arc::new(Mutex::new(None));
    let mut policy = Policy::observing("fold", &seen);
    let for_after = Arc::clone(&folded);
    policy.after = Box::new(move |outcome| {
        *for_after.lock().expect("folded") = Some(outcome.clone());
        Verdict::Keep
    });
    let layered = ErasedHandler::new(Streamer).layered(policy);
    let tap = Arc::new(Tapped::default());
    let (events, receiver) = mpsc::channel(8);
    let sink = tapped(OutcomeSink::stream(EffectId::from_raw(9), events), &tap);
    layered.handle(custom(json!(1)), sink).await;
    let items: Vec<_> = receiver.collect().await;
    assert_eq!(
        items.len(),
        5,
        "a block start, two deltas, a block end, a final: {items:?}"
    );
    let folded = folded.lock().expect("folded").clone().expect("after ran");
    let Ok(Outcome::Completion(response)) = folded else {
        panic!("a folded completion, not {folded:?}");
    };
    assert_eq!(
        response.choice,
        vec![crate::message::AssistantContent::text("hello")]
    );
    // The record holds the fold and the events, tapped on the inner hop.
    let outcomes = tap.outcomes.lock().expect("outcomes");
    assert!(matches!(&outcomes[0], Ok(Outcome::Completion(_))));
    assert_eq!(tap.events.lock().expect("events").len(), items.len());
}

#[tokio::test]
async fn an_error_replacing_a_streamed_answer_follows_its_events() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let mut policy = Policy::observing("cut", &seen);
    policy.after = Box::new(|_| {
        Verdict::Replace(Err(ErrorReport::new(
            ErrorKind::Cancelled,
            "the program stops",
        )))
    });
    let layered = ErasedHandler::new(Streamer).layered(policy);
    let tap = Arc::new(Tapped::default());
    let (events, receiver) = mpsc::channel(8);
    let sink = tapped(OutcomeSink::stream(EffectId::from_raw(9), events), &tap);
    layered.handle(custom(json!(1)), sink).await;
    let items: Vec<_> = receiver.collect().await;
    let last = items.last().expect("an ending");
    assert!(
        matches!(last, Err(report) if report.kind == ErrorKind::Cancelled),
        "{last:?}"
    );
    assert!(
        items[..items.len() - 1].iter().all(Result::is_ok),
        "the events were delivered as they came"
    );
    // The record holds the handler's real answer.
    {
        let outcomes = tap.outcomes.lock().expect("outcomes");
        assert!(matches!(&outcomes[0], Ok(Outcome::Completion(_))));
    }
    // A replacement answer cannot follow events already delivered.
    let mut policy = Policy::observing("swap", &seen);
    policy.after = Box::new(|_| Verdict::Replace(Ok(Outcome::Custom(json!("late")))));
    let layered = ErasedHandler::new(Streamer).layered(policy);
    let (events, receiver) = mpsc::channel(8);
    layered
        .handle(
            custom(json!(1)),
            OutcomeSink::stream(EffectId::from_raw(9), events),
        )
        .await;
    let items: Vec<_> = receiver.collect().await;
    let last = items.last().expect("an ending");
    assert!(
        matches!(last, Err(report) if report.kind == ErrorKind::Internal && report.message.contains("layer `swap`")),
        "{last:?}"
    );
}

#[tokio::test]
async fn a_layer_serves_inline_too() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let (handler, _) = echo();
    let mut policy = Policy::observing("gate", &seen);
    policy.before = Box::new(|_| Decision::deny("inline too"));
    let layered = handler.layered(policy);
    let report = super::super::serve_inline(&layered, custom(json!(1)))
        .await
        .expect_err("denied");
    assert_eq!(report.kind, ErrorKind::Denied);
}

#[test]
fn decisions_and_verdicts_are_data() {
    let deny = Decision::deny("no");
    let json = serde_json::to_value(&deny).expect("serializes");
    assert_eq!(json["decision"], "deny");
    assert_eq!(json["kind"], "denied");
    let back: Decision = serde_json::from_value(json.clone()).expect("restores");
    assert_eq!(serde_json::to_value(&back).expect("serializes"), json);
    let keep = serde_json::to_value(Verdict::Keep).expect("serializes");
    assert_eq!(keep, json!({"verdict": "keep"}));
}
