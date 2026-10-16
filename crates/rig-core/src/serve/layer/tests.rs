use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};

use futures::StreamExt;
use serde_json::json;

use super::*;
use crate::{
    effect::{FamilyDescriptor, HandlerKey},
    operation::Finish,
    streaming::{Item, StreamEvent},
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

    async fn serve(&self, kind: EffectKind, _dispatch: Dispatch) -> Reply {
        self.served.fetch_add(1, Ordering::SeqCst);
        let outcome = match kind {
            EffectKind::Custom { payload, .. } => Ok(Answered::Custom { payload }),
            other => Err(ErrorReport::new(
                ErrorKind::Internal,
                format!("echo received {}", other.name()),
            )),
        };
        Reply::Outcome(outcome)
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

    async fn serve(&self, _kind: EffectKind, _dispatch: Dispatch) -> Reply {
        Reply::written(
            crate::message::Origin::new("test", "test", "test"),
            |mut writer| async move {
                writer.text("hel").await.expect("open");
                writer.text("lo").await.expect("open");
                writer.finish(Finish::default()).await.expect("open");
            },
        )
    }
}

type Before = Box<dyn Fn(&EffectKind) -> BeforeServe + Send + Sync>;
type After = Box<dyn Fn(&Result<Answered, ErrorReport>) -> AfterServe + Send + Sync>;

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
            before: Box::new(|_| BeforeServe::Proceed),
            after: Box::new(|_| AfterServe::Keep),
        }
    }
}

impl Intercept for Policy {
    fn name(&self) -> String {
        self.name.to_owned()
    }

    async fn before(&self, _id: EffectId, kind: &EffectKind) -> BeforeServe {
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
        outcome: &Result<Answered, ErrorReport>,
    ) -> AfterServe {
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
    outcomes: Mutex<Vec<Result<Answered, ErrorReport>>>,
    events: Mutex<Vec<Item<StreamEvent>>>,
    discarded: AtomicUsize,
    patched: Mutex<Vec<EffectKind>>,
}

impl super::super::Observe for Arc<Tapped> {
    fn origin(&mut self, _origin: &crate::message::Origin) {}
    fn outcome(&mut self, outcome: Result<Answered, ErrorReport>) {
        self.outcomes.lock().expect("outcomes").push(outcome);
    }

    fn keep_events(&self) -> bool {
        true
    }

    fn event(&mut self, event: &Item<StreamEvent>) {
        self.events.lock().expect("events").push(event.clone());
    }

    fn discard(&mut self, _: &str) {
        self.discarded.fetch_add(1, Ordering::SeqCst);
    }

    fn patch(&mut self, kind: &EffectKind) {
        self.patched.lock().expect("patched").push(kind.clone());
    }
}

fn tapped(dispatch: Dispatch, tapped: &Arc<Tapped>) -> Dispatch {
    dispatch.with_observer(Box::new(Arc::clone(tapped)))
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
) -> (Result<Answered, ErrorReport>, Arc<Tapped>) {
    let tap = Arc::new(Tapped::default());
    let dispatch = tapped(Dispatch::new(EffectId::from_raw(7), false), &tap);
    let outcome = handler.handle(kind, dispatch).await.into_outcome().await;
    (outcome, tap)
}

#[tokio::test]
async fn a_patch_of_another_family_is_internal_and_never_a_dispatch() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let (handler, served) = echo();
    let mut policy = Policy::observing("wrong", &seen);
    policy.before = Box::new(|_| {
        BeforeServe::Patch(EffectKind::ToolCall {
            name: "read".into(),
            args: "{}".into(),
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
    assert!(report.message.contains("custom") && report.message.contains("tool_call"));
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
async fn a_replacement_of_a_streamed_answer_ends_it_after_its_events() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let mut policy = Policy::observing("cut", &seen);
    policy.after = Box::new(|_| {
        AfterServe::Replace(Err(ErrorReport::new(
            ErrorKind::Cancelled,
            "the program stops",
        )))
    });
    let layered = ErasedHandler::new(Streamer).layered(policy);
    let tap = Arc::new(Tapped::default());
    let dispatch = tapped(Dispatch::new(EffectId::from_raw(9), true), &tap);
    let items: Vec<_> = layered
        .handle(custom(json!(1)), dispatch)
        .await
        .into_stream()
        .collect()
        .await;
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
        assert!(matches!(&outcomes[0], Ok(Answered::Completion(_))));
    }
    // A replacement completion is the reply's end: the consumer's fold
    // returns it, after the events delivered as they came; the record keeps
    // the handler's.
    let mut policy = Policy::observing("rewrite", &seen);
    policy.after = Box::new(|outcome| match outcome {
        Ok(Answered::Completion(response)) => {
            let mut response = response.clone();
            std::sync::Arc::make_mut(&mut response).choice =
                vec![crate::message::AssistantContent::text("rewritten")];
            AfterServe::Replace(Ok(Answered::Completion(response)))
        }
        _ => AfterServe::Keep,
    });
    let tap = Arc::new(Tapped::default());
    let dispatch = tapped(Dispatch::new(EffectId::from_raw(9), true), &tap);
    let outcome = ErasedHandler::new(Streamer)
        .layered(policy)
        .handle(custom(json!(1)), dispatch)
        .await
        .into_outcome()
        .await;
    assert!(
        matches!(&outcome, Ok(Answered::Completion(response)) if response.text() == "rewritten"),
        "{outcome:?}"
    );
    {
        let outcomes = tap.outcomes.lock().expect("outcomes");
        assert!(
            matches!(&outcomes[0], Ok(Answered::Completion(response)) if response.text() == "hello")
        );
    }
    // A replacement of another family cannot end a streamed completion.
    let mut policy = Policy::observing("swap", &seen);
    policy.after = Box::new(|_| {
        AfterServe::Replace(Ok(Answered::Custom {
            payload: json!("late"),
        }))
    });
    let layered = ErasedHandler::new(Streamer).layered(policy);
    let items: Vec<_> = layered
        .handle(custom(json!(1)), Dispatch::new(EffectId::from_raw(9), true))
        .await
        .into_stream()
        .collect()
        .await;
    let last = items.last().expect("an ending");
    assert!(
        matches!(last, Err(report) if report.kind == ErrorKind::Internal && report.message.contains("layer `swap`")),
        "{last:?}"
    );
}

#[test]
fn decisions_and_verdicts_are_data() {
    let deny = BeforeServe::deny("no");
    let json = serde_json::to_value(&deny).expect("serializes");
    assert_eq!(json["decision"], "deny");
    assert_eq!(json["kind"], "denied");
    let back: BeforeServe = serde_json::from_value(json.clone()).expect("restores");
    assert_eq!(serde_json::to_value(&back).expect("serializes"), json);
    let keep = serde_json::to_value(AfterServe::Keep).expect("serializes");
    assert_eq!(keep, json!({"verdict": "keep"}));
}

/// Retargeting must be refused before a nested policy can authorize a different name.
#[tokio::test]
async fn tool_target_patch_is_refused_before_inner_policy_in_unary_and_streaming() {
    use crate::{
        message::ToolName,
        tool::{DynamicTool, ToolContext, ToolOutput},
        wasm_compat::WasmBoxedFuture,
    };
    for streamed in [false, true] {
        let served = Arc::new(AtomicUsize::new(0));
        let counter = served.clone();
        let tool = DynamicTool::new_with_context(
            ToolName::new("original").expect("a tool name"),
            "bound original",
            json!({"type":"object"}),
            move |_context: &mut ToolContext, _args: serde_json::Value| {
                counter.fetch_add(1, Ordering::SeqCst);
                Box::pin(async { Ok(ToolOutput::text("executed original")) })
                    as WasmBoxedFuture<'_, _>
            },
        );
        let seen = Arc::new(Mutex::new(Vec::new()));
        let mut inner = Policy::observing("inner", &seen);
        inner.before = Box::new(|kind| match kind {
            EffectKind::ToolCall { name, .. } if name == "allowed" => BeforeServe::Proceed,
            _ => BeforeServe::Deny(ErrorReport::new(ErrorKind::Denied, "original is forbidden")),
        });
        let mut outer = Policy::observing("outer", &seen);
        outer.before = Box::new(|_| {
            BeforeServe::Patch(EffectKind::ToolCall {
                name: "allowed".into(),
                args: "{}".into(),
            })
        });
        let handler = ErasedHandler::new(tool).layered(inner).layered(outer);
        let kind = EffectKind::ToolCall {
            name: "original".into(),
            args: "{}".into(),
        };
        let tap = if streamed {
            let tap = Arc::new(Tapped::default());
            let events: Vec<_> = handler
                .handle(
                    kind,
                    tapped(Dispatch::new(EffectId::from_raw(7), true), &tap),
                )
                .await
                .into_stream()
                .collect()
                .await;
            assert!(events.iter().any(Result::is_err));
            tap
        } else {
            let (outcome, tap) = unary(&handler, kind).await;
            assert!(outcome.is_err());
            tap
        };
        assert_eq!(
            served.load(Ordering::SeqCst),
            0,
            "never execute the original under an authorized replacement name"
        );
        assert_eq!(*seen.lock().expect("policy trace"), ["outer.before"]);
        assert!(tap.patched.lock().expect("patches").is_empty());
        assert_eq!(tap.discarded.load(Ordering::SeqCst), 1);
    }
}

#[tokio::test]
async fn tool_argument_patches_keep_the_bound_target_and_reach_inner_policy() {
    use crate::{
        message::ToolName,
        tool::{DynamicTool, ToolContext, ToolOutput},
        wasm_compat::WasmBoxedFuture,
    };
    let tool = DynamicTool::new_with_context(
        ToolName::new("original").expect("a tool name"),
        "bound original",
        json!({"type":"object"}),
        |_context: &mut ToolContext, args: serde_json::Value| {
            assert_eq!(args, json!({"patched":true}));
            Box::pin(async { Ok(ToolOutput::text("executed patched arguments")) })
                as WasmBoxedFuture<'_, _>
        },
    );
    let seen = Arc::new(Mutex::new(Vec::new()));
    let mut inner = Policy::observing("inner", &seen);
    inner.before = Box::new(|kind| {
        assert!(
            matches!(kind, EffectKind::ToolCall { name, args } if name == "original" && args == r#"{"patched":true}"#)
        );
        BeforeServe::Proceed
    });
    let mut outer = Policy::observing("outer", &seen);
    outer.before = Box::new(|_| {
        BeforeServe::Patch(EffectKind::ToolCall {
            name: "original".into(),
            args: r#"{"patched":true}"#.into(),
        })
    });
    let handler = ErasedHandler::new(tool).layered(inner).layered(outer);
    let (outcome, tap) = unary(
        &handler,
        EffectKind::ToolCall {
            name: "original".into(),
            args: "{}".into(),
        },
    )
    .await;
    assert!(outcome.is_ok());
    assert_eq!(
        *seen.lock().expect("policy trace"),
        ["outer.before", "inner.before", "inner.after", "outer.after"]
    );
    assert_eq!(tap.patched.lock().expect("patches").len(), 1);
}

/// Says every effect is independent of every other.
struct Independent;

impl Serve for Independent {
    type Family = family::Dynamic;

    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: HandlerKey::from("independent"),
            family: FamilyDescriptor::Custom {
                kind: "test".into(),
            },
            layers: Vec::new(),
        }
    }

    async fn serve(&self, _kind: EffectKind, _dispatch: Dispatch) -> Reply {
        Reply::Outcome(Ok(Answered::Custom {
            payload: json!(null),
        }))
    }

    fn footprint(&self, _kind: &EffectKind) -> Option<crate::tool::Touches> {
        Some(crate::tool::Touches::Independent)
    }
}

/// A footprint passes through sharing, erasure, re-erasure and a layer; a
/// handler that says nothing stays silent through them.
#[test]
fn a_footprint_passes_through_every_wrapper() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let wrapped = |inner: ErasedHandler| {
        ErasedHandler::new(Layer::new(
            ErasedHandler::new(inner),
            Policy::observing("p", &seen),
        ))
    };
    let kind = custom(json!(1));
    let independent = wrapped(ErasedHandler::new(Arc::new(Independent)));
    assert_eq!(
        independent.footprint(&kind),
        Some(crate::tool::Touches::Independent)
    );
    let (echo, _) = echo();
    assert_eq!(wrapped(echo).footprint(&kind), None);
}
