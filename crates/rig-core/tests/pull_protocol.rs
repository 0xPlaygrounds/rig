//! The returned-reply boundary, including recording before suspended verdicts.
#![allow(clippy::panic, clippy::indexing_slicing, clippy::unwrap_used)]

use futures::{StreamExt, channel::oneshot, executor::block_on};
use rig_core::{
    completion::{CompletionResponse, Usage},
    effect::{
        EffectId, EffectKind, FamilyDescriptor, HandlerDescriptor, HandlerKey, Outcome, family,
    },
    error::{ErrorKind, ErrorReport},
    message::{AssistantContent, DocumentSourceKind, Image},
    operation::Finish,
    serve::{Decision, Dispatch, ErasedHandler, Intercept, Observe, Reply, Serve, Verdict},
    streaming::{Item, Relayed, StreamEvent},
};
use serde_json::{Value, json};
use std::sync::{Arc, Mutex};

fn kind(value: Value) -> EffectKind {
    EffectKind::Custom {
        kind: Arc::from("proof"),
        payload: value,
    }
}

#[derive(Default)]
struct Seen {
    outcomes: Vec<Value>,
    events: Vec<Value>,
    patches: Vec<Value>,
    discarded: usize,
}
struct Observer(Arc<Mutex<Seen>>);
impl Observe for Observer {
    fn origin(&mut self, _origin: &rig_core::message::Origin) {}
    fn outcome(&mut self, outcome: &Result<Outcome, ErrorReport>) {
        self.0
            .lock()
            .unwrap()
            .outcomes
            .push(serde_json::to_value(outcome).unwrap());
    }
    fn keep_events(&self) -> bool {
        true
    }
    fn event(&mut self, event: &Item<StreamEvent>) {
        self.0
            .lock()
            .unwrap()
            .events
            .push(serde_json::to_value(event).unwrap());
    }
    fn stream_error(&mut self, error: &ErrorReport) {
        self.0
            .lock()
            .unwrap()
            .events
            .push(serde_json::to_value(error).unwrap());
    }
    fn discard(&mut self, _: &str) {
        self.0.lock().unwrap().discarded += 1;
    }
    fn patch(&mut self, kind: &EffectKind) {
        self.0
            .lock()
            .unwrap()
            .patches
            .push(serde_json::to_value(kind).unwrap());
    }
}
fn dispatch(streaming: bool, seen: &Arc<Mutex<Seen>>) -> Dispatch {
    Dispatch::new(EffectId::from_raw(1), streaming).with_observer(Box::new(Observer(seen.clone())))
}

struct Answer {
    streaming: bool,
}
impl Serve for Answer {
    type Family = family::Dynamic;
    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: HandlerKey::from("proof"),
            family: FamilyDescriptor::Custom {
                kind: "proof".into(),
            },
            layers: vec![],
        }
    }
    async fn serve(&self, kind: EffectKind, dispatch: Dispatch) -> Reply {
        if let Some(scope) = dispatch.scope::<String>() {
            assert_eq!(&*scope, "scope");
        }
        if self.streaming {
            Reply::written(
                rig_core::message::Origin::new("proof", "proof", "proof"),
                |mut out| async move {
                    out.text("original").await.unwrap();
                    out.finish(Finish::default()).await.unwrap();
                },
            )
        } else {
            let EffectKind::Custom { payload, .. } = kind else {
                panic!("custom")
            };
            Reply::Outcome(Ok(Outcome::Custom { payload }))
        }
    }
}

struct Policy {
    decision: Decision,
    verdict: Verdict,
    gate: Mutex<Option<oneshot::Receiver<()>>>,
    calls: Arc<Mutex<usize>>,
}
impl Intercept for Policy {
    fn name(&self) -> String {
        "proof-policy".into()
    }
    async fn before(&self, _: EffectId, _: &EffectKind) -> Decision {
        self.decision.clone()
    }
    async fn after(
        &self,
        _: EffectId,
        _: &EffectKind,
        _: &Result<Outcome, ErrorReport>,
    ) -> Verdict {
        *self.calls.lock().unwrap() += 1;
        let gate = self.gate.lock().unwrap().take();
        if let Some(gate) = gate {
            gate.await.unwrap();
        }
        self.verdict.clone()
    }
}
fn policy(decision: Decision, verdict: Verdict) -> Policy {
    Policy {
        decision,
        verdict,
        gate: Mutex::new(None),
        calls: Arc::default(),
    }
}

struct ImageAnswer;
impl Serve for ImageAnswer {
    type Family = family::Dynamic;
    fn descriptor(&self) -> HandlerDescriptor {
        Answer { streaming: false }.descriptor()
    }
    async fn serve(&self, _: EffectKind, _: Dispatch) -> Reply {
        Reply::Outcome(Ok(Outcome::Completion(CompletionResponse::new(
            vec![AssistantContent::Image(Image {
                data: DocumentSourceKind::base64("aW1hZ2U="),
                ..Image::default()
            })],
            Usage::default(),
            rig_core::message::Origin::new("test.api", "proof", ""),
            serde_json::json!({}),
        ))))
    }
}

#[test]
fn unary_image_recording_survives_stream_projection_and_replacement() {
    let seen = Arc::default();
    let handler = ErasedHandler::new(ImageAnswer).layered(policy(
        Decision::Proceed,
        Verdict::Replace(Err(ErrorReport::new(ErrorKind::Denied, "replacement"))),
    ));
    let items = block_on(
        block_on(handler.handle(kind(json!(null)), dispatch(true, &seen)))
            .into_stream()
            .collect::<Vec<_>>(),
    );
    assert!(items.iter().any(|item| matches!(
        item,
        Ok(Relayed::Item(Item::Event(StreamEvent::End {
            content: AssistantContent::Image(_),
            ..
        })))
    )));
    assert!(items.last().unwrap().is_err());
    let expected = block_on(
        block_on(ImageAnswer.serve(
            kind(json!(null)),
            Dispatch::new(EffectId::from_raw(1), false),
        ))
        .into_outcome(),
    );
    assert_eq!(
        seen.lock().unwrap().outcomes,
        vec![serde_json::to_value(expected).unwrap()]
    );
    // The recorded projection ends with the image part the answer carried.
    let last = seen.lock().unwrap().events.last().unwrap().to_string();
    assert!(
        last.contains(r#""event":"end""#) && last.contains("image"),
        "{last}"
    );
}

#[test]
fn deferred_answer_reports_drop_and_late_resolution() {
    let (resolver, waiting) = rig_core::serve::deferred();
    drop(resolver);
    assert_eq!(
        block_on(waiting).unwrap_err().to_string(),
        "the handler dropped its outcome sink without answering"
    );
    let (resolver, waiting) = rig_core::serve::deferred();
    drop(waiting);
    assert!(resolver.is_closed());
    assert!(
        resolver
            .resolve(Ok(Outcome::Custom {
                payload: json!(null)
            }))
            .is_err()
    );
}

#[test]
fn a_layer_does_not_repoll_an_exhausted_non_fused_stream() {
    struct Empty;
    impl Serve for Empty {
        type Family = family::Dynamic;
        fn descriptor(&self) -> HandlerDescriptor {
            HandlerDescriptor {
                key: "empty".into(),
                family: FamilyDescriptor::Custom {
                    kind: "empty".into(),
                },
                layers: vec![],
            }
        }
        async fn serve(&self, _: EffectKind, _: Dispatch) -> Reply {
            Reply::Stream(Box::pin(futures::stream::unfold((), |_| async {
                None::<(Result<Relayed, ErrorReport>, ())>
            })))
        }
    }
    struct Keep;
    impl Intercept for Keep {
        fn name(&self) -> String {
            "keep".into()
        }
        async fn before(&self, _: EffectId, _: &EffectKind) -> Decision {
            Decision::Proceed
        }
        async fn after(
            &self,
            _: EffectId,
            _: &EffectKind,
            _: &Result<Outcome, ErrorReport>,
        ) -> Verdict {
            Verdict::Keep
        }
    }
    block_on(async {
        let handler = ErasedHandler::new(Empty).layered(Keep);
        let mut stream = handler
            .handle(
                EffectKind::Custom {
                    kind: "empty".into(),
                    payload: serde_json::Value::Null,
                },
                Dispatch::new(EffectId::from_raw(0), true),
            )
            .await
            .into_stream();
        assert_eq!(
            stream.next().await.unwrap().unwrap_err().message,
            rig_core::serve::stream_truncated().message
        );
        assert!(stream.next().await.is_none());
    });
}
