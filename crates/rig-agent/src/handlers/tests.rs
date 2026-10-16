use std::sync::Arc;

use futures::StreamExt;
use rig_core::{
    completion::CompletionRequest,
    effect::{Answered, EffectKind, FamilyDescriptor, HandlerDescriptor, HandlerKey},
    error::ErrorKind,
    serve::{Dispatch, RecordBuffer, Reply, Serve},
    streaming::Relayed,
};

use super::Handlers;

fn note(payload: serde_json::Value) -> EffectKind {
    EffectKind::Custom {
        kind: Arc::from("test:note"),
        payload,
    }
}

fn completion(stream: bool) -> EffectKind {
    EffectKind::Completion {
        request: CompletionRequest::new("hi").into(),
        stream,
    }
}

fn descriptor(key: &str, family: FamilyDescriptor) -> HandlerDescriptor {
    HandlerDescriptor {
        key: HandlerKey::from(key),
        family,
        layers: Vec::new(),
    }
}

fn custom() -> FamilyDescriptor {
    FamilyDescriptor::Custom {
        kind: "test:note".into(),
    }
}

/// Answers a note with its payload; a note with `"nest": true` first sends a
/// note of its own through the handlers it is served with.
struct Notes;

impl Serve for Notes {
    type Family = rig_core::effect::family::Dynamic;

    fn descriptor(&self) -> HandlerDescriptor {
        descriptor("notes", custom())
    }

    async fn serve(&self, kind: EffectKind, dispatch: Dispatch) -> Reply {
        let EffectKind::Custom { payload, .. } = kind else {
            return Reply::Outcome(Ok(Answered::Custom {
                payload: serde_json::Value::Null,
            }));
        };
        if payload["nest"] == true {
            let handlers = Handlers::serving(&dispatch).expect("served by a table");
            assert_eq!(handlers.parent(), Some(dispatch.id()));
            let inner = handlers
                .dispatch(&HandlerKey::from("notes"), note(serde_json::json!("inner")))
                .await;
            return Reply::Outcome(inner);
        }
        Reply::Outcome(Ok(Answered::Custom { payload }))
    }
}

/// A completion stream that ends with neither its response nor an error,
/// or, unary, never answers.
struct Cut;

impl Serve for Cut {
    type Family = rig_core::effect::family::Dynamic;

    fn descriptor(&self) -> HandlerDescriptor {
        descriptor(
            "cut",
            FamilyDescriptor::Completion {
                model: "cut".into(),
                capabilities: Default::default(),
            },
        )
    }

    async fn serve(&self, _kind: EffectKind, dispatch: Dispatch) -> Reply {
        if dispatch.is_stream() {
            return Reply::Stream(Box::pin(futures::stream::empty()));
        }
        std::future::pending().await
    }
}

#[tokio::test]
async fn a_nested_dispatch_is_recorded_under_the_one_that_made_it() {
    let handlers = Handlers::default();
    handlers.register("notes", Notes).expect("a fresh key");
    let records = RecordBuffer::new();
    handlers.record_to(records.clone());

    let answer = handlers
        .dispatch(
            &HandlerKey::from("notes"),
            note(serde_json::json!({"nest": true})),
        )
        .await
        .expect("answered");
    assert!(matches!(answer, Answered::Custom { payload } if payload == "inner"));

    let records = records.take_resolved();
    let shape: Vec<_> = records
        .iter()
        .map(|record| (record.id.as_u64(), record.parent.map(|id| id.as_u64())))
        .collect();
    assert_eq!(shape, [(1, None), (2, Some(1))]);
}

#[tokio::test]
async fn a_missing_key_answers_handler_unavailable_and_records_nothing() {
    let handlers = Handlers::default();
    let records = RecordBuffer::new();
    handlers.record_to(records.clone());
    let key = HandlerKey::from("absent");

    let unary = handlers.dispatch(&key, note(serde_json::Value::Null)).await;
    assert_eq!(
        unary.err().map(|report| report.kind),
        Some(ErrorKind::HandlerUnavailable)
    );
    let streamed: Vec<_> = handlers
        .dispatch_stream(&key, completion(true))
        .collect()
        .await;
    assert!(matches!(
        streamed.as_slice(),
        [Err(report)] if report.kind == ErrorKind::HandlerUnavailable
    ));
    assert_eq!(records.in_flight() + records.resolved(), 0);
}

#[test]
fn a_registration_keeps_the_family_of_its_key() {
    let handlers = Handlers::default();
    handlers.register("notes", Notes).expect("a fresh key");
    let refused = handlers.register("notes", Cut).expect_err("another family");
    assert_eq!(refused.kind, ErrorKind::HandlerUnavailable);
    assert!(handlers.deregister(&HandlerKey::from("notes")));
    handlers.register("notes", Cut).expect("a free key again");
}

#[tokio::test]
async fn a_stream_cut_short_ends_truncated() {
    let handlers = Handlers::default();
    handlers.register("cut", Cut).expect("a fresh key");
    let items: Vec<_> = handlers
        .dispatch_stream(&HandlerKey::from("cut"), completion(true))
        .collect()
        .await;
    assert!(matches!(
        items.as_slice(),
        [Err(report)] if report.kind == rig_core::serve::stream_truncated().kind
    ));
    let unary: Vec<Result<Relayed, _>> = handlers
        .dispatch_stream(&HandlerKey::from("cut"), completion(false))
        .collect()
        .await;
    assert!(matches!(unary.as_slice(), [Err(report)] if report.kind == ErrorKind::Request));
}

#[tokio::test]
async fn dropping_a_dispatch_cancels_it_and_records_the_cancellation() {
    let handlers = Handlers::default();
    handlers.register("cut", Cut).expect("a fresh key");
    let records = RecordBuffer::new();
    handlers.record_to(records.clone());

    let mut pending = Box::pin(handlers.dispatch(&HandlerKey::from("cut"), completion(false)));
    // The first poll yields, so the dispatches made beside it begin first.
    assert!(futures::poll!(pending.as_mut()).is_pending());
    assert_eq!(records.in_flight(), 0);
    assert!(futures::poll!(pending.as_mut()).is_pending());
    assert_eq!(records.in_flight(), 1);
    drop(pending);

    let records = records.take_resolved();
    assert!(matches!(
        records.as_slice(),
        [record] if record.outcome.as_ref().err().map(|report| report.kind) == Some(ErrorKind::Cancelled)
    ));
}
