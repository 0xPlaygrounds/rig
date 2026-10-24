use std::sync::Arc;

use futures::executor::block_on;

use super::*;
use crate::{
    completion::{CompletionRequest, Message},
    effect::{FamilyDescriptor, family, tool_key},
    operation::Finish,
    serve::{Dispatch, ErasedHandler, Reply, Serve, catch_panics},
    tool::ToolResult,
};

fn tool(name: &str) -> EffectKind {
    EffectKind::ToolCall {
        name: name.to_owned(),
        args: "{}".to_owned(),
    }
}

fn answer(text: &str) -> Result<Answered, ErrorReport> {
    Ok(Answered::ToolResult {
        result: ToolResult::skipped(text),
    })
}

fn json<T: serde::Serialize>(value: &T) -> serde_json::Value {
    serde_json::to_value(value).expect("serializes")
}

fn ids(records: &[EffectRecord]) -> Vec<u64> {
    records.iter().map(|record| record.id.as_u64()).collect()
}

/// Streams two text deltas and a terminal, whatever it is asked.
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

/// Records come out in the order their dispatches began, whatever order
/// they resolve in, with where they came from; one in flight stays.
#[test]
fn records_keep_dispatch_order_and_origin_whatever_order_they_resolve_in() {
    let buffer = RecordBuffer::new();
    let origin = DispatchContext {
        parent: Some(EffectId::from_raw(9)),
        scope: Some(Arc::from("agent-1")),
        purpose: Some("summary".into()),
    };
    for raw in 1..=3 {
        let id = EffectId::from_raw(raw);
        buffer.begin(id, tool_key("t"), tool("t"), origin.clone());
    }
    buffer.resolve(EffectId::from_raw(3), answer("three"));
    buffer.resolve(EffectId::from_raw(1), answer("one"));
    assert_eq!((buffer.resolved(), buffer.in_flight()), (2, 1));
    assert_eq!(ids(&buffer.records()), [1, 3], "a copy leaves them");
    assert!(
        buffer.take(EffectId::from_raw(2)).is_none(),
        "still in flight"
    );

    let taken = buffer.take_resolved();
    assert_eq!(ids(&taken), [1, 3]);
    assert_eq!(
        format!("{buffer:?}"),
        "RecordBuffer { resolved: 0, in_flight: 1 }"
    );
    let first = taken.first().expect("taken");
    assert_eq!(first.parent, Some(EffectId::from_raw(9)));
    assert_eq!(first.scope.as_deref(), Some("agent-1"));
    assert_eq!(first.purpose.as_deref(), Some("summary"));
    assert_eq!(first.disposition, Disposition::Ran);
    assert!(first.reviewed.is_none());
    assert_eq!(
        (first.started_ms, first.finished_ms),
        (None, None),
        "untimed"
    );

    buffer.resolve(EffectId::from_raw(2), answer("two"));
    let second = buffer.take(EffectId::from_raw(2)).expect("resolved");
    assert_eq!(json(&second.outcome), json(&answer("two")));
    assert_eq!(buffer.resolved() + buffer.in_flight(), 0);
}

/// A filtered take moves out only the resolved records it accepts, in
/// dispatch order, and leaves the rest resolved or in flight.
#[test]
fn a_filtered_take_leaves_the_records_it_refuses() {
    let buffer = RecordBuffer::new();
    for raw in 1..=4 {
        let id = EffectId::from_raw(raw);
        buffer.begin(id, tool_key("t"), tool("t"), DispatchContext::default());
    }
    for raw in [4, 2, 1] {
        buffer.resolve(EffectId::from_raw(raw), answer("done"));
    }

    let taken = buffer.take_resolved_where(|id| id != EffectId::from_raw(2));
    assert_eq!(ids(&taken), [1, 4], "accepted ones, in dispatch order");
    assert_eq!((buffer.resolved(), buffer.in_flight()), (1, 1));
    assert_eq!(ids(&buffer.take_resolved()), [2], "the refused one stays");
}

/// A layer's discard drops the record, and its patch replaces the request
/// recorded; tool output counts only before the outcome.
#[test]
fn discard_drops_patch_replaces_and_tool_output_lands_before_the_outcome() {
    let buffer = RecordBuffer::new();
    let (patched, dropped) = (EffectId::from_raw(1), EffectId::from_raw(2));
    buffer.begin(
        patched,
        tool_key("a"),
        tool("a"),
        DispatchContext::default(),
    );
    buffer.begin(
        dropped,
        tool_key("b"),
        tool("b"),
        DispatchContext::default(),
    );
    buffer.discard(dropped);
    assert!(!buffer.contains(dropped) && buffer.contains(patched));
    assert_eq!(buffer.remove(dropped), None, "already gone");

    buffer.patch(patched, tool("c"));
    buffer.tool_output(patched, ToolResultContext::default());
    buffer.resolve(patched, answer("done"));
    buffer.tool_output(patched, ToolResultContext::default());
    let record = buffer.take(patched).expect("resolved");
    assert_eq!(json(&record.kind), json(&tool("c")));
    assert_eq!(record.published, Some(ToolResultContext::default()));

    buffer.begin(
        dropped,
        tool_key("b"),
        tool("b"),
        DispatchContext::default(),
    );
    assert_eq!(buffer.remove(dropped), Some(tool_key("b")));
}

/// Streams `handler`'s reply through a dispatch `buffer` records.
fn record_stream(buffer: &RecordBuffer, id: EffectId) -> Result<Answered, ErrorReport> {
    let kind = EffectKind::Completion {
        request: CompletionRequest::new(Message::user("hi")).into(),
        stream: true,
    };
    let handler = ErasedHandler::new(Streamer);
    let key = handler.descriptor().key;
    buffer.begin(id, key, kind.clone(), DispatchContext::default());
    let recorded = Arc::new(buffer.clone());
    let dispatch = Dispatch::new(id, true).recorded_by(recorded);
    block_on(async { handler.handle(kind, dispatch).await.into_outcome().await })
}

/// A buffer recording a real streamed dispatch keeps its items and who it
/// was from, and its outcome is the fold; a timed one stamps its start,
/// first item and end, whether it keeps the items or not.
#[test]
fn a_recorded_stream_keeps_its_items_and_a_timed_buffer_stamps_its_ends() {
    let buffer = RecordBuffer::new().keeping_stream_events().timed();
    assert!(buffer.keep_events());
    let id = EffectId::from_raw(4);
    let outcome = record_stream(&buffer, id);
    assert!(
        matches!(outcome, Ok(Answered::Completion(_))),
        "{outcome:?}"
    );
    assert!(
        buffer.kept_events(id).is_some_and(|items| items >= 3),
        "two deltas and the end"
    );
    assert_eq!(buffer.kept_events(EffectId::from_raw(5)), None);

    let record = buffer.take(id).expect("resolved");
    assert!(record.events.is_some(), "the items are kept");
    let from = record
        .stream_origin
        .as_ref()
        .expect("the relay said who it is");
    assert_eq!(from, &crate::message::Origin::new("test", "test", "test"));
    let timed = RecordBuffer::new().timed();
    assert!(record_stream(&timed, id).is_ok());
    let unkept = timed.take(id).expect("resolved");
    assert!(unkept.events.is_none());
    for record in [record, unkept] {
        let stamps = (record.started_ms, record.first_item_ms, record.finished_ms);
        let (Some(started), Some(first), Some(finished)) = stamps else {
            panic!("timed: {stamps:?}");
        };
        assert!(started <= first && first <= finished, "{stamps:?}");
        assert!(started > 1_600_000_000_000, "milliseconds since the epoch");
    }
}

/// A unary dispatch keeps no items, even when the buffer keeps them.
#[test]
fn a_unary_dispatch_keeps_no_items() {
    let buffer = RecordBuffer::new().keeping_stream_events();
    let id = EffectId::from_raw(1);
    buffer.begin(id, tool_key("t"), tool("t"), DispatchContext::default());
    let origin = crate::message::Origin::new("test", "test", "test");
    buffer.origin(id, &origin);
    buffer.handlers(Vec::new());
    buffer.resolve(id, answer("done"));
    let record = buffer.take(id).expect("resolved");
    assert!(record.events.is_none() && record.stream_origin.is_none());
}

/// Work that panics before it answers.
async fn boom() -> Result<(), ErrorReport> {
    explode();
    Ok(())
}

fn explode() {
    panic!("boom")
}

/// A panic in the work is an internal error, recorded as the dispatch's
/// outcome when there is a recorder.
#[test]
fn a_panic_is_an_internal_error_recorded_as_the_outcome() {
    let buffer = RecordBuffer::new();
    let id = EffectId::from_raw(1);
    buffer.begin(id, tool_key("t"), tool("t"), DispatchContext::default());
    let recorder: Arc<dyn Recorder + Send + Sync> = Arc::new(buffer.clone());
    let error = block_on(catch_panics(Some(recorder), id, boom())).expect_err("panicked");
    assert_eq!(error.kind, crate::error::ErrorKind::Internal);
    let record = buffer.take(id).expect("resolved");
    assert_eq!(json(&record.outcome), json(&Err::<Answered, _>(error)));

    let unrecorded = block_on(catch_panics(None, id, boom()));
    assert!(unrecorded.is_err_and(|error| error.to_string().contains("panicked: boom")));
    assert_eq!(
        block_on(catch_panics(None, id, async { Ok(7) })).ok(),
        Some(7)
    );
}
