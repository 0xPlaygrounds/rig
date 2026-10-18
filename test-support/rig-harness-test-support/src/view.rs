//! A plugin's agent half and view half in one test: what the agent half
//! publishes ([`WorkerOutbox`]) as the records a view applies
//! ([`records`]), or straight to the app's own view halves ([`loopback`]).

use rig_ecs::prelude::*;
use rig_harness::view::Applied;
use rig_harness::view::protocol::methods::WorkerEvent;
use rig_harness::view::protocol::{EventRecord, Seq, SessionId};
use rig_harness::worker::WorkerOutbox;

/// `events` as a session's records, numbered from `first`, as `rigd`
/// journals them, for `rig_harness::view::mirror::apply`.
pub fn records(events: impl IntoIterator<Item = WorkerEvent>, first: u64) -> Vec<EventRecord> {
    events
        .into_iter()
        .zip(first..)
        .map(|(event, seq)| EventRecord {
            session: SessionId("s-1".into()),
            seq: Seq(seq),
            at_ms: 0,
            kind: event.kind,
            data: event.data,
            source: event.source,
        })
        .collect()
}

/// Makes `app` its own view: the events its agent halves publish reach
/// its view halves as [`Applied`] the next frame. Plugin events only: the
/// app's agents are themselves, so no mirror is built.
pub fn loopback(app: &mut App) -> &mut App {
    app.init_resource::<WorkerOutbox>()
        .add_message::<Applied>()
        .add_systems(First, reflect)
}

/// Turns the outbox into [`Applied`] records.
fn reflect(
    mut outbox: ResMut<WorkerOutbox>,
    mut applied: MessageWriter<Applied>,
    mut next: Local<u64>,
) {
    let events = std::mem::take(&mut outbox.events);
    let first = (*next).max(1);
    for record in records(events, first) {
        *next = record.seq.0 + 1;
        applied.write(Applied(record));
    }
}
