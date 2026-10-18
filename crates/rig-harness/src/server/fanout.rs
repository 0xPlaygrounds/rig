//! Fan-out: each subscription has a cursor, and each frame it is sent the
//! records after it, from the session's recent ring or, behind that, its
//! journal file, at most [`PER_FRAME`] at a time and only while its
//! client's out-queue has room. Replay, live delivery and backpressure are
//! the one loop: a slow client's subscription waits with its cursor held
//! and costs only its own disk reads once it drains.

use std::time::{Duration, Instant};

use bevy_ecs::prelude::*;
use rig_protocol::methods::{EVENT, EventNotice};
use rig_protocol::{EventRecord, Frame, Notification, Seq};

use super::methods::{SubscriptionNumber, Until};
use super::net;
use super::{Client, Journal, SubscribedTo, Subscription};

/// The most records sent on one subscription per frame.
pub const PER_FRAME: usize = 256;
/// How soon a subscription held back by a full queue is tried again.
const RETRY: Duration = Duration::from_millis(20);

/// Sends every subscription what it is behind on; returns when to try
/// again for one a full queue held back.
pub fn fan_out(world: &mut World) -> Option<Instant> {
    let mut subs = world.query::<(
        Entity,
        &Subscription,
        &SubscribedTo,
        &SubscriptionNumber,
        Option<&Until>,
    )>();
    let subs: Vec<_> = subs
        .iter(world)
        .map(|(e, sub, to, n, until)| (e, sub.client, sub.after, to.0, n.0, until.map(|u| u.0)))
        .collect();
    let mut retry = None;
    for (entity, client, after, session, number, until) in subs {
        let Some(out) = world.get::<Client>(client).map(|c| c.out.0.clone()) else {
            continue;
        };
        let Some(journal) = world.get::<Journal>(session) else {
            continue;
        };
        let last = Seq(journal.next_seq.0.saturating_sub(1));
        let end = until.map_or(last, |u| u.min(last));
        if after >= end {
            if until.is_some() {
                world.despawn(entity);
            }
            continue;
        }
        let records = behind(journal, after, end);
        let mut sent = after;
        for record in records {
            if !net::has_room(&out) {
                retry = Some(Instant::now() + RETRY);
                break;
            }
            let seq = record.seq;
            let notice = EventNotice {
                subscription: number,
                record,
            };
            let params = serde_json::to_value(notice).unwrap_or_default();
            let frame = Frame::Notification(Notification::new(EVENT, params));
            if net::offer(&out, frame) != net::Offered::Queued {
                break;
            }
            sent = seq;
        }
        if let Some(mut sub) = world.get_mut::<Subscription>(entity) {
            sub.after = sent;
        }
        if sent < end && retry.is_none() {
            // More than a frame's worth: the next frame goes on.
            retry = Some(Instant::now());
        }
    }
    retry
}

/// At most [`PER_FRAME`] records after `after` up to `end`.
fn behind(journal: &Journal, after: Seq, end: Seq) -> Vec<EventRecord> {
    let in_ring = journal
        .recent
        .front()
        .is_some_and(|first| first.seq.0 <= after.0 + 1);
    let records: Vec<EventRecord> = if in_ring {
        journal
            .recent
            .iter()
            .filter(|r| r.seq > after)
            .take(PER_FRAME)
            .cloned()
            .collect()
    } else {
        match &journal.file {
            Some(file) => file.read_after(after, PER_FRAME).unwrap_or_default(),
            None => Vec::new(),
        }
    };
    records.into_iter().filter(|r| r.seq <= end).collect()
}
