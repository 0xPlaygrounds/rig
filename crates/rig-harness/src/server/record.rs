//! Journaling: every event of a session gets its `seq` here, is written to
//! the session's journal (fsynced) and kept in its recent ring, and moves
//! what `rigd` knows of the session (its state, its open asks, its last
//! answer) before anyone sees it.

use std::collections::{BTreeMap, VecDeque};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use bevy_ecs::prelude::*;
use rig_protocol::event::{AskKind, AskOpened, AskResolved, TurnEnded, TurnStatus, Usage};
use rig_protocol::methods::{AskInfo, Unload, WorkerCommand};
use rig_protocol::{CommandSeq, EventRecord, RpcError, Seq, SourceKey, codes};

use super::{Journal, Session, SessionState};

/// The most records a session keeps in memory for its subscribers.
pub const RING: usize = 256;

/// What `rigd` keeps of a session beside its journal.
#[derive(Component, Default)]
pub struct Track {
    /// Accepted commands the worker has not applied yet, in order.
    pub commands: VecDeque<WorkerCommand>,
    /// The last command the worker said it applied.
    pub applied: CommandSeq,
    /// The last saved-log record journaled, per log.
    pub known: BTreeMap<String, u64>,
    /// The last turn that ended.
    pub ended: Option<Ended>,
    /// Open asks, oldest first.
    pub asks: Vec<AskInfo>,
    pub tokens: u64,
    pub cost_usd: Option<f64>,
    pub unload: Option<Unload>,
    /// Restarts of the worker, for the crash limit.
    pub crashes: VecDeque<Instant>,
    /// Asked to stop the whole session: its worker exits and stays down.
    pub stopping: bool,
}

/// A turn's end.
#[derive(Clone, Debug)]
pub struct Ended {
    pub through: CommandSeq,
    pub status: TurnStatus,
    pub answer: Option<String>,
    pub error: Option<String>,
}

/// Now, in milliseconds since the epoch.
pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
        .unwrap_or_default()
}

/// One event before it gets its seq.
pub struct Draft {
    pub kind: String,
    pub data: serde_json::Value,
    pub source: Option<SourceKey>,
}

impl Draft {
    pub fn new(kind: &str, data: impl serde::Serialize) -> Draft {
        Draft {
            kind: kind.to_owned(),
            data: serde_json::to_value(data).unwrap_or_default(),
            source: None,
        }
    }
}

/// Journals `drafts` on `session` as one fsynced batch, in order, and
/// returns the last seq. Durable drafts already journaled (their source at
/// or before what is known) are dropped.
pub fn publish(world: &mut World, session: Entity, drafts: Vec<Draft>) -> Result<Seq, RpcError> {
    let Ok(mut entity) = world.get_entity_mut(session) else {
        return Err(RpcError::new(codes::NOT_FOUND, "no such session"));
    };
    let Some(id) = entity.get::<Session>().map(|s| s.id.clone()) else {
        return Err(RpcError::new(codes::NOT_FOUND, "no such session"));
    };
    let mut records = Vec::with_capacity(drafts.len());
    {
        let Some(mut track) = entity.get_mut::<Track>() else {
            return Err(RpcError::new(codes::NOT_FOUND, "no such session"));
        };
        // One saved record may export several events under one key, so a
        // re-export is judged against what was known before this batch.
        let before = track.known.clone();
        for draft in drafts {
            if let Some(source) = &draft.source {
                let known = before.get(&source.log).copied();
                if known.is_some_and(|known| source.index <= known) {
                    continue;
                }
                let known = track.known.entry(source.log.clone()).or_default();
                *known = (*known).max(source.index);
            }
            records.push(draft);
        }
    }
    let Some(mut journal) = entity.get_mut::<Journal>() else {
        return Err(RpcError::new(codes::NOT_FOUND, "no such session"));
    };
    let first = journal.next_seq.0.max(1);
    let at_ms = now_ms();
    let records: Vec<EventRecord> = records
        .into_iter()
        .zip(first..)
        .map(|(draft, seq)| EventRecord {
            session: id.clone(),
            seq: Seq(seq),
            at_ms,
            kind: draft.kind,
            data: draft.data,
            source: draft.source,
        })
        .collect();
    if records.is_empty() {
        return Ok(Seq(first - 1));
    }
    if let Some(file) = journal.file.as_mut() {
        file.append(&records)
            .map_err(|e| RpcError::new(codes::INTERNAL, format!("the journal: {e}")))?;
    }
    let last = first + records.len() as u64 - 1;
    journal.next_seq = Seq(last + 1);
    for record in &records {
        journal.recent.push_back(record.clone());
        if journal.recent.len() > RING {
            journal.recent.pop_front();
        }
    }
    for record in &records {
        note(&mut entity, record);
    }
    Ok(Seq(last))
}

/// Moves the session's state for one journaled record: what replay on
/// start-up does too.
pub fn note(entity: &mut EntityWorldMut, record: &EventRecord) {
    let id = entity.get::<Session>().map(|s| s.id.clone());
    let Some(mut track) = entity.get_mut::<Track>() else {
        return;
    };
    if let Some(source) = &record.source {
        let known = track.known.entry(source.log.clone()).or_default();
        *known = (*known).max(source.index);
    }
    let state = match record.kind.as_str() {
        "turn.started" => Some(SessionState::Running),
        "command.accepted" => Some(SessionState::Running),
        "turn.ended" => {
            if let Ok(end) = serde_json::from_value::<TurnEnded>(record.data.clone()) {
                track.ended = Some(Ended {
                    through: end.through,
                    status: end.status,
                    answer: end.answer,
                    error: end.error,
                });
            }
            Some(if track.commands.is_empty() {
                SessionState::Idle
            } else {
                SessionState::Running
            })
        }
        "ask.opened" => {
            let opened = serde_json::from_value::<AskOpened>(record.data.clone()).ok();
            opened.zip(id).map(|(ask, session_id)| {
                let state = match ask.kind {
                    AskKind::Approval => SessionState::NeedsApproval,
                    AskKind::Question => SessionState::NeedsAnswer,
                };
                track.asks.retain(|a| a.ask.ask_id != ask.ask_id);
                track.asks.push(AskInfo { session_id, ask });
                state
            })
        }
        "ask.resolved" => {
            if let Ok(resolved) = serde_json::from_value::<AskResolved>(record.data.clone()) {
                track.asks.retain(|a| a.ask.ask_id != resolved.ask_id);
            }
            track.asks.is_empty().then_some(SessionState::Running)
        }
        "usage" => {
            if let Ok(usage) = serde_json::from_value::<Usage>(record.data.clone()) {
                track.tokens += usage.tokens;
                if let Some(cost) = usage.cost_usd {
                    track.cost_usd = Some(track.cost_usd.unwrap_or_default() + cost);
                }
            }
            None
        }
        "session.stopped" => Some(SessionState::Stopped),
        "session.faulted" => Some(SessionState::Faulted),
        _ => None,
    };
    if let Some(state) = state {
        entity.insert(state);
    }
}
