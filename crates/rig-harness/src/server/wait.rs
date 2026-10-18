//! `rig/session/wait`: answered when the session's turn covering the
//! awaited command ends, an ask opens, the session stops or faults, or the
//! wait times out. A wait that can't be answered yet is an entity, settled
//! each frame after the frame's events are journaled.

use std::time::{Duration, Instant};

use bevy_ecs::prelude::*;
use rig_protocol::event::TurnStatus;
use rig_protocol::methods::{AskInfo, Wait, WaitResult, WaitStatus};
use rig_protocol::{CommandSeq, RequestId, RpcError, codes};

use super::dispatch::{find, not_found, respond};
use super::record::Track;
use super::{Journal, SessionState};
use crate::channel::{Caller, Reply};

/// A wait not answered yet.
#[derive(Component)]
pub struct PendingWait {
    pub client: Entity,
    pub id: RequestId,
    pub session: Entity,
    pub after: CommandSeq,
    pub deadline: Option<Instant>,
}

/// Answers a wait at once when it can; otherwise keeps it and answers
/// nothing now.
pub fn begin(
    world: &mut World,
    caller: Caller,
    id: RequestId,
    params: serde_json::Value,
) -> Option<Reply> {
    let wait: Wait = match serde_json::from_value(params) {
        Ok(wait) => wait,
        Err(e) => {
            return Some(Err(RpcError::new(
                codes::INVALID_PARAMS,
                format!("rig/session/wait: {e}"),
            )));
        }
    };
    let Some(session) = find(world, &wait.session_id) else {
        return Some(Err(not_found(&wait.session_id.0)));
    };
    let latest = world
        .get::<Journal>(session)
        .map_or(CommandSeq(0), |journal| journal.last_message);
    let after = wait.after_command.unwrap_or(latest);
    if let Some(result) = settled(world, session, after) {
        return Some(encode(result));
    }
    let deadline = wait
        .timeout_ms
        .map(|ms| Instant::now() + Duration::from_millis(ms));
    world.spawn(PendingWait {
        client: caller.client,
        id,
        session,
        after,
        deadline,
    });
    None
}

fn encode(result: WaitResult) -> Reply {
    serde_json::to_value(result).map_err(|e| RpcError::new(codes::INTERNAL, e.to_string()))
}

/// The answer to a wait for `after` on `session`, if it has one now.
pub fn settled(world: &World, session: Entity, after: CommandSeq) -> Option<WaitResult> {
    let track = world.get::<Track>(session)?;
    let state = *world.get::<SessionState>(session)?;
    let journal = world.get::<Journal>(session)?;
    let result = |status, answer: Option<String>, ask: Option<AskInfo>, error| WaitResult {
        status,
        answer,
        ask,
        error,
        seq: rig_protocol::Seq(journal.next_seq.0.saturating_sub(1)),
        command_seq: journal.last_command,
    };
    if let Some(ask) = track.asks.first() {
        let status = match ask.ask.kind {
            rig_protocol::event::AskKind::Approval => WaitStatus::NeedsApproval,
            rig_protocol::event::AskKind::Question => WaitStatus::NeedsAnswer,
        };
        return Some(result(status, None, Some(ask.clone()), None));
    }
    if let Some(end) = &track.ended
        && end.through >= after
    {
        let status = match end.status {
            TurnStatus::Answered => WaitStatus::Answered,
            TurnStatus::Failed => WaitStatus::Failed,
            TurnStatus::Stopped => WaitStatus::Stopped,
        };
        return Some(result(status, end.answer.clone(), None, end.error.clone()));
    }
    match state {
        SessionState::Stopped => Some(result(WaitStatus::Stopped, None, None, None)),
        SessionState::Faulted => Some(result(
            WaitStatus::Failed,
            None,
            None,
            Some("the session's worker crashed repeatedly; rig log shows why".to_owned()),
        )),
        SessionState::Idle if after == CommandSeq(0) => {
            let answer = track.ended.as_ref().and_then(|end| end.answer.clone());
            Some(result(WaitStatus::Answered, answer, None, None))
        }
        _ => None,
    }
}

/// Answers every wait that can be answered or has timed out; returns the
/// earliest deadline still pending.
pub fn settle(world: &mut World) -> Option<Instant> {
    let mut waits = world.query::<(Entity, &PendingWait)>();
    let waits: Vec<(Entity, Entity, CommandSeq, Option<Instant>)> = waits
        .iter(world)
        .map(|(entity, wait)| (entity, wait.session, wait.after, wait.deadline))
        .collect();
    let now = Instant::now();
    let mut next = None::<Instant>;
    for (entity, session, after, deadline) in waits {
        let answer = settled(world, session, after).or_else(|| {
            deadline.filter(|d| *d <= now).map(|_| {
                let journal = world.get::<Journal>(session);
                WaitResult {
                    status: WaitStatus::TimedOut,
                    answer: None,
                    ask: None,
                    error: None,
                    seq: rig_protocol::Seq(journal.map_or(0, |j| j.next_seq.0.saturating_sub(1))),
                    command_seq: journal.map_or(CommandSeq(0), |j| j.last_command),
                }
            })
        });
        match answer {
            Some(answer) => {
                if let Some(wait) = world.entity_mut(entity).take::<PendingWait>() {
                    world.despawn(entity);
                    respond(world, wait.client, wait.id, encode(answer));
                }
            }
            None => {
                if let Some(deadline) = deadline {
                    next = Some(next.map_or(deadline, |n| n.min(deadline)));
                }
            }
        }
    }
    next
}
