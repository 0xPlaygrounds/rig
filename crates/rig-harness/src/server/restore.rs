//! A starting `rigd` rebuilds every session from its journal: the session,
//! its tree, its state, its open asks, its keys and the commands its worker
//! may not have applied, and restarts the worker of every session that was
//! busy when the last server died.

use std::time::{Duration, Instant};

use bevy_ecs::prelude::*;
use rig_protocol::event::{CommandAccepted, SessionStarted};
use rig_protocol::methods::{Command, WorkerCommand};
use rig_protocol::{Seq, SessionId};

use super::dispatch::Sessions;
use super::policy::{AskDetails, OpenAsk};
use super::record::{RING, Track, note};
use super::tree::{ParentSession, SessionGrant, SessionPolicy, Spent, TreeBudget, default_policy};
use super::{Journal, Keyed, Server, Session, SessionState};

/// Loads every journaled session of the server's home.
pub fn restore(world: &mut World) {
    let Some(home) = world.get_resource::<Server>().map(|s| s.home.clone()) else {
        return;
    };
    // A dead server's workers went with it; their tools go too, before
    // any worker of this server starts (each carries the same mark).
    super::workers::kill_home_strays(&home);
    let mut parents = Vec::new();
    // Every session with a journal, saved by its worker or not.
    let ids = std::fs::read_dir(home.sessions())
        .into_iter()
        .flatten()
        .filter_map(|entry| entry.ok()?.file_name().to_str()?.parse().ok());
    for id in ids {
        let dir = home.session(&id);
        if dir.is_deleting() || !dir.path().join(rig_memory::journal::FILE).is_file() {
            continue;
        }
        let id: crate::harness_protocol::SessionId = id;
        let id = SessionId(id.as_str().to_owned());
        let file = match rig_memory::journal::Journal::open(dir.path(), &id) {
            Ok(file) => file,
            Err(e) => {
                bevy_log::warn!("rigd: session {} is not loaded: {e}", id.0);
                continue;
            }
        };
        let records = file.read_after(Seq(0), usize::MAX).unwrap_or_default();
        let Some(started) = records
            .iter()
            .find(|r| r.kind == "session.started")
            .and_then(|r| serde_json::from_value::<SessionStarted>(r.data.clone()).ok())
        else {
            continue;
        };
        let session = world
            .spawn((
                Session {
                    id: id.clone(),
                    cwd: started.cwd.clone(),
                    host: started.host.clone(),
                    build: started.build.clone(),
                    name: started.name.clone(),
                },
                SessionPolicy(default_policy(started.parent.is_some())),
                Spent::default(),
            ))
            .id();
        if let Some(grant) = started.grant.clone() {
            world.entity_mut(session).insert(SessionGrant(grant));
        }
        if started.parent.is_none() {
            world.entity_mut(session).insert(TreeBudget::default());
        }
        parents.extend(started.parent.map(|p| (session, p)));
        let mut journal = Journal {
            next_seq: Seq(file.last().0 + 1),
            ..Journal::default()
        };
        let mut entity = world.entity_mut(session);
        for record in &records {
            note(&mut entity, record);
            if record.kind != "command.accepted" {
                continue;
            }
            let Ok(accepted) = serde_json::from_value::<CommandAccepted>(record.data.clone())
            else {
                continue;
            };
            journal.last_command = journal.last_command.max(accepted.command_seq);
            let Some(command) = accepted.command else {
                continue;
            };
            if matches!(command, Command::Send(_)) {
                journal.last_message = journal.last_message.max(accepted.command_seq);
            }
            if let Some((params, result)) = keyed(&command, accepted.command_seq) {
                journal.keys.insert(
                    accepted.key.clone(),
                    Keyed {
                        method: accepted.method.clone(),
                        params,
                        result,
                    },
                );
            }
            if let Some(mut track) = entity.get_mut::<Track>() {
                track.commands.push_back(WorkerCommand {
                    command_seq: accepted.command_seq,
                    command,
                });
            }
        }
        let skip = records.len().saturating_sub(RING);
        journal.recent = records.into_iter().skip(skip).collect();
        journal.file = Some(file);
        entity.insert(journal);
        world
            .get_resource_or_init::<Sessions>()
            .0
            .insert(id, session);
    }
    for (child, parent) in parents {
        if let Some(parent) = super::find(world, &parent) {
            world.entity_mut(child).insert(ParentSession(parent));
        }
    }
    let mut sessions = world.query::<(Entity, &SessionState, &Track)>();
    let sessions: Vec<(Entity, SessionState, Vec<AskDetails>)> = sessions
        .iter(world)
        .map(|(e, state, track)| {
            let asks = track
                .asks
                .iter()
                .filter_map(|ask| serde_json::to_value(&ask.ask).ok())
                .filter_map(|data| AskDetails::from_data(&data))
                .collect();
            (e, *state, asks)
        })
        .collect();
    for (session, state, asks) in sessions {
        for details in asks {
            let timeout = super::tree::policy_of(world, session).timeout_s;
            let ask = world
                .spawn((
                    OpenAsk {
                        id: details.opened.ask_id.clone(),
                        session,
                        agent: details.opened.agent_id.clone(),
                        deadline: Instant::now() + Duration::from_secs(timeout),
                        answerers: details.opened.answerers,
                    },
                    details,
                ))
                .id();
            super::policy::on_ask(world, ask);
        }
        let busy = matches!(
            state,
            SessionState::Running
                | SessionState::NeedsApproval
                | SessionState::NeedsAnswer
                | SessionState::Stopping
        );
        if busy && let Err(e) = super::workers::spawn_worker(world, session) {
            bevy_log::warn!(
                "rigd: a restored session's worker did not start: {}",
                e.message
            );
        }
    }
}

/// The params and result a repeated key is answered with.
fn keyed(
    command: &Command,
    seq: rig_protocol::CommandSeq,
) -> Option<(serde_json::Value, serde_json::Value)> {
    use rig_protocol::methods::{Disposition, Resolved, Sent, Stopping};
    let (params, result) = match command {
        Command::Send(send) => (
            serde_json::to_value(send).ok()?,
            serde_json::to_value(Sent {
                command_seq: seq,
                disposition: Disposition::Started,
            })
            .ok()?,
        ),
        Command::Answer(answer) => (
            serde_json::to_value(answer).ok()?,
            serde_json::to_value(Resolved { resolved: true }).ok()?,
        ),
        Command::Stop(stop) => (
            serde_json::to_value(stop).ok()?,
            serde_json::to_value(Stopping {
                stopping: vec![stop.session_id.clone()],
            })
            .ok()?,
        ),
        Command::Reload(reload) => (
            serde_json::to_value(reload).ok()?,
            serde_json::to_value(rig_protocol::methods::Reloading { command_seq: seq }).ok()?,
        ),
        Command::Plugin { .. } => return None,
    };
    Some((params, result))
}
