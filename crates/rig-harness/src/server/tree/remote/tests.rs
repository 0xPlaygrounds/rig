use bevy_ecs::prelude::*;
use rig_protocol::methods::{Command, Stop, StopScope, WorkerCommand};
use rig_protocol::{CommandSeq, EventRecord, Key, Seq, SessionId};
use serde_json::json;

use super::*;
use crate::server::record::Track;

fn record(kind: &str, data: serde_json::Value) -> EventRecord {
    EventRecord {
        session: SessionId("remote".into()),
        seq: Seq(1),
        at_ms: 0,
        kind: kind.into(),
        data,
        source: None,
    }
}

fn proxied(world: &mut World) -> (Entity, Receiver<WorkerCommand>) {
    let (commands, queued) = crossbeam_channel::unbounded();
    let session = world
        .spawn((
            Track::default(),
            RemoteSession {
                host: "bench".into(),
                remote: Some(SessionId("remote".into())),
                commands,
                // The remote numbered the local sends 3 and 5 as 1 and 2.
                sends: vec![(1, 3), (2, 5)],
            },
        ))
        .id();
    (session, queued)
}

#[test]
fn a_remote_turns_end_answers_the_local_command_it_covers() {
    let mut world = World::new();
    let (session, _) = proxied(&mut world);
    let through = |world: &World, remote: u64| {
        mirrored(
            world,
            session,
            record("turn.ended", json!({ "through": remote })),
        )
        .and_then(|d| d.data.get("through").and_then(serde_json::Value::as_u64))
    };
    assert_eq!(through(&world, 1), Some(3));
    assert_eq!(through(&world, 2), Some(5));
    assert_eq!(through(&world, 0), Some(0));
}

#[test]
fn the_mirror_keeps_only_what_the_local_journal_does_not_make_itself() {
    let mut world = World::new();
    let (session, _) = proxied(&mut world);
    for kind in [
        "session.started",
        "command.accepted",
        "worker.ready",
        "worker.exited",
    ] {
        assert!(
            mirrored(&world, session, record(kind, json!({}))).is_none(),
            "{kind}"
        );
    }
    assert!(mirrored(&world, session, record("message", json!({ "text": "hi" }))).is_some());
    // An ask answered here already resolved it; the remote's echo is dropped.
    let resolved = record(
        "ask.resolved",
        json!({ "askId": "a-1", "by": "client:c-1" }),
    );
    assert!(mirrored(&world, session, resolved).is_none());
}

#[test]
fn commands_reach_the_proxy_in_order() {
    let mut world = World::new();
    let (session, queued) = proxied(&mut world);
    for n in 1..=3 {
        let stop = Stop {
            session_id: SessionId("local".into()),
            scope: StopScope::Turn,
            key: Key(format!("k{n}")),
        };
        let command = WorkerCommand {
            command_seq: CommandSeq(n),
            command: Command::Stop(stop),
        };
        assert!(forward(&mut world, session, command));
    }
    let order: Vec<u64> = queued.try_iter().map(|c| c.command_seq.0).collect();
    assert_eq!(order, [1, 2, 3]);
    let local = world.spawn_empty().id();
    let stop = Stop {
        session_id: SessionId("x".into()),
        scope: StopScope::Turn,
        key: Key("k".into()),
    };
    assert!(!forward(
        &mut world,
        local,
        WorkerCommand {
            command_seq: CommandSeq(1),
            command: Command::Stop(stop)
        }
    ));
}
