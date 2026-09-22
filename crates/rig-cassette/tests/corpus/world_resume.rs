//! Resume as a scene load (CONTRACT §13): the world interpreter takes the
//! program through `tool_turns` tool turns' results, saves the world as a
//! `WorldScene`, and a fresh world over the log's tail loads it and ticks
//! to the golden's ending. The checkpoint is a cut of the log with the
//! scene as its `state`, round-tripped as JSON — a process image: nothing
//! of the first world survives but the two strings.
//!
//! What a resumed world does that the frozen engine's resumed run did
//! not: it keeps its state. The run loaded from the scene is still
//! `Remembering`, so the append is the resumed run's; the tail's records
//! are the golden's from the cut, one record sequence with the head.

use std::time::Instant;

use bevy_ecs::prelude::*;
use rig_cassette::ecs::EffectLogResource;
use rig_cassette::ecs::identity::stamp_run;
use rig_cassette::effect_log::{Checkpoint, EffectLog, RequestCheck};
use rig_ecs::{
    agent::{Cursor, Failed, MessageParts, Run, RunPhase, Settled},
    bus::{EffectOutcome, IdCounter, RigSchedule},
    checkpoint::{RestoreMode, load_world, save_world},
    systems::{Fresh, RunCommands},
};

use super::{
    Against, Program, assert_same_records, golden,
    world::{GUARD, Opened, assert_ending, drive, open, spawn_agent},
};

/// The world interpreter resumed: `program` through `tool_turns` tool
/// turns in one world, the rest in another, over the checkpoint's
/// continuation replayed under `check`. `Against::FullLog` asserts the
/// refusal and stops.
pub fn world_resume_reproduces(
    program: &Program,
    tool_turns: usize,
    check: RequestCheck,
    against: Against,
) {
    assert!(
        program.second_prompt.is_none(),
        "a resumed program is one run"
    );
    let log = golden(program.fixture);
    let start = Instant::now();

    // The head: the program's world, to the cut.
    let Opened {
        mut app,
        handlers,
        reached: _,
        asks: _,
    } = open(program, &log, check);
    let world = app.world_mut();
    super::world_hooks::install(world, program);
    let agent = spawn_agent(world, program, &handlers);
    let history: Vec<MessageParts> = program
        .history
        .map(|history| {
            history()
                .iter()
                .filter_map(MessageParts::from_message)
                .collect()
        })
        .unwrap_or_default();
    let run = world.spawn_run(
        agent,
        &history,
        program.prompt,
        program.streamed,
        program.max_turns,
    );
    if let Some(concurrency) = program.tool_concurrency {
        world
            .entity_mut(run)
            .insert(rig_ecs::agent::ToolPolicy { concurrency });
    }
    stamp_run(world, run, &world.resource::<EffectLogResource>().0.clone())
        .expect("the run stamps its program identity");
    // One pass of the schedule at a time (an `update` runs it to
    // quiescence), until `tool_turns` batches landed and the run wants its
    // next turn: `Assembling`, the cursor at `tool_turns`, no fresh turn
    // yet — the moment `land_batch` leaves, before `Advance`.
    loop {
        app.world_mut().run_schedule(RigSchedule);
        let world = app.world_mut();
        assert!(
            world.get::<Settled>(run).is_none() && world.get::<Failed>(run).is_none(),
            "{}: the run ended before its {tool_turns} tool turn(s): {:?}",
            program.fixture,
            world.get::<Failed>(run)
        );
        let at_cut = world.get::<RunPhase>(run) == Some(&RunPhase::Assembling)
            && world
                .get::<Cursor>(run)
                .is_some_and(|cursor| cursor.turn == tool_turns)
            && !world
                .query_filtered::<&ChildOf, With<Fresh>>()
                .iter(world)
                .any(|child_of| child_of.parent() == run);
        if at_cut {
            break;
        }
        assert!(
            start.elapsed() < GUARD,
            "{}: the cut was not reached within {GUARD:?}",
            program.fixture
        );
        std::thread::yield_now();
    }
    // No open record at the cut: every answered effect has been settled
    // (`InFlight` gone), so the head's log holds every answer given.
    let unsettled = app
        .world_mut()
        .query_filtered::<(), (With<EffectOutcome>, With<rig_ecs::bus::InFlight>)>()
        .iter(app.world())
        .count();
    assert_eq!(
        unsettled, 0,
        "{}: {unsettled} record(s) open at the cut",
        program.fixture
    );
    let next_id = app.world().resource::<IdCounter>().0;
    let scene = save_world(app.world_mut()).expect("every component serializes");
    let head = app.world().resource::<EffectLogResource>().log();
    drop(app);

    // The process image: the head's log and the checkpoint (the scene as
    // its state) as JSON, and nothing else.
    let head: EffectLog = serde_json::from_str(&serde_json::to_string(&head).expect("serde"))
        .expect("the head log restores");
    let at = head.records.len();
    let (checkpoint, tail) = log.checkpoint(at, scene);
    let checkpoint: Checkpoint<rig_ecs::checkpoint::Checkpoint> =
        serde_json::from_str(&serde_json::to_string(&checkpoint).expect("serde"))
            .expect("a checkpoint restores");
    assert_eq!(checkpoint.at, at);
    let scene = checkpoint.state.clone();
    let continuation = match against {
        Against::Tail => {
            EffectLog::from_checkpoint(&checkpoint, tail).expect("the tail follows its checkpoint")
        }
        Against::FullLog => {
            let refused = EffectLog::from_checkpoint(&checkpoint, log.clone())
                .expect_err("a full log is not the tail");
            assert!(
                refused.message.starts_with(&format!(
                    "resume refused: the checkpoint at {at} expects record"
                )) && refused
                    .message
                    .ends_with(&format!("the tail begins at {}", log.records[0].id)),
                "{}",
                refused.message
            );
            return;
        }
    };
    // The head's records are the golden's, to the cut.
    let mut prefix = log.clone();
    prefix.records.truncate(at);
    assert_same_records(&head, &prefix, "world resume (head)");

    // The tail: a fresh world over the continuation's replayers, the scene
    // loaded, the hooks installed after it (no run-start fires), the run
    // ticked to the golden's ending.
    let Opened {
        mut app,
        handlers: tail_handlers,
        reached,
        asks,
    } = open(program, &continuation, check);
    // A key the head dispatched to before the cut and the continuation
    // never does (an outcome hook's note) has no replayer in the tail
    // world, yet the checkpoint still requires it. The host builds those
    // from the recording — never from a live provider — and hands them to
    // the load, which installs them with the state they serve.
    let served: std::collections::HashSet<_> =
        tail_handlers.iter().map(|(key, _)| key.clone()).collect();
    let supplied: Vec<_> = scene
        .requirements()
        .expect("the scene's requirements")
        .into_iter()
        .filter(|descriptor| !served.contains(&descriptor.key))
        .map(|descriptor| {
            let handler =
                super::world::replayer_handler(program, &log, check, &descriptor.key, &asks);
            (descriptor.key, handler)
        })
        .collect();
    let world = app.world_mut();
    world.resource_mut::<IdCounter>().0 = next_id;
    // `Strict`: a resumed world serves exactly the implementations the head
    // saved, whether it already had them or the host supplied them here.
    let loaded = load_world(&scene, world, RestoreMode::Strict, supplied).unwrap_or_else(|error| {
        panic!(
            "{}: the scene's handlers are bound: {error}",
            program.fixture
        )
    });
    let run = loaded
        .with::<Run>(world)
        .first()
        .copied()
        .expect("the scene holds the run");
    super::world_hooks::install(world, program);
    stamp_run(world, run, &world.resource::<EffectLogResource>().0.clone())
        .expect("the run stamps its program identity");
    assert!(
        drive(&mut app, program, run, start, &continuation, &reached),
        "{}: a resumed program does not cancel when reached",
        program.fixture
    );
    assert_ending(&app, program, run, &log);
    let tail = app.world().resource::<EffectLogResource>().log();
    assert_same_records(&tail, &continuation, "world resume (tail)");
    // One record sequence: head then tail is the golden.
    let mut whole = head;
    whole.records.extend(tail.records);
    assert_same_records(&whole, &log, "world resume");
}
