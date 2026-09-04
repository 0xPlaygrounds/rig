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
use rig_ecs::{
    agent::{
        Assembling, Cursor, Failed, MessageParts, Run, RunOf, Settled,
        scene::{WorldScene, load_world, save_world},
    },
    bus::{EffectLogResource, EffectOutcome, IdCounter, RigSchedule},
    replay::{stamp_header, stamp_run},
    systems::{Fresh, spawn_run},
};
use rig_effect_log::{Checkpoint, EffectLog, RequestCheck};

use super::{
    Against, Program, assert_same_records, golden, program_hooks,
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
    } = open(program, &log, check);
    let world = app.world_mut();
    super::world_hooks::install(world, program);
    let agent = spawn_agent(world, program, &handlers);
    stamp_header(
        world,
        agent,
        &world.resource::<EffectLogResource>().0.clone(),
        log.header.bus,
        program_hooks(program, program.owner),
    );
    let history: Vec<MessageParts> = program
        .history
        .map(|history| {
            history()
                .iter()
                .filter_map(MessageParts::from_message)
                .collect()
        })
        .unwrap_or_default();
    let run = spawn_run(
        world,
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
    stamp_run(world, run, &world.resource::<EffectLogResource>().0.clone());
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
        let at_cut = world.get::<Assembling>(run).is_some()
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
    let (checkpoint, tail) = log.checkpoint(at, serde_json::to_value(&scene).expect("serde"));
    let checkpoint: Checkpoint =
        serde_json::from_str(&serde_json::to_string(&checkpoint).expect("serde"))
            .expect("a checkpoint restores");
    assert_eq!(checkpoint.at, at);
    let scene: WorldScene =
        serde_json::from_value(checkpoint.state.clone()).expect("the state is a world scene");
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
    assert_partial_header(&head, &log, program, "head");

    // The tail: a fresh world over the continuation's replayers, the scene
    // loaded, the hooks installed after it (no run-start fires), the run
    // ticked to the golden's ending.
    let Opened {
        mut app,
        handlers: _,
        reached,
    } = open(program, &continuation, check);
    let world = app.world_mut();
    world.resource_mut::<IdCounter>().0 = next_id;
    let loaded = load_world(&scene, world).expect("the scene's handlers are bound");
    let run = loaded
        .graph
        .iter()
        .copied()
        .find(|entity| world.get::<Run>(*entity).is_some())
        .expect("the scene holds the run");
    let agent = world.get::<RunOf>(run).expect("the run's agent").0;
    super::world_hooks::install(world, program);
    stamp_header(
        world,
        agent,
        &world.resource::<EffectLogResource>().0.clone(),
        log.header.bus,
        program_hooks(program, program.owner),
    );
    stamp_run(world, run, &world.resource::<EffectLogResource>().0.clone());
    assert!(
        drive(&mut app, program, run, start, &continuation, &reached),
        "{}: a resumed program does not cancel when reached",
        program.fixture
    );
    assert_ending(&app, program, run, &log);
    let tail = app.world().resource::<EffectLogResource>().log();
    assert_same_records(&tail, &continuation, "world resume (tail)");
    assert_partial_header(&tail, &log, program, "tail");
    // One record sequence: head then tail is the golden.
    let mut whole = head;
    whole.records.extend(tail.records);
    assert_same_records(&whole, &log, "world resume");
}

/// A part's header is the golden's but for the signature, which is read
/// off the records: the part's keys are among the golden's (a route the
/// tail selects, a note the head's outcome hook dispatched, are one
/// side's).
fn assert_partial_header(part: &EffectLog, log: &EffectLog, program: &Program, which: &str) {
    assert_eq!(
        part.header.run_spec, log.header.run_spec,
        "{}: the {which}'s spec hash",
        program.fixture
    );
    assert_eq!(
        part.header.hooks, log.header.hooks,
        "{}: the {which}'s hook list",
        program.fixture
    );
    assert_eq!(
        part.header.required, log.header.required,
        "{}: the {which}'s required row",
        program.fixture
    );
    for (key, family) in part.header.signature.iter() {
        assert_eq!(
            log.header.signature.get(key),
            Some(family),
            "{}: the {which} performed `{key}`, which the golden did not",
            program.fixture
        );
    }
}
