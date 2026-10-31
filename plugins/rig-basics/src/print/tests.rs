use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use bevy_tasks::futures_lite::stream;
use bevy_tasks::{IoTaskPool, TaskPool};
use rig_core::completion::{CompletionResponse, Message, Usage};
use rig_core::message::{AssistantContent, Origin as Made, Text};
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness::harness_protocol::Invocation;
use rig_harness::host::exit::ExitPlugin;
use rig_harness_test_support::{
    self as test_support, MemoryStore, app_on, calls, connect, first_agent, reply, scripted_models,
    start_call,
};

use super::*;
use crate::approval::{ApprovalPlugin, NeedsApproval};

mod tools;

/// The model the scripted runs choose.
const MODEL: &str = "test/model";

/// How the runs ended, and each turn that ended.
#[derive(Resource, Default)]
struct Ends(Vec<HeadlessEnded>, Vec<HeadlessTurnEnded>);

/// An app on `store` that keeps what [`Ends`] holds, `invoked` as its
/// command line.
fn noting(store: &MemoryStore, invoked: Invocation) -> (App, Receiver<()>) {
    let (mut app, wakes) = app_on(store);
    app.insert_resource(Invoked {
        args: invoked,
        terminal: false,
    })
    .add_plugins(AgentPlugin)
    .init_resource::<Ends>()
    .add_observer(|ended: On<HeadlessEnded>, mut ends: ResMut<Ends>| {
        ends.0.push(ended.event().clone());
    })
    .add_observer(|ended: On<HeadlessTurnEnded>, mut ends: ResMut<Ends>| {
        ends.1.push(ended.event().clone());
    });
    (app, wakes)
}

/// A run of `prompt` after its first frame and `setup`; with `model`,
/// named on the command line and connected.
fn started(
    prompt: &str,
    model: Option<&MockCompletionModel>,
    setup: impl FnOnce(&mut World, Entity),
) -> App {
    let named = model.map(|_| "deepseek/deepseek-flash".to_owned());
    let invoked = Invocation {
        model: named,
        ..Invocation::default()
    };
    let (mut app, _) = noting(&MemoryStore::default(), invoked);
    app.add_plugins(ExitPlugin)
        .run_headless(Some(prompt.to_owned()))
        .update();
    let world = app.world_mut();
    let agent = first_agent(world).unwrap_or(Entity::PLACEHOLDER);
    if let Some(connection) = model.and_then(connect) {
        world.entity_mut(agent).insert(connection);
    }
    setup(world, agent);
    app
}

/// How the run of [`started`] ended: each end's answer, code and why.
fn run(
    prompt: &str,
    model: Option<&MockCompletionModel>,
    setup: impl FnOnce(&mut World, Entity),
) -> Vec<(Option<String>, u8, Option<String>)> {
    let mut app = started(prompt, model, setup);
    ended(&mut app)
}

/// Runs `app` to its end, then how it ended, as [`run`] says it.
fn ended(app: &mut App) -> Vec<(Option<String>, u8, Option<String>)> {
    run_until(app, |_| false);
    // Nothing more once it ended.
    app.update();
    let ends = app.world().resource::<Ends>().0.iter().cloned();
    let how = |ended: HeadlessEnded| (ended.answer, ended.code, ended.why);
    ends.map(how).collect()
}

/// Runs `app` until `done`, it exits, or ten seconds passed.
fn run_until(app: &mut App, done: impl Fn(&mut World) -> bool) {
    let started = Instant::now();
    while started.elapsed() < Duration::from_secs(10)
        && app.should_exit().is_none()
        && !done(app.world_mut())
    {
        app.update();
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// A model that is never asked.
fn silent() -> MockCompletionModel {
    MockCompletionModel::from_stream_turns(Vec::<Vec<MockStreamEvent>>::new())
}

/// A model that answers each of `answers` in turn.
fn answering(answers: &[&str]) -> MockCompletionModel {
    MockCompletionModel::from_stream_turns(answers.iter().map(|answer| reply(answer)))
}

/// An answered run's end.
fn answered(answer: &str) -> Vec<(Option<String>, u8, Option<String>)> {
    vec![(Some(answer.to_owned()), 0, None)]
}

#[test]
fn the_prompt_is_answered_and_the_run_ends_beside_an_open_stream() {
    let model = answering(&["hello"]);
    let mut app = started("hi", Some(&model), |world, _| {
        // A source that never ends, such as a file watch, is no work.
        let pool = IoTaskPool::get_or_init(TaskPool::default);
        let wake = world.resource::<Wake>().clone();
        world.spawn(Running::stream(pool, &wake, stream::pending::<u32>()));
    });
    let agent = first_agent(app.world_mut());
    assert_eq!(ended(&mut app), answered("hello"));
    let ends = &app.world().resource::<Ends>().0;
    assert_eq!(ends.first().and_then(|end| end.agent), agent);
}

#[test]
fn a_turn_compacted_while_it_runs_still_prints_its_answer() {
    let model = answering(&["hello"]);
    let ends = run("hi", Some(&model), |world, primary| {
        let earlier = |In(agent): In<Entity>, mut agents: Query<&mut Conversation>| {
            if let Ok(mut conversation) = agents.get_mut(agent) {
                for n in 1..=2 {
                    conversation.push(Message::user(format!("question {n}")));
                    conversation.push(Message::assistant(format!("answer {n}")));
                }
            }
        };
        assert!(world.run_system_cached_with(earlier, primary).is_ok());
        // A compaction as the turn starts: the conversation shrinks to a
        // summary.
        world.add_observer(
            move |_: On<Add<TurnOf>>, mut agents: Query<&mut Conversation>| {
                if let Ok(mut conversation) = agents.get_mut(primary) {
                    let summary = Message::user("Summary: two questions, then hi.");
                    let compaction = Author::plugin("compaction");
                    let summary = Entry::new(summary).by(compaction.clone());
                    conversation.splice(.., [summary], compaction);
                }
            },
        );
    });
    assert_eq!(ends, answered("hello"));
}

#[test]
fn a_warning_about_the_agent_does_not_fail_the_run_and_an_error_does() {
    let codes = [NoticeLevel::Warn, NoticeLevel::Error].map(|level| {
        // A stopped turn neither answers nor fails, so only the notice decides.
        let model = silent();
        let ends = run("hi", Some(&model), |world, primary| {
            let stop = move |calls: Query<(), (ReadyToStart, With<TurnCall>)>,
                             mut commands: Commands| {
                if !calls.is_empty() {
                    commands.trigger(StopTurn::new(primary));
                }
            };
            let mut schedules = world.resource_mut::<Schedules>();
            schedules.add_systems(AgentPass, stop.in_set(Stage::Gate));
            world.add_observer(move |_: On<Add<TurnOf>>, mut commands: Commands| {
                for key in [1, 2] {
                    let text = format!("key {key} refused; using key {}", key + 1);
                    let agent = Some(primary);
                    commands.write_message(Notice { agent, text, level });
                }
            });
        });
        ends.into_iter()
            .map(|(_, code, why)| (code, why))
            .collect::<Vec<_>>()
    });
    // The last error notice is why.
    let why = "key 2 refused; using key 3".to_owned();
    assert_eq!(codes, [vec![(0, None)], vec![(1, Some(why))]]);
}

#[test]
fn a_failure_before_the_prompt_went_out_does_not_fail_an_answered_run() {
    // As a turn the restored session left fails within the first frame.
    let model = answering(&["done"]);
    let ends = run("go on", Some(&model), |world, _| {
        let mut run = world.resource_mut::<Run>();
        run.fail(1, "The restored turn failed".to_owned(), true);
    });
    assert_eq!(ends, answered("done"));
}

#[test]
fn a_failed_headless_run_names_the_unanswered_message() {
    let model = MockCompletionModel::from_stream_turns([[MockStreamEvent::error("no credit")]]);
    let lines = Lines::default();
    let invoked = Invocation {
        print: Some("Explain this failure. OPENAI_API_KEY=sk-proj-abcdefghijklmnop1234".to_owned()),
        ..Invocation::default()
    };
    let (mut app, _) = noting(&MemoryStore::default(), invoked);
    app.insert_resource(scripted_models([(MODEL, &model)]))
        .insert_resource(Printed::to(lines.clone()))
        .add_plugins((ExitPlugin, PrintPlugin));
    app.finish();
    app.cleanup();
    run_until(&mut app, |_| false);
    assert_eq!(app.should_exit(), Some(AppExit::from_code(1)));
    assert_eq!(model.request_count(), 1);
    let bytes = lines
        .0
        .lock()
        .map(|lines| lines.clone())
        .unwrap_or_default();
    let stderr = String::from_utf8_lossy(&bytes);
    assert!(
        stderr.lines().any(
            |line| line == "rig: not answered: Explain this failure. OPENAI_API_KEY=[REDACTED]"
        ),
        "{stderr}"
    );
    // There is no input to send it again from.
    assert!(!stderr.contains("back in the input"), "{stderr}");
}

#[test]
fn a_restored_model_that_cannot_be_used_fails_the_run_once() {
    let ends = run("hi", None, |world, primary| {
        let chosen = ModelChoice("nobody/nothing".to_owned());
        world.entity_mut(primary).insert(chosen);
    });
    // The notice of the refused choice is why.
    let why = "Cannot use nobody/nothing: the catalog lists no model `nobody/nothing`.";
    assert_eq!(ends, [(None, 1, Some(why.to_owned()))]);
    // Also after an earlier error about something else.
    let ends = run("hi", None, |world, primary| {
        world.write_message(Notice::error(None, "A plugin did not start."));
        let chosen = ModelChoice("nobody/nothing".to_owned());
        world.entity_mut(primary).insert(chosen);
    });
    assert_eq!(ends, [(None, 1, Some(why.to_owned()))]);
}

#[test]
fn an_empty_prompt_says_why_unless_the_run_was_kept_or_restarted() {
    let model = silent();
    let why = run(" ", Some(&model), |_, _| {});
    assert!(matches!(why.as_slice(), [(None, 2, Some(_))]), "{why:?}");
    // A run a plugin kept going, whose work ended and let it go before the
    // run first looked, as a one-shot plugin's fast turn can, had work.
    let kept = run(" ", Some(&model), |world, _| {
        let keep = world.spawn(KeepRunning).id();
        world.despawn(keep);
    });
    // A run the launcher restarted on a reload, with nothing left to do.
    let restarted = run(" ", Some(&model), |world, _| {
        world.insert_resource(AppStart(StartReason::Reloaded));
    });
    assert_eq!([kept, restarted], [[(None, 0, None)], [(None, 0, None)]]);
}

/// Whether the hold of [`the_run_waits_for_held_effects`] was released.
#[derive(Resource, Default)]
struct Released(bool);

/// The hold on the effect the run waits for.
#[derive(Component)]
struct TestHold;

/// Releases the hold a few frames after the turn ended.
fn release_late(
    turns: Query<(), With<TurnOf>>,
    holds: Query<Entity, With<TestHold>>,
    mut idle_frames: Local<u32>,
    mut released: ResMut<Released>,
    mut commands: Commands,
) {
    if !turns.is_empty() {
        return;
    }
    *idle_frames += 1;
    if *idle_frames > 5 {
        for hold in &holds {
            commands.entity(hold).despawn();
            released.0 = true;
        }
    }
}

#[test]
fn the_run_waits_for_held_effects() {
    // A plugin's effect held at Gate, then one answered and held for
    // review.
    for held_answered in [false, true] {
        let model = answering(&["hello"]);
        let ends = run("hi", Some(&model), |world, _| {
            let kind = rig_core::effect::EffectKind::Custom {
                kind: "note".into(),
                payload: serde_json::Value::Null,
            };
            let mut effect = world.spawn(EffectRequest {
                kind,
                handler: None,
            });
            if held_answered {
                effect.insert(Outcome::custom("noted"));
            }
            let effect = effect.id();
            world.spawn((TestHold, HoldOf(effect)));
            world.init_resource::<Released>();
            world
                .resource_mut::<Schedules>()
                .add_systems(Update, release_late);
            world.add_observer(|_: On<HeadlessEnded>, released: Res<Released>| {
                assert!(released.0, "the run ended while an effect was held");
            });
        });
        assert_eq!(ends, answered("hello"), "answered: {held_answered}");
    }
}

#[test]
fn a_kept_run_answers_each_prompt_and_ends_once_nothing_keeps_it() {
    let model = answering(&["one", "two"]);
    let mut app = started("hi", Some(&model), |world, _| {
        let keep = world.spawn(KeepRunning).id();
        // A plugin sends a second prompt once the first is answered, then
        // lets the run end. It counts the turns itself: the order in which
        // two observers of one event run is not specified.
        world.add_observer(
            move |ended: On<HeadlessTurnEnded>, mut seen: Local<usize>, mut commands: Commands| {
                *seen += 1;
                match *seen {
                    1 => commands.trigger(Deliver::new(ended.agent, "again", DeliveryMode::Steer)),
                    _ => commands.entity(keep).despawn(),
                }
            },
        );
    });
    assert_eq!(ended(&mut app), answered("two"));
    assert_eq!(turns(&app), [Some("one"), Some("two")]);
}

/// The answers of the turns that ended.
fn turns(app: &App) -> Vec<Option<&str>> {
    let ends = &app.world().resource::<Ends>().1;
    ends.iter().map(|turn| turn.answer.as_deref()).collect()
}

/// A turn that starts and ends between two frames of the run, as a fast
/// model's can within one run of the agent loop, is still reported, and
/// the run ends once nothing keeps it.
#[test]
fn a_turn_the_run_never_saw_working_is_still_reported() {
    let model = answering(&["one", "two"]);
    let mut app = started("hi", Some(&model), |world, _| {
        let keep = world.spawn(KeepRunning).id();
        world.add_observer(
            move |_: On<HeadlessTurnEnded>, mut seen: Local<usize>, mut commands: Commands| {
                *seen += 1;
                if *seen == 2 {
                    commands.entity(keep).despawn();
                }
            },
        );
    });
    run_until(&mut app, |world| world.resource::<Ends>().1.len() == 1);
    // The second prompt's whole turn runs in the agent loop alone.
    let world = app.world_mut();
    let agent = first_agent(world).unwrap_or(Entity::PLACEHOLDER);
    world.trigger(Deliver::new(agent, "again", DeliveryMode::Steer));
    let started = Instant::now();
    let answered_alone = |world: &mut World| {
        let mut turns = world.query_filtered::<(), With<TurnOf>>();
        let length = world
            .get::<Conversation>(agent)
            .map(|it| it.messages().len());
        length == Some(4) && turns.iter(world).next().is_none()
    };
    while started.elapsed() < Duration::from_secs(10) && !answered_alone(world) {
        world.run_schedule(AgentPass);
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(answered_alone(world), "the second turn ran");
    assert_eq!(ended(&mut app), answered("two"));
    assert_eq!(turns(&app), [Some("one"), Some("two")]);
}

/// Holds every model call of a turn, as an approval nobody gave.
fn hold_turn_calls(calls: Query<Entity, (ReadyToStart, With<TurnCall>)>, mut commands: Commands) {
    for call in &calls {
        commands.spawn(HoldOf(call));
    }
}

/// Runs a frame of `app`, then one more for each wake, as the host loop
/// does, up to 50 frames.
fn run_while_woken(app: &mut App, wakes: &Receiver<()>) {
    app.update();
    for _ in 0..50 {
        if wakes.try_recv().is_err() {
            return;
        }
        app.update();
    }
}

/// A run of `prompt` on `store`, answered by `model` as [`MODEL`], named on
/// the command line when `named`, with `systems` in its agent loop, before
/// its first frame; its loop's wakes.
fn scripted<M>(
    store: &MemoryStore,
    (model, named): (&MockCompletionModel, bool),
    prompt: Option<&str>,
    systems: impl IntoScheduleConfigs<bevy_ecs::system::ScheduleSystem, M>,
) -> (App, Receiver<()>) {
    let invoked = Invocation {
        model: named.then(|| MODEL.to_owned()),
        ..Invocation::default()
    };
    let (mut app, wakes) = noting(store, invoked);
    app.insert_resource(scripted_models([(MODEL, model)]))
        .add_plugins((SessionLogPlugin, ExitPlugin))
        .add_systems(AgentPass, systems)
        .run_headless(prompt.map(str::to_owned));
    app.finish();
    (app, wakes)
}

/// A run whose model it chooses itself sends its prompt without waiting for
/// something else to wake the loop.
#[test]
fn the_prompt_goes_out_without_waiting_for_a_wake() {
    let (model, store) = (silent(), MemoryStore::default());
    let held = hold_turn_calls.in_set(Stage::Gate);
    let (mut app, wakes) = scripted(&store, (&model, false), Some("hi"), held);
    run_while_woken(&mut app, &wakes);
    let world = app.world_mut();
    let agent = first_agent(world).unwrap_or(Entity::PLACEHOLDER);
    let messages = world.get::<Conversation>(agent).map(|conversation| {
        let messages = conversation.messages().iter();
        messages.map(|m| (**m).clone()).collect::<Vec<Message>>()
    });
    assert_eq!(messages, Some(vec![Message::user("hi")]));
}

/// `rig -r <id> -p` with no prompt, after a run that exited mid-turn: the
/// exit stopped that turn, so nothing runs and the run has no prompt.
#[test]
fn a_resumed_run_without_a_prompt_runs_nothing_of_a_turn_an_exit_stopped() {
    let (first, store) = (silent(), MemoryStore::default());
    let (mut app, wakes) = app_on(&store);
    app.insert_resource(scripted_models([(MODEL, &first)]))
        .add_plugins((AgentPlugin, SessionLogPlugin))
        .add_systems(AgentPass, hold_turn_calls.in_set(Stage::Gate));
    app.finish();
    app.update();
    let world = app.world_mut();
    let agent = first_agent(world).unwrap_or(Entity::PLACEHOLDER);
    let model = MODEL.to_owned();
    world.trigger(SetModel {
        entity: agent,
        model,
    });
    world.trigger(Deliver::new(agent, "hi", DeliveryMode::Steer));
    test_support::run_until(&mut app, &wakes, |world| {
        let held = world
            .iter_entities()
            .any(|entity| entity.contains::<HoldOf>());
        held && world.get::<ActiveTurn>(agent).is_some()
    });
    app.world_mut().write_message(AppExit::Success);
    app.update();
    drop(app);

    let second = answering(&["hello"]);
    let (mut app, _) = scripted(&store, (&second, false), None, || {});
    assert_eq!(ended(&mut app), no_prompt());
    assert_eq!((first.request_count(), second.request_count()), (0, 0));
}

/// How a run with no prompt and nothing to carry on ends.
fn no_prompt() -> Vec<(Option<String>, u8, Option<String>)> {
    let why = "no prompt: give one after --print, or pipe it in";
    vec![(None, 2, Some(why.to_owned()))]
}

/// A run asked to exit while a plugin holds the start exits without running
/// anything, and leaves the turn it restored as the session has it: the
/// next run's plugin carries it on.
#[test]
fn an_exit_during_a_held_start_leaves_the_restored_turn() {
    // The exit comes while the start is held, and as its last hold goes.
    for with_the_last in [false, true] {
        let store = MemoryStore::default();
        killed_two_frames_in(&store);
        let (second, third) = (silent(), silent());
        let (mut app, _) = scripted(&store, (&second, false), Some("again"), || {});
        let starting = app.world_mut().spawn(Starting).id();
        app.update();
        app.world_mut().trigger(Exit(ExitReason::Signal));
        if with_the_last {
            app.world_mut().despawn(starting);
        }
        run_until(&mut app, |_| false);
        assert_eq!(app.should_exit(), Some(ExitReason::Signal.exit()));
        assert_eq!(second.request_count(), 0);
        drop(app);
        let in_place = (carry_on, answer_in_place).in_set(Stage::Gate);
        let (mut app, _) = scripted(&store, (&third, false), None, in_place);
        assert_eq!(ended(&mut app), answered("hello"), "{with_the_last}");
        assert_eq!(third.request_count(), 0);
    }
}

/// A `/command` waits for the start as any prompt does: a skill's starts a
/// turn, which must see the tools that come during it.
#[test]
fn a_skill_run_waits_for_the_start() {
    let scratch = test_support::scratch_home("print-skill").root().to_owned();
    let dir = scratch.join("repo");
    let skill = dir.join(".agents/skills/haiku");
    assert!(std::fs::create_dir_all(dir.join(".git")).is_ok());
    assert!(std::fs::create_dir_all(&skill).is_ok());
    let body = "---\nname: haiku\ndescription: Answer in a haiku.\n---\nWrite three lines.\n";
    assert!(std::fs::write(skill.join("SKILL.md"), body).is_ok());
    let model = answering(&["done"]);
    let (mut app, _) = noting(&MemoryStore::default(), Invocation::default());
    app.insert_resource(ProjectRoot(dir))
        .insert_resource(RigHome(Home::at(scratch.join("home"))))
        .insert_resource(scripted_models([(MODEL, &model)]))
        .add_plugins((SessionLogPlugin, ExitPlugin, crate::skills::SkillsPlugin))
        .run_headless(Some("/skill:haiku".to_owned()));
    let starting = app.world_mut().spawn(Starting).id();
    app.finish();
    for _ in 0..20 {
        app.update();
    }
    assert_eq!(model.request_count(), 0);
    app.world_mut().despawn(starting);
    assert_eq!(ended(&mut app), answered("done"));
    assert_eq!(model.request_count(), 1);
}

/// Nothing runs while a plugin holds the start: the turn the session
/// restored is decided, here stopped, and the prompt calls the model, only
/// once the last [`Starting`] is gone.
#[test]
fn a_held_start_holds_the_restored_turn_and_the_prompt() {
    let store = MemoryStore::default();
    killed_two_frames_in(&store);
    let second = answering(&["prompted"]);
    let (mut app, _) = scripted(&store, (&second, false), Some("again"), || {});
    let starting = app.world_mut().spawn(Starting).id();
    for _ in 0..20 {
        app.update();
    }
    let world = app.world_mut();
    let mut turns = world.query_filtered::<(), (With<Restored>, With<TurnOf>)>();
    let restored = turns.iter(world).count();
    assert_eq!((restored, second.request_count()), (1, 0));
    assert!(app.world().resource::<Ends>().0.is_empty());
    app.world_mut().despawn(starting);
    assert_eq!(ended(&mut app), answered("prompted"));
    assert_eq!(second.request_count(), 1);
    let world = app.world_mut();
    let restored = turns.iter(world).count();
    assert_eq!(restored, 0);
}

/// `rig -r <id> -p --json` signalled while a plugin holds the start: the
/// turn it restored, which it leaves to the next run, still gets its
/// `turn_end` before the `exit` line.
#[test]
fn a_json_exit_during_a_held_start_closes_the_restored_turn_it_leaves() {
    let store = MemoryStore::default();
    killed_two_frames_in(&store);
    let second = silent();
    let held = |app: &mut App| {
        app.world_mut().spawn(Starting);
        let signal = |mut frames: Local<u8>, mut commands: Commands| {
            *frames += 1;
            if *frames == 2 {
                commands.trigger(Exit(ExitReason::Signal));
            }
        };
        app.add_systems(Update, signal);
    };
    let (app, lines) = one_shot_on(&store, "again", None, &second, true, held);
    assert_eq!(app.should_exit(), Some(ExitReason::Signal.exit()));
    assert_eq!(second.request_count(), 0);
    let closing = ["turn_start", "turn_end", "exit"].map(Value::from);
    let parsed = lines.parsed();
    let kinds = parsed.iter().map(|line| at(line, "type"));
    let kinds: Vec<_> = kinds.filter(|kind| closing.contains(kind)).collect();
    assert_eq!(kinds, closing.iter().collect::<Vec<_>>(), "{parsed:?}");
    let ends = lines.of("turn_end");
    let outcome = ends.first().map(|end| at(end, "outcome").clone());
    assert_eq!(outcome, Some(json!("stopped")));
    drop(app);
    let third = silent();
    let in_place = (carry_on, answer_in_place).in_set(Stage::Gate);
    let (mut app, _) = scripted(&store, (&third, false), None, in_place);
    assert_eq!(ended(&mut app), answered("hello"));
}

/// `rig -m <MODEL> -p hi` on `store`, killed two frames in, before any
/// reply; its model, which got no request.
fn killed_two_frames_in(store: &MemoryStore) -> MockCompletionModel {
    let first = silent();
    // `--model` chooses the model in `First`, as `rig_basics::models` does.
    let choose =
        |agents: Query<Entity, With<Agent>>, mut done: Local<bool>, mut commands: Commands| {
            if let Some(entity) = agents.iter().next()
                && !*done
            {
                *done = true;
                let model = MODEL.to_owned();
                commands.trigger(SetModel { entity, model });
            }
        };
    let held = hold_turn_calls.in_set(Stage::Gate);
    let (mut app, _) = scripted(store, (&first, true), Some("hi"), held);
    app.add_systems(First, choose);
    app.update();
    app.update();
    // Killed: no exit, nothing more written.
    drop(app);
    first
}

/// `rig -m <model> -p hi` killed two frames in, before any reply: the
/// restart stops that turn, so `rig -r <id> -p` has no prompt, and
/// `rig -r <id> -p again` answers the new prompt, the model told that
/// `hi` was stopped.
#[test]
fn a_run_killed_two_frames_in_is_resumed_stopped() {
    let store = MemoryStore::default();
    let first = killed_two_frames_in(&store);
    let second = answering(&["hello"]);
    let (mut app, _) = scripted(&store, (&second, false), None, || {});
    assert_eq!(ended(&mut app), no_prompt());
    assert_eq!((first.request_count(), second.request_count()), (0, 0));
    drop(app);
    // Resumed with a prompt straight away, from a session of its own.
    let store = MemoryStore::default();
    let first = killed_two_frames_in(&store);
    let (mut app, _) = scripted(&store, (&second, false), Some("again"), || {});
    assert_eq!(ended(&mut app), answered("hello"));
    assert_eq!((first.request_count(), second.request_count()), (0, 1));
    let sent = serde_json::to_string(&second.requests()).unwrap_or_default();
    let order = [
        "\"hi\"",
        "Do not carry it out unless asked again",
        "\"again\"",
    ]
    .map(|text| sent.find(text));
    assert!(
        order.iter().all(Option::is_some) && order.is_sorted(),
        "{sent}"
    );
}

/// Lets each restored turn carry on, as a plugin that recovers it does.
fn carry_on(turns: Query<Entity, (With<Restored>, Without<CarryOn>)>, mut commands: Commands) {
    for turn in &turns {
        commands.entity(turn).insert(CarryOn);
    }
}

/// Answers each model call where it is asked for, with "hello", as a plugin
/// standing in for the model does: nothing is sent.
fn answer_in_place(mut calls: Query<&mut Decision, (ReadyToStart, ModelCalls, With<TurnCall>)>) {
    for mut decision in &mut calls {
        let choice = vec![AssistantContent::Text(Text::new("hello".to_owned()))];
        let made = Made::new("plugin", "test", "answer");
        let reply =
            CompletionResponse::new(choice, Usage::default(), made, serde_json::Value::Null);
        decision.decide(
            Decider::plugin("test"),
            Outcome(Ok(rig_core::effect::Answered::Completion(reply.into()))),
        );
    }
}

/// The resumed turn a plugin carries on ends inside the run's first frame,
/// before the run looked at its agent: its answer is the run's all the
/// same.
#[test]
fn a_resumed_turn_that_ends_in_the_first_frame_answers_the_run() {
    let store = MemoryStore::default();
    killed_two_frames_in(&store);
    let model = silent();
    let in_place = (carry_on, answer_in_place).in_set(Stage::Gate);
    let (mut app, wakes) = scripted(&store, (&model, false), None, in_place);
    // Only the frames the run asks for, as the host loop runs them.
    run_while_woken(&mut app, &wakes);
    let ends = &app.world().resource::<Ends>().0;
    let how: Vec<_> = ends
        .iter()
        .map(|end| (end.answer.as_deref(), end.code))
        .collect();
    assert_eq!(how, [(Some("hello"), 0)]);
    assert_eq!(model.request_count(), 0);
}

/// Refuses each model call where it is asked for, as a plugin's gate does.
fn refuse_in_place(mut calls: Query<&mut Decision, (ReadyToStart, ModelCalls, With<TurnCall>)>) {
    for mut decision in &mut calls {
        decision.decide(Decider::plugin("test"), Outcome::refused("not now"));
    }
}

/// The resumed turn a plugin carries on fails inside the run's first
/// frame, with no prompt given: its failure is the run's.
#[test]
fn a_resumed_turn_that_fails_in_the_first_frame_fails_the_run() {
    let store = MemoryStore::default();
    killed_two_frames_in(&store);
    let model = silent();
    let in_place = (carry_on, refuse_in_place).in_set(Stage::Gate);
    let (mut app, wakes) = scripted(&store, (&model, false), None, in_place);
    run_while_woken(&mut app, &wakes);
    let ends = &app.world().resource::<Ends>().0;
    let how: Vec<_> = ends.iter().map(|end| (end.code, end.why.clone())).collect();
    assert_eq!(how, [(1, Some("not now".to_owned()))]);
    assert_eq!(model.request_count(), 0);
}

/// A resumed run whose restored model can no longer be used, and whose
/// restored turn failed first: the model's refusal, which leaves the run
/// nothing to send with, is why it fails, and plain `-p` prints it once.
#[test]
fn a_refused_model_is_why_a_resumed_run_fails_over_its_restored_turn() {
    let store = MemoryStore::default();
    killed_two_frames_in(&store);
    let model = silent();
    // `rig -r <id> -p` with no prompt.
    let invoked = Invocation {
        print: Some(String::new()),
        ..Invocation::default()
    };
    let (mut app, wakes) = noting(&store, invoked);
    let stderr = Lines::default();
    app.insert_resource(scripted_models([("test/other", &model)]))
        .add_plugins((SessionLogPlugin, ExitPlugin, PrintPlugin))
        .add_systems(AgentPass, refuse_in_place.in_set(Stage::Gate))
        .insert_resource(Printed::to(stderr.clone()));
    app.finish();
    run_while_woken(&mut app, &wakes);
    let ends = &app.world().resource::<Ends>().0;
    let how: Vec<_> = ends.iter().map(|end| (end.code, end.why.clone())).collect();
    assert!(
        matches!(how.as_slice(), [(1, Some(why))] if why.starts_with("Cannot use test/model")),
        "{how:?}"
    );
    let lines = stderr.text();
    let said = lines
        .iter()
        .filter(|line| line.starts_with("rig: Cannot use test/model"));
    assert_eq!(said.count(), 1, "{lines:?}");
}

/// A print run with `--answer=answers` when given, after its first frame.
fn print_run(answers: Option<&str>) -> App {
    let flags = answers.map(|ids| (ANSWER.to_owned(), Some(ids.to_owned())));
    let invoked = Invocation {
        print: Some("hi".to_owned()),
        flags: flags.into_iter().collect(),
        ..Invocation::default()
    };
    let (mut app, _) = noting(&MemoryStore::default(), invoked);
    app.add_plugins(PrintPlugin);
    app.finish();
    app.cleanup();
    app.update();
    app
}

/// How each of `asks` was answered after a frame, and how many answer
/// questions: `Some(None)` when refused, `None` when not at all.
fn answers(
    app: &mut App,
    asks: impl IntoIterator<Item = Ask>,
) -> (usize, Vec<Option<Option<Answer>>>) {
    let asks: Vec<Entity> = asks
        .into_iter()
        .map(|ask| app.world_mut().spawn(ask).id())
        .collect();
    app.update();
    let world = app.world_mut();
    let answerers = world.query::<&Answerer>().iter(world).count();
    let outcome = |ask: &Entity| world.get::<Outcome>(*ask).map(Answer::of);
    (answerers, asks.iter().map(outcome).collect())
}

#[test]
fn questions_are_answered_from_answer_or_refused() {
    let approval = || {
        Ask::new("Run rm -rf build?")
            .choice("deny", "Deny")
            .choice("allow", "Allow")
    };
    let several = Ask::yes_no("Which?").choice("allow", "Allow").multiple();
    let asks = [approval(), Ask::new("Name?").free_text(), several];
    let allowed = Some(Answer::choice("allow"));
    let both = Some(Answer::choices(["yes", "allow"]));
    // With `--answer`, plugins may ask; without, they apply their own
    // policy.
    let given = answers(&mut print_run(Some("allow,yes")), asks);
    assert_eq!(given, (1, vec![Some(allowed), Some(None), Some(both)]));
    let none = answers(&mut print_run(None), [approval()]);
    assert_eq!(none, (0, vec![Some(None)]));
}

#[test]
fn sections_follow_the_answer_by_tag() {
    let sections = [
        PrintSection::new("todo", "todo 1/2\n[x] read\n[ ] fix\n"),
        PrintSection::new("cost", "$0.02"),
        PrintSection::new("empty", ""),
    ];
    let printed = output(Some("Fixed.\n"), &sections);
    assert_eq!(
        printed.as_deref(),
        Some("Fixed.\n\n$0.02\n\ntodo 1/2\n[x] read\n[ ] fix")
    );
    assert_eq!(output(None, &[]), None);
}

#[test]
fn a_prompt_takes_what_is_piped_in_and_never_waits_on_a_silent_stdin() {
    let wait = Duration::from_millis(50);
    let on = || false;
    let piped = |prompt: &str, text: &'static [u8]| with_piped(prompt, text, wait, on);
    assert_eq!(piped("Review", b"diff").as_deref(), Some("Review\n\ndiff"));
    assert_eq!(piped("", b"diff").as_deref(), Some("diff"));
    // A pipe whose writer stays open and silent, as an agent's shell may
    // leave stdin.
    let Ok((silent, writer)) = std::io::pipe() else {
        return;
    };
    assert_eq!(
        with_piped("Review", silent, wait, on).as_deref(),
        Some("Review")
    );
    drop(writer);
}

/// A run waiting for its whole prompt on stdin, which may never come, stops
/// waiting once it is interrupted, as by a signal or its launcher's death.
#[test]
fn an_interrupted_run_stops_waiting_for_stdin() {
    let Ok((silent, writer)) = std::io::pipe() else {
        return;
    };
    let since = std::time::Instant::now();
    let interrupted = move || since.elapsed() > Duration::from_millis(100);
    assert_eq!(with_piped("", silent, Duration::MAX, interrupted), None);
    drop(writer);
}

/// A long recalled text, such as a big input piped in after the prompt,
/// shows its first lines on stderr and how much was left out.
#[test]
fn a_long_recalled_text_is_shortened() {
    assert_eq!(excerpt("Fix it.\n\nshort"), "Fix it.\n\nshort");
    // Ten lines and the blank ones after them leave nothing out.
    let ten: String = (1..=10).map(|n| format!("{n}\n")).collect();
    assert_eq!(
        excerpt(&format!("{ten}{}", "\n".repeat(5000))),
        ten.trim_end()
    );
    let numbers: String = (1..=1000).map(|n| format!("{n}\n")).collect();
    let text = format!("Count:\n\n{numbers}");
    let kept = "Count:\n\n1\n2\n3\n4\n5\n6\n7\n8";
    let left = text.len() - kept.len();
    assert_eq!(excerpt(&text), format!("{kept}\n… ({left} more bytes)"));
    // Never inside a character: byte 1000 is the second of an `é`.
    let wide = format!("a{}", "é".repeat(600));
    assert_eq!(
        excerpt(&wide),
        format!("a{}\n… (202 more bytes)", "é".repeat(499))
    );
}

/// The lines written, shared with the test.
#[derive(Clone, Default)]
struct Lines(Arc<Mutex<Vec<u8>>>);

impl Write for Lines {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if let Ok(mut lines) = self.0.lock() {
            lines.extend_from_slice(bytes);
        }
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl Lines {
    /// Each line, parsed; `Null` for one that is not JSON.
    fn parsed(&self) -> Vec<Value> {
        let bytes = self.0.lock().map(|lines| lines.clone()).unwrap_or_default();
        String::from_utf8_lossy(&bytes)
            .lines()
            .map(|line| serde_json::from_str(line).unwrap_or(Value::Null))
            .collect()
    }

    /// Each line, as text.
    fn text(&self) -> Vec<String> {
        let bytes = self.0.lock().map(|lines| lines.clone()).unwrap_or_default();
        String::from_utf8_lossy(&bytes)
            .lines()
            .map(str::to_owned)
            .collect()
    }

    /// The lines of type `kind`.
    fn of(&self, kind: &str) -> Vec<Value> {
        let lines = self.parsed().into_iter();
        lines.filter(|line| at(line, "type") == kind).collect()
    }
}

/// The field `key` of `line`, `Null` without one.
fn at<'a>(line: &'a Value, key: &str) -> &'a Value {
    line.get(key).unwrap_or(&Value::Null)
}

/// On a call of `wait`: the test answers it.
#[derive(Component)]
struct Waiting;

/// A long-lived `--json` run on `model` whose `wait` tool the test
/// answers, each call approved first with `approve`; its lines, and the
/// sender of its request lines.
fn json_run(
    model: &MockCompletionModel,
    approve: bool,
) -> (App, Receiver<()>, Lines, Sender<String>) {
    let lines = Lines::default();
    let (input, requests) = JsonIn::channel();
    let invoked = Invocation {
        flags: vec![(JSON.to_owned(), None)],
        ..Invocation::default()
    };
    let (mut app, wakes) = noting(&MemoryStore::default(), invoked);
    app.insert_resource(JsonOut::new(lines.clone()))
        .insert_resource(requests)
        .add_plugins((SessionLogPlugin, ExitPlugin, PrintPlugin))
        .add_open_tool(
            "wait",
            "Wait.",
            ToolOptions::default(),
            |called: On<ToolCalled<Value>>, mut commands: Commands| {
                commands.entity(called.call).insert(Waiting);
            },
        );
    if approve {
        let ask = |calls: Query<Entity, (ReadyToStart, ToolCalls)>, mut commands: Commands| {
            for call in &calls {
                let approval = NeedsApproval::add("test", "It waits.");
                commands.entity(call).queue(approval);
            }
        };
        let ask = ask
            .in_set(Stage::Gate)
            .before(crate::approval::ApprovalSystems);
        app.add_plugins(ApprovalPlugin).add_systems(AgentPass, ask);
    }
    app.finish();
    app.update();
    let world = app.world_mut();
    if let (Some(agent), Some(connection)) = (first_agent(world), connect(model)) {
        world.entity_mut(agent).insert(connection);
    }
    (app, wakes, lines, input)
}

/// Sends `line` as a request.
fn send(input: &Sender<String>, line: Value) {
    assert!(input.try_send(line.to_string()).is_ok(), "the run reads");
}

/// Whether a `wait` call runs.
fn waits(world: &World) -> bool {
    world
        .iter_entities()
        .any(|entity| entity.contains::<Waiting>())
}

/// A driver's request that comes while a plugin holds the start waits for
/// it: no turn starts until the last [`Starting`] is gone.
#[test]
fn a_held_start_holds_a_drivers_first_request() {
    let model = answering(&["hello"]);
    let lines = Lines::default();
    let (input, requests) = JsonIn::channel();
    let invoked = Invocation {
        flags: vec![(JSON.to_owned(), None)],
        ..Invocation::default()
    };
    let (mut app, wakes) = noting(&MemoryStore::default(), invoked);
    app.insert_resource(JsonOut::new(lines.clone()))
        .insert_resource(requests)
        .add_plugins((SessionLogPlugin, ExitPlugin, PrintPlugin));
    let starting = app.world_mut().spawn(Starting).id();
    app.finish();
    app.update();
    let world = app.world_mut();
    if let (Some(agent), Some(connection)) = (first_agent(world), connect(&model)) {
        world.entity_mut(agent).insert(connection);
    }
    send(
        &input,
        json!({ "v": 1, "type": "send", "id": 1, "text": "hi" }),
    );
    for _ in 0..20 {
        app.update();
    }
    assert_eq!(model.request_count(), 0);
    assert!(lines.of("turn_start").is_empty());
    app.world_mut().despawn(starting);
    test_support::run_until(&mut app, &wakes, |_| !lines.of("turn_end").is_empty());
    assert_eq!(model.request_count(), 1);
}

#[test]
fn a_driver_sends_a_message_mid_turn_and_reads_the_answer() {
    let model = MockCompletionModel::from_stream_turns([
        calls("c1", "wait", json!({})),
        reply("Both done."),
    ]);
    let (mut app, wakes, lines, input) = json_run(&model, false);
    let text = "Fix the tests.";
    send(
        &input,
        json!({ "v": 1, "type": "send", "id": 1, "text": text }),
    );
    test_support::run_until(&mut app, &wakes, waits);
    // While the tool call runs: it reaches the turn's next model call.
    let text = "Also the docs.";
    send(
        &input,
        json!({ "v": 1, "type": "send", "id": 2, "text": text }),
    );
    test_support::run_until(&mut app, &wakes, |_| lines.of("reply").len() == 2);
    let world = app.world_mut();
    let mut waiting = world.query_filtered::<&mut Decision, With<Waiting>>();
    for mut decision in waiting.iter_mut(world) {
        let waited = rig_core::tool::ToolResult::success("Waited.".into());
        decision.decide(Decider::plugin("test"), Outcome::from(waited));
    }
    test_support::run_until(&mut app, &wakes, |_| !lines.of("turn_end").is_empty());

    let parsed = lines.parsed();
    let versioned = parsed.iter().all(|line| at(line, "v") == VERSION);
    assert!(versioned, "{parsed:?}");
    let kinds: Vec<&str> = parsed
        .iter()
        .filter_map(|line| at(line, "type").as_str())
        .collect();
    let mut rest = kinds.iter();
    let order = [
        "ready",
        "turn_start",
        "tool_call",
        "tool_result",
        "text",
        "turn_end",
    ];
    for kind in order {
        let found = rest.any(|seen| *seen == kind);
        assert!(found, "{kind} in order in {kinds:?}");
    }
    let end = lines.of("turn_end");
    let answers: Vec<&Value> = end.iter().map(|end| at(end, "answer")).collect();
    assert_eq!(answers, ["Both done."], "one turn: {end:?}");
    let ms = |kind, key| {
        lines
            .of(kind)
            .first()
            .and_then(|line| at(line, key).as_u64())
    };
    let (started, first) = (ms("turn_start", "ms"), ms("turn_end", "first_token_ms"));
    assert!(
        started.zip(first).is_some_and(|(s, f)| s <= f),
        "{started:?} {first:?}"
    );
    let second = model
        .requests()
        .get(1)
        .map(|request| format!("{:?}", request.chat_history));
    let read = second.is_some_and(|history| history.contains("Also the docs."));
    assert!(read, "the next call reads it");
}

#[test]
fn an_approval_is_answered_over_the_pipe_and_eof_ends_the_run() {
    let model =
        MockCompletionModel::from_stream_turns([calls("c1", "wait", json!({})), reply("ok")]);
    let (mut app, wakes, lines, input) = json_run(&model, true);
    send(&input, json!({ "v": 1, "type": "send", "text": "Wait." }));
    test_support::run_until(&mut app, &wakes, |_| !lines.of("ask").is_empty());
    let ask = lines.of("ask").first().cloned().unwrap_or_default();
    let choices = at(&ask, "choices").as_array().into_iter().flatten();
    let ids: Vec<&str> = choices.filter_map(|c| at(c, "id").as_str()).collect();
    assert_eq!(ids, ["deny", "allow"], "{ask}");
    send(&input, json!({ "v": 1, "type": "state", "id": "s" }));
    let id = at(&ask, "ask");
    send(
        &input,
        json!({ "v": 1, "type": "answer", "id": "a", "ask": id, "choices": ["allow"] }),
    );
    test_support::run_until(&mut app, &wakes, waits);
    let replies = lines.of("reply");
    let reply = |id: &str| replies.iter().find(|reply| at(reply, "id") == id);
    let state = reply("s").and_then(|reply| reply.pointer("/state/asks/0/ask"));
    assert_eq!(state, Some(id));
    assert_eq!(
        reply("a").map(|reply| at(reply, "ok")),
        Some(&Value::Bool(true))
    );

    // The driver goes away mid-turn: the agent exits at once, closing the
    // call and the turn it cut off, with no `exit` line.
    drop(input);
    test_support::run_until(&mut app, &wakes, |_| !lines.of("turn_end").is_empty());
    let exit = app.world().get_resource::<ExitReason>();
    assert_eq!(exit, Some(&ExitReason::Quit));
    let result = lines.of("tool_result").first().cloned().unwrap_or_default();
    assert_eq!(at(&result, "cancelled"), &json!(true), "{result}");
    let end = lines.of("turn_end").first().cloned().unwrap_or_default();
    let why = (at(&end, "outcome"), at(&end, "error"));
    assert_eq!(why, (&json!("stopped"), &json!("the run ended")));
    assert!(lines.of("exit").is_empty());
}

#[test]
fn a_line_of_another_version_is_refused_naming_the_side_to_update() {
    let model = MockCompletionModel::from_stream_turns([reply("ok")]);
    let (mut app, wakes, lines, input) = json_run(&model, false);
    send(&input, json!({ "v": 2, "type": "state", "id": 1 }));
    send(&input, json!({ "type": "state", "id": 2 }));
    send(&input, json!({ "v": 1, "type": "sned", "id": 3 }));
    test_support::run_until(&mut app, &wakes, |_| lines.of("reply").len() == 3);
    let errors: Vec<String> = lines
        .of("reply")
        .iter()
        .map(|reply| format!("{} {}", at(reply, "ok"), at(reply, "error")))
        .collect();
    let [newer, older, unknown] = errors.as_slice() else {
        return assert_eq!(errors.len(), 3, "{errors:?}");
    };
    assert!(
        newer.starts_with("false") && newer.contains("update rig"),
        "{newer}"
    );
    let driver = older.starts_with("false") && older.contains("update the driver");
    assert!(driver, "{older}");
    assert!(unknown.contains("sned"), "{unknown}");
}

/// A one-shot run of `prompt` on `model`, `--model named` when given,
/// `--json` when `json`, after `setup`, run to its end; its JSON lines.
fn one_shot(
    prompt: &str,
    named: Option<&str>,
    model: &MockCompletionModel,
    json: bool,
    setup: impl FnOnce(&mut App),
) -> (App, Lines) {
    let store = MemoryStore::default();
    one_shot_on(&store, prompt, named, model, json, setup)
}

/// [`one_shot`], its session on `store`.
fn one_shot_on(
    store: &MemoryStore,
    prompt: &str,
    named: Option<&str>,
    model: &MockCompletionModel,
    json: bool,
    setup: impl FnOnce(&mut App),
) -> (App, Lines) {
    let lines = Lines::default();
    let invoked = Invocation {
        print: Some(prompt.to_owned()),
        model: named.map(str::to_owned),
        flags: json.then(|| (JSON.to_owned(), None)).into_iter().collect(),
        ..Invocation::default()
    };
    let (mut app, _) = noting(store, invoked);
    app.insert_resource(JsonOut::new(lines.clone()))
        .insert_resource(scripted_models([(MODEL, model)]))
        .add_plugins((SessionLogPlugin, ExitPlugin, PrintPlugin));
    if named.is_some() {
        app.add_plugins(crate::models::ModelsPlugin);
    }
    setup(&mut app);
    app.finish();
    app.cleanup();
    // As the runner does, nothing after the frame that exits.
    run_until(&mut app, |_| false);
    (app, lines)
}

/// A model whose provider refuses the call for good.
fn out_of_credit() -> MockCompletionModel {
    let refused = rig_core::ProviderResponseError::without_status("the account is out of credit")
        .with_transient(Some(false));
    MockCompletionModel::from_stream_turns([vec![MockStreamEvent::Error(
        rig_core::test_utils::MockError::ProviderResponse(refused),
    )]])
}

#[test]
fn a_failed_one_shot_run_ends_with_exit_last_naming_its_error() {
    // A provider refusal fails the turn; a `--model` the catalog lacks is
    // refused before it, as a command line.
    for (named, model, code, asked) in [
        (None, out_of_credit(), 1, 1),
        (Some("nobody/nothing"), silent(), 2, 0),
    ] {
        let parsed = one_shot("hi", named, &model, true, |_| {}).1.parsed();
        // The refusal is the scripted model's, not another provider's.
        assert_eq!(model.request_count(), asked, "{parsed:?}");
        let exit = parsed.last().cloned().unwrap_or_default();
        assert_eq!(at(&exit, "type"), "exit", "{parsed:?}");
        assert_eq!(at(&exit, "code"), code, "{exit}");
        let error = |line: &&Value| at(line, "type") == "notice" && at(line, "level") == "error";
        // The error notice, perhaps with advice after it, comes first.
        let notice = parsed
            .iter()
            .find(error)
            .and_then(|line| at(line, "text").as_str());
        let why = at(&exit, "error").as_str().unwrap_or("no error");
        assert!(
            notice.is_some_and(|text| text.starts_with(why)),
            "{parsed:?}"
        );
        // The prompt that went out was not answered: it is recalled, out
        // of the session.
        let recalled = parsed
            .iter()
            .any(|line| at(line, "type") == "recalled" && at(line, "text") == "hi");
        assert_eq!(recalled, named.is_none(), "{parsed:?}");
        if let Some(end) = parsed.iter().find(|line| at(line, "type") == "turn_end") {
            assert_eq!(at(end, "error"), at(&exit, "error"));
            assert!(
                at(end, "first_token_ms").is_null(),
                "nothing streamed: {end}"
            );
        }
        let ms: Vec<u64> = parsed.iter().filter_map(|l| at(l, "ms").as_u64()).collect();
        assert_eq!(ms.len(), parsed.len(), "every line is timed");
        assert!(ms.is_sorted(), "{ms:?}");
    }
}

#[test]
fn a_kernel_failure_reads_the_same_in_exit_turn_end_and_its_notice() {
    // The kernel's reason for a reply with no message, which no notice
    // tells.
    let empty = MockCompletionModel::from_stream_turns([vec![
        MockStreamEvent::final_response_with_default_usage(),
    ]]);
    let parsed = one_shot("hi", None, &empty, true, |_| {}).1.parsed();
    let of = |kind: &str| parsed.iter().find(|line| at(line, "type") == kind);
    let why = "the model ended the turn without a message";
    assert_eq!(
        of("exit").map(|line| at(line, "error").clone()),
        Some(why.into())
    );
    assert_eq!(
        of("turn_end").map(|line| at(line, "error").clone()),
        Some(why.into())
    );
}

/// The kernel takes a reply with no final text as the turn's answer.
#[derive(Resource, Default)]
struct Answered(bool);

#[test]
fn a_turn_that_ends_without_text_fails_the_run() {
    let reasoning = MockStreamEvent::Reasoning {
        id: "r1".to_owned(),
        text: "Thinking it over.".to_owned(),
    };
    let only_reasoning = vec![
        reasoning,
        MockStreamEvent::final_response_with_default_usage(),
    ];
    for events in [only_reasoning, reply(" \n")] {
        let model = MockCompletionModel::from_stream_turns([events.clone(), events]);
        let noted = |app: &mut App| {
            app.init_resource::<Answered>().add_observer(
                |settled: On<Add<Settled>>,
                 turns: Query<&TurnOutcome>,
                 mut seen: ResMut<Answered>| {
                    let outcome = turns.get(settled.entity);
                    seen.0 |= matches!(outcome, Ok(TurnOutcome::Answered(_)));
                },
            );
        };
        let (app, lines) = one_shot("hi", Some(MODEL), &model, true, noted);
        assert!(app.world().resource::<Answered>().0, "the kernel answered");
        let end = lines.of("turn_end").first().cloned().unwrap_or_default();
        assert_eq!(
            (at(&end, "outcome"), at(&end, "answer"), at(&end, "error")),
            (&json!("failed"), &Value::Null, &json!(NO_ANSWER)),
            "{end}"
        );
        let exit = lines.parsed().last().cloned().unwrap_or_default();
        assert_eq!(
            (at(&exit, "code"), at(&exit, "error")),
            (&json!(1), &json!(NO_ANSWER))
        );
        // Answered, as far as the session goes: its prompt is not taken back.
        assert!(lines.of("recalled").is_empty(), "{:?}", lines.parsed());
        // Plain: the reason once on stderr, exit 1.
        let stderr = Lines::default();
        let (app, _) = one_shot("hi", Some(MODEL), &model, false, |app| {
            app.insert_resource(Printed::to(stderr.clone()));
        });
        assert_eq!(app.should_exit(), Some(AppExit::from_code(1)));
        assert_eq!(stderr.text(), [format!("rig: {NO_ANSWER}")]);
    }
}

/// How the `wait` call stands when the signal comes.
#[derive(Clone, Copy, Debug)]
enum Cut {
    /// It waits.
    Running,
    /// It waits, its output so far kept for a stop ([`IfStopped`]).
    WithOutput,
    /// It answered, and a hold keeps it from settling.
    Answered,
}

#[test]
fn a_one_shot_json_run_cut_off_by_a_signal_closes_what_it_started() {
    // A turn a plugin carries on across the exit is left for the restart,
    // and closed all the same.
    let cuts = [Cut::Running, Cut::WithOutput, Cut::Answered];
    for (cut, carried) in cuts.into_iter().flat_map(|cut| [(cut, false), (cut, true)]) {
        let store = MemoryStore::default();
        let model = MockCompletionModel::from_stream_turns([calls("c1", "wait", json!({}))]);
        let (app, lines) = one_shot_on(&store, "Wait.", Some(MODEL), &model, true, |app| {
            // The signal comes while the call runs, as Ctrl+C or the
            // launcher's death does.
            app.add_open_tool(
                "wait",
                "Wait.",
                ToolOptions::default(),
                move |called: On<ToolCalled<Value>>,
                      turns: Query<&CallOf>,
                      mut commands: Commands| {
                    if let Ok(&CallOf(turn)) = turns.get(called.call)
                        && carried
                    {
                        commands.entity(turn).insert(CarryOn);
                    }
                    let result = |text: &str| rig_core::tool::ToolResult::success(text.into());
                    let mut call = commands.entity(called.call);
                    match cut {
                        Cut::Running => call.insert(Waiting),
                        Cut::WithOutput => {
                            let so_far = result("half ghp_abcdefghijklmnopqrstuvwxyz0123456789");
                            call.insert((Waiting, IfStopped(Outcome::from(so_far))))
                        }
                        Cut::Answered => call.insert(Outcome::from(result("Waited."))),
                    };
                    commands.spawn((TestHold, HoldOf(called.call)));
                    commands.trigger(Exit(ExitReason::Signal));
                },
            );
        });
        let case = format!("{cut:?}, carried: {carried}");
        assert_eq!(app.should_exit(), Some(AppExit::from_code(130)), "{case}");
        let parsed = lines.parsed();
        let kinds: Vec<&str> = parsed
            .iter()
            .filter_map(|l| at(l, "type").as_str())
            .collect();
        let at_end = kinds.get(kinds.len().saturating_sub(4)..);
        assert_eq!(
            at_end,
            Some(&["tool_call", "tool_result", "turn_end", "exit"][..]),
            "{case} {kinds:?}"
        );
        // The result the session keeps, once.
        let (ok, cancelled, output) = match cut {
            Cut::Running => (
                false,
                true,
                "Stopped: the run was interrupted. It may have partly run; check before running it again.",
            ),
            // A key it printed is redacted, on the line and in the session.
            Cut::WithOutput => (true, true, "half [REDACTED]"),
            Cut::Answered => (true, false, "Waited."),
        };
        let results = lines.of("tool_result");
        let [result] = results.as_slice() else {
            return assert_eq!(results.len(), 1, "{case} {results:?}");
        };
        let fields =
            ["id", "ok", "refused", "cancelled", "output"].map(|key| at(result, key).clone());
        assert_eq!(
            fields,
            [
                json!("c1"),
                json!(ok),
                json!(false),
                json!(cancelled),
                json!(output)
            ],
            "{case}"
        );
        let why = "the run was interrupted";
        let end = lines.of("turn_end").first().cloned().unwrap_or_default();
        assert_eq!(
            (at(&end, "outcome"), at(&end, "error")),
            (&json!("stopped"), &json!(why))
        );
        let exit = parsed.last().cloned().unwrap_or_default();
        assert_eq!(
            (at(&exit, "code"), at(&exit, "error")),
            (&json!(130), &json!(why))
        );
        drop(app);
        if carried {
            continue;
        }

        // The session keeps the turn stopped, with the result the line
        // had: a resumed run runs nothing of it, and its model reads that
        // result and that the request was stopped.
        let second = answering(&["done"]);
        let (mut app, _) = scripted(&store, (&second, false), Some("go on"), || {});
        assert_eq!(ended(&mut app), answered("done"), "{cut:?}");
        let history = second
            .requests()
            .first()
            .map(|r| format!("{:?}", r.chat_history))
            .unwrap_or_default();
        assert!(
            history.contains(output) && history.contains("Do not carry it out"),
            "{cut:?} {history}"
        );
        assert_eq!(second.request_count(), 1, "{cut:?}");
    }
}

#[test]
fn a_tool_call_of_no_turn_cut_off_by_a_signal_is_closed_too() {
    let store = MemoryStore::default();
    let model = silent();
    let (app, lines) = one_shot_on(&store, "Wait.", Some(MODEL), &model, true, |app| {
        app.add_systems(AgentPass, hold_turn_calls.in_set(Stage::Gate))
            // A plugin's own call, of no turn, that runs when the signal
            // comes.
            .add_open_tool(
                "wait",
                "Wait.",
                ToolOptions::default(),
                |called: On<ToolCalled<Value>>, mut commands: Commands| {
                    commands.entity(called.call).insert(Waiting);
                    commands.trigger(Exit(ExitReason::Signal));
                },
            )
            .add_systems(Update, |world: &mut World, mut done: Local<bool>| {
                let agent = first_agent(world);
                if let Some(agent) = agent.filter(|_| !*done) {
                    *done = true;
                    start_call(world, agent, ("wait", "s1"), json!({}));
                }
            });
    });
    assert_eq!(app.should_exit(), Some(AppExit::from_code(130)));
    let results = lines.of("tool_result");
    let closed = results.iter().find(|line| at(line, "id") == "s1");
    let fields = closed.map(|line| ["cancelled", "output"].map(|key| at(line, key).clone()));
    let cut =
        "Stopped: the run was interrupted. It may have partly run; check before running it again.";
    assert_eq!(
        fields,
        Some([json!(true), json!(cut)]),
        "{:?}",
        lines.parsed()
    );
}

/// Holds each exit asked for until the run's own end, as a plugin that
/// finishes its work before the app exits does.
fn hold_exits(
    exits: Query<Entity, Added<ExitRequest>>,
    holds: Query<Entity, With<TestHold>>,
    ends: Res<Ends>,
    mut commands: Commands,
) {
    for exit in &exits {
        commands.spawn((TestHold, HoldOf(exit)));
    }
    if !ends.0.is_empty() {
        holds
            .iter()
            .for_each(|hold| commands.entity(hold).despawn());
    }
}

#[test]
fn a_signal_before_the_run_ends_gives_the_exit_its_code() {
    let model = answering(&["hello"]);
    let (app, lines) = one_shot("hi", Some(MODEL), &model, true, |app| {
        app.add_systems(Update, hold_exits).add_observer(
            |_: On<Add<TurnOf>>, mut commands: Commands| {
                commands.trigger(Exit(ExitReason::Signal));
            },
        );
    });
    // The run answered, then exited as the signal asked first.
    let ends = &app.world().resource::<Ends>().0;
    assert_eq!(ends.first().map(|end| end.code), Some(0));
    assert_eq!(app.should_exit(), Some(AppExit::from_code(130)));
    let exits = lines.of("exit");
    let [exit] = exits.as_slice() else {
        return assert_eq!(exits.len(), 1, "{exits:?}");
    };
    let fields = ["code", "answer", "error"].map(|key| at(exit, key).clone());
    assert_eq!(
        fields,
        [json!(130), json!("hello"), json!("the run was interrupted")]
    );
    assert_eq!(lines.parsed().last(), Some(exit));
}

/// A run that writes to stdout stops once nobody reads it; one whose lines
/// the app sends elsewhere does not watch stdout.
#[test]
fn a_run_on_stdout_watches_its_reader() {
    let model = silent();
    let (app, _) = one_shot("/help", None, &model, false, |_| {});
    assert!(app.world().contains_resource::<ExitWhenUnread>());
    let (app, _) = one_shot("/help", None, &model, true, |_| {});
    assert!(!app.world().contains_resource::<ExitWhenUnread>());
    // Its lines on stdout, which the test captures.
    let (app, _) = one_shot("/help", None, &model, true, |app| {
        app.world_mut().remove_resource::<JsonOut>();
    });
    assert!(app.world().contains_resource::<ExitWhenUnread>());
}

/// A run whose output nobody reads any more, as after `| head -1`, stops as
/// on a signal before its model is asked.
#[test]
fn a_run_nobody_reads_stops_as_on_a_signal() {
    fn gone() -> bool {
        true
    }
    let model = answering(&["hello"]);
    let (app, lines) = one_shot("hi", Some(MODEL), &model, true, |app| {
        app.insert_resource(ExitWhenUnread(gone))
            .add_plugins(rig_harness::host::signals::ExitOnSignalPlugin);
    });
    assert_eq!(app.should_exit(), Some(AppExit::from_code(130)));
    assert_eq!(model.request_count(), 0);
    let exit = lines.parsed().last().cloned().unwrap_or_default();
    assert_eq!(
        (at(&exit, "type"), at(&exit, "code")),
        (&json!("exit"), &json!(130))
    );
}

/// A plain run cut off by a signal says so on stderr, once.
#[test]
fn a_plain_run_cut_off_by_a_signal_says_so() {
    let model = answering(&["hello"]);
    let stderr = Lines::default();
    let (app, _) = one_shot("hi", Some(MODEL), &model, false, |app| {
        app.insert_resource(Printed::to(stderr.clone()))
            .add_observer(|_: On<Add<TurnOf>>, mut commands: Commands| {
                commands.trigger(Exit(ExitReason::Signal));
            });
    });
    assert_eq!(app.should_exit(), Some(AppExit::from_code(130)));
    assert_eq!(stderr.text(), ["rig: the run was interrupted"]);
}

#[test]
fn a_turn_without_text_fails_the_run_only_when_no_later_turn_answers() {
    let (silent, said) = (
        TurnOutcome::Answered(Message::assistant(" ")),
        TurnOutcome::Answered(Message::assistant("done")),
    );
    let no_answer = Some(Failure {
        code: 1,
        why: NO_ANSWER.to_owned(),
        told: false,
    });
    // As a turn that started a subagent, then the one its report started.
    for (turns, answer, fails) in [
        (vec![&silent, &said], Some("done"), None),
        (vec![&said, &silent], None, no_answer),
    ] {
        let mut run = Run::default();
        turns.into_iter().for_each(|turn| run.turn_ended(turn));
        assert_eq!(run.settled().as_deref(), answer);
        assert_eq!(run.failed, fails);
    }
}

/// An error notice the work recovered from, answering after it, is spent
/// with that settlement: a later one with no answer and no error of its
/// own does not fail the run with it.
#[test]
fn a_recovered_error_does_not_fail_a_later_settlement() {
    let mut run = Run {
        notice: Some("Compaction failed: the summary was cut".to_owned()),
        ..Run::default()
    };
    run.turn_ended(&TurnOutcome::Answered(Message::assistant("done")));
    assert_eq!(run.settled().as_deref(), Some("done"));
    run.turn_ended(&TurnOutcome::Stopped(None));
    assert_eq!(run.settled(), None);
    assert_eq!(run.failed, None);
}

#[test]
fn a_failed_turn_fails_the_run_whatever_turn_settles_with_it() {
    let (answered, failed, stopped) = (
        TurnOutcome::Answered(Message::assistant("done")),
        TurnOutcome::Failed("out of credit".to_owned()),
        TurnOutcome::Stopped(None),
    );
    let failure = Some(Failure {
        code: 1,
        why: "out of credit".to_owned(),
        told: true,
    });
    for (turns, answer, fails) in [
        ([&answered, &failed], None, failure.clone()),
        ([&failed, &answered], Some("done"), failure),
        ([&answered, &stopped], Some("done"), None),
    ] {
        let mut run = Run::default();
        turns.into_iter().for_each(|turn| run.turn_ended(turn));
        assert_eq!(run.settled().as_deref(), answer);
        assert_eq!(run.failed, fails);
    }
}

#[test]
fn a_failed_plain_run_writes_its_reason_once() {
    // The third call of a tool nobody has fails the turn with a notice that
    // wraps the reason.
    let unknown = MockCompletionModel::from_stream_turns(
        ["c1", "c2", "c3"].map(|id| calls(id, "bash", json!({}))),
    );
    // A reply without a message fails the turn, with no notice, after an
    // error the turn got over.
    let empty = MockCompletionModel::from_stream_turns([vec![
        MockStreamEvent::final_response_with_default_usage(),
    ]]);
    let earlier = "Compaction failed: the summary was cut";
    for (named, model, recovered) in [
        (None, out_of_credit(), false),
        (Some("nobody/nothing"), silent(), false),
        (None, unknown, false),
        (None, empty, true),
    ] {
        let stderr = Lines::default();
        let (app, _) = one_shot("hi", named, &model, false, |app| {
            app.insert_resource(Printed::to(stderr.clone()));
            if recovered {
                app.add_observer(
                    move |turn: On<Add<TurnOf>>, turns: Query<&TurnOf>, mut commands: Commands| {
                        let agent = turns.get(turn.entity).ok().map(|of| of.0);
                        commands.write_message(Notice::error(agent, earlier));
                    },
                );
            }
        });
        let ended = app.world().resource::<Ends>().0.last().cloned();
        let why = ended.and_then(|ended| ended.why).unwrap_or_default();
        let errors = stderr.text();
        let lower = why.to_lowercase();
        let said = errors
            .iter()
            .filter(|text| text.to_lowercase().contains(&lower));
        assert!(!why.is_empty() && said.count() == 1, "{why:?} {errors:?}");
        let shown = errors.iter().any(|text| *text == format!("rig: {earlier}"));
        assert_eq!(shown, recovered, "{errors:?}");
    }
}

#[test]
fn nothing_is_written_after_exit() {
    let lines = Lines::default();
    let mut world = World::new();
    world.insert_resource(JsonOut::new(lines.clone()));
    world.init_resource::<LinesRead>();
    world.init_resource::<Messages<Notice>>();
    world.init_resource::<Messages<Recalled>>();
    world.init_resource::<Messages<StreamDelta>>();
    world.write_message(Notice::error(None, "Before exit."));
    assert!(world.run_system_cached(flush).is_ok());
    world.resource_mut::<JsonOut>().closed = true;
    world.write_message(Notice::error(None, "After exit."));
    assert!(world.run_system_cached(flush).is_ok());
    let texts: Vec<Value> = lines
        .parsed()
        .iter()
        .map(|l| at(l, "text").clone())
        .collect();
    assert_eq!(texts, ["Before exit."]);
}

#[test]
fn a_pending_notice_comes_before_pending_text() {
    let lines = Lines::default();
    let mut world = World::new();
    world.insert_resource(JsonOut::new(lines.clone()));
    world.init_resource::<LinesRead>();
    world.init_resource::<Messages<Notice>>();
    world.init_resource::<Messages<Recalled>>();
    world.init_resource::<Messages<StreamDelta>>();
    let effect = world.spawn(TurnCall).id();
    let text = StreamEvent::Text {
        part: rig_core::streaming::Part::new(0),
        text: "Hel".to_owned(),
    };
    world.write_message(StreamDelta {
        effect,
        agent: None,
        event: text,
    });
    world.write_message(Notice::error(None, "The model call failed."));
    assert!(world.run_system_cached(flush).is_ok());
    let kinds: Vec<Value> = lines
        .parsed()
        .iter()
        .map(|l| at(l, "type").clone())
        .collect();
    assert_eq!(kinds, ["notice", "text"]);
}

#[test]
fn a_command_does_not_fail_on_an_unusable_model() {
    let (app, _) = one_shot("/echo", Some("nobody/nothing"), &silent(), false, |app| {
        app.add_command("echo", "Echoes", |_: In<CommandArgs>| {});
    });
    let ends = &app.world().resource::<Ends>().0;
    let codes: Vec<u8> = ends.iter().map(|ended| ended.code).collect();
    assert_eq!(codes, [0], "{ends:?}");
}

#[test]
fn an_error_before_the_run_looked_fails_it_only_when_the_agent_never_connects() {
    // A stopped turn neither answers nor fails, so only the notice decides.
    let early = |mut notices: MessageWriter<Notice>| {
        notices.write(Notice::error(None, "a plugin failed to start"));
    };
    let stop = |calls: Query<&ForAgent, (ReadyToStart, With<TurnCall>)>, mut commands: Commands| {
        for &ForAgent(agent) in &calls {
            commands.trigger(StopTurn::new(agent));
        }
    };
    let (app, _) = one_shot("hi", None, &silent(), false, |app| {
        app.add_systems(Startup, early);
        let mut schedules = app.world_mut().resource_mut::<Schedules>();
        schedules.add_systems(AgentPass, stop.in_set(Stage::Gate));
    });
    let ends = &app.world().resource::<Ends>().0;
    let codes: Vec<u8> = ends.iter().map(|ended| ended.code).collect();
    assert_eq!(codes, [0], "{ends:?}");
}
