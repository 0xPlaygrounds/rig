use std::time::{Duration, Instant};

use bevy_app::TaskPoolPlugin;
use bevy_tasks::futures_lite::stream;
use bevy_tasks::{IoTaskPool, TaskPool};
use rig_core::catalog::Catalog;
use rig_core::operation::Completion;
use rig_core::serve::ErasedHandler;
use rig_core::serve::adapters::ModelAdapter;
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_ecs::AgentPlugin;
use rig_ecs::effects::Handler;
use rig_harness_protocol::Invocation;
use rig_harness_test_support::{self as test_support, app_on, first_agent, scripted_models};
use rig_memory::MemoryStore;

use super::*;
use crate::host::exit::ExitPlugin;

/// How the runs ended, and each turn that ended.
#[derive(Resource, Default)]
struct Ends(Vec<HeadlessEnded>, Vec<HeadlessTurnEnded>);

/// A headless run of `prompt`, its model named on the command line when
/// `model`, until it exits or ten seconds passed; how it ended.
fn run(prompt: &str, model: bool, setup: impl FnOnce(&mut World, Entity)) -> Vec<HeadlessEnded> {
    let mut app = run_app(prompt, model, setup);
    app.world_mut().resource_mut::<Ends>().0.drain(..).collect()
}

/// The app of [`run`], after the run.
fn run_app(prompt: &str, model: bool, setup: impl FnOnce(&mut World, Entity)) -> App {
    let mut app = started(prompt, model, setup);
    run_to_end(&mut app);
    app
}

/// The app of [`run`] after its first frame and `setup`, before the run.
fn started(prompt: &str, model: bool, setup: impl FnOnce(&mut World, Entity)) -> App {
    let mut app = App::new();
    let args = Invocation {
        model: model.then(|| "deepseek/deepseek-flash".to_owned()),
        ..Invocation::default()
    };
    app.insert_resource(Invoked {
        args,
        terminal: false,
    })
    .add_plugins((TaskPoolPlugin::default(), AgentPlugin, ExitPlugin))
    .init_resource::<Ends>()
    .add_observer(|ended: On<HeadlessEnded>, mut ends: ResMut<Ends>| {
        ends.0.push(ended.event().clone());
    })
    .add_observer(|ended: On<HeadlessTurnEnded>, mut ends: ResMut<Ends>| {
        ends.1.push(ended.event().clone());
    })
    .run_headless(Some(prompt.to_owned()));
    app.update();
    let world = app.world_mut();
    let agent = world
        .query_filtered::<Entity, With<Agent>>()
        .iter(world)
        .next();
    setup(world, agent.unwrap_or(Entity::PLACEHOLDER));
    app
}

/// Runs `app` until it exits or ten seconds passed, then a frame more.
fn run_to_end(app: &mut App) {
    run_until(app, |_| false);
    // Nothing more once it ended.
    app.update();
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

/// `model` as the catalog's DeepSeek model.
fn connection(model: &MockCompletionModel) -> Option<Connection> {
    let spec = Catalog::builtin()
        .resolve("deepseek/deepseek-flash")
        .ok()?
        .shared();
    let handler = ModelAdapter::<Completion>::new(spec.reference(), model.clone());
    Some(Connection {
        spec,
        handler: Handler(ErasedHandler::new(handler)),
    })
}

#[test]
fn the_prompt_is_answered_and_the_run_ends_beside_an_open_stream() {
    let model = MockCompletionModel::from_stream_turns([vec![
        MockStreamEvent::text("hello"),
        MockStreamEvent::final_response_with_default_usage(),
    ]]);
    let mut agent = None;
    let ends = run("hi", true, |world, primary| {
        agent = Some(primary);
        if let Some(connection) = connection(&model) {
            world.entity_mut(primary).insert(connection);
        }
        // A source that never ends, such as a file watch, is no work.
        let pool = IoTaskPool::get_or_init(TaskPool::default);
        let wake = world.resource::<Wake>().clone();
        world.spawn(Running::stream(pool, &wake, stream::pending::<u32>()));
    });
    let answered = HeadlessEnded {
        agent,
        answer: Some("hello".to_owned()),
        why: None,
        exit: ExitReason::Quit,
    };
    assert_eq!(ends, [answered]);
}

#[test]
fn a_turn_compacted_while_it_runs_still_prints_its_answer() {
    let model = MockCompletionModel::from_stream_turns([vec![
        MockStreamEvent::text("hello"),
        MockStreamEvent::final_response_with_default_usage(),
    ]]);
    let ends = run("hi", true, |world, primary| {
        if let Some(connection) = connection(&model) {
            world.entity_mut(primary).insert(connection);
        }
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
    let answers = ends
        .iter()
        .map(|ended| (ended.answer.as_deref(), &ended.exit));
    let answers: Vec<_> = answers.collect();
    assert_eq!(answers, [(Some("hello"), &ExitReason::Quit)]);
}

#[test]
fn a_warning_about_the_agent_does_not_fail_the_run_and_an_error_does() {
    let exits = [NoticeLevel::Warn, NoticeLevel::Error].map(|level| {
        // A reply without text: nothing answers, so only the notice decides.
        let model = MockCompletionModel::from_stream_turns([vec![
            MockStreamEvent::final_response_with_default_usage(),
        ]]);
        let ends = run("hi", true, |world, primary| {
            if let Some(connection) = connection(&model) {
                world.entity_mut(primary).insert(connection);
            }
            world.add_observer(move |_: On<Add<TurnOf>>, mut commands: Commands| {
                commands.write_message(Notice {
                    agent: Some(primary),
                    text: "key 2 refused; using key 3".to_owned(),
                    level,
                });
            });
        });
        ends.into_iter().map(|ended| ended.exit).collect::<Vec<_>>()
    });
    assert_eq!(exits, [vec![ExitReason::Quit], vec![ExitReason::Failed(1)]]);
}

#[test]
fn a_restored_model_that_cannot_be_used_fails_the_run_once() {
    let mut agent = None;
    let ends = run("hi", false, |world, primary| {
        agent = Some(primary);
        world
            .entity_mut(primary)
            .insert(ModelChoice("nobody/nothing".to_owned()));
    });
    let failed = HeadlessEnded {
        agent,
        answer: None,
        why: None,
        exit: ExitReason::Failed(1),
    };
    assert_eq!(ends, [failed]);
}

#[test]
fn an_empty_prompt_says_why() {
    let model = MockCompletionModel::from_stream_turns(Vec::<Vec<MockStreamEvent>>::new());
    let ends = run(" ", true, |world, primary| {
        if let Some(connection) = connection(&model) {
            world.entity_mut(primary).insert(connection);
        }
    });
    let codes: Vec<(ExitReason, bool)> = ends
        .into_iter()
        .map(|ended| (ended.exit, ended.why.is_some()))
        .collect();
    assert_eq!(codes, [(ExitReason::Failed(2), true)]);
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
    for answered in [false, true] {
        let model = MockCompletionModel::from_stream_turns([vec![
            MockStreamEvent::text("hello"),
            MockStreamEvent::final_response_with_default_usage(),
        ]]);
        let ends = run("hi", true, |world, primary| {
            if let Some(connection) = connection(&model) {
                world.entity_mut(primary).insert(connection);
            }
            let kind = rig_core::effect::EffectKind::Custom {
                kind: "note".into(),
                payload: serde_json::Value::Null,
            };
            let mut effect = world.spawn(EffectRequest {
                kind,
                handler: None,
            });
            if answered {
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
        let exits: Vec<_> = ends.into_iter().map(|ended| ended.exit).collect();
        assert_eq!(exits, [ExitReason::Quit], "answered: {answered}");
    }
}

#[test]
fn a_kept_run_answers_each_prompt_and_ends_once_nothing_keeps_it() {
    let model = MockCompletionModel::from_stream_turns(["one", "two"].map(|answer| {
        vec![
            MockStreamEvent::text(answer),
            MockStreamEvent::final_response_with_default_usage(),
        ]
    }));
    let mut app = run_app("hi", true, |world, primary| {
        if let Some(connection) = connection(&model) {
            world.entity_mut(primary).insert(connection);
        }
        let keep = world.spawn(KeepRunning).id();
        // A plugin sends a second prompt once the first is answered, then
        // lets the run end. It counts the turns itself: the order in which
        // two observers of one event run is not specified, so `Ends` may or
        // may not hold this turn yet.
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
    let ends = app.world_mut().resource_mut::<Ends>();
    let turns: Vec<_> = ends
        .1
        .iter()
        .map(|turn| (turn.answer.as_deref(), turn.failed))
        .collect();
    assert_eq!(turns, [(Some("one"), false), (Some("two"), false)]);
    let ended: Vec<_> = ends
        .0
        .iter()
        .map(|ended| (ended.answer.as_deref(), &ended.exit))
        .collect();
    assert_eq!(ended, [(Some("two"), &ExitReason::Quit)]);
}

/// A turn that starts and ends between two frames of the run, as a fast
/// model's can within one run of the agent loop, is still reported, and
/// the run ends once nothing keeps it.
#[test]
fn a_turn_the_run_never_saw_working_is_still_reported() {
    let model = MockCompletionModel::from_stream_turns(["one", "two"].map(|answer| {
        vec![
            MockStreamEvent::text(answer),
            MockStreamEvent::final_response_with_default_usage(),
        ]
    }));
    let mut agent = Entity::PLACEHOLDER;
    let mut app = started("hi", true, |world, primary| {
        agent = primary;
        if let Some(connection) = connection(&model) {
            world.entity_mut(primary).insert(connection);
        }
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
    world.trigger(Deliver::new(agent, "again", DeliveryMode::Steer));
    let started = Instant::now();
    let answered = |world: &mut World| {
        let mut turns = world.query_filtered::<(), With<TurnOf>>();
        let length = world
            .get::<Conversation>(agent)
            .map(|it| it.messages().len());
        length == Some(4) && turns.iter(world).next().is_none()
    };
    while started.elapsed() < Duration::from_secs(10) && !answered(world) {
        world.run_schedule(AgentPass);
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(answered(world), "the second turn ran");
    run_to_end(&mut app);
    let ends = app.world().resource::<Ends>();
    let turns: Vec<_> = ends.1.iter().map(|turn| turn.answer.as_deref()).collect();
    assert_eq!(turns, [Some("one"), Some("two")]);
    let ended: Vec<_> = ends.0.iter().map(|ended| &ended.exit).collect();
    assert_eq!(ended, [&ExitReason::Quit]);
}

/// Holds every model call of a turn, as an approval nobody gave.
fn hold_turn_calls(calls: Query<Entity, (ReadyToStart, With<TurnCall>)>, mut commands: Commands) {
    for call in &calls {
        commands.spawn(HoldOf(call));
    }
}

/// Runs a frame of `app`, then one more for each wake, as the host loop
/// does, up to 50 frames.
fn run_while_woken(app: &mut App, wakes: &std::sync::mpsc::Receiver<()>) {
    app.update();
    for _ in 0..50 {
        if wakes.try_recv().is_err() {
            return;
        }
        app.update();
    }
}

/// A run whose model it chooses itself sends its prompt without waiting for
/// something else to wake the loop.
#[test]
fn the_prompt_goes_out_without_waiting_for_a_wake() {
    const MODEL: &str = "test/model";
    let store = MemoryStore::default();
    let model = MockCompletionModel::from_stream_turns(Vec::<Vec<MockStreamEvent>>::new());
    let (mut app, wakes) = app_on(&store);
    app.insert_resource(scripted_models([(MODEL, &model)]))
        .add_plugins((AgentPlugin, ExitPlugin))
        .add_systems(AgentPass, hold_turn_calls.in_set(Stage::Gate))
        .run_headless(Some("hi".to_owned()));
    app.finish();
    run_while_woken(&mut app, &wakes);
    let world = app.world_mut();
    let agent = first_agent(world).unwrap_or(Entity::PLACEHOLDER);
    let messages: Vec<Message> = world
        .get::<Conversation>(agent)
        .map(|conversation| {
            conversation
                .messages()
                .iter()
                .map(|m| (**m).clone())
                .collect()
        })
        .unwrap_or_default();
    assert_eq!(messages, [Message::user("hi")]);
}

/// `rig -r <id> -p` with no prompt, after a run that stopped mid-turn: the
/// restored turn finishes, its answer is the run's, and the run ends.
#[test]
fn a_resumed_run_without_a_prompt_finishes_the_turn_it_restored() {
    const MODEL: &str = "test/model";
    let store = MemoryStore::default();
    let first = MockCompletionModel::from_stream_turns(Vec::<Vec<MockStreamEvent>>::new());
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

    let second = MockCompletionModel::from_stream_turns([vec![
        MockStreamEvent::text("hello"),
        MockStreamEvent::final_response_with_default_usage(),
    ]]);
    let (mut app, _) = app_on(&store);
    app.insert_resource(scripted_models([(MODEL, &second)]))
        .add_plugins((AgentPlugin, SessionLogPlugin, ExitPlugin))
        .init_resource::<Ends>()
        .add_observer(|ended: On<HeadlessEnded>, mut ends: ResMut<Ends>| {
            ends.0.push(ended.event().clone());
        })
        .run_headless(None);
    app.finish();
    run_to_end(&mut app);
    let ends = &app.world().resource::<Ends>().0;
    let ended: Vec<_> = ends
        .iter()
        .map(|ended| (ended.answer.as_deref(), &ended.exit, ended.why.as_deref()))
        .collect();
    assert_eq!(ended, [(Some("hello"), &ExitReason::Quit, None)]);
    assert_eq!((first.request_count(), second.request_count()), (0, 1));
}

/// `rig -m <model> -p hi` killed two frames in, before any reply: its
/// session holds the prompt and the model, and `rig -r <id> -p` answers it.
#[test]
fn a_run_killed_two_frames_in_leaves_its_prompt_to_resume() {
    const MODEL: &str = "test/model";
    let store = MemoryStore::default();
    let first = MockCompletionModel::from_stream_turns(Vec::<Vec<MockStreamEvent>>::new());
    let (mut app, _) = app_on(&store);
    let args = Invocation {
        model: Some(MODEL.to_owned()),
        ..Invocation::default()
    };
    // `--model` chooses the model in `First`, as `rig_basics::models` does.
    let choose =
        |agents: Query<Entity, With<Agent>>, mut done: Local<bool>, mut commands: Commands| {
            if let Some(agent) = agents.iter().next()
                && !*done
            {
                *done = true;
                let model = MODEL.to_owned();
                commands.trigger(SetModel {
                    entity: agent,
                    model,
                });
            }
        };
    app.insert_resource(scripted_models([(MODEL, &first)]))
        .insert_resource(Invoked {
            args,
            terminal: false,
        })
        .add_plugins((AgentPlugin, SessionLogPlugin, ExitPlugin))
        .add_systems(First, choose)
        .add_systems(AgentPass, hold_turn_calls.in_set(Stage::Gate))
        .run_headless(Some("hi".to_owned()));
    app.finish();
    app.update();
    app.update();
    // Killed: no exit, nothing more written.
    drop(app);

    let second = MockCompletionModel::from_stream_turns([vec![
        MockStreamEvent::text("hello"),
        MockStreamEvent::final_response_with_default_usage(),
    ]]);
    let (mut app, _) = app_on(&store);
    app.insert_resource(scripted_models([(MODEL, &second)]))
        .add_plugins((AgentPlugin, SessionLogPlugin, ExitPlugin))
        .init_resource::<Ends>()
        .add_observer(|ended: On<HeadlessEnded>, mut ends: ResMut<Ends>| {
            ends.0.push(ended.event().clone());
        })
        .run_headless(None);
    app.finish();
    run_to_end(&mut app);
    let ends = &app.world().resource::<Ends>().0;
    let ended: Vec<_> = ends
        .iter()
        .map(|ended| (ended.answer.as_deref(), &ended.exit, ended.why.as_deref()))
        .collect();
    assert_eq!(ended, [(Some("hello"), &ExitReason::Quit, None)]);
    assert_eq!((first.request_count(), second.request_count()), (0, 1));
}

/// A run the launcher restarted on a reload, without its prompt, with
/// nothing left to do ends with code 0.
#[test]
fn a_run_restarted_on_a_reload_with_nothing_to_do_ends_quietly() {
    let model = MockCompletionModel::from_stream_turns(Vec::<Vec<MockStreamEvent>>::new());
    let ends = run(" ", true, |world, primary| {
        if let Some(connection) = connection(&model) {
            world.entity_mut(primary).insert(connection);
        }
        world.insert_resource(AppStart(StartReason::Reloaded));
    });
    let codes: Vec<(ExitReason, Option<String>)> = ends
        .into_iter()
        .map(|ended| (ended.exit, ended.why))
        .collect();
    assert_eq!(codes, [(ExitReason::Quit, None)]);
}
