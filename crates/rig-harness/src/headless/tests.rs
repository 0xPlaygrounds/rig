use std::time::{Duration, Instant};

use bevy_app::TaskPoolPlugin;
use bevy_tasks::futures_lite::stream;
use bevy_tasks::{IoTaskPool, TaskPool};
use rig::harness_protocol::Invocation;
use rig_core::catalog::Catalog;
use rig_core::operation::Completion;
use rig_core::serve::ErasedHandler;
use rig_core::serve::adapters::ModelAdapter;
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_ecs::AgentPlugin;
use rig_ecs::effects::Handler;

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
    let started = Instant::now();
    while started.elapsed() < Duration::from_secs(10) && app.should_exit().is_none() {
        app.update();
        std::thread::sleep(Duration::from_millis(1));
    }
    // Nothing more once it ended.
    app.update();
    app
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
        // lets the run end.
        world.add_observer(
            move |ended: On<HeadlessTurnEnded>, ends: Res<Ends>, mut commands: Commands| match ends
                .1
                .len()
            {
                0 => commands.trigger(Deliver::new(ended.agent, "again", DeliveryMode::Steer)),
                _ => commands.entity(keep).despawn(),
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
