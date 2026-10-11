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

/// How the runs ended.
#[derive(Resource, Default)]
struct Ends(Vec<HeadlessEnded>);

/// A headless run of `prompt`, its model named on the command line when
/// `model`, until it exits or ten seconds passed; how it ended.
fn run(prompt: &str, model: bool, setup: impl FnOnce(&mut World, Entity)) -> Vec<HeadlessEnded> {
    let mut app = App::new();
    let args = Invocation {
        model: model.then(|| "deepseek/deepseek-flash".to_owned()),
        ..Invocation::default()
    };
    app.insert_resource(Invoked {
        args,
        terminal: false,
    })
    .add_plugins((TaskPoolPlugin::default(), AgentPlugin))
    .init_resource::<Ends>()
    .add_observer(|ended: On<HeadlessEnded>, mut ends: ResMut<Ends>| {
        ends.0.push(ended.event().clone());
    })
    .run_headless(prompt);
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
    app.world_mut().resource_mut::<Ends>().0.drain(..).collect()
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
        exit: AppExit::Success,
    };
    assert_eq!(ends, [answered]);
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
        exit: AppExit::from_code(1),
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
    let codes: Vec<(AppExit, bool)> = ends
        .into_iter()
        .map(|ended| (ended.exit, ended.why.is_some()))
        .collect();
    assert_eq!(codes, [(AppExit::from_code(2), true)]);
}
