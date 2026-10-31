//! The helpers do what plugin tests lean on them for.

use std::time::SystemTime;

use rig_core::catalog::{Catalog, ModelSpec};
use rig_core::providers::registry::ConnectOptions;
use rig_core::test_utils::MockCompletionModel;
use rig_harness::harness_protocol::SessionId;
use rig_harness::prelude::*;
use rig_harness_test_support::{
    Collected, Ended, MemoryStore, ToolOutputs, advance, answer_open, app_in, app_on, calls,
    collect, first_agent, log_ended, log_tool_outputs, pass_wall_clock, reply, run_until,
    scratch_home, scripted_models,
};

/// Answers a `look` call at once.
fn look(called: On<ToolCalled<serde_json::Value>>, mut commands: Commands) {
    let seen = Outcome::from(ToolResult::success("seen".into()));
    commands.entity(called.call).insert_if_new(seen);
}

#[test]
fn a_scripted_catalog_model_answers_in_a_session_directory_and_its_work_is_collected() {
    let model = MockCompletionModel::from_stream_turns([
        calls("c1", "look", serde_json::json!({})),
        reply("done"),
    ]);
    let home = scratch_home("helpers-home");
    let id = SessionId::generate();
    let (mut app, wakes) = app_in(&home, &id);
    app.insert_resource(scripted_models([("corpus-test/one", &model)]))
        .add_plugins((AgentPlugin, SessionLogPlugin))
        .add_open_tool("look", "Look.", ToolOptions::default(), look)
        .init_resource::<ToolOutputs>()
        .add_observer(log_tool_outputs)
        .init_resource::<Ended>()
        .add_observer(log_ended);
    collect::<EffectRecorded>(&mut app);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    let world = app.world_mut();
    let model_name = "corpus-test/one".to_owned();
    world.trigger(SetModel {
        entity: agent,
        model: model_name,
    });
    world.trigger(Deliver::new(agent, "Look.", DeliveryMode::Steer));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });
    // The records of the last frame are collected in its `Last`.
    app.update();
    let outputs = &app.world().resource::<ToolOutputs>().0;
    assert_eq!(outputs.get("c1").map(String::as_str), Some("seen"));
    assert_eq!(model.request_count(), 2);
    let kinds: Vec<String> = app
        .world()
        .resource::<Collected<EffectRecorded>>()
        .0
        .iter()
        .map(|EffectRecorded(record)| record.kind.name().to_owned())
        .collect();
    assert!(kinds.iter().any(|kind| kind == "tool_call"), "{kinds:?}");
    assert!(home.session(&id).path().is_dir());
}

#[test]
fn a_question_is_answered_later_and_the_clocks_move_on() {
    let mut app = App::new();
    app.add_plugins((bevy_time::TimePlugin, rig_harness::clock::ClockPlugin));
    let world = app.world_mut();
    let ask = world.spawn(Ask::new("Run it?")).id();
    assert!(!answer_open(world, "Delete it?", &Answer::choice("yes")));
    assert!(answer_open(world, "Run it?", &Answer::choice("yes")));
    // Decided; the agent loop applies it as the question's outcome.
    let answer = world
        .get::<Decision>(ask)
        .and_then(Decision::get)
        .and_then(Answer::of);
    assert_eq!(answer, Some(Answer::choice("yes")));

    let hour = Duration::from_secs(3_600);
    let alarm = world.spawn(WakeAt(SystemTime::now() + hour)).id();
    app.update();
    assert!(!app.world().entity(alarm).contains::<Due>());
    pass_wall_clock(app.world_mut(), hour);
    app.update();
    assert!(app.world().entity(alarm).contains::<Due>());

    advance(&mut app, hour);
    assert!(app.world().resource::<Time<Virtual>>().elapsed() >= hour);
}

#[test]
fn a_scripted_app_lists_only_its_models_and_sends_no_http() {
    let model = MockCompletionModel::from_stream_turns([reply("done")]);
    let Models(mut connector) = scripted_models([("corpus-test/one", &model)]);
    let references = |specs: Vec<&ModelSpec>| -> Vec<String> {
        specs.into_iter().map(ModelSpec::reference).collect()
    };
    assert_eq!(
        references(connector.catalog().iter().collect()),
        ["corpus-test/one"]
    );
    assert_eq!(references(connector.reachable()), ["corpus-test/one"]);
    // A real provider's model a test adds itself, with a key, is refused
    // before it reaches the network.
    let openai = Catalog::builtin().resolve("openai/gpt-5").ok();
    assert!(openai.is_some(), "the catalog lists openai/gpt-5");
    let Some(openai) = openai.map(|found| found.spec.clone()) else {
        return;
    };
    connector.provide(openai.provider, ConnectOptions::new().api_key("sk-fake"));
    connector.catalog_mut().insert(openai.clone());
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.insert_resource(Models(connector))
        .add_plugins((AgentPlugin, SessionLogPlugin))
        .init_resource::<Ended>()
        .add_observer(log_ended);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    let world = app.world_mut();
    world.trigger(SetModel {
        entity: agent,
        model: openai.reference(),
    });
    world.trigger(Deliver::new(agent, "Hi.", DeliveryMode::Steer));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });
    let ended = &app.world().resource::<Ended>().0;
    assert!(
        matches!(ended.as_slice(), [TurnOutcome::Failed(why)] if why.contains("sends no HTTP")),
        "{ended:?}"
    );
    assert_eq!(model.request_count(), 0);
}
