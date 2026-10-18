use rig_basics::subagents::{SubagentsPlugin, TASK};
use rig_basics::telemetry::UsagePlugin;
use rig_core::catalog::ModelSpec;
use rig_core::completion::Usage;
use rig_core::providers::registry::ProviderId;
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness_test_support::MemoryStore;
use rig_harness_test_support::{
    Collected, Ended, app_on, collect, log_ended, run_until, scripted_models,
};
use rig_tui::TuiPlugin;
use serde_json::json;

use super::*;

fn used(input: u64, output: u64) -> MockStreamEvent {
    MockStreamEvent::final_response(Usage::new().input_tokens(input).output_tokens(output))
}

fn answer(text: &str, input: u64, output: u64) -> Vec<MockStreamEvent> {
    vec![MockStreamEvent::text(text), used(input, output)]
}

/// The gauge an agent's status line shows, by text and tone.
fn gauge(world: &World, agent: Entity) -> Option<(String, Tone)> {
    let items = world.get::<StatusItems>(agent)?;
    let mut at_place = items
        .iter()
        .filter(|item| (item.side, item.order) == (GAUGE.side, GAUGE.order));
    let item = at_place.next()?;
    assert!(at_place.next().is_none(), "one item at the gauge's place");
    Some((item.text.clone(), item.tone))
}

fn warnings(world: &World) -> usize {
    let notices = &world.resource::<Collected<Notice>>().0;
    notices
        .iter()
        .filter(|notice| notice.text.contains("nearly full"))
        .count()
}

fn shows(world: &World, agent: Entity, text: &str, tone: Tone) -> bool {
    gauge(world, agent) == Some((text.to_owned(), tone))
}

#[test]
fn the_gauge_follows_tokens_models_and_focus_and_warns_once() {
    let task =
        json!({ "description": "Look around", "prompt": "Look around.", "model": "scripted/sub" });
    let parent = MockCompletionModel::from_stream_turns([
        answer("Read it all.", 140_000, 10_000),
        answer("More.", 185_000, 0),
        answer("Even more.", 190_000, 0),
        vec![
            MockStreamEvent::tool_call("t1", TASK, task),
            used(190_000, 0),
        ],
        answer("Started it.", 190_000, 0),
        answer("It reported.", 190_000, 0),
        answer("Spare.", 190_000, 0),
    ]);
    let sub = MockCompletionModel::from_stream_turns([answer("Nothing here.", 20_000, 0)]);
    let idle = MockCompletionModel::from_stream_turns(Vec::<Vec<MockStreamEvent>>::new());
    let mut models = scripted_models([
        ("scripted/w200k", &parent),
        ("scripted/w1m", &idle),
        ("scripted/unknown", &idle),
        ("scripted/sub", &sub),
    ]);
    let Ok(vendor) = ProviderId::declare("scripted") else {
        return;
    };
    for (name, window) in [("w200k", 200_000), ("w1m", 1_000_000), ("sub", 200_000)] {
        let spec = ModelSpec::new(vendor, name)
            .with_tools(true)
            .with_context_window(window);
        models.0.catalog_mut().insert(spec);
    }

    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.insert_resource(models)
        .add_plugins((
            AgentPlugin,
            SessionLogPlugin,
            TuiPlugin,
            UsagePlugin,
            SubagentsPlugin,
            ContextGaugePlugin,
        ))
        .init_resource::<Ended>()
        .add_observer(log_ended);
    collect::<Notice>(&mut app);
    app.finish();
    app.update();
    let agent = (app.world_mut())
        .spawn((Agent, ModelChoice("scripted/w200k".to_owned())))
        .id();
    app.update();
    assert!(
        shows(app.world(), agent, "ctx —", Tone::Dim),
        "{:?}",
        gauge(app.world(), agent)
    );

    let turn = |app: &mut App, ended: usize| {
        app.world_mut()
            .trigger(Deliver::new(agent, "go on", DeliveryMode::Steer));
        run_until(app, &wakes, |world| {
            world.resource::<Ended>().0.len() >= ended
        });
        app.update();
    };
    let set_model = |app: &mut App, model: &str| {
        let model = model.to_owned();
        app.world_mut().trigger(SetModel {
            entity: agent,
            model,
        });
        app.update();
    };

    // 140k in and 10k out of 200k: a warning, over the default meter's item.
    turn(&mut app, 1);
    let at_75 = "ctx 75% · 150k/200k";
    assert!(
        shows(app.world(), agent, at_75, Tone::Yellow),
        "{:?}",
        gauge(app.world(), agent)
    );

    // A bigger window, at once, with the same tokens; then no known window.
    set_model(&mut app, "scripted/w1m");
    assert!(
        shows(app.world(), agent, "ctx 15% · 150k/1.0M", Tone::Dim),
        "{:?}",
        gauge(app.world(), agent)
    );
    set_model(&mut app, "scripted/unknown");
    assert!(
        shows(app.world(), agent, "ctx 150k", Tone::Dim),
        "{:?}",
        gauge(app.world(), agent)
    );
    set_model(&mut app, "scripted/w200k");
    assert!(shows(app.world(), agent, at_75, Tone::Yellow));
    assert_eq!(warnings(app.world()), 0);

    // Above 90%: red, and one notice; not a second one at 95%.
    turn(&mut app, 2);
    assert!(
        shows(app.world(), agent, "ctx 92% · 185k/200k", Tone::Red),
        "{:?}",
        gauge(app.world(), agent)
    );
    assert_eq!(warnings(app.world()), 1);
    turn(&mut app, 3);
    assert!(shows(app.world(), agent, "ctx 95% · 190k/200k", Tone::Red));
    assert_eq!(warnings(app.world()), 1);

    // A subagent has its own gauge, which its line shows when focused.
    app.world_mut()
        .trigger(Deliver::new(agent, "ask a subagent", DeliveryMode::Steer));
    let subagent = |world: &mut World| {
        let mut spawned = world.query_filtered::<Entity, With<SpawnedBy>>();
        spawned.iter(world).next()
    };
    run_until(&mut app, &wakes, |world| {
        world.resource::<Ended>().0.len() >= 6
    });
    app.update();
    let child = subagent(app.world_mut());
    assert!(child.is_some(), "no subagent");
    let Some(child) = child else {
        return;
    };
    assert!(
        shows(app.world(), child, "ctx 10% · 20k/200k", Tone::Dim),
        "{:?}",
        gauge(app.world(), child)
    );
    assert!(shows(app.world(), agent, "ctx 95% · 190k/200k", Tone::Red));
    assert_eq!(warnings(app.world()), 1);
}
