use std::sync::mpsc::Receiver;

use rig_cassette::journal::MemoryStore;
use rig_core::catalog::Catalog;
use rig_core::completion::Usage;
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness_test_support::{Ended, app_on, connect, log_ended, run_until};

use super::*;

/// The notices written so far.
#[derive(Resource, Default)]
struct Seen(Vec<String>);

/// The budget plugin on `store`, with a `[budget]` default of $5, after
/// the first frame restored the session, with its loop's wakes.
fn start(store: &MemoryStore) -> (App, Receiver<()>) {
    let (mut app, wakes) = app_on(store);
    app.insert_resource(Settings::from_toml("[budget]\ndefault = 5.0"))
        .add_plugins((AgentPlugin, JournalPlugin, CostBudgetPlugin))
        .init_resource::<Seen>()
        .init_resource::<Ended>()
        .add_observer(log_ended)
        .add_systems(
            Last,
            |mut notices: MessageReader<Notice>, mut seen: ResMut<Seen>| {
                seen.0
                    .extend(notices.read().map(|notice| notice.text.clone()));
            },
        );
    app.finish();
    app.update();
    (app, wakes)
}

/// The app's first agent.
fn first_agent(app: &mut App) -> Option<Entity> {
    let world = app.world_mut();
    world
        .query_filtered::<Entity, With<Agent>>()
        .iter(world)
        .next()
}

/// Runs `/line` for `agent`, and returns the notices it wrote.
fn command(app: &mut App, agent: Entity, line: &str) -> Vec<String> {
    let before = app.world().resource::<Seen>().0.len();
    let line = line.to_owned();
    app.world_mut().trigger(RunCommand {
        entity: agent,
        line,
    });
    app.update();
    let seen = &app.world().resource::<Seen>().0;
    seen.get(before..)
        .map(<[String]>::to_vec)
        .unwrap_or_default()
}

/// The end of a reply of a million output tokens: $0.60 at the catalog's
/// DeepSeek price.
fn costly() -> MockStreamEvent {
    let mut usage = Usage::default();
    usage.input_tokens = Some(0);
    usage.output_tokens = Some(1_000_000);
    MockStreamEvent::final_response(usage)
}

#[test]
fn a_spent_budget_stops_the_next_request_and_is_kept_with_the_session() {
    let spec = Catalog::builtin().resolve("deepseek/deepseek-flash");
    assert!(spec.is_ok(), "the catalog lists DeepSeek");
    let Ok(spec) = spec.map(|found| found.shared()) else {
        return;
    };
    let model = MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::tool_call("1", "look", serde_json::json!({})),
            costly(),
        ],
        vec![
            MockStreamEvent::tool_call("2", "look", serde_json::json!({})),
            costly(),
        ],
        vec![MockStreamEvent::text("done"), costly()],
    ])
    .priced_as(spec);
    let store = MemoryStore::default();
    let (mut app, wakes) = start(&store);
    let (agent, connection) = (first_agent(&mut app), connect(&model));
    assert!(
        agent.is_some() && connection.is_some(),
        "an agent on the scripted model"
    );
    let (Some(agent), Some(connection)) = (agent, connection) else {
        return;
    };
    app.world_mut().entity_mut(agent).insert(connection);
    assert_eq!(command(&mut app, agent, "budget"), ["Budget: $0.00/$5.00."]);
    assert_eq!(
        command(&mut app, agent, "budget 1.00"),
        ["Budget: $0.00/$1.00."]
    );

    app.world_mut()
        .trigger(Deliver::new(agent, "Refactor it.", DeliveryMode::Queue));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });

    assert_eq!(model.request_count(), 2, "the third request was not sent");
    let ended = &app.world().resource::<Ended>().0;
    assert!(matches!(ended.as_slice(), [TurnOutcome::Failed(why)] if why.contains("budget")));
    let seen = &app.world().resource::<Seen>().0;
    let warnings = seen.iter().filter(|text| text.contains("nearly spent"));
    assert_eq!(warnings.count(), 1, "{seen:?}");
    let reached = "budget of $1.00 reached ($1.20 spent)";
    assert!(seen.iter().any(|text| text.contains(reached)), "{seen:?}");
    let shown = command(&mut app, agent, "budget");
    assert_eq!(shown, ["Budget: $1.20/$1.00."]);
    drop(app);

    // The restored session keeps its budget, not the default, and its
    // spending, and does not answer the stopped request by itself.
    let (mut restarted, wakes) = start(&store);
    let (agent, connection) = (first_agent(&mut restarted), connect(&model));
    assert!(agent.is_some(), "the agent was restored");
    let (Some(agent), Some(connection)) = (agent, connection) else {
        return;
    };
    assert_eq!(command(&mut restarted, agent, "budget"), shown);
    assert!(restarted.world().resource::<Ended>().0.is_empty());

    // A raised budget lets `/retry` send the request that was refused.
    restarted.world_mut().entity_mut(agent).insert(connection);
    command(&mut restarted, agent, "budget 2.00");
    // What rig-basics' `/retry` triggers.
    restarted.world_mut().trigger(Retry { entity: agent });
    run_until(&mut restarted, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });
    assert_eq!(model.request_count(), 3);
}
