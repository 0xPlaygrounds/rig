use std::sync::mpsc::Receiver;

use rig_core::catalog::Catalog;
use rig_core::completion::Usage;
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness_test_support::MemoryStore;
use rig_harness_test_support::{
    Collected, Ended, app_on, collect, connect, first_agent, log_ended, run_until,
};

use super::*;

/// The notices written so far.
fn seen(app: &App) -> Vec<String> {
    let notices = &app.world().resource::<Collected<Notice>>().0;
    notices.iter().map(|notice| notice.text.clone()).collect()
}

/// The budget plugin on `store`, with a `[budget]` default of $5, after
/// the first frame restored the session, with its loop's wakes.
fn start(store: &MemoryStore) -> (App, Receiver<()>) {
    let (mut app, wakes) = app_on(store);
    app.insert_resource(Settings::from_toml("[budget]\ndefault = 5.0"))
        .add_plugins((AgentPlugin, SessionLogPlugin, CostBudgetPlugin))
        .init_resource::<Ended>()
        .add_observer(log_ended);
    collect::<Notice>(&mut app);
    app.finish();
    app.update();
    (app, wakes)
}

/// Runs `/line` for `agent`, and returns the notices it wrote.
fn command(app: &mut App, agent: Entity, line: &str) -> Vec<String> {
    let before = seen(app).len();
    app.world_mut()
        .trigger(RunCommand::new(agent, line.to_owned()));
    app.update();
    seen(app).into_iter().skip(before).collect()
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
    let (agent, connection) = (first_agent(app.world_mut()), connect(&model));
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
    let seen = seen(&app);
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
    let (agent, connection) = (first_agent(restarted.world_mut()), connect(&model));
    assert!(agent.is_some(), "the agent was restored");
    let (Some(agent), Some(connection)) = (agent, connection) else {
        return;
    };
    assert_eq!(command(&mut restarted, agent, "budget"), shown);
    assert!(restarted.world().resource::<Ended>().0.is_empty());

    // A raised budget lets the next message go on from the tool results.
    restarted.world_mut().entity_mut(agent).insert(connection);
    command(&mut restarted, agent, "budget 2.00");
    restarted
        .world_mut()
        .trigger(Deliver::new(agent, "Go on.", DeliveryMode::Queue));
    run_until(&mut restarted, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });
    assert_eq!(model.request_count(), 3);
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(CostBudgetPlugin);
}
