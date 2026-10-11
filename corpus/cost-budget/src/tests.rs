use std::sync::mpsc::Receiver;

use rig_cassette::journal::MemoryStore;
use rig_core::catalog::Catalog;
use rig_core::completion::Usage;
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness_test_support::{app_on, connect, run_until};

use super::*;

/// The notices written and the turns ended.
#[derive(Resource, Default)]
struct Seen {
    notices: Vec<String>,
    ended: usize,
}

/// The kernel and the budget plugin on `store`, after the first frame
/// restored the session, with its loop's wakes.
fn start(store: &MemoryStore) -> (App, Receiver<()>) {
    let (mut app, wakes) = app_on(store);
    app.add_plugins((AgentPlugin, JournalPlugin, CostBudgetPlugin))
        .init_resource::<Seen>()
        .add_systems(
            Last,
            |mut notices: MessageReader<Notice>, mut seen: ResMut<Seen>| {
                let texts = notices.read().map(|notice| notice.text.clone());
                seen.notices.extend(texts);
            },
        )
        .add_observer(|_: On<TurnEnded>, mut seen: ResMut<Seen>| seen.ended += 1);
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
    let before = app.world().resource::<Seen>().notices.len();
    app.world_mut().trigger(RunCommand {
        entity: agent,
        line: line.to_owned(),
    });
    app.update();
    let notices = &app.world().resource::<Seen>().notices;
    notices
        .get(before..)
        .map(<[String]>::to_vec)
        .unwrap_or_default()
}

/// The end of a reply whose million output tokens cost $0.60 at the
/// catalog's DeepSeek price. A real provider's wire prices its reply so;
/// the scripted model's adapter does not, so the test does it.
fn costly() -> MockStreamEvent {
    let mut usage = Usage::default();
    usage.input_tokens = Some(0);
    usage.output_tokens = Some(1_000_000);
    let catalog = Catalog::builtin();
    let pricing = catalog
        .resolve("deepseek/deepseek-flash")
        .ok()
        .and_then(|found| found.spec.pricing);
    MockStreamEvent::final_response(usage.cost(pricing.and_then(|pricing| pricing.cost(&usage))))
}

#[test]
fn a_spent_budget_stops_the_next_request_and_is_kept_with_the_session() {
    let store = MemoryStore::default();
    let (mut app, wakes) = start(&store);
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
    ]);
    let agent = first_agent(&mut app);
    let connection = connect(&model);
    assert!(
        agent.is_some() && connection.is_some(),
        "the catalog lists the model"
    );
    let (Some(agent), Some(connection)) = (agent, connection) else {
        return;
    };
    app.world_mut().entity_mut(agent).insert(connection);
    assert_eq!(
        command(&mut app, agent, "budget 1.00"),
        ["Budget: $0.00/$1.00."]
    );

    app.world_mut()
        .trigger(Deliver::new(agent, "Refactor it.", DeliveryMode::Queue));
    run_until(&mut app, &wakes, |world| world.resource::<Seen>().ended > 0);

    assert_eq!(model.request_count(), 2, "the third request was not sent");
    let notices = &app.world().resource::<Seen>().notices;
    let warnings = notices.iter().filter(|text| text.contains("nearly spent"));
    assert_eq!(warnings.count(), 1, "{notices:?}");
    assert!(
        notices
            .iter()
            .any(|text| text.contains("budget of $1.00 reached ($1.20 spent)")),
        "{notices:?}"
    );
    let shown = command(&mut app, agent, "budget");
    assert_eq!(shown, ["Budget: $1.20/$1.00."]);
    drop(app);

    // The restored session keeps counting from what it spent.
    let (mut restarted, wakes) = start(&store);
    let agent = first_agent(&mut restarted);
    let connection = connect(&model);
    assert!(agent.is_some(), "the agent was restored");
    let (Some(agent), Some(connection)) = (agent, connection) else {
        return;
    };
    assert_eq!(command(&mut restarted, agent, "budget"), shown);
    assert_eq!(
        restarted.world().resource::<Seen>().ended,
        0,
        "the stopped turn stays stopped"
    );

    // A raised budget lets `/retry` send the request that was held.
    restarted.world_mut().entity_mut(agent).insert(connection);
    command(&mut restarted, agent, "budget 2.00");
    // What rig-basics' `/retry` triggers.
    restarted.world_mut().trigger(Retry { entity: agent });
    run_until(&mut restarted, &wakes, |world| {
        world.resource::<Seen>().ended > 0
    });
    assert_eq!(model.request_count(), 3);
}
