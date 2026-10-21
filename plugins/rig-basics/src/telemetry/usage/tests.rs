use std::sync::Arc;
use std::sync::mpsc::Receiver;

use rig_core::completion::{CompletionRequest, Message};
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness_test_support::MemoryStore;
use rig_harness_test_support::{Ended, app_on, connect, log_ended, reply, run_until};

use super::*;

/// An app on `store` that counts spending, after its first frame.
fn start(store: &MemoryStore) -> (App, Receiver<()>) {
    let (mut app, wakes) = app_on(store);
    app.add_plugins((AgentPlugin, SessionLogPlugin, UsagePlugin))
        .init_resource::<Ended>()
        .add_observer(log_ended);
    app.finish();
    app.update();
    (app, wakes)
}

/// The text of every notice written since the last drain.
fn notices(world: &mut World) -> Vec<String> {
    let mut notices = world.resource_mut::<Messages<Notice>>();
    notices.drain().map(|notice| notice.text).collect()
}

#[test]
fn spending_is_split_per_model_and_purpose_and_outlives_its_agent() {
    let model = MockCompletionModel::from_stream_turns([reply("hi"), reply("a summary")]);
    let store = MemoryStore::default();
    let (mut app, wakes) = start(&store);
    let Some(connection) = connect(&model) else {
        return;
    };
    let reference = connection.spec.reference();
    let world = app.world_mut();
    let agent = world.spawn((Agent, connection)).id();
    world.trigger(Deliver::new(agent, "hello", DeliveryMode::Steer));
    run_until(&mut app, &wakes, |world| {
        world.resource::<Ended>().0.len() == 1
    });
    // A plugin's own call for the agent, outside any turn.
    let summary = (
        ModelRequest::new(CompletionRequest::new(Message::user("Summarize."))),
        Effect::default().with_purpose("summary"),
        ForAgent(agent),
    );
    app.world_mut().spawn(summary);
    let spent = |world: &World| world.get::<AgentUsage>(agent).cloned().unwrap_or_default();
    run_until(&mut app, &wakes, |world| spent(world).total.calls == 2);

    let spending = spent(app.world());
    let calls =
        |split: &BTreeMap<String, UsageTotals>, key: &str| split.get(key).map(|spent| spent.calls);
    assert_eq!(spending.total.calls, 2);
    assert_eq!(calls(&spending.models, &reference), Some(2));
    assert_eq!(calls(&spending.purposes, "turn"), Some(1));
    assert_eq!(calls(&spending.purposes, "summary"), Some(1));
    assert_eq!(app.world().resource::<SessionUsage>().0, spending);

    // `/usage` splits the session per purpose and adds other plugins' rows.
    let world = app.world_mut();
    world.spawn(UsageRow::new("budget", "Budget: 2 of 10 calls."));
    notices(world);
    world.trigger(RunCommand::new(agent, "usage".to_owned()));
    world.flush();
    let shown = notices(world).join("\n");
    assert!(shown.contains("  summary: "), "{shown}");
    assert!(shown.ends_with("Budget: 2 of 10 calls."), "{shown}");

    // The session keeps what a despawned agent spent, and so does a
    // restart.
    app.world_mut().despawn(agent);
    app.update();
    let session = app.world().resource::<SessionUsage>().0.clone();
    assert_eq!(session.total.calls, 2);
    drop(app);
    let (restarted, _wakes) = start(&store);
    assert_eq!(restarted.world().resource::<SessionUsage>().0, session);
}

#[test]
fn usage_labels_catalog_costs_as_estimates_not_subscription_charges() {
    let store = MemoryStore::default();
    let (mut app, _wakes) = start(&store);
    let mut spent = AgentUsage::default();
    spent.record(
        Some("claude-cloak/claude-opus-5-5"),
        Some("turn"),
        &Usage::new().cost(rig_core::completion::Cost::from_total(0.176192)),
    );
    let world = app.world_mut();
    world.resource_mut::<SessionUsage>().0 = spent.clone();
    let agent = world.spawn((Agent, spent)).id();
    notices(world);
    world.trigger(RunCommand::new(agent, "usage".to_owned()));
    world.flush();
    let shown = notices(world).join("\n");
    assert!(shown.contains("$0.176"), "{shown}");
    assert!(shown.contains("API list-price estimates"), "{shown}");
    assert!(
        shown.contains("not subscription charges or verified invoices"),
        "{shown}"
    );
}

/// Sends each turn call to its connection instead of the agent's model, as
/// a router or a race does.
#[derive(Resource)]
struct RouteTo(Connection);

fn route(mut calls: Query<&mut ModelRequest, (With<TurnCall>, Added<Admitted>)>, to: Res<RouteTo>) {
    for mut call in &mut calls {
        call.connection = Some(to.0.clone());
    }
}

#[test]
fn the_context_is_against_the_window_of_the_model_a_call_went_to() {
    let model = MockCompletionModel::from_stream_turns([vec![
        MockStreamEvent::text("hi"),
        MockStreamEvent::final_response_with_total_tokens(500),
    ]]);
    let store = MemoryStore::default();
    let (mut app, wakes) = start(&store);
    let Some(connection) = connect(&model) else {
        return;
    };
    let routed = Connection {
        spec: Arc::new((*connection.spec).clone().with_context_window(2_000)),
        handler: connection.handler.clone(),
    };
    let reference = routed.spec.reference();
    app.insert_resource(RouteTo(routed))
        .add_systems(AgentPass, route.in_set(PrepareSystems::Edit));
    let world = app.world_mut();
    let agent = world.spawn((Agent, connection)).id();
    world.trigger(Deliver::new(agent, "hello", DeliveryMode::Steer));
    run_until(&mut app, &wakes, |world| {
        world.resource::<Ended>().0.len() == 1
    });
    let world = app.world_mut();
    let spent = world.get::<AgentUsage>(agent).cloned().unwrap_or_default();
    assert_eq!(spent.models.keys().collect::<Vec<_>>(), [&reference]);
    notices(world);
    world.trigger(RunCommand::new(agent, "usage".to_owned()));
    world.flush();
    let shown = notices(world).join("\n");
    assert!(shown.contains("Context: 500/2.0k (25%) tokens."), "{shown}");

    // After `/model`, the next request goes to the new model: its window
    // stands until it replies.
    let Some(other) = connect(&model) else {
        return;
    };
    let other = Connection {
        spec: Arc::new((*other.spec).clone().with_context_window(10_000)),
        handler: other.handler,
    };
    world.entity_mut(agent).insert(other);
    world.flush();
    world.trigger(RunCommand::new(agent, "usage".to_owned()));
    world.flush();
    let shown = notices(world).join("\n");
    assert!(shown.contains("Context: 500/10k (5%) tokens."), "{shown}");
}

/// A call that settled without a reply, such as a race's loser, counts what
/// its CallUsage says, as a reply's usage counts.
#[test]
fn a_superseded_call_counts_its_call_usage_under_the_model_it_went_to() {
    let model = MockCompletionModel::from_stream_turns([reply("unused")]);
    let store = MemoryStore::default();
    let (mut app, wakes) = start(&store);
    let Some(connection) = connect(&model) else {
        return;
    };
    let reference = connection.spec.reference();
    let world = app.world_mut();
    let agent = world.spawn(Agent).id();
    let turn = world.spawn(TurnOf(agent)).id();
    let winner = world.spawn_empty().id();
    let mut used = Usage::default();
    used.input_tokens = Some(100);
    world.spawn((
        ModelRequest::new(CompletionRequest::new(Message::user("race"))),
        Effect::default().with_purpose("turn"),
        (ForAgent(agent), CallOf(turn), SentTo(connection)),
        CallUsage(used),
        SupersededBy(winner),
    ));
    let input = |spent: &UsageTotals| spent.tokens.input_tokens;
    let spent = |world: &World| world.get::<AgentUsage>(agent).cloned().unwrap_or_default();
    run_until(&mut app, &wakes, |world| spent(world).total.calls == 1);

    let spending = spent(app.world());
    assert_eq!(input(&spending.total), Some(100));
    assert_eq!(spending.models.get(&reference).and_then(input), Some(100));
    let session = &app.world().resource::<SessionUsage>().0;
    assert_eq!(session.models.get(&reference).and_then(input), Some(100));
    let turn = app
        .world()
        .get::<TurnUsage>(turn)
        .map(|TurnUsage(spent)| input(spent));
    assert_eq!(turn, Some(Some(100)));
}
