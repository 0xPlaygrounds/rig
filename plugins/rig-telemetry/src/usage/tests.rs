use std::sync::mpsc::Receiver;

use rig_core::completion::{CompletionRequest, Message};
use rig_core::test_utils::MockCompletionModel;
use rig_harness_test_support::MemoryStore;
use rig_harness_test_support::{Ended, app_on, connect, log_ended, reply, run_until};

use super::*;

/// An app on `store` that counts spending, after its first frame.
fn start(store: &MemoryStore) -> (App, Receiver<()>) {
    let (mut app, wakes) = app_on(store);
    app.add_plugins((AgentPlugin, JournalPlugin, UsagePlugin))
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
    let spent = |world: &World| world.get::<Spending>(agent).cloned().unwrap_or_default();
    run_until(&mut app, &wakes, |world| spent(world).total.calls == 2);

    let spending = spent(app.world());
    let calls =
        |split: &BTreeMap<String, UsageTotals>, key: &str| split.get(key).map(|spent| spent.calls);
    assert_eq!(spending.total.calls, 2);
    assert_eq!(calls(&spending.models, &reference), Some(2));
    assert_eq!(calls(&spending.purposes, "turn"), Some(1));
    assert_eq!(calls(&spending.purposes, "summary"), Some(1));
    assert_eq!(app.world().resource::<SessionSpending>().0, spending);

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
    let session = app.world().resource::<SessionSpending>().0.clone();
    assert_eq!(session.total.calls, 2);
    drop(app);
    let (restarted, _wakes) = start(&store);
    assert_eq!(restarted.world().resource::<SessionSpending>().0, session);
}
