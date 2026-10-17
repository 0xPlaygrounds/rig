use std::sync::mpsc::Receiver;

use rig_basics::compaction::CompactionPlugin;
use rig_basics::subagents::SubagentsPlugin;
use rig_core::catalog::{Catalog, Connector};
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness_test_support::MemoryStore;

use super::*;
use rig_harness_test_support::{
    Collected, Ended, app_on, collect, connect, first_agent, log_ended, reply, run_until,
};

/// The kernel, compaction, subagents and pinned notes on `store`, after
/// the first frame restored the session, with its loop's wakes and its
/// agent, answered by `model` when given.
fn start(
    store: &MemoryStore,
    model: Option<&MockCompletionModel>,
) -> Option<(App, Receiver<()>, Entity)> {
    let (mut app, wakes) = app_on(store);
    app.set_error_handler(bevy_ecs::error::warn)
        .insert_resource(Models(Connector::new(Catalog::default())))
        .add_plugins((
            AgentPlugin,
            SessionLogPlugin,
            CompactionPlugin,
            SubagentsPlugin,
            PinnedNotesPlugin,
        ))
        .init_resource::<Ended>()
        .add_observer(log_ended);
    if let Some(connection) = model.and_then(connect) {
        let choice = ModelChoice(connection.spec.reference());
        app.world_mut().spawn((Agent, connection, choice));
    }
    collect::<Notice>(&mut app).finish();
    app.update();
    let agent = first_agent(app.world_mut())?;
    Some((app, wakes, agent))
}

/// Runs `/line` for `agent`.
fn run(app: &mut App, agent: Entity, line: &str) {
    let line = line.to_owned();
    app.world_mut().trigger(RunCommand::new(agent, line));
}

/// Asks `agent` something and runs frames until its turn ended.
fn ask(app: &mut App, wakes: &Receiver<()>, agent: Entity, text: &str) {
    let ended = app.world().resource::<Ended>().0.len();
    app.world_mut()
        .trigger(Deliver::new(agent, text, DeliveryMode::Steer));
    run_until(app, wakes, |world| {
        world.resource::<Ended>().0.len() > ended
    });
}

#[test]
fn pins_go_with_every_request_survive_compaction_and_come_back() {
    let model = MockCompletionModel::from_stream_turns([
        reply("First answer."),
        reply("## Goal\nAnswer questions."),
        reply("Second answer."),
    ]);
    let store = MemoryStore::default();
    let started = start(&store, Some(&model));
    assert!(started.is_some(), "the session has one agent");
    let Some((mut app, wakes, agent)) = started else {
        return;
    };

    run(&mut app, agent, "pin Use approach B");
    ask(&mut app, &wakes, agent, "Which approach?");
    run(&mut app, agent, "compact");
    run_until(&mut app, &wakes, |world| {
        world
            .get::<Conversation>(agent)
            .is_some_and(|conversation| rig_basics::compaction::summarized_prefix(conversation) > 0)
    });
    ask(&mut app, &wakes, agent, "And now?");

    let sent: Vec<String> = model
        .requests()
        .iter()
        .map(|request| serde_json::to_string(request).unwrap_or_default())
        .collect();
    assert_eq!(
        sent.len(),
        3,
        "a request, a summary and a request: {sent:?}"
    );
    let [first, summary, after] = sent.as_slice() else {
        return;
    };
    assert!(
        first.contains("Pinned notes") && first.contains("1. Use approach B"),
        "{first}"
    );
    assert!(
        summary.contains("do not repeat them in the summary:\\n1. Use approach B"),
        "{summary}"
    );
    assert!(after.contains("## Goal"), "after compaction: {after}");
    assert!(after.contains("1. Use approach B"), "{after}");
    drop(app);

    // A restart restores the agent with its pins.
    let restarted = start(&store, None);
    assert!(restarted.is_some(), "the agent was restored");
    let Some((mut app, _wakes, agent)) = restarted else {
        return;
    };
    run(&mut app, agent, "pins");
    app.update();
    let notices = &app.world().resource::<Collected<Notice>>().0;
    let said: Vec<&str> = notices.iter().map(|notice| notice.text.as_str()).collect();
    assert!(
        said.contains(&"Pinned notes (1):\n1. Use approach B"),
        "{said:?}"
    );
}

#[test]
fn only_a_task_with_inherit_pins_passes_the_pins_on() {
    let task = |id: &str, inherit: bool| {
        let prompt = format!("Look, inheriting: {inherit}.");
        let args =
            serde_json::json!({ "description": "Look", "prompt": prompt, "inherit_pins": inherit });
        MockStreamEvent::tool_call(id, TASK, args)
    };
    let model = MockCompletionModel::from_stream_turns([
        vec![
            task("t1", true),
            task("t2", false),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        reply("Started both."),
        reply("Looked."),
        reply("Looked."),
        reply("Both looked."),
    ]);
    let started = start(&MemoryStore::default(), Some(&model));
    assert!(started.is_some(), "the session has one agent");
    let Some((mut app, wakes, parent)) = started else {
        return;
    };
    run(&mut app, parent, "pin Keep the API stable");
    app.world_mut()
        .trigger(Deliver::new(parent, "Look twice.", DeliveryMode::Steer));
    run_until(&mut app, &wakes, |world| {
        world.resource::<Ended>().0.len() >= 4
    });

    let sent: Vec<String> = model
        .requests()
        .iter()
        .map(|request| serde_json::to_string(request).unwrap_or_default())
        .collect();
    assert_eq!(
        sent.len(),
        5,
        "the parent's two turns (three requests) and each subagent's"
    );
    assert!(
        sent.first()
            .is_some_and(|first| first.contains(INHERIT_PINS)),
        "the model is offered `inherit_pins` on `task`"
    );
    // Each subagent's requests: whether it was asked to inherit, and
    // whether its prompt has the pin. The parent's all have it.
    let pin = "1. Keep the API stable";
    let mut children: Vec<(bool, bool)> = sent
        .iter()
        .filter(|request| request.contains("<subagent_role>"))
        .map(|request| (request.contains("inheriting: true"), request.contains(pin)))
        .collect();
    children.sort_unstable();
    assert_eq!(children, [(false, false), (true, true)], "{sent:?}");
    let parent = sent
        .iter()
        .filter(|request| !request.contains("<subagent_role>"));
    assert!(
        parent.clone().all(|request| request.contains(pin)),
        "{sent:?}"
    );
    assert_eq!(parent.count(), 3, "{sent:?}");
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(PinnedNotesPlugin);
}
