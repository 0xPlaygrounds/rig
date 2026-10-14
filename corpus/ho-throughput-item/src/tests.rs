use bevy_time::TimeUpdateStrategy;
use rig_core::completion::{CompletionRequest, Message, Usage};
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness_test_support::MemoryStore;
use rig_harness_test_support::{
    Collected, Ended, app_on, calls, collect, connect, log_ended, run_until,
};
use rig_tui::TuiPlugin;

use super::*;

/// Each frame is this much later on Bevy's clock: the scripted stream has
/// no delays of its own.
const FRAME: Duration = Duration::from_millis(300);

fn item(world: &World, agent: Entity) -> Option<StatusItem> {
    let items = world.get::<StatusItems>(agent)?;
    items
        .iter()
        .find(|item| (item.side, item.order) == (SPEED.side, SPEED.order))
        .cloned()
}

#[test]
fn a_turns_call_shows_its_speed_and_a_plugins_call_only_counts() {
    let mut usage = Usage::default();
    usage.output_tokens = Some(37);
    let turn = vec![
        MockStreamEvent::reasoning("Let me think about it."),
        MockStreamEvent::text("Hello there, how can I help?"),
        MockStreamEvent::final_response(usage),
    ];
    // The plugin's call answers with a tool call and reports no usage: its
    // tokens are estimated from the arguments streamed.
    let arguments = serde_json::json!({ "path": "src/a/rather/long/path/to/a/file.rs" });
    let estimate = (arguments.to_string().len() / 4) as u64;
    let summary = calls("c1", "read", arguments);
    let model = MockCompletionModel::from_stream_turns([turn, summary]);
    let Some(connection) = connect(&model) else {
        return;
    };
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.add_plugins((AgentPlugin, SessionLogPlugin, TuiPlugin, ThroughputPlugin))
        .insert_resource(TimeUpdateStrategy::ManualDuration(FRAME))
        .init_resource::<Ended>()
        .add_observer(log_ended);
    collect::<Notice>(&mut app);
    app.finish();
    app.update();
    let agent = app.world_mut().spawn((Agent, connection)).id();
    app.world_mut()
        .trigger(Deliver::new(agent, "hello", DeliveryMode::Steer));
    run_until(&mut app, &wakes, |world| {
        world.resource::<Ended>().0.len() == 1
    });
    app.update();

    let model = "deepseek/deepseek-flash";
    let stats = |world: &World| world.resource::<SpeedTable>().0.get(model).cloned();
    let after_turn = stats(app.world()).unwrap_or_default();
    assert_eq!(after_turn.calls, 1, "{after_turn:?}");
    assert_eq!(after_turn.output_tokens, 37);
    assert!(after_turn.first_token.iter().all(|first| !first.is_zero()));
    assert_eq!(after_turn.first_token.len(), 1);
    let shown = item(app.world(), agent);
    let says_rate = shown
        .as_ref()
        .is_some_and(|item| item.text.contains("tok/s"));
    assert!(says_rate, "{shown:?}");
    assert_eq!(shown.as_ref().map(|item| item.tone), Some(Tone::Dim));

    // A plugin's own call for the agent counts, but is not the agent's.
    let summary = (
        ModelRequest::new(CompletionRequest::new(Message::user("Summarize."))),
        ForAgent(agent),
    );
    app.world_mut().spawn(summary);
    run_until(&mut app, &wakes, |world| {
        stats(world).is_some_and(|speed| speed.calls == 2)
    });
    app.update();
    let after_call = stats(app.world()).unwrap_or_default();
    assert_eq!(after_call.calls, 2);
    assert_eq!(after_call.output_tokens, 37 + estimate, "{after_call:?}");
    assert_eq!(item(app.world(), agent), shown);

    app.world_mut()
        .trigger(RunCommand::new(agent, "speed".to_owned()));
    app.update();
    let notices = &app.world().resource::<Collected<Notice>>().0;
    let listed = notices
        .iter()
        .any(|notice| notice.text.contains("deepseek-flash: 2 calls"));
    assert!(listed, "{notices:?}");
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(ThroughputPlugin);
}
