use rig_core::completion::AssistantContent;
use rig_ecs::agent::TurnOutcome;
use rig_ecs::prelude::{Agent, AgentPlugin, Deliver, DeliveryMode, SessionLogPlugin};
use rig_harness_test_support::{Ended, MemoryStore, app_on, log_ended, run_until};

use super::*;

fn turns(json: serde_json::Value) -> Vec<Turn> {
    serde_json::from_value(json).expect("a script")
}

fn answer(outcome: &TurnOutcome) -> String {
    let TurnOutcome::Answered(Message::Assistant(message)) = outcome else {
        return format!("{outcome:?}");
    };
    message
        .content
        .iter()
        .filter_map(|block| match block {
            AssistantContent::Text(text) => Some(text.text.clone()),
            _ => None,
        })
        .collect()
}

#[test]
fn each_conversation_follows_its_own_turns_in_order() {
    let script = turns(serde_json::json!([
        {"reply": "one"},
        {"reply": "two"},
        {"when": "be the helper", "reasoning": "thinking", "reply": "helping"},
        {"when": "call a tool", "calls": [{"tool": "no_such_tool", "args": {"x": 1}}]},
        {"when": "call a tool", "reply": "called it"},
        {"when": "fail now", "fail": "scripted failure"},
    ]));
    let store = MemoryStore::default();
    let (mut app, wakes) = app_on(&store);
    app.add_plugins((
        AgentPlugin,
        SessionLogPlugin,
        ScriptedModelPlugin::with_script(script),
    ))
    .init_resource::<Ended>()
    .add_observer(log_ended);
    app.finish();
    app.update();
    let world = app.world_mut();
    let main = world.spawn(Agent).id();
    let helper = world.spawn(Agent).id();
    let tools = world.spawn(Agent).id();
    let fails = world.spawn(Agent).id();
    assert_eq!(
        world.get::<ModelChoice>(main).map(|m| m.0.as_str()),
        Some(MODEL),
        "an agent without a model gets the scripted one"
    );
    world.trigger(Deliver::new(main, "hello", DeliveryMode::Steer));
    world.trigger(Deliver::new(
        helper,
        "please be the helper",
        DeliveryMode::Steer,
    ));
    world.trigger(Deliver::new(tools, "call a tool now", DeliveryMode::Steer));
    world.trigger(Deliver::new(fails, "fail now", DeliveryMode::Steer));
    run_until(&mut app, &wakes, |world| {
        world.resource::<Ended>().0.len() == 4
    });
    app.world_mut()
        .trigger(Deliver::new(main, "and again", DeliveryMode::Steer));
    run_until(&mut app, &wakes, |world| {
        world.resource::<Ended>().0.len() == 5
    });
    let ended = &app.world().resource::<Ended>().0;
    let mut answers: Vec<String> = ended.iter().map(answer).collect();
    answers.sort();
    let failed = ended.iter().filter(
        |outcome| matches!(outcome, TurnOutcome::Failed(why) if why.contains("scripted failure")),
    );
    assert_eq!(failed.count(), 1, "{ended:?}");
    answers.retain(|answer| !answer.starts_with("Failed"));
    assert_eq!(answers, ["called it", "helping", "one", "two"]);
}
