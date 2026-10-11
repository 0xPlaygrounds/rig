use rig_harness::rig_core::effect::EffectKind;
use rig_harness::rig_core::message::{ToolCall, ToolFunction, ToolName};

use super::*;

/// The agent's status after a frame.
fn status(app: &mut App, agent: Entity) -> Option<Status> {
    app.update();
    let activity = app.world().get::<Activity>(agent);
    activity.map(|activity| activity.status.clone())
}

#[test]
fn a_question_and_a_plugins_work_outside_the_turn_show() {
    let mut app = App::new();
    app.add_plugins((TaskPoolPlugin::default(), AgentPlugin, ActivityPlugin));
    let world = app.world_mut();
    let agent = world.spawn(Agent).id();
    let review = EffectRequest {
        kind: EffectKind::Custom {
            kind: "review".into(),
            payload: serde_json::Value::Null,
        },
        handler: None,
    };
    let review = world
        .spawn((review, ForAgent(agent), Name::new("reviewing")))
        .id();
    let reviewing = Some(Status::Busy("reviewing".to_owned()));
    assert_eq!(status(&mut app, agent), reviewing);

    let ask = Ask::yes_no("Keep the changes?");
    let ask = app.world_mut().spawn((ask, ForAgent(agent))).id();
    let asking = Some(Status::Asking("Keep the changes?".to_owned()));
    assert_eq!(status(&mut app, agent), asking);

    let yes = Answer::Choice("yes".to_owned()).outcome();
    app.world_mut().entity_mut(ask).insert(yes);
    assert_eq!(status(&mut app, agent), reviewing);
    app.world_mut()
        .entity_mut(review)
        .insert(Outcome::custom("done"));
    assert_eq!(status(&mut app, agent), Some(Status::Idle));
}

#[test]
fn a_held_tool_call_names_its_holds() {
    let mut app = App::new();
    app.add_plugins(ActivityPlugin);
    let world = app.world_mut();
    let agent = world.spawn(Agent).id();
    let turn = world.spawn(TurnOf(agent)).id();
    let Ok(name) = ToolName::new("edit") else {
        return;
    };
    let call = ToolCall::from_wire("c1", ToolFunction::new(name, serde_json::json!({})));
    let call = world.spawn((ToolRequest { call }, CallOf(turn))).id();
    world.spawn((Name::new("waiting for lease on src/a.rs"), HoldOf(call)));
    world.spawn(HoldOf(call));
    app.update();
    let activity = app.world().get::<Activity>(agent);
    let held_by = activity
        .and_then(|activity| activity.tools.first())
        .map(|tool| &tool.held_by);
    let names = ["waiting for lease on src/a.rs", UNNAMED_HOLD].map(str::to_owned);
    assert_eq!(held_by, Some(&names.to_vec()));
}
