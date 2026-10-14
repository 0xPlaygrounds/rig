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

    let yes = Answer::choice("yes").outcome();
    app.world_mut().entity_mut(ask).insert(yes);
    assert_eq!(status(&mut app, agent), reviewing);
    app.world_mut()
        .entity_mut(review)
        .insert(Outcome::custom("done"));
    assert_eq!(status(&mut app, agent), Some(Status::Idle));
}

#[test]
fn a_tool_calls_activity_names_its_holds_and_its_progress() {
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

    let progress = ToolProgress("12 of 40 files".to_owned());
    app.world_mut().entity_mut(call).insert(progress);
    app.update();
    let activity = app.world().get::<Activity>(agent);
    let progress = activity
        .and_then(|activity| activity.tools.first())
        .and_then(|tool| tool.progress.as_deref());
    assert_eq!(progress, Some("12 of 40 files"));
}

#[test]
fn an_inline_question_leaves_a_running_turn_its_status() {
    let mut app = App::new();
    app.add_plugins((TaskPoolPlugin::default(), AgentPlugin, ActivityPlugin));
    let world = app.world_mut();
    let agent = world.spawn(Agent).id();
    let ask = Ask::yes_no("Try the next step?").inline();
    world.spawn((ask, ForAgent(agent)));
    let asking = Some(Status::Asking("Try the next step?".to_owned()));
    assert_eq!(status(&mut app, agent), asking);

    app.world_mut().spawn(TurnOf(agent));
    assert_eq!(status(&mut app, agent), Some(Status::Thinking));
}

#[test]
fn a_session_rewrite_reaches_the_message_feed() {
    let mut app = App::new();
    app.add_plugins((TaskPoolPlugin::default(), AgentPlugin, ActivityPlugin));
    let agent = app.world_mut().spawn(Agent).id();
    app.world_mut()
        .resource_mut::<MessageFeed>()
        .entries
        .push_back(FedMessage {
            to: agent,
            from: None,
            author: Author::user(),
            text: "my key is sk-9".to_owned(),
        });
    let redact = Strings(|text: &mut String| {
        let hit = text.contains("sk-9");
        if hit {
            *text = text.replace("sk-9", "[REDACTED]");
        }
        hit
    });
    app.world_mut().write_message(SessionRewritten {
        by: Author::plugin("redact"),
        asker: None,
        with: std::sync::Arc::new(redact),
        result: Ok(Rewritten::default()),
    });
    app.update();
    let feed = app.world().resource::<MessageFeed>();
    let texts: Vec<&str> = feed.iter().map(|fed| fed.text.as_str()).collect();
    assert_eq!(texts, ["my key is [REDACTED]"]);
}
