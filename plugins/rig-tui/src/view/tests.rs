use bevy_ecs::system::RunSystemOnce;
use rig_harness::harness_protocol::{Home, SessionDir};
use rig_harness::prelude::*;

use super::*;
use crate::render::{ShownAgent, lay_out_transcript};
use crate::tests::{scratch, view_app};
use crate::transcript::{Scroll, Transcript};

/// The shown agent's transcript, as text, a line per row.
fn transcript(app: &mut App, agent: Entity) -> Vec<String> {
    let world = app.world_mut();
    let mut agents = world.query::<(&AgentId, &Conversation)>();
    let Ok((id, conversation)) = agents.get(world, agent) else {
        return Vec::new();
    };
    let shown = ShownAgent {
        entity: agent,
        id,
        conversation,
        changed: true,
        inbox: &[],
    };
    let mut rows = Transcript::default();
    let view = world.resource::<TuiView>();
    let (renderers, calls) = (Default::default(), Default::default());
    let parts = lay_out_transcript(
        &mut rows,
        view,
        Some(&shown),
        (&renderers, &calls, None),
        80,
    );
    let (lines, _) = rows.visible(&parts, (80, 100), &mut Scroll::default());
    lines.iter().map(ToString::to_string).collect()
}

/// Where in `lines` the first that contains `text` is.
fn line_of(lines: &[String], text: &str) -> Option<usize> {
    lines.iter().position(|line| line.contains(text))
}

#[test]
fn a_notice_stays_before_a_prompt_that_joins_the_message_it_followed() {
    let (mut app, agent) = view_app();
    let note = |app: &mut App, text: &str| {
        let note = Deliver::new(agent, text, DeliveryMode::Note);
        app.world_mut().trigger(note);
        app.update();
    };
    note(&mut app, "first");
    app.world_mut()
        .write_message(Notice::info(agent, "Interrupted."));
    app.update();
    // Left unanswered, the first note takes in the second.
    note(&mut app, "second");
    let messages = app
        .world()
        .get::<Conversation>(agent)
        .map(|c| c.messages().len());
    assert_eq!(messages, Some(1));
    let lines = transcript(&mut app, agent);
    let order = ["first", "Interrupted.", "second"].map(|text| line_of(&lines, text));
    assert!(order.iter().all(Option::is_some), "{lines:#?}");
    assert!(order.is_sorted(), "{lines:#?}");
}

#[test]
fn notices_are_kept_for_the_next_build_of_the_session() {
    let home = Home::at(scratch("notices"));
    let Ok(id) = "1-1".parse() else { return };
    let session = home.session(&id);
    let build = |session: &SessionDir| {
        let (mut app, agent) = view_app();
        app.world_mut()
            .entity_mut(agent)
            .insert(AgentId("a1".to_owned()));
        app.insert_resource(SessionPaths(session.clone()));
        let loaded = app.world_mut().run_system_once(load_notices);
        assert!(loaded.is_ok());
        (app, agent)
    };
    let (mut app, agent) = build(&session);
    app.world_mut()
        .write_message(Notice::warn(agent, "Kept across a reload."));
    app.update();
    drop(app);
    // The view's file is no agent log: the session has none yet.
    assert!(!session.is_saved());
    let (mut app, agent) = build(&session);
    let lines = transcript(&mut app, agent);
    assert!(
        line_of(&lines, "Kept across a reload.").is_some(),
        "{lines:#?}"
    );
}
