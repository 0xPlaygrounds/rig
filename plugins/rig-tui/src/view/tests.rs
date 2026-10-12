use bevy_ecs::system::RunSystemOnce;
use rig_harness::harness_protocol::{Home, SessionDir};
use rig_harness::prelude::*;
use rig_harness::rig_ecs::journal::SessionStore;

use super::*;
use crate::tests::{scratch, transcript, view_app};

/// The store of `session`, as the harness opens it.
fn store(session: &SessionDir) -> Option<SessionStore> {
    SessionStores::default()
        .open(session)
        .ok()
        .map(SessionStore)
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
    let Some(store) = store(&session) else { return };
    let build = || {
        let (mut app, agent) = view_app();
        app.world_mut()
            .entity_mut(agent)
            .insert(AgentId("a1".to_owned()));
        app.insert_resource(store.clone());
        let loaded = app.world_mut().run_system_once(load_notices);
        assert!(loaded.is_ok());
        (app, agent)
    };
    let (mut app, agent) = build();
    app.world_mut()
        .write_message(Notice::warn(agent, "Kept across a reload."));
    app.update();
    drop(app);
    // The view's file is no agent log: the session is not saved yet.
    assert!(!session.is_saved());
    let (mut app, agent) = build();
    let lines = transcript(&mut app, agent);
    assert!(
        line_of(&lines, "Kept across a reload.").is_some(),
        "{lines:#?}"
    );
}

#[test]
fn what_was_typed_and_not_sent_is_kept_across_a_reload() {
    let home = Home::at(scratch("draft"));
    let Ok(id) = "1-1".parse() else { return };
    let session = home.session(&id);
    let Some(store) = store(&session) else { return };
    let (mut app, _) = view_app();
    app.insert_resource(store.clone())
        .insert_resource(ExitReason::Reload);
    app.world_mut().resource_mut::<TuiInput>().set("/tools");
    assert!(app.world_mut().run_system_once(keep_draft).is_ok());
    drop(app);
    let (mut app, _) = view_app();
    app.insert_resource(store.clone());
    assert!(app.world_mut().run_system_once(load_draft).is_ok());
    assert_eq!(app.world().resource::<TuiInput>().text(), "/tools");
    // Once: the build after this one starts empty.
    let (mut app, _) = view_app();
    app.insert_resource(store);
    assert!(app.world_mut().run_system_once(load_draft).is_ok());
    assert!(app.world().resource::<TuiInput>().is_empty());
}
