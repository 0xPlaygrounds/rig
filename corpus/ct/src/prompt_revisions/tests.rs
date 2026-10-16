use std::sync::mpsc::Receiver;

use rig_core::test_utils::MockCompletionModel;
use rig_harness::ask::AskPlugin;
use rig_harness::harness_protocol::SessionId;
use rig_harness::rig_ecs::session_log::SessionLogPlugin;
use rig_harness_test_support::{
    AnswerAsks, Answers, Ended, app_in, connect, first_agent, log_ended, press, reply, run_until,
    scratch_home, tui_view,
};
use rig_tui::{MainPane, TranscriptItem, TranscriptView};

use super::*;

fn start(home: &Home, id: &SessionId, view: bool) -> (App, Receiver<()>, Entity) {
    let (mut app, wakes) = app_in(home, id);
    if view {
        tui_view(&mut app);
    }
    app.add_plugins((AgentPlugin, SessionLogPlugin, AskPlugin, AnswerAsks))
        .add_plugins(PromptRevisionsPlugin)
        .init_resource::<Ended>()
        .add_observer(log_ended);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    (app, wakes, agent)
}

fn turns(app: &App) -> usize {
    app.world().resource::<Ended>().0.len()
}

fn say(app: &mut App, wakes: &Receiver<()>, agent: Entity, text: &str) {
    let ended = turns(app);
    let typed = Deliver::new(agent, text, DeliveryMode::Queue).with_author(Author::user());
    app.world_mut().trigger(typed);
    run_until(app, wakes, |world| {
        world.resource::<Ended>().0.len() > ended
    });
}

/// Runs `/line`, then until `turns` turns ended.
fn run(app: &mut App, wakes: &Receiver<()>, agent: Entity, line: &str, turns: usize) {
    app.world_mut().trigger(RunCommand::new(agent, line));
    run_until(app, wakes, |world| {
        world.resource::<Ended>().0.len() >= turns
    });
    for _ in 0..3 {
        app.update();
    }
}

/// The active line's text.
fn active(app: &App, agent: Entity) -> Vec<String> {
    let Some(conversation) = app.world().get::<Conversation>(agent) else {
        return Vec::new();
    };
    let said = conversation.messages().iter().enumerate();
    let said = said.map(|(at, message)| match &**message {
        Message::Assistant(reply) => reply.text(),
        _ => typed(conversation, at).unwrap_or_default(),
    });
    said.collect()
}

/// Each user message's version count, and the badges shown.
fn versions(app: &App, agent: Entity) -> (Vec<usize>, Vec<String>) {
    let Some(conversation) = app.world().get::<Conversation>(agent) else {
        return Default::default();
    };
    let counts = slots(conversation)
        .iter()
        .map(|slot| slot.versions.len())
        .collect();
    let notes = conversation
        .asides()
        .filter_map(|(_, aside)| Note::of(aside));
    (counts, notes.map(|note| note.title).collect())
}

#[test]
fn revisions_switch_whole_continuations_nest_and_survive_a_restart() {
    let home = scratch_home("ct-prompt-revisions");
    let id = SessionId::generate();
    let model =
        MockCompletionModel::from_stream_turns([reply("A"), reply("B"), reply("C"), reply("D")]);
    let (mut app, wakes, agent) = start(&home, &id, false);
    if let Some(connection) = connect(&model) {
        app.world_mut().entity_mut(agent).insert(connection);
    }
    say(&mut app, &wakes, agent, "add logging");
    say(&mut app, &wakes, agent, "now tests");

    // Opened for editing, the message is answered with the new text.
    *app.world_mut().resource_mut::<Answers>() = Answers::new([Answer::text("add tracing")]);
    run(&mut app, &wakes, agent, "revise 1", 3);
    assert_eq!(active(&app, agent), ["add tracing", "C"]);
    assert_eq!(versions(&app, agent), (vec![2], vec!["‹ 2/2 ›".to_owned()]));

    // Switching calls no model.
    let asked = model.requests().len();
    run(&mut app, &wakes, agent, "version 1 1", 3);
    let first = ["add logging", "A", "now tests", "B"];
    assert_eq!(active(&app, agent), first);
    assert_eq!(model.requests().len(), asked);

    // A revision inside version 1 has versions of its own.
    run(&mut app, &wakes, agent, "revise 2 now benchmarks", 4);
    let nested = ["add logging", "A", "now benchmarks", "D"];
    assert_eq!(active(&app, agent), nested);
    let badges = vec!["‹ 1/2 ›".to_owned(), "‹ 2/2 ›".to_owned()];
    assert_eq!(versions(&app, agent), (vec![2, 2], badges.clone()));

    // The outer switch away and back keeps the inner choice.
    run(&mut app, &wakes, agent, "version 2 1", 4);
    assert_eq!(active(&app, agent), ["add tracing", "C"]);
    run(&mut app, &wakes, agent, "version 1", 4);
    assert_eq!(active(&app, agent), nested);
    assert_eq!(model.requests().len(), asked + 1);

    // Restarted, the same path, versions and state.
    drop(app);
    let (mut app, wakes, agent) = start(&home, &id, true);
    run_until(&mut app, &wakes, |world| {
        world
            .get::<Conversation>(agent)
            .is_some_and(|it| it.messages().len() == 4)
    });
    app.update();
    assert_eq!(active(&app, agent), nested);
    assert_eq!(versions(&app, agent), (vec![2, 2], badges));
    run(&mut app, &wakes, agent, "version 1 2", 0);
    assert_eq!(active(&app, agent), first);

    // Right on the selected first message shows its next version.
    let message = app
        .world()
        .get::<Conversation>(agent)
        .and_then(|it| it.id(0));
    let selected = message.map(TranscriptItem::Message);
    let world = app.world_mut();
    let mut main = world.query_filtered::<&mut TranscriptView, With<MainPane>>();
    let picked = main
        .single_mut(world)
        .map(|mut view| view.selected = selected);
    assert!(picked.is_ok(), "the default screen has a main pane");
    app.world_mut().insert_resource(ShownAgent(Some(agent)));
    press(&mut app, "right");
    for _ in 0..3 {
        app.update();
    }
    assert_eq!(active(&app, agent), ["add tracing", "C"]);
}
