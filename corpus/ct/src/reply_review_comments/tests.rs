use std::sync::mpsc::Receiver;

use rig_core::test_utils::MockCompletionModel;
use rig_harness::harness_protocol::SessionId;
use rig_harness::rig_ecs::session_log::SessionLogPlugin;
use rig_harness_test_support::{
    Ended, app_in, connect, first_agent, log_ended, reply, run_until, scratch_home,
};

use super::*;

const PLAN: &str = "Plan:\nUse a global mutex for the cache.\nShip it.";
const NEXT: &str = "Next:\nThen delete the old config loader.";

/// The test's removal command: takes the agent's first reply out of its
/// conversation.
fn forget(In(args): In<CommandArgs>, mut agents: Query<&mut Conversation>) {
    let Ok(mut conversation) = agents.get_mut(args.agent) else {
        return;
    };
    let first = conversation
        .entries()
        .find(|entry| matches!(entry.message, Message::Assistant(_)))
        .map(|entry| entry.index);
    if let Some(at) = first {
        conversation.splice(at..=at, [], Author::plugin("forget"));
    }
}

fn comments_of(app: &App, agent: Entity) -> Vec<Comment> {
    let comments = app
        .world()
        .get::<Conversation>(agent)
        .map(|it| Comment::of(it));
    comments
        .unwrap_or_default()
        .into_iter()
        .map(|(_, comment)| comment)
        .collect()
}

fn drafts(app: &App, agent: Entity) -> usize {
    comments_of(app, agent)
        .iter()
        .filter(|comment| !comment.sent)
        .count()
}

/// The open picker whose title starts with `title`.
fn picker(app: &mut App, wakes: &Receiver<()>, title: &str) -> Option<(Entity, Picker)> {
    let open = |world: &World| {
        world
            .iter_entities()
            .filter_map(|entity| Some((entity.id(), entity.get::<Picker>()?.clone())))
            .find(|(_, picker)| picker.title.starts_with(title))
    };
    run_until(app, wakes, |world| open(world).is_some());
    open(app.world())
}

/// Types `text` to `agent`, as the user.
fn say(app: &mut App, agent: Entity, text: &str) {
    app.world_mut()
        .trigger(Deliver::new(agent, text, DeliveryMode::Steer));
}

/// `/comment`, picking the reply whose label or detail has `which`, then
/// `passage`, then typing `text`.
fn comment_on(
    app: &mut App,
    wakes: &Receiver<()>,
    agent: Entity,
    (which, passage, text): (&str, Chosen, &str),
) {
    let before = drafts(app, agent);
    app.world_mut().trigger(RunCommand::new(agent, "comment"));
    let Some((replies, list)) = picker(app, wakes, "Comment on which reply") else {
        return;
    };
    let item = list.items.iter().position(|item| {
        item.label.contains(which) || item.detail.as_deref().is_some_and(|it| it.contains(which))
    });
    app.world_mut()
        .trigger(Pick::person(replies, item.map(Chosen::item)));
    let Some((lines, _)) = picker(app, wakes, "Comment on turn") else {
        return;
    };
    app.world_mut().trigger(Pick::person(lines, Some(passage)));
    run_until(app, wakes, |world| {
        world.iter_entities().any(|entity| entity.contains::<Ask>())
    });
    say(app, agent, text);
    run_until(app, wakes, |world| {
        world
            .get::<Conversation>(agent)
            .is_some_and(|it| Comment::drafts(it).len() > before)
    });
}

/// The texts of the last user message `model` was sent.
fn last_user_message(model: &MockCompletionModel) -> Vec<String> {
    let requests = model.requests();
    let history = requests.last().map(|request| request.chat_history.clone());
    let last = history
        .unwrap_or_default()
        .into_iter()
        .rev()
        .find_map(|message| match std::sync::Arc::unwrap_or_clone(message) {
            Message::User { content } => Some(content),
            _ => None,
        });
    let texts = last
        .unwrap_or_default()
        .into_iter()
        .filter_map(|item| match item {
            UserContent::Text(text) => Some(text.text),
            _ => None,
        });
    texts.collect()
}

#[test]
fn comments_go_with_the_next_message() {
    let model = MockCompletionModel::from_stream_turns([
        reply(PLAN),
        reply(NEXT),
        reply("Revised."),
        reply("Noted."),
    ]);
    let home = scratch_home("ct-reply-review-comments");
    let (mut app, wakes) = app_in(&home, &SessionId::generate());
    app.add_plugins((AgentPlugin, SessionLogPlugin, ReviewCommentsPlugin))
        .add_command("forget", "Take the first reply out", forget)
        .init_resource::<Ended>()
        .add_observer(log_ended);
    app.world_mut().spawn(Answerer::default());
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    let connection = connect(&model);
    assert!(connection.is_some());
    if let Some(connection) = connection {
        app.world_mut().entity_mut(agent).insert(connection);
    }
    let turns = |app: &App| app.world().resource::<Ended>().0.len();

    // Two turns: the plan, then the next step.
    for (done, text) in (1..).zip(["Plan the cache.", "Go on."]) {
        say(&mut app, agent, text);
        run_until(&mut app, &wakes, |world| {
            world.resource::<Ended>().0.len() == done
        });
    }

    // A comment on a line ticked, and one on a passage typed.
    let mutex = Chosen::item(1);
    comment_on(
        &mut app,
        &wakes,
        agent,
        ("Turn 1", mutex, "A global mutex serializes every read."),
    );
    let loader = Chosen::text("Then delete the old config loader.");
    comment_on(
        &mut app,
        &wakes,
        agent,
        ("Turn 2", loader, "Other crates still use it."),
    );
    assert_eq!(drafts(&app, agent), 2);

    // `/comments` lists both, to edit or delete.
    app.world_mut().trigger(RunCommand::new(agent, "comments"));
    let listed = picker(&mut app, &wakes, "Draft comments");
    let edits = listed.as_ref().map(|(_, it)| {
        it.items
            .iter()
            .filter(|item| item.label.starts_with("Edit"))
            .count()
    });
    assert_eq!(edits, Some(2));
    if let Some((list, _)) = listed {
        app.world_mut().trigger(Pick::person(list, None));
    }

    // The next message carries both, before its text.
    say(&mut app, agent, "please revise");
    run_until(&mut app, &wakes, |world| {
        world.resource::<Ended>().0.len() == 3
    });
    let sent = last_user_message(&model);
    let (block, typed) = (
        sent.first().cloned().unwrap_or_default(),
        sent.last().cloned(),
    );
    for expected in [
        "> Use a global mutex for the cache.",
        "turn 1, line 2",
        "A global mutex serializes every read.",
        "> Then delete the old config loader.",
        "turn 2",
        "Other crates still use it.",
    ] {
        assert!(block.contains(expected), "{expected} in {block}");
    }
    assert!(!block.contains("no longer in context"), "{block}");
    assert_eq!(typed.as_deref(), Some("please revise"));
    assert_eq!(drafts(&app, agent), 0);
    let comments = comments_of(&app, agent);
    assert!(comments.len() == 2 && comments.iter().all(|comment| comment.sent));
    let conversation = app.world().get::<Conversation>(agent);
    let notes = conversation.map(|conversation| {
        let notes = conversation
            .asides()
            .filter_map(|(_, aside)| Note::of(aside));
        notes
            .filter(|note| note.title.starts_with("Comment (sent)"))
            .count()
    });
    assert_eq!(notes, Some(2));

    // The plan taken out of context, a comment on it from its log.
    app.world_mut().trigger(RunCommand::new(agent, "forget"));
    app.update();
    app.update();
    let ship = Chosen::text("Ship it.");
    comment_on(
        &mut app,
        &wakes,
        agent,
        ("no longer in context", ship, "Not before the tests pass."),
    );
    assert_eq!(drafts(&app, agent), 1);
    app.world_mut()
        .trigger(RunCommand::new(agent, "comments send"));
    run_until(&mut app, &wakes, |world| {
        world.resource::<Ended>().0.len() == 4
    });
    let block = last_user_message(&model).join("\n");
    assert!(
        block.contains("> Ship it.") && block.contains("Not before the tests pass."),
        "{block}"
    );
    assert!(block.contains("no longer in context"), "{block}");
    assert_eq!((drafts(&app, agent), turns(&app)), (0, 4));
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(ReviewCommentsPlugin);
}
