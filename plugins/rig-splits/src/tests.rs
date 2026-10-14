use std::sync::mpsc::Receiver;

use rig_core::completion::Message;
use rig_core::test_utils::MockCompletionModel;
use rig_harness::commands::RunCommand;
use rig_harness::rig_memory::{DirStore, load, save};
use rig_harness_test_support::{
    Collected, Ended, app_in, collect, commit_message, connect, first_agent, log_ended, reply,
    run_until, scratch_home,
};

use super::*;

/// The sessions the app asked the launcher to run next.
#[derive(Resource, Default)]
struct Switched(Vec<SessionId>);

fn keep_switch(exit: On<Exit>, mut switched: ResMut<Switched>) {
    if let ExitReason::Switch(Some(to)) = &exit.0 {
        switched.0.push(to.clone());
    }
}

/// The app of session `id` of `home` with the split plugin, its agent
/// answered by `model`, and its loop's wakes.
fn start(
    home: &Home,
    id: &SessionId,
    model: &MockCompletionModel,
) -> Option<(App, Receiver<()>, Entity)> {
    let (mut app, wakes) = app_in(home, id);
    app.add_plugins((AgentPlugin, JournalPlugin, SplitsPlugin))
        .init_resource::<Switched>()
        .init_resource::<Ended>()
        .add_observer(keep_switch)
        .add_observer(log_ended);
    collect::<Notice>(&mut app);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut())?;
    let connection = connect(model)?;
    app.world_mut().entity_mut(agent).insert(connection);
    Some((app, wakes, agent))
}

/// Runs `/split <line>` for `agent`, and the frame after it.
fn split(app: &mut App, agent: Entity, line: &str) {
    app.world_mut()
        .trigger(RunCommand::new(agent, format!("split {line}")));
    app.update();
    app.update();
}

/// The ids of the active line of `agent`.
fn ids(app: &App, agent: Entity) -> Vec<MessageId> {
    let conversation = app.world().get::<Conversation>(agent);
    let entries = conversation
        .into_iter()
        .flat_map(|conversation| conversation.entries());
    entries.map(|entry| entry.id).collect()
}

/// The split pointers of the active line of `agent`.
fn pointers(app: &App, agent: Entity) -> Vec<SplitPointer> {
    let conversation = app.world().get::<Conversation>(agent);
    let asides = conversation
        .into_iter()
        .flat_map(|conversation| conversation.asides());
    asides
        .filter(|(_, aside)| aside.kind == SPLIT)
        .filter_map(|(_, aside)| serde_json::from_value(aside.value.clone()).ok())
        .collect()
}

/// What the error notices said.
fn errors(app: &App) -> Vec<String> {
    let notices = app.world().resource::<Collected<Notice>>();
    let errors = notices
        .0
        .iter()
        .filter(|notice| notice.level == NoticeLevel::Error);
    errors.map(|notice| notice.text.clone()).collect()
}

/// A conversation split at an assistant reply goes on in the split; the
/// split splits again; a range moved into a new session leaves a pointer;
/// switching goes from line to line; merged back, the moved range is where
/// it was with its ids, what the split added after it follows with new
/// ones, and a kept branch's entries come at the end. A split into a new
/// session runs it next.
#[test]
fn splits_split_again_move_switch_and_merge_back() {
    let home = scratch_home("rig-splits-plugin");
    let Ok(id) = "1700000000-40".parse::<SessionId>() else {
        return;
    };
    let model = MockCompletionModel::from_stream_turns([reply("Red.")]);
    let Some((mut app, wakes, agent)) = start(&home, &id, &model) else {
        return;
    };
    app.world_mut()
        .trigger(Deliver::new(agent, "Name a colour.", DeliveryMode::Steer));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });
    for message in [Message::user("Another?"), Message::assistant("Green.")] {
        assert_eq!(commit_message(app.world_mut(), agent, message), Ok(()));
    }
    app.update();
    let first = ids(&app, agent);
    assert_eq!(first.len(), 4, "{first:?}");
    let red = first.get(1).copied().unwrap_or_default();

    // At the assistant's reply: the rest is kept as `main`.
    split(&mut app, agent, &format!("--at {red} blue"));
    assert_eq!(errors(&app), Vec::<String>::new());
    assert_eq!(ids(&app, agent), first.get(..2).unwrap_or_default());
    let title = |app: &App| {
        let conversation = app.world().get::<Conversation>(agent);
        conversation.map(|conversation| conversation.branch_title().to_owned())
    };
    assert_eq!(title(&app).as_deref(), Some("blue"));
    for message in [Message::user("Blue?"), Message::assistant("Blue.")] {
        assert_eq!(commit_message(app.world_mut(), agent, message), Ok(()));
    }
    // The split splits again, at its newest entry.
    split(&mut app, agent, "nested");
    for message in [Message::user("Why?"), Message::assistant("Because.")] {
        assert_eq!(commit_message(app.world_mut(), agent, message), Ok(()));
    }
    app.update();
    let nested = ids(&app, agent);
    let listed: Vec<String> = app
        .world()
        .get::<Conversation>(agent)
        .map(|conversation| {
            lines(conversation)
                .into_iter()
                .map(|line| line.title)
                .collect()
        })
        .unwrap_or_default();
    assert_eq!(listed, ["nested", "main", "blue"]);

    // The last exchange moves into a new session, a pointer in its place.
    let (Some(why), Some(because)) = (nested.get(4), nested.get(5)) else {
        return;
    };
    split(&mut app, agent, &format!("move {why} {because} new why"));
    assert_eq!(errors(&app), Vec::<String>::new());
    assert_eq!(ids(&app, agent), nested.get(..4).unwrap_or_default());
    let moved = pointers(&app, agent);
    assert_eq!(
        moved
            .iter()
            .map(|pointer| (pointer.moved, pointer.merged))
            .collect::<Vec<_>>(),
        [(2, false)]
    );
    let Some(away) = moved
        .first()
        .and_then(|pointer| pointer.session.parse::<SessionId>().ok())
    else {
        return;
    };
    assert!(
        app.world().resource::<Switched>().0.is_empty(),
        "a move stays"
    );
    // The split goes on there.
    let store = DirStore::open(home.session(&away).path()).ok();
    let mut there = store
        .as_ref()
        .and_then(|store| load(store, &agent_name(&app, agent)).ok());
    if let Some(there) = there.as_mut() {
        there.push(Message::user("And then?"));
    }
    let saved = store
        .as_ref()
        .zip(there.as_mut())
        .map(|(store, there)| save(store, &agent_name(&app, agent), there).is_ok());
    assert_eq!(saved, Some(true));

    // Switching to `blue` and back to `nested`.
    split(&mut app, agent, "switch blue");
    assert_eq!(title(&app).as_deref(), Some("blue"));
    split(&mut app, agent, "switch nested");
    assert_eq!(title(&app).as_deref(), Some("nested"));
    assert_eq!(errors(&app), Vec::<String>::new());

    // Merged back: the moved range where it was, then what the split added.
    split(&mut app, agent, &format!("merge {away}"));
    assert_eq!(errors(&app), Vec::<String>::new());
    let back = ids(&app, agent);
    assert_eq!(back.get(..6), Some(nested.as_slice()));
    assert_eq!(back.len(), 7);
    let last = app
        .world()
        .get::<Conversation>(agent)
        .and_then(|conversation| conversation.messages().last().cloned());
    assert_eq!(
        last,
        Some(Message::user("And then?")),
        "what the split added follows"
    );
    let merged = pointers(&app, agent);
    assert_eq!(
        merged
            .iter()
            .map(|pointer| pointer.merged)
            .collect::<Vec<_>>(),
        [true]
    );

    // A kept branch merged: its own entries at the end, with their ids.
    split(&mut app, agent, "merge main");
    let all = ids(&app, agent);
    assert_eq!(all.get(7..), first.get(2..));

    // A split into a new session runs it next.
    split(&mut app, agent, &format!("new --at {red} fresh"));
    assert_eq!(errors(&app), Vec::<String>::new());
    let switched = app.world().resource::<Switched>().0.clone();
    assert_eq!(switched.len(), 1);
    let copied = switched.first().and_then(|to| {
        let store = DirStore::open(home.session(to).path()).ok()?;
        load(&store, &agent_name(&app, agent)).ok()
    });
    let copied: Vec<MessageId> = copied
        .iter()
        .flat_map(|copied| copied.entries().map(|entry| entry.id))
        .collect();
    assert_eq!(copied, first.get(..2).unwrap_or_default());
    std::fs::remove_dir_all(home.root()).ok();
}

/// The id of `agent`, its log's name.
fn agent_name(app: &App, agent: Entity) -> String {
    app.world()
        .get::<AgentId>(agent)
        .map(|id| id.0.clone())
        .unwrap_or_default()
}

#[test]
fn the_grammar_reads_where_new_and_the_title() {
    let asked = parse("move m4 m9 new --at m2 --from 1-2 side quest");
    let (Ok(first), Ok(last), Ok(at)) = ("m4".parse(), "m9".parse(), "m2".parse()) else {
        return;
    };
    assert_eq!(
        asked,
        Ok(Asked {
            action: Action::Move {
                first,
                last,
                new: true
            },
            place: Where {
                from: Some("1-2".to_owned()),
                at: Some(at),
                ..Where::default()
            },
            title: "side quest".to_owned(),
        })
    );
    assert!(parse("move m4").is_err());
    assert!(parse("--at x").is_err());
    assert_eq!(
        parse("switch a3").map(|asked| asked.action),
        Ok(Action::Switch("a3".to_owned()))
    );
}
