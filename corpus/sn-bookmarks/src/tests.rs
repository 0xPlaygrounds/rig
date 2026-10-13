use std::fs;
use std::sync::mpsc::Receiver;

use rig_core::test_utils::MockCompletionModel;
use rig_harness::harness_protocol::SessionId;
use rig_harness_test_support::{
    Ended, app_in, connect, first_agent, log_ended, reply, run_until, scratch_home,
};

use super::*;

const AGREED: &str = "We agreed: read takes a path and returns bytes.";
const ROOT_CAUSE: &str = "The root cause is a stale cache entry.";

/// The kernel, the session service and bookmarks on session `id` of
/// `home`, after its first frame restored it, with its first agent: for a
/// new session, one of the id `new`, which every request carries.
fn start(home: &Home, id: &SessionId, new: Option<&str>) -> Option<(App, Receiver<()>, Entity)> {
    let (mut app, wakes) = app_in(home, id);
    if let Some(new) = new {
        app.world_mut()
            .spawn((Name::new("agent"), Agent, AgentId(new.to_owned())));
    }
    app.init_resource::<TranscriptView>()
        .init_resource::<Ended>()
        .add_observer(log_ended)
        .add_plugins((AgentPlugin, JournalPlugin, BookmarksPlugin));
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut())?;
    Some((app, wakes, agent))
}

/// Sends `line` as the user typed it: a command runs in that frame, a
/// message's turn is run to its end.
fn send(app: &mut App, wakes: &Receiver<()>, agent: Entity, line: &str) {
    let ended = app.world().resource::<Ended>().0.len();
    let typed = Deliver::new(agent, line, DeliveryMode::Queue).with_author(Author::user());
    app.world_mut().trigger(typed);
    if line.starts_with('/') {
        app.update();
    } else {
        run_until(app, wakes, |world| {
            world.resource::<Ended>().0.len() > ended
        });
    }
}

/// Three exchanges on a new session of `home`, with `/bookmark api-design`
/// after the first and `/bookmark` after the second when `marking`: the
/// app, its agent, the `/bookmarks` picker's items then, and every
/// request the model got.
fn three_exchanges(
    home: &Home,
    marking: bool,
) -> Option<(App, SessionId, Vec<PickItem>, Vec<String>)> {
    let id: SessionId = "1000-1".parse().ok()?;
    let model =
        MockCompletionModel::from_stream_turns([reply(AGREED), reply(ROOT_CAUSE), reply("Done.")]);
    let (mut app, wakes, agent) = start(home, &id, Some("main"))?;
    app.world_mut().entity_mut(agent).insert(connect(&model)?);
    send(&mut app, &wakes, agent, "How should read work?");
    if marking {
        send(&mut app, &wakes, agent, "/bookmark api-design");
    }
    send(&mut app, &wakes, agent, "Why did the test fail?");
    let mut items = Vec::new();
    if marking {
        send(&mut app, &wakes, agent, "/bookmark");
        items = take_picker(&mut app, &wakes, agent)?.1;
    }
    send(&mut app, &wakes, agent, "Fix it.");
    let requests = model.requests();
    let requests = requests
        .iter()
        .map(|request| serde_json::to_string(request).unwrap_or_default());
    Some((app, id, items, requests.collect()))
}

/// Runs `/bookmarks` and returns its picker and items.
fn take_picker(
    app: &mut App,
    wakes: &Receiver<()>,
    agent: Entity,
) -> Option<(Entity, Vec<PickItem>)> {
    send(app, wakes, agent, "/bookmarks");
    let world = app.world_mut();
    let mut pickers = world.query_filtered::<(Entity, &Picker), With<Choices>>();
    let (picker, items) = pickers
        .iter(world)
        .map(|(entity, picker)| (entity, picker.items.clone()))
        .next()?;
    Some((picker, items))
}

/// Opens bookmark `label` from a `/bookmarks` picker; the item the view
/// selected then.
fn open_bookmark(
    app: &mut App,
    wakes: &Receiver<()>,
    agent: Entity,
    label: &str,
) -> Option<TranscriptItem> {
    let (picker, items) = take_picker(app, wakes, agent)?;
    let chosen = items.iter().position(|item| item.label == label)?;
    app.world_mut().trigger(Picked {
        entity: picker,
        chosen: Some(Chosen::item(chosen)),
    });
    app.world_mut().despawn(picker);
    app.update();
    app.world().resource::<TranscriptView>().selected.clone()
}

fn labels(app: &App) -> Vec<String> {
    app.world()
        .resource::<Bookmarks>()
        .0
        .iter()
        .map(|bookmark| bookmark.label.clone())
        .collect()
}

/// Fails the test: the restored conversation is not two exchanges.
fn unreachable_ids(ids: &[Option<MessageId>]) {
    assert!(
        ids.iter().all(Option::is_some),
        "two exchanges restored: {ids:?}"
    );
}

fn ids(app: &mut App, agent: Entity) -> Vec<Option<MessageId>> {
    let conversation = app.world().get::<Conversation>(agent);
    conversation.map_or_else(Vec::new, |conversation| {
        (0..4).map(|at| conversation.id(at)).collect()
    })
}

/// The fork of a session in `home`, once its parent handed it the
/// bookmarks.
fn fork_of(home: &Home) -> Option<SessionId> {
    let sessions = fs::read_dir(home.sessions())
        .into_iter()
        .flatten()
        .flatten();
    sessions
        .filter_map(|entry| entry.file_name().to_str()?.parse::<SessionId>().ok())
        .find(|id| {
            let store = SessionStores::default().open(&home.session(id));
            store.is_ok_and(|store| store.read_file(HANDED).is_ok())
        })
}

/// Condenses exchange 1 of `agent` into a summary, as compaction does.
fn condense(In(agent): In<Entity>, mut agents: Query<&mut Conversation>) {
    let by = Author::plugin("summary");
    let summary = Entry::new(Message::user("Summary: read was designed.")).by(by.clone());
    if let Ok(mut conversation) = agents.get_mut(agent) {
        conversation.splice(0..2, [summary], by);
    }
}

/// The note `id` of `agent`'s conversation.
fn note(app: &App, agent: Entity, id: AsideId) -> Option<Note> {
    let conversation = app.world().get::<Conversation>(agent)?;
    Note::of(conversation.aside(id)?)
}

#[test]
fn bookmarks_list_open_survive_resume_fork_and_condensing_and_never_reach_the_model() {
    let home = scratch_home("sn-bookmarks-home");
    // The same session in another home, with no bookmarks: what the model
    // gets depends on the session's id.
    let plain_home = scratch_home("sn-bookmarks-plain");
    let marked = three_exchanges(&home, true);
    let plain = three_exchanges(&plain_home, false);
    assert!(
        marked.is_some() && plain.is_some(),
        "two sessions with an agent and a model"
    );
    let (Some((app, id, items, requests)), Some((plain_app, _, _, plain_requests))) =
        (marked, plain)
    else {
        return;
    };
    drop((app, plain_app));

    // `/bookmarks` listed both, with what each message begins with.
    let listed: Vec<(&str, bool)> = items
        .iter()
        .map(|item| {
            (
                item.label.as_str(),
                item.detail
                    .as_ref()
                    .is_some_and(|detail| detail.ends_with(AGREED)),
            )
        })
        .collect();
    assert_eq!(
        listed,
        [("api-design", true), ("The root cause is a", false)]
    );
    assert!(
        items
            .get(1)
            .and_then(|item| item.detail.as_ref())
            .is_some_and(|detail| detail.ends_with(ROOT_CAUSE))
    );
    // The model never saw them: request 3 is the plain run's, byte for byte.
    assert_eq!(requests.len(), 3);
    assert_eq!(requests.get(2), plain_requests.get(2));

    // Resumed, both are back, at exchange 1's and 2's answers, with their markers.
    let restored = start(&home, &id, None);
    assert!(restored.is_some(), "the session was restored");
    let Some((mut app, wakes, agent)) = restored else {
        return;
    };
    let ids = ids(&mut app, agent);
    let (Some(Some(answer_1)), Some(Some(question_2)), Some(Some(answer_2))) =
        (ids.get(1), ids.get(2), ids.get(3))
    else {
        unreachable_ids(&ids);
        return;
    };
    assert_eq!(labels(&app), ["api-design", "The root cause is a"]);
    let marks: Vec<MessageId> = app
        .world()
        .resource::<Bookmarks>()
        .0
        .iter()
        .map(|bookmark| bookmark.message)
        .collect();
    assert_eq!(marks, [*answer_1, *answer_2]);
    let conversation = app.world().get::<Conversation>(agent);
    let markers = conversation.into_iter().flat_map(|conversation| {
        let marker = Author::plugin(ORIGIN);
        let asides = conversation.asides();
        asides
            .filter(move |(_, aside)| aside.author == marker && Note::of(aside).is_some())
            .map(|(_, aside)| aside.after)
    });
    let mut drawn: Vec<_> = markers.collect();
    drawn.sort();
    assert_eq!(drawn, [Some(*answer_1), Some(*answer_2)]);
    // `api-design` opens exchange 1's answer.
    let opened = open_bookmark(&mut app, &wakes, agent, "api-design");
    assert_eq!(opened, Some(TranscriptItem::Message(*answer_1)));

    // Fork before exchange 2: the fork keeps only `api-design`.
    let agent_id = app.world().get::<AgentId>(agent).cloned();
    let before = agent_id.map(|agent_id| (agent_id, *question_2));
    app.world_mut().trigger(ForkSession {
        before,
        asker: None,
    });
    run_until(&mut app, &wakes, |_| fork_of(&home).is_some());
    let fork = fork_of(&home);
    assert!(fork.is_some(), "a fork with the bookmarks handed over");
    let Some(fork) = fork else {
        return;
    };

    // Condensed, exchange 1 still opens, showing the saved history.
    assert!(
        app.world_mut()
            .run_system_cached_with(condense, agent)
            .is_ok()
    );
    app.update();
    let opened = open_bookmark(&mut app, &wakes, agent, "api-design");
    let history = match opened {
        Some(TranscriptItem::Note(id)) => note(&app, agent, id),
        _ => None,
    };
    let text = history.map(|history| history.text).unwrap_or_default();
    assert!(text.contains(&format!("◆ model: {AGREED}")), "{text}");
    assert!(text.contains("user: How should read work?"), "{text}");
    drop(app);

    let forked = start(&home, &fork, None);
    assert!(forked.is_some(), "the fork was restored");
    let Some((app, _, _)) = forked else {
        return;
    };
    assert_eq!(labels(&app), ["api-design"]);
    drop(app);
    fs::remove_dir_all(home.root()).ok();
    fs::remove_dir_all(plain_home.root()).ok();
}
