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
    // The terminal view's main pane, its view alone.
    app.world_mut().spawn((TranscriptView::default(), MainPane));
    app.init_resource::<Ended>()
        .add_observer(log_ended)
        .add_plugins((AgentPlugin, SessionLogPlugin, BookmarksPlugin));
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
    let world = app.world_mut();
    let mut panes = world.query_filtered::<&TranscriptView, With<MainPane>>();
    panes
        .single(world)
        .ok()
        .and_then(|view| view.selected.clone())
}

/// Every bookmark's label and message, oldest first.
fn marks(app: &mut App) -> Vec<(String, MessageId)> {
    let world = app.world_mut();
    let mut agents = world.query::<(Entity, &Conversation)>();
    let found = all(agents
        .iter(world)
        .map(|(agent, conversation)| (agent, &conversation.0)));
    found
        .into_iter()
        .map(|(_, _, message, bookmark)| (bookmark.label, message))
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

/// Condenses exchange 1 of `agent` into a summary, as compaction does.
fn condense(In(agent): In<Entity>, mut agents: Query<&mut Conversation>) {
    let by = Author::plugin("summary");
    let summary = Entry::new(Message::user("Summary: read was designed.")).by(by.clone());
    if let Ok(mut conversation) = agents.get_mut(agent) {
        conversation.splice(0..2, [summary], by);
    }
}

#[test]
fn bookmarks_list_open_survive_resume_and_condensing_and_never_reach_the_model() {
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

    // Resumed, both are back, at exchange 1's and 2's answers.
    let restored = start(&home, &id, None);
    assert!(restored.is_some(), "the session was restored");
    let Some((mut app, wakes, agent)) = restored else {
        return;
    };
    let ids = ids(&mut app, agent);
    let (Some(Some(answer_1)), Some(Some(answer_2))) = (ids.get(1), ids.get(3)) else {
        unreachable_ids(&ids);
        return;
    };
    assert_eq!(
        marks(&mut app),
        [
            ("api-design".to_owned(), *answer_1),
            ("The root cause is a".to_owned(), *answer_2)
        ]
    );
    // `api-design` opens exchange 1's answer.
    let opened = open_bookmark(&mut app, &wakes, agent, "api-design");
    assert_eq!(opened, Some(TranscriptItem::Message(*answer_1)));

    // Condensed, exchange 1 still opens, in its saved history.
    assert!(
        app.world_mut()
            .run_system_cached_with(condense, agent)
            .is_ok()
    );
    app.update();
    open_bookmark(&mut app, &wakes, agent, "api-design");
    let world = app.world_mut();
    let mut panes = world.query_filtered::<(&TranscriptPane, &TranscriptView), With<HistoryPane>>();
    let shown = panes.single(world).ok().and_then(|(pane, view)| {
        let TranscriptSource::Value(then) = &pane.source else {
            return None;
        };
        let at = then.index_of(*answer_1)?;
        let text = text_of(then.messages().get(at)?);
        Some((text, view.selected.clone()))
    });
    assert_eq!(
        shown,
        Some((AGREED.to_owned(), Some(TranscriptItem::Message(*answer_1))))
    );
    drop(app);

    fs::remove_dir_all(home.root()).ok();
    fs::remove_dir_all(plain_home.root()).ok();
}
