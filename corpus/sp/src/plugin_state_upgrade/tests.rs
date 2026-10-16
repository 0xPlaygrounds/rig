use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::mpsc::Receiver;

use rig_core::test_utils::MockCompletionModel;
use rig_harness::harness_protocol::{Home, SessionId};
use rig_harness::rig_memory::DirStore;
use rig_harness_test_support::{
    Collected, Ended, app_in, collect, connect, first_agent, log_ended, reply, run_until,
    scratch_home,
};
use rig_tui::TuiTestScreen;
use serde_json::{Value, json};

use super::v1::WorklistV1Plugin;
use super::v2::WorklistV2Plugin;
use super::*;

/// The kind of the asides a followed value is saved in.
const FOLLOWS: &str = "rig.follows";

/// Which worklist an app runs, if any.
#[derive(Clone, Copy)]
enum Version {
    None,
    V1,
    V2,
}

/// The kernel, the session service and the terminal view on a test screen
/// on session `id` of `home`, with `version` of the worklist and a scripted
/// model, after its first frames; with its main agent.
fn start(home: &Home, id: &SessionId, version: Version) -> (App, Receiver<()>, Entity) {
    let (mut app, wakes) = app_in(home, id);
    app.insert_resource(TuiTestScreen::new(100, 20));
    rig_harness_test_support::tui_view(&mut app);
    app.add_plugins((AgentPlugin, SessionLogPlugin))
        .init_resource::<Ended>()
        .add_observer(log_ended);
    collect::<Notice>(&mut app);
    match version {
        Version::None => {}
        Version::V1 => {
            app.add_plugins(WorklistV1Plugin);
        }
        Version::V2 => {
            app.add_plugins(WorklistV2Plugin);
        }
    }
    app.finish();
    app.cleanup();
    for _ in 0..3 {
        app.update();
    }
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    let model = MockCompletionModel::from_stream_turns([reply("On it.")]);
    if let Some(connection) = connect(&model) {
        app.world_mut().entity_mut(agent).insert(connection);
    }
    (app, wakes, agent)
}

/// Runs the command `line` (without its `/`), and a few frames.
fn run(app: &mut App, agent: Entity, line: &str) {
    app.world_mut().trigger(RunCommand::new(agent, line));
    for _ in 0..3 {
        app.update();
    }
}

/// Sends `text` as the user and runs the scripted turn to its end.
fn say(app: &mut App, wakes: &Receiver<()>, agent: Entity, text: &str) {
    let ended = app.world().resource::<Ended>().0.len();
    let typed = Deliver::new(agent, text, DeliveryMode::Queue).with_author(Author::user());
    app.world_mut().trigger(typed);
    run_until(app, wakes, |world| {
        world.resource::<Ended>().0.len() > ended
    });
}

/// The worklist panel's rows, as the test screen drew them.
fn panel(app: &App) -> Vec<String> {
    let rows = app.world().resource::<TuiTestScreen>().rows();
    rows.iter()
        .filter_map(|row| row.find('│').map(|at| row.get(at..).unwrap_or_default()))
        .map(|row| {
            row.trim_matches(|c: char| c == '│' || c.is_whitespace())
                .to_owned()
        })
        .filter(|row| row.starts_with(|c: char| c.is_ascii_digit()))
        .collect()
}

/// Whether the worklist panel says it is read-only.
fn read_only(app: &App) -> bool {
    let rows = app.world().resource::<TuiTestScreen>().rows();
    rows.iter().any(|row| row.contains("worklist (read-only)"))
}

/// The worklist warnings written.
fn warnings(app: &App) -> Vec<String> {
    let notices = &app.world().resource::<Collected<Notice>>().0;
    let warned = notices
        .iter()
        .filter(|notice| notice.level == NoticeLevel::Warn);
    warned
        .map(|notice| notice.text.clone())
        .filter(|text| text.starts_with("worklist"))
        .collect()
}

/// Every file of the session, by path, with its bytes.
fn files(home: &Home, id: &SessionId) -> BTreeMap<PathBuf, Vec<u8>> {
    fn walk(dir: &Path, into: &mut BTreeMap<PathBuf, Vec<u8>>) {
        for entry in fs::read_dir(dir).into_iter().flatten().flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, into);
            } else if let Ok(bytes) = fs::read(&path) {
                into.insert(path, bytes);
            }
        }
    }
    let mut files = BTreeMap::new();
    walk(home.session(id).path(), &mut files);
    files
}

/// Each saved worklist value (`{"v": n, "value": …}`) in the session's
/// main conversation, in the order kept.
fn saved(home: &Home, id: &SessionId) -> Vec<Value> {
    let Ok(store) = DirStore::open(home.session(id).path()) else {
        return Vec::new();
    };
    let Ok(session) = Session::read(&store) else {
        return Vec::new();
    };
    let Some(agent) = session.agents.first() else {
        return Vec::new();
    };
    let asides = agent.conversation.asides().filter_map(|(_, aside)| {
        let ours = aside.kind == FOLLOWS && aside.value.get("key") == Some(&json!(KEY));
        ours.then(|| json!({ "v": aside.value.get("v"), "value": aside.value.get("value") }))
    });
    asides.collect()
}

#[test]
fn an_upgrade_reads_old_lists_and_a_downgrade_leaves_new_ones_alone() {
    let home = scratch_home("sp-plugin-state-upgrade");
    let original = SessionId::generate();

    // Version 1: two items, a turn, then the first item done.
    let (mut app, wakes, agent) = start(&home, &original, Version::V1);
    run(&mut app, agent, "work add a");
    run(&mut app, agent, "work add b");
    say(&mut app, &wakes, agent, "Let's go.");
    run(&mut app, agent, "work done 1");
    assert_eq!(panel(&app), ["1. [x] a", "2. b"]);
    drop(app);
    assert_eq!(
        saved(&home, &original).last(),
        Some(&json!({ "v": 1, "value": ["[x] a", "b"] }))
    );

    // Version 2 reads it, and writes nothing on open.
    let before = files(&home, &original);
    let (app, _, _) = start(&home, &original, Version::V2);
    assert_eq!(panel(&app), ["1. [x] a (turn ?)", "2. [ ] b (turn ?)"]);
    assert!(warnings(&app).is_empty(), "{:?}", warnings(&app));
    drop(app);
    assert_eq!(
        files(&home, &original),
        before,
        "the session as version 1 left it"
    );

    // Its first change writes the new shape.
    let (mut app, _, agent) = start(&home, &original, Version::V2);
    run(&mut app, agent, "work doing 2");
    assert_eq!(panel(&app), ["1. [x] a (turn ?)", "2. [~] b (turn ?)"]);
    drop(app);
    let items = json!([
        { "id": 1, "text": "a", "status": "done", "added_turn": null },
        { "id": 2, "text": "b", "status": "doing", "added_turn": null },
    ]);
    assert_eq!(
        saved(&home, &original).last(),
        Some(&json!({ "v": 2, "value": items }))
    );

    // Version 1 shows the newer list read-only and changes nothing.
    let before = files(&home, &original);
    let (mut app, _, agent) = start(&home, &original, Version::V1);
    assert!(read_only(&app), "a read-only panel");
    assert!(panel(&app).is_empty(), "{:?}", panel(&app));
    let warned = warnings(&app);
    assert!(
        warned.len() == 1
            && warned
                .iter()
                .all(|it| it.contains("saved by a newer worklist")),
        "{warned:?}"
    );
    run(&mut app, agent, "work add x");
    assert!(panel(&app).is_empty(), "{:?}", panel(&app));
    drop(app);
    assert_eq!(files(&home, &original), before, "version 1 wrote nothing");

    // Version 2 again: everything as it left it.
    let (app, _, _) = start(&home, &original, Version::V2);
    assert_eq!(panel(&app), ["1. [x] a (turn ?)", "2. [~] b (turn ?)"]);
    drop(app);
    fs::remove_dir_all(home.root()).ok();
}

#[test]
fn an_unreadable_list_starts_empty_and_stays_saved() {
    let home = scratch_home("sp-plugin-state-upgrade-broken");

    // A session that never had the worklist: an empty list, no warning.
    let plain = SessionId::generate();
    let (mut app, wakes, agent) = start(&home, &plain, Version::None);
    say(&mut app, &wakes, agent, "Hello.");
    drop(app);
    let (mut app, _, agent) = start(&home, &plain, Version::V2);
    assert!(panel(&app).is_empty(), "{:?}", panel(&app));
    assert!(warnings(&app).is_empty(), "{:?}", warnings(&app));
    run(&mut app, agent, "work add c");
    assert_eq!(panel(&app), ["1. [ ] c (turn 1)"]);
    drop(app);

    // A session whose worklist state is `{"weird": true}`.
    let broken = SessionId::generate();
    let (mut app, wakes, agent) = start(&home, &broken, Version::None);
    say(&mut app, &wakes, agent, "Hello.");
    let world = app.world_mut();
    if let Some(mut conversation) = world.get_mut::<Conversation>(agent) {
        let last = conversation.entries().next_back().map(|entry| entry.id);
        let value = json!({ "key": KEY, "v": 2, "value": { "weird": true } });
        conversation.add_aside(Aside::new(FOLLOWS, Author::rig(), value).after(last));
    }
    app.update();
    drop(app);

    let before = files(&home, &broken);
    let (mut app, _, agent) = start(&home, &broken, Version::V2);
    assert!(panel(&app).is_empty(), "{:?}", panel(&app));
    assert_eq!(warnings(&app).len(), 1, "{:?}", warnings(&app));
    run(&mut app, agent, "work add c");
    assert!(panel(&app).is_empty(), "refused until /work reset");
    let conversation = app.world().get::<Conversation>(agent);
    assert_eq!(conversation.map(|it| it.messages().len()), Some(2));
    drop(app);
    assert_eq!(files(&home, &broken), before, "the saved data as it was");
    fs::remove_dir_all(home.root()).ok();
}
