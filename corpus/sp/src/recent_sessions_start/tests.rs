use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::sync::mpsc::Receiver;

use rig_core::test_utils::MockCompletionModel;
use rig_harness::commands::CommandPlugin;
use rig_harness::pick::PickPlugin;
use rig_harness::rig_core::message::Message;
use rig_harness_test_support::{
    Ended, app_in, first_agent, log_ended, press, reply, run_until, scratch_home, scripted_models,
    tui_view,
};
use rig_sessions::SessionsPlugin;
use rig_tui::TuiTestScreen;

use super::*;

/// A word only the first prompt of session `n` of the project has.
const WORDS: [&str; 10] = [
    "otter", "quokka", "walrus", "ibis", "lynx", "marmot", "pangolin", "tapir", "vole", "yak",
];

/// The model every app talks to.
const MODEL: &str = "test/model";

/// Runs a session in `project` under `home` whose user says `ask` once;
/// its id. The next one is newer by its listing's time.
fn saved_session(home: &Home, project: &Path, ask: &str) -> SessionId {
    let id = SessionId::generate();
    let (mut app, wakes) = app_in(home, &id);
    fs::write(
        home.session(&id).directory(),
        project.to_string_lossy().as_bytes(),
    )
    .ok();
    let model = MockCompletionModel::from_stream_turns([reply("Done.")]);
    app.insert_resource(scripted_models([(MODEL, &model)]))
        .insert_resource(ProjectRoot(project.to_owned()))
        .add_plugins((
            AgentPlugin,
            SessionLogPlugin,
            SessionsPlugin,
            RecentSessionsPlugin,
        ))
        .init_resource::<Ended>()
        .add_observer(log_ended);
    app.finish();
    app.update();
    talk(&mut app, &wakes, ask);
    std::thread::sleep(Duration::from_millis(5));
    id
}

/// Sends `ask` to the app's agent on the test model and waits for the
/// turn's end.
fn talk(app: &mut App, wakes: &Receiver<()>, ask: &str) {
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    let model = MODEL.to_owned();
    app.world_mut().trigger(SetModel {
        entity: agent,
        model,
    });
    let before = app.world().resource::<Ended>().0.len();
    app.world_mut()
        .trigger(Deliver::new(agent, ask, DeliveryMode::Steer));
    run_until(app, wakes, |world| {
        world.resource::<Ended>().0.len() > before
    });
    app.update();
}

/// The running agent at the terminal on a test screen, in a new session
/// of `home` working in `project`, its sessions read; the screen's
/// entity, if it opened.
fn at_terminal(
    home: &Home,
    project: &Path,
    model: &MockCompletionModel,
) -> (App, Receiver<()>, Option<Entity>) {
    let (mut app, wakes) = app_in(home, &SessionId::generate());
    app.insert_resource(scripted_models([(MODEL, model)]))
        .insert_resource(ProjectRoot(project.to_owned()))
        .insert_resource(TuiTestScreen::new(120, 40))
        .add_plugins((AgentPlugin, SessionLogPlugin, CommandPlugin, PickPlugin))
        .add_plugins((SessionsPlugin, RecentSessionsPlugin))
        .init_resource::<Ended>()
        .add_observer(log_ended)
        .init_resource::<Exits>()
        .add_observer(|exit: On<Exit>, mut exits: ResMut<Exits>| exits.0.push(exit.0.clone()));
    tui_view(&mut app);
    app.finish();
    app.cleanup();
    app.update();
    run_until(&mut app, &wakes, |world| {
        world.get_resource::<SessionRestored>().is_some()
            && world
                .try_query_filtered::<(), With<Reading>>()
                .is_some_and(|mut q| q.iter(world).next().is_none())
    });
    app.update();
    let screen = {
        let world = app.world_mut();
        let mut screens = world.query_filtered::<Entity, With<StartScreen>>();
        screens.iter(world).next()
    };
    (app, wakes, screen)
}

/// Every exit asked for.
#[derive(Resource, Default)]
struct Exits(Vec<ExitReason>);

/// The labels the screen lists.
fn labels(app: &App, screen: Option<Entity>) -> Vec<String> {
    let picker = screen.and_then(|screen| app.world().get::<Picker>(screen));
    let items = picker
        .map(|picker| picker.items.clone())
        .unwrap_or_default();
    items.into_iter().map(|item| item.label).collect()
}

/// What the terminal shows.
fn shown(app: &App) -> String {
    app.world().resource::<TuiTestScreen>().rows().join("\n")
}

/// Every file of the sessions `ids`, with its contents.
fn files(home: &Home, ids: &[SessionId]) -> BTreeMap<String, Vec<u8>> {
    let mut found = BTreeMap::new();
    let mut dirs: Vec<_> = ids
        .iter()
        .map(|id| home.session(id).path().to_owned())
        .collect();
    while let Some(dir) = dirs.pop() {
        for entry in fs::read_dir(&dir).into_iter().flatten().flatten() {
            let path = entry.path();
            if path.is_dir() {
                dirs.push(path);
            } else {
                let bytes = fs::read(&path).unwrap_or_default();
                found.insert(path.to_string_lossy().into_owned(), bytes);
            }
        }
    }
    found
}

#[test]
fn lists_filters_and_resumes_recent_sessions_without_writing() {
    let home = scratch_home("sp-recent-sessions-start");
    let project = home.root().join("api");
    let elsewhere = home.root().join("web");
    fs::create_dir_all(&project).ok();
    fs::create_dir_all(&elsewhere).ok();

    // Ten of this project, oldest first, then two of another project.
    let ours: Vec<SessionId> = (WORDS.iter())
        .map(|word| saved_session(&home, &project, &format!("Fix the {word} retry test")))
        .collect();
    let theirs: Vec<SessionId> = ["Style the header", "Fix the footer"]
        .iter()
        .map(|ask| saved_session(&home, &elsewhere, ask))
        .collect();
    // The newest, a session of this project whose files do not read.
    let broken = SessionId::generate();
    let dir = home.session(&broken);
    fs::create_dir_all(dir.path()).ok();
    fs::write(dir.directory(), project.to_string_lossy().as_bytes()).ok();
    let store = StoreOpener::default().open(&dir);
    assert!(store.is_ok_and(|store| {
        store.append_log("a1", b"{ not a record\n").is_ok()
            && store.write_file(META, b"{ not a listing").is_ok()
    }));
    // A rig runs the third newest of ours.
    let busy = ours.get(7).cloned().unwrap_or_else(SessionId::generate);
    let _lock = home.session(&busy).hold();
    let all: Vec<SessionId> = ours
        .iter()
        .chain(&theirs)
        .chain([&broken])
        .cloned()
        .collect();
    let before = files(&home, &all);
    let saved = home.saved_sessions().len();
    assert_eq!(saved, 13);

    // The screen: the 8 newest of this project, newest first.
    let model = MockCompletionModel::from_stream_turns([reply("Fine.")]);
    let (mut app, _wakes, screen) = at_terminal(&home, &project, &model);
    let rows = labels(&app, screen);
    let newest: Vec<String> = (WORDS.iter().rev().take(7))
        .map(|word| format!("Fix the {word} retry test"))
        .collect();
    assert_eq!(rows.len(), SHOWN, "{rows:?}");
    assert_eq!(
        rows.first().map(String::as_str),
        Some(format!("unreadable · session {broken}").as_str())
    );
    assert_eq!(rows.get(1..), Some(&newest[..]));
    let screen_text = shown(&app);
    assert!(screen_text.contains("unreadable"), "{screen_text}");
    assert!(
        screen_text.contains("Fix the yak retry test"),
        "{screen_text}"
    );
    let details = screen
        .and_then(|screen| app.world().get::<Picker>(screen))
        .map(|picker| {
            picker
                .items
                .iter()
                .filter_map(|item| item.detail.clone())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let second = details.get(1).cloned().unwrap_or_default();
    assert!(second.contains("1 turn · test/model"), "{second}");
    assert!(
        details
            .get(3)
            .is_some_and(|detail| detail.contains("open elsewhere")),
        "{details:?}"
    );

    // Enter on the unreadable one says why and stays.
    press(&mut app, "enter");
    assert_eq!(labels(&app, screen).len(), SHOWN);
    assert!(
        shown(&app).contains("Cannot resume that session"),
        "{}",
        shown(&app)
    );

    // `/` filters every session of the project: the 9th newest is found.
    press(&mut app, "/ q u o k k a");
    assert_eq!(labels(&app, screen), ["Fix the quokka retry test"]);
    press(
        &mut app,
        "backspace backspace backspace backspace backspace backspace backspace",
    );

    // `a`: another project's sessions too.
    press(&mut app, "a");
    let rows = labels(&app, screen);
    assert!(rows.iter().any(|row| row == "Fix the footer"), "{rows:?}");
    press(&mut app, "a");

    // Enter on the second row resumes it.
    press(&mut app, "down enter");
    app.update();
    let yak = ours.last().cloned().unwrap_or_else(SessionId::generate);
    let exits = app.world().resource::<Exits>().0.clone();
    assert_eq!(exits, [ExitReason::Switch(Some(yak.clone()))]);
    drop(app);
    // Nothing of the saved sessions changed, and no session was added.
    assert_eq!(files(&home, &all), before);
    assert_eq!(home.saved_sessions().len(), saved);

    // Enter on the session open elsewhere: read it only, then back.
    let model = MockCompletionModel::from_stream_turns([reply("Fine.")]);
    let (mut app, wakes, screen) = at_terminal(&home, &project, &model);
    assert!(screen.is_some());
    press(&mut app, "down down down enter");
    let asked = shown(&app);
    assert!(asked.contains("is open in another rig"), "{asked}");
    press(&mut app, "enter");
    run_until(&mut app, &wakes, |world| {
        world
            .try_query_filtered::<(), (With<ReadOnly>, With<TuiPanel>)>()
            .is_some_and(|mut views| views.iter(world).next().is_some())
    });
    app.update();
    let read = shown(&app);
    assert!(
        read.contains("read-only") && read.contains("Fix the tapir retry test"),
        "{read}"
    );
    press(&mut app, "esc");
    let screen = {
        let world = app.world_mut();
        let mut screens = world.query_filtered::<Entity, With<StartScreen>>();
        screens.iter(world).next()
    };
    assert_eq!(labels(&app, screen).len(), SHOWN);

    // Esc: the new session is saved only once a prompt is sent.
    press(&mut app, "esc");
    assert_eq!(labels(&app, screen), Vec::<String>::new());
    assert_eq!(home.saved_sessions().len(), saved);
    assert_eq!(files(&home, &all), before);
    talk(&mut app, &wakes, "A new task.");
    assert_eq!(home.saved_sessions().len(), saved + 1);
    assert_eq!(files(&home, &all), before);
    drop(app);

    // The launcher then runs the session picked: its messages go with the
    // next prompt.
    let model = MockCompletionModel::from_stream_turns([reply("Again.")]);
    let (mut app, wakes) = app_in(&home, &yak);
    app.insert_resource(scripted_models([(MODEL, &model)]))
        .insert_resource(ProjectRoot(project.clone()))
        .add_plugins((
            AgentPlugin,
            SessionLogPlugin,
            SessionsPlugin,
            RecentSessionsPlugin,
        ))
        .init_resource::<Ended>()
        .add_observer(log_ended);
    app.finish();
    app.update();
    talk(&mut app, &wakes, "Keep going.");
    let requests = model.requests();
    let history = requests
        .last()
        .map(|request| request.chat_history.clone())
        .unwrap_or_default();
    let said = |message: &Message| format!("{message:?}").contains("Fix the yak retry test");
    assert!(history.iter().any(|message| said(message)), "{history:?}");
    drop(app);

    // With no saved session, the input comes first.
    let empty = scratch_home("sp-recent-sessions-start-empty");
    let model = MockCompletionModel::from_stream_turns([reply("Fine.")]);
    let (app, _wakes, screen) = at_terminal(&empty, &project, &model);
    assert_eq!(screen, None);
    drop(app);
    fs::remove_dir_all(home.root()).ok();
    fs::remove_dir_all(empty.root()).ok();
}
