use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::mpsc::Receiver;

use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness::commands::CommandPlugin;
use rig_harness::harness_protocol::Home;
use rig_harness_test_support::{
    Ended, app_in, calls, first_agent, log_ended, press, reply, run_until, scratch_home,
    scripted_models, tui_view,
};
use rig_sessions::SessionsPlugin;
use rig_tui::ratatui::text::Line as Drawn;
use rig_tui::{
    AppToolRenderersExt, MainEditor, MainPane, ShowAgent, TranscriptView, TuiTestScreen,
};

use super::*;

/// The exits asked for: switching sessions needs the launcher, which a
/// test has none of.
#[derive(Resource, Default)]
struct Exits(Vec<ExitReason>);

/// An open tool that answers at once.
fn note(called: On<ToolCalled<serde_json::Value>>, mut calls: Query<&mut Decision, ToolCalls>) {
    if let Ok(mut call) = calls.get_mut(called.call) {
        call.decide(Decider::plugin("note"), ToolResult::success("noted".into()));
    }
}

/// An open tool that never answers: its turn runs on.
fn wait(_: On<ToolCalled<serde_json::Value>>) {}

/// An app on session `id` of `home`, working in `project`, with `model`,
/// the open tools, the terminal view drawing `[NOTE: …]` for `note` calls,
/// and the browser; its agent, shown.
fn start(
    home: &Home,
    id: &SessionId,
    project: &Path,
    model: &MockCompletionModel,
) -> (App, Receiver<()>, Entity) {
    let (mut app, wakes) = app_in(home, id);
    fs::write(
        home.session(id).directory(),
        project.to_string_lossy().as_bytes(),
    )
    .ok();
    app.insert_resource(ProjectRoot(project.to_owned()))
        .insert_resource(scripted_models([("test/model", model)]))
        .insert_resource(TuiTestScreen::new(180, 30))
        .add_plugins((AgentPlugin, SessionLogPlugin, CommandPlugin, SessionsPlugin))
        .add_open_tool("note", "Take a note.", ToolOptions::default(), note)
        .add_open_tool("wait", "Wait.", ToolOptions::default(), wait)
        .init_resource::<Ended>()
        .add_observer(log_ended)
        .init_resource::<Exits>()
        .add_observer(|exit: On<Exit>, mut exits: ResMut<Exits>| exits.0.push(exit.0.clone()));
    tui_view(&mut app);
    app.add_plugins(SessionTreeBrowserPlugin)
        .add_tool_renderer("note", |call| {
            vec![Drawn::from(format!(
                "[NOTE: {}]",
                call.argument("text").unwrap_or_default()
            ))]
        });
    app.finish();
    app.cleanup();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    let model = "test/model".to_owned();
    app.world_mut().trigger(SetModel {
        entity: agent,
        model,
    });
    app.world_mut().trigger(ShowAgent { entity: agent });
    app.update();
    (app, wakes, agent)
}

/// Sends `prompt` and runs its turn to its end.
fn ask(app: &mut App, wakes: &Receiver<()>, agent: Entity, prompt: &str) {
    let ended = app.world().resource::<Ended>().0.len();
    app.world_mut()
        .trigger(Deliver::new(agent, prompt, DeliveryMode::Queue));
    run_until(app, wakes, |world| {
        world.resource::<Ended>().0.len() > ended
    });
    app.update();
}

/// A saved session of `project` whose user asked `prompts`, the model
/// answering with `script`; with `branch`, a branch after its first turn
/// continued with those prompts too.
fn saved(
    home: &Home,
    project: &Path,
    prompts: &[&str],
    branch: &[&str],
    script: Vec<Vec<MockStreamEvent>>,
) -> SessionId {
    let id = SessionId::generate();
    let model = MockCompletionModel::from_stream_turns(script);
    let (mut app, wakes, agent) = start(home, &id, project, &model);
    for prompt in prompts {
        ask(&mut app, &wakes, agent, prompt);
    }
    if !branch.is_empty() {
        let conversation = app.world_mut().get_mut::<Conversation>(agent);
        let branched = conversation.and_then(|mut conversation| {
            let second = conversation
                .entries()
                .filter(|entry| entry.is_users_own())
                .nth(1);
            let at = second.map(|entry| entry.index)?;
            conversation.branch_off(at, "retry", Author::user()).ok()
        });
        assert!(branched.is_some());
        for prompt in branch {
            ask(&mut app, &wakes, agent, prompt);
        }
    }
    app.update();
    std::thread::sleep(Duration::from_millis(5));
    id
}

/// Every file of the session `id`, by path.
fn files(home: &Home, id: &SessionId) -> BTreeMap<PathBuf, Vec<u8>> {
    let mut found = BTreeMap::new();
    let mut dirs = vec![home.session(id).path().to_owned()];
    while let Some(dir) = dirs.pop() {
        for entry in fs::read_dir(&dir).into_iter().flatten().flatten() {
            let path = entry.path();
            if path.is_dir() {
                dirs.push(path);
            } else {
                found.insert(path.clone(), fs::read(&path).unwrap_or_default());
            }
        }
    }
    found
}

/// The browser's state.
fn state(app: &mut App) -> Option<&Browser> {
    let entity = browser(app.world_mut())?;
    app.world().get::<Browser>(entity)
}

/// Whether the browser is open and `check` holds for it.
fn browsed(world: &World, check: impl Fn(&Browser) -> bool) -> bool {
    let Some(mut browsers) = world.try_query::<&Browser>() else {
        return false;
    };
    browsers.iter(world).any(check)
}

/// Opens the browser, and waits until it listed the sessions.
fn browse(app: &mut App, wakes: &Receiver<()>, agent: Entity) {
    app.world_mut().trigger(RunCommand::new(agent, "browse"));
    run_until(app, wakes, |world| {
        browsed(world, |browser| browser.sessions.is_some())
    });
    app.update();
}

/// Searches for `text` and waits until the session found is read; then
/// the keys go to the tree.
fn pick(app: &mut App, wakes: &Receiver<()>, text: &str) {
    let typed: Vec<String> = text.chars().map(String::from).collect();
    press(app, &format!("/ {} enter", typed.join(" ")));
    run_until(app, wakes, |world| {
        browsed(world, |browser| browser.tree.is_some())
    });
    press(app, "enter");
}

/// The selected tree row's title.
fn at(app: &mut App) -> String {
    let Some(browser) = state(app) else {
        return String::new();
    };
    let tree = browser.tree.as_ref().map(|(_, tree)| tree);
    match browser.rows.get(browser.row) {
        Some(TreeRow::Node { node, .. }) => tree
            .and_then(|tree| tree.nodes.get(*node))
            .map(|node| node.title.clone())
            .unwrap_or_default(),
        _ => String::new(),
    }
}

/// Moves up the tree until the selected row is `title`.
fn up_to(app: &mut App, title: &str) {
    for _ in 0..10 {
        if at(app) == title {
            return;
        }
        press(app, "k");
    }
}

/// The texts of the preview's messages.
fn previewed(app: &mut App) -> String {
    let preview = state(app)
        .map(|browser| browser.panes)
        .map(|[.., preview]| preview);
    let pane = preview.and_then(|preview| app.world().get::<TranscriptPane>(preview));
    match pane.map(|pane| &pane.source) {
        Some(TranscriptSource::Value(line)) => {
            serde_json::to_string(line.messages()).unwrap_or_default()
        }
        _ => String::new(),
    }
}

/// The screen as text.
fn screen(app: &App) -> String {
    app.world().resource::<TuiTestScreen>().rows().join("\n")
}

/// How many leaves the main agent's tree has.
fn leaves(app: &App, agent: Entity) -> usize {
    let conversation = app.world().get::<Conversation>(agent);
    let tree = conversation.map(|conversation| build(conversation));
    tree.map_or(0, |tree| {
        tree.nodes
            .iter()
            .filter(|node| node.children.is_empty())
            .count()
    })
}

#[test]
fn browses_sessions_and_branches_and_resumes_at_a_turn() {
    let home = scratch_home("sp-session-tree-browser");
    let project = home.root().join("project");
    fs::create_dir_all(&project).ok();
    let s1 = saved(
        &home,
        &project,
        &["Alpha one", "Alpha two"],
        &[],
        vec![reply("A1"), reply("A2")],
    );
    let note_call = calls(
        "n1",
        "note",
        serde_json::json!({ "text": "the migration plan" }),
    );
    let s2 = saved(
        &home,
        &project,
        &["Bravo one", "Bravo two", "Bravo three"],
        &["Bravo four", "Bravo five"],
        vec![
            reply("B1"),
            reply("B2"),
            note_call,
            reply("B3"),
            reply("B4"),
            reply("B5"),
        ],
    );
    saved(&home, &project, &["Charlie one"], &[], vec![reply("C1")]);

    // A draft, then the browser: three panes, the three sessions and this one.
    let model = MockCompletionModel::from_stream_turns(Vec::<Vec<MockStreamEvent>>::new());
    let (mut app, wakes, agent) = start(&home, &SessionId::generate(), &project, &model);
    press(&mut app, "h a l f space t y p e d");
    let top = |app: &mut App| {
        let world = app.world_mut();
        let mut panes = world.query_filtered::<&TranscriptView, With<MainPane>>();
        panes
            .iter(world)
            .next()
            .map(|view| (view.top().cloned(), view.width()))
    };
    let before = top(&mut app);
    browse(&mut app, &wakes, agent);
    let shown = screen(&app);
    for text in [
        " sessions ",
        " tree ",
        " preview ",
        "Alpha one",
        "Bravo one",
        "Charlie one",
        "current",
    ] {
        assert!(shown.contains(text), "{text}:\n{shown}");
    }

    // S2: two leaves; turn 1 previews alone, the main line's leaf its note.
    pick(&mut app, &wakes, "bravo");
    let tree = state(&mut app).and_then(|browser| browser.tree.clone());
    let tree = tree.map(|(_, tree)| tree).unwrap_or_default();
    assert_eq!(
        tree.nodes
            .iter()
            .filter(|node| node.children.is_empty())
            .count(),
        2
    );
    assert_eq!(at(&mut app), "Bravo five");
    up_to(&mut app, "Bravo one");
    assert_eq!(at(&mut app), "Bravo one");
    let preview = previewed(&mut app);
    assert!(
        preview.contains("Bravo one") && preview.contains("B1"),
        "{preview}"
    );
    assert!(
        !preview.contains("Bravo two") && !preview.contains("Bravo four"),
        "{preview}"
    );
    press(&mut app, "j j");
    assert_eq!(at(&mut app), "Bravo three");
    app.update();
    assert!(
        screen(&app).contains("[NOTE: the migration plan]"),
        "{}",
        screen(&app)
    );

    // Esc: the main view as it was, the draft kept.
    press(&mut app, "esc");
    assert!(state(&mut app).is_none());
    let world = app.world_mut();
    let mut editors = world.query_filtered::<&rig_tui::Editor, With<MainEditor>>();
    let draft = editors.iter(world).next().map(|editor| editor.text());
    assert_eq!(draft, Some("half typed"));
    assert_eq!(app.world().resource::<ShownAgent>().0, Some(agent));
    assert_eq!(top(&mut app), before);

    // `o` on S1 shows it whole, read-only; S1's files stay as they were.
    let s1_files = files(&home, &s1);
    browse(&mut app, &wakes, agent);
    pick(&mut app, &wakes, "alpha");
    press(&mut app, "o");
    assert!(state(&mut app).is_some_and(|browser| browser.open));
    let shown = screen(&app);
    assert!(
        shown.contains("read-only") && shown.contains("A2"),
        "{shown}"
    );
    assert!(!shown.contains(" tree "), "{shown}");
    press(&mut app, "esc esc");
    assert!(state(&mut app).is_none());
    assert_eq!(files(&home, &s1), s1_files);

    // S2's turn 1, Enter: S2 runs next, from there.
    browse(&mut app, &wakes, agent);
    pick(&mut app, &wakes, "bravo");
    up_to(&mut app, "Bravo one");
    press(&mut app, "enter");
    let said = state(&mut app).and_then(|browser| browser.said.clone());
    let switched = app.world().resource::<Exits>().0.clone();
    assert_eq!(switched, [ExitReason::Switch(Some(s2.clone()))], "{said:?}");
    drop(app);

    // As the launcher would: S2 runs; a prompt goes on from turn 1 alone.
    let model =
        MockCompletionModel::from_stream_turns([calls("w1", "wait", serde_json::json!({}))]);
    let (mut app, wakes, agent) = start(&home, &s2, &project, &model);
    app.world_mut()
        .trigger(Deliver::new(agent, "Bravo again", DeliveryMode::Queue));
    run_until(&mut app, &wakes, |_| model.request_count() > 0);
    let sent = model.requests();
    let sent = serde_json::to_string(&sent.first().map(|request| &request.chat_history));
    let sent = sent.unwrap_or_default();
    assert!(
        sent.contains("Bravo one") && sent.contains("Bravo again"),
        "{sent}"
    );
    for left in ["Bravo two", "Bravo three", "Bravo four", "Bravo five"] {
        assert!(!sent.contains(left), "{left}: {sent}");
    }
    assert_eq!(leaves(&app, agent), 3);

    // While its turn runs, resuming another session is refused.
    browse(&mut app, &wakes, agent);
    pick(&mut app, &wakes, "alpha");
    let s1_files = files(&home, &s1);
    press(&mut app, "enter");
    app.update();
    assert!(app.world().resource::<Exits>().0.is_empty());
    assert!(
        screen(&app).contains("An agent is running"),
        "{}",
        screen(&app)
    );
    assert_eq!(files(&home, &s1), s1_files);
    fs::remove_dir_all(home.root()).ok();
}
