use std::process::Command;
use std::sync::mpsc::Receiver;

use rig_basics::coding_tools::{ReadTool, WriteTool};
use rig_basics::subagents::{SpawnSubagent, SubagentsPlugin};
use rig_core::message::Message;
use rig_core::test_utils::MockCompletionModel;
use rig_harness::harness_protocol::Invocation;
use rig_harness::host::exit::ExitPlugin;
use rig_harness::load;
use rig_harness_test_support::{
    Collected, Ended, ToolOutputs, app_in, calls, collect, commit_message, log_ended,
    log_tool_outputs, reply, run_until, scratch_home, scripted_models,
};
use serde_json::json;

use super::*;

/// A plugin that keeps a counter on each agent, saved with the session.
#[derive(Default)]
struct CounterPlugin;

impl Plugin for CounterPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<Counter>().keeps_saved::<Counter>();
    }
}

#[derive(Component, Reflect, Default, Debug, PartialEq, Eq)]
#[reflect(Component, Saved, Default)]
struct Counter(u32);

fn git(dir: &Path, args: &[&str]) -> String {
    let output = Command::new("git").current_dir(dir).args(args).output();
    output.map_or_else(
        |failure| format!("failed: {failure}"),
        |output| String::from_utf8_lossy(&output.stdout).trim().to_owned(),
    )
}

/// A git project at `dir` whose `src/lib.rs` is committed.
fn project(dir: &Path) {
    fs::create_dir_all(dir.join("src")).ok();
    fs::write(dir.join("src/lib.rs"), "pub fn start() {}\n").ok();
    git(dir, &["init", "-q", "-b", "main"]);
    git(dir, &["add", "."]);
    let who = ["-c", "user.name=t", "-c", "user.email=t@t"];
    git(dir, &[&who[..], &["commit", "-q", "-m", "first"]].concat());
}

/// An app on the session `id` of `home`, in the project `root`, with the
/// file tools, subagents and the bundle plugin; the counter plugin when
/// `counter`.
fn start(
    home: &Home,
    id: &SessionId,
    root: &Path,
    model: &MockCompletionModel,
    counter: bool,
) -> (App, Receiver<()>) {
    let (mut app, wakes) = app_in(home, id);
    app.insert_resource(ProjectRoot(root.to_owned()))
        .insert_resource(scripted_models([("test/model", model)]))
        .add_plugins((AgentPlugin, SessionLogPlugin))
        .init_resource::<Ended>()
        .add_observer(log_ended)
        .init_resource::<ToolOutputs>()
        .add_observer(log_tool_outputs);
    load::<WriteTool>(&mut app, "rig-basics", "");
    load::<ReadTool>(&mut app, "rig-basics", "");
    load::<SubagentsPlugin>(&mut app, "rig-basics", "");
    if counter {
        load::<CounterPlugin>(&mut app, "corpus-oc-session-bundle", "");
    }
    load::<SessionBundlePlugin>(&mut app, "corpus-oc-session-bundle", "");
    collect::<BundleWritten>(&mut app);
    collect::<Recalled>(&mut app);
    app.finish();
    app.update();
    (app, wakes)
}

/// The agent the user talks to: the one no agent spawned.
fn main_agent(app: &mut App) -> Entity {
    let world = app.world_mut();
    let mut agents = world.query_filtered::<Entity, (With<Agent>, Without<SpawnedBy>)>();
    agents.iter(world).next().unwrap_or(Entity::PLACEHOLDER)
}

fn turn(app: &mut App, wakes: &Receiver<()>, agent: Entity, text: &str) {
    let ended = app.world().resource::<Ended>().0.len();
    app.world_mut()
        .trigger(Deliver::new(agent, text, DeliveryMode::Queue));
    run_until(app, wakes, |world| {
        world.resource::<Ended>().0.len() > ended
    });
}

/// Why each command was refused, as a `--print` run fails with (exit 2).
fn refusals(app: &App) -> Vec<String> {
    let recalled = &app.world().resource::<Collected<Recalled>>().0;
    recalled
        .iter()
        .filter_map(|recalled| recalled.why.clone())
        .collect()
}

#[test]
fn a_bundle_moves_a_session_to_another_checkout() {
    let home = scratch_home("oc-session-bundle");
    let (a, b) = (home.root().join("project-a"), home.root().join("project-b"));
    project(&a);
    let lib_a = a.join("src/lib.rs");
    let path_a = lib_a.display().to_string();
    let edited = "pub fn edited() -> u8 {\n    3\n}\n";

    // Project A: two turns edit `src/lib.rs`, a helper subagent answers,
    // and the counter plugin saves 3.
    let model = MockCompletionModel::from_stream_turns([
        calls(
            "w1",
            "write",
            json!({ "path": path_a, "content": "pub fn edited() {}\n" }),
        ),
        reply("Edited."),
        calls("w2", "write", json!({ "path": path_a, "content": edited })),
        reply("Edited again."),
        reply("Helper here."),
    ]);
    let session_a = SessionId::generate();
    let (mut app, wakes) = start(&home, &session_a, &a, &model, true);
    let agent = main_agent(&mut app);
    app.world_mut().trigger(SetModel {
        entity: agent,
        model: "test/model".to_owned(),
        kind: ChoiceKind::ForThisRun,
    });
    turn(&mut app, &wakes, agent, "Edit src/lib.rs.");
    turn(&mut app, &wakes, agent, "Edit it again.");
    let ended = app.world().resource::<Ended>().0.len();
    let helper = SpawnSubagent::helper("helper").brief("Say hello.").once();
    app.world_mut().spawn((helper, ForAgent(agent)));
    run_until(&mut app, &wakes, |world| {
        world.resource::<Ended>().0.len() > ended
    });
    app.world_mut().entity_mut(agent).insert(Counter(3));
    app.update();
    app.world_mut()
        .trigger(RunCommand::new(agent, "bundle out.rigbundle"));
    app.update();
    app.update();
    let written = app.world().resource::<Collected<BundleWritten>>().0.clone();
    let manifest = written
        .first()
        .map(|written| written.manifest.clone())
        .unwrap_or_default();
    // Each plugin is named with what it says it keeps, though the counter
    // and the bundle plugin share a crate.
    let kept = |plugin: &str| {
        let kept = manifest
            .plugins
            .iter()
            .find(|(name, _)| name.ends_with(plugin));
        kept.map(|(_, kept)| kept.saved.clone()).unwrap_or_default()
    };
    assert_eq!(
        kept("::CounterPlugin"),
        [Counter::type_path()],
        "{manifest:?} {:?}",
        refusals(&app)
    );
    assert_eq!(kept("::SessionBundlePlugin"), [BundleOrigin::type_path()]);
    assert_eq!(manifest.rig_version, VERSION);
    assert_eq!(manifest.project_root, a);
    assert_eq!(manifest.files, ["src/lib.rs"]);
    assert_eq!(manifest.models, ["test/model"]);
    assert_eq!(
        manifest.commit.as_deref(),
        Some(git(&a, &["rev-parse", "HEAD"]).as_str())
    );
    drop(app);

    // Project B: a clone at the same commit, without the edit, imported
    // with `rig bundle open … --apply` in a build without the counter and
    // without the model the session used.
    git(
        home.root(),
        &[
            "clone",
            "-q",
            &a.display().to_string(),
            &b.display().to_string(),
        ],
    );
    let (mut open, _wakes) = app_in(&home, &SessionId::generate());
    let arguments = [
        "open",
        &a.join("out.rigbundle").display().to_string(),
        "--project",
        &b.display().to_string(),
        "--apply",
    ];
    open.insert_resource(Invoked {
        args: Invocation {
            command: Some("bundle".to_owned()),
            arguments: arguments.map(str::to_owned).to_vec(),
            ..Invocation::default()
        },
        terminal: false,
    })
    .insert_resource(scripted_models([("test/other", &model)]))
    .add_plugins((ExitPlugin, AgentPlugin, SessionLogPlugin));
    load::<SessionBundlePlugin>(&mut open, "corpus-oc-session-bundle", "");
    collect::<BundleImported>(&mut open);
    open.finish();
    open.update();
    assert_eq!(
        open.world().get_resource::<ExitReason>(),
        Some(&ExitReason::Quit)
    );
    let imported = open
        .world()
        .resource::<Collected<BundleImported>>()
        .0
        .clone();
    assert_eq!(imported.len(), 1);
    let Some(imported) = imported.first().cloned() else {
        return;
    };
    assert_eq!(
        fs::read_to_string(b.join("src/lib.rs")).unwrap_or_default(),
        edited
    );
    assert_eq!(imported.applied, ["src/lib.rs"]);
    // Only the lacking plugins that kept saved values are named, not the
    // file tools this build also lacks.
    assert_eq!(
        imported.missing_plugins,
        [
            "corpus_oc::session_bundle::tests::CounterPlugin",
            "rig_basics::subagents::SubagentsPlugin"
        ]
    );
    assert!(
        imported
            .lines()
            .iter()
            .any(|line| line.starts_with("Missing plugins")),
        "{imported:?}"
    );
    assert_eq!(
        imported.notes,
        ["test/model is not available here: its agents use the default model."]
    );

    // The imported session lists the subagent, and its conversation's
    // paths are under B.
    let store = StoreOpener::default().open(&home.session(&imported.session));
    let saved = store
        .and_then(|store| Session::read(&*store))
        .unwrap_or_default();
    assert_eq!(
        saved
            .agents
            .iter()
            .filter(|agent| agent.parent.is_some())
            .count(),
        1
    );
    let text = serde_json::to_string(
        &saved
            .agents
            .iter()
            .map(|agent| agent.conversation.messages().to_vec())
            .collect::<Vec<Vec<std::sync::Arc<Message>>>>(),
    )
    .unwrap_or_default();
    assert!(
        text.contains(&b.join("src/lib.rs").display().to_string()),
        "{text}"
    );
    assert!(!text.contains(&path_a), "{text}");
    drop(open);

    // Resumed in B with the counter plugin installed again: it reads 3,
    // the agent has no model of its own (the default), and a read by the
    // remapped absolute path succeeds.
    let path_b = b.join("src/lib.rs").display().to_string();
    let model = MockCompletionModel::from_stream_turns([
        calls("r1", "read", json!({ "path": path_b })),
        reply("It returns 3."),
    ]);
    let (mut app, wakes) = start(&home, &imported.session, &b, &model, true);
    let agent = main_agent(&mut app);
    assert_eq!(app.world().get::<Counter>(agent), Some(&Counter(3)));
    assert_eq!(app.world().get::<ModelChoice>(agent), None);
    app.world_mut().trigger(SetModel {
        entity: agent,
        model: "test/model".to_owned(),
        kind: ChoiceKind::ForThisRun,
    });
    turn(&mut app, &wakes, agent, "What does src/lib.rs return?");
    let outputs = &app.world().resource::<ToolOutputs>().0;
    let output = outputs.get("r1").cloned().unwrap_or_default();
    assert!(output.contains("edited"), "{output}");
    drop(app);

    // A secret typed into the conversation never reaches the session's
    // logs (they are redacted as written), so the bundle made from them
    // holds none and is written without asking.
    let (mut app, _wakes) = start(&home, &SessionId::generate(), &a, &model, false);
    let agent = main_agent(&mut app);
    commit_message(
        app.world_mut(),
        agent,
        Message::user("export AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMIK7MDENG"),
    )
    .ok();
    app.update();
    app.world_mut()
        .trigger(RunCommand::new(agent, "bundle secret.rigbundle"));
    app.update();
    app.update();
    let bundle = std::fs::read(a.join("secret.rigbundle")).unwrap_or_default();
    assert!(!bundle.is_empty(), "{:?}", refusals(&app));
    let secret = b"wJalrXUtnFEMIK7MDENG";
    assert!(!bundle.windows(secret.len()).any(|window| window == secret));
}
