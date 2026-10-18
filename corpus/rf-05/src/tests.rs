//! The plugin's suite, on published crates only: rig-harness (with the
//! rig-memory store it re-exports), rig-core's scripted model
//! (`test-utils`) and, for the stage check, rig-basics and rig-tui. None
//! of rig's unpublished test-support crates: what they would give is
//! written out here from public API.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, channel};
use std::time::{Duration, Instant};

use rig_core::catalog::Catalog;
use rig_core::operation::Completion;
use rig_core::serve::ErasedHandler;
use rig_core::serve::adapters::ModelAdapter;
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness::harness_protocol::Invocation;
use rig_harness::rig_ecs::effects::Handler;
use rig_harness::rig_ecs::session_log::SessionStore;
use rig_harness::rig_memory::MemoryStore;
use serde_json::json;

use super::*;

/// How long a test waits for the scripted model's off-thread work.
const DEADLINE: Duration = Duration::from_secs(30);

/// A directory of the test's own beside the test binary (never the
/// developer's `RIG_HOME` or the system's temporary directory), emptied,
/// with a `home` and a `project` holding `files`.
fn scratch(name: &str, files: &[(&str, &[u8])]) -> PathBuf {
    let exe = std::env::current_exe().unwrap_or_default();
    let root = exe.with_file_name(format!("rig-wordcount-{name}"));
    std::fs::remove_dir_all(&root).ok();
    let project = root.join("project");
    assert!(std::fs::create_dir_all(&project).is_ok());
    assert!(std::fs::create_dir_all(root.join("home")).is_ok());
    for (file, bytes) in files {
        assert!(std::fs::write(project.join(file), bytes).is_ok());
    }
    root
}

/// How each turn ended, and each tool call's output and count by call id.
#[derive(Resource, Default)]
struct Seen {
    turns: Vec<TurnOutcome>,
    outputs: HashMap<String, (String, Option<Counted>)>,
}

fn see(
    settled: On<Add<Settled>>,
    turns: Query<&TurnOutcome>,
    calls: Query<(&ToolRequest, &Outcome, Option<&Published>)>,
    mut seen: ResMut<Seen>,
) {
    seen.turns.extend(turns.get(settled.entity).cloned());
    if let Ok((request, outcome, published)) = calls.get(settled.entity) {
        let output = outcome.tool_result().output().render();
        let counted = published.and_then(|it| it.0.get::<Counted>().ok().flatten());
        seen.outputs
            .insert(request.call.id.to_string(), (output, counted));
    }
}

/// `model` as the catalog's DeepSeek model, on every agent as it spawns.
fn connect(app: &mut App, model: &MockCompletionModel) {
    let spec = Catalog::builtin().resolve("deepseek/deepseek-flash");
    assert!(spec.is_ok(), "the catalog has its DeepSeek model");
    let Ok(spec) = spec else { return };
    let spec = spec.shared();
    let handler = ModelAdapter::<Completion>::new(spec.reference(), model.clone());
    let connection = Connection {
        spec,
        handler: Handler(ErasedHandler::new(handler)),
    };
    app.add_observer(move |agent: On<Add<Agent>>, mut commands: Commands| {
        commands.entity(agent.entity).insert(connection.clone());
    });
}

/// The kernel, the plugin and `model` on the session in `store`, in the
/// project and home of `root`, with no user settings; its loop's wakes.
fn app(
    store: &MemoryStore,
    root: &std::path::Path,
    model: &MockCompletionModel,
) -> (App, Receiver<()>) {
    let (sender, wakes) = channel();
    let mut app = App::new();
    app.add_plugins(TaskPoolPlugin {
        task_pool_options: rig_harness::task_pools(),
    })
    .insert_resource(Wake::new(move || {
        sender.send(()).ok();
    }))
    .insert_resource(Settings::default())
    .insert_resource(RigHome(Home::at(root.join("home"))))
    .insert_resource(ProjectRoot(root.join("project")))
    .insert_resource(SessionStore::new(store.clone()))
    .add_plugins((AgentPlugin, SessionLogPlugin, WordCountPlugin))
    .init_resource::<Seen>()
    .add_observer(see);
    connect(&mut app, model);
    (app, wakes)
}

/// Runs frames, waiting between them for the loop's wake (the model's
/// off-thread work), until `done`; fails after [`DEADLINE`].
#[track_caller]
fn run_until(app: &mut App, wakes: &Receiver<()>, done: impl Fn(&mut World) -> bool) {
    let started = Instant::now();
    while !done(app.world_mut()) && started.elapsed() < DEADLINE {
        app.update();
        wakes.recv_timeout(Duration::from_millis(100)).ok();
    }
    assert!(done(app.world_mut()), "waited {DEADLINE:?} in vain");
}

fn first_agent(world: &mut World) -> Option<Entity> {
    world
        .query_filtered::<Entity, With<Agent>>()
        .iter(world)
        .next()
}

fn total(world: &mut World) -> Option<u64> {
    let agent = first_agent(world)?;
    world.get::<WordsCounted>(agent).map(|it| it.0)
}

fn reply(text: &str) -> Vec<MockStreamEvent> {
    vec![
        MockStreamEvent::text(text),
        MockStreamEvent::final_response_with_default_usage(),
    ]
}

fn calls(id: &str, path: &str) -> Vec<MockStreamEvent> {
    vec![
        MockStreamEvent::tool_call(id, TOOL, json!({ "path": path })),
        MockStreamEvent::final_response_with_default_usage(),
    ]
}

/// One turn in which the model counts `path`, then says "done", in a
/// project holding `files`; `more` adds to the app first.
fn one_turn(
    name: &str,
    files: &[(&str, &[u8])],
    path: &str,
    more: impl FnOnce(&mut App),
) -> (App, PathBuf) {
    let root = scratch(name, files);
    let model = MockCompletionModel::from_stream_turns([calls("c1", path), reply("done")]);
    let (mut app, wakes) = app(&MemoryStore::default(), &root, &model);
    more(&mut app);
    app.finish();
    app.cleanup();
    app.update();
    let agent = first_agent(app.world_mut());
    assert!(agent.is_some(), "the kernel spawns an agent");
    let Some(agent) = agent else {
        return (app, root);
    };
    app.world_mut().trigger(Deliver::new(
        agent,
        format!("Count {path}."),
        DeliveryMode::Steer,
    ));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Seen>().turns.is_empty()
    });
    (app, root)
}

/// R2: the model calls `word_count`, it runs on the project, the model
/// replies; the turn ended, with the count saved on the agent, and
/// `/wc-reset` sets it back to 0.
#[test]
fn a_turn_counts_the_words_of_a_file() {
    let (mut app, root) = one_turn(
        "turn",
        &[("notes.txt", b"one two\nthree\n")],
        "notes.txt",
        |_| {},
    );
    let seen = app.world().resource::<Seen>();
    let output = seen.outputs.get("c1").cloned();
    assert_eq!(
        output,
        Some(("notes.txt: 3 words".to_owned(), Some(Counted { words: 3 })))
    );
    assert!(
        matches!(seen.turns.as_slice(), [TurnOutcome::Answered(_)]),
        "{:?}",
        seen.turns
    );
    assert_eq!(total(app.world_mut()), Some(3));

    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    app.world_mut().trigger(RunCommand::new(agent, "wc-reset"));
    app.update();
    assert_eq!(total(app.world_mut()), Some(0));
    std::fs::remove_dir_all(root).ok();
}

/// R3: a call on a 2 MiB file is refused with the plugin's reason, and the
/// tool never ran: it counted nothing and published nothing.
#[test]
fn a_file_over_one_mib_is_refused_before_the_tool_runs() {
    let big = vec![b'a'; 2 << 20];
    let (mut app, root) = one_turn("gate", &[("big.txt", &big)], "big.txt", |_| {});
    let output = app.world().resource::<Seen>().outputs.get("c1").cloned();
    assert!(
        output
            .as_ref()
            .is_some_and(|(text, counted)| text.contains(TOO_BIG) && counted.is_none()),
        "{output:?}"
    );
    assert_eq!(total(app.world_mut()), Some(0));
    std::fs::remove_dir_all(root).ok();
}

/// R4: the total survives a second app on the same store, and a restored
/// call already answered, a `word_count` that settled while the reply's
/// other call still ran when the first app was killed, is not counted
/// again.
#[test]
fn the_total_survives_a_restart_without_counting_twice() {
    let root = scratch("restart", &[("notes.txt", b"one two three\n")]);
    let store = MemoryStore::default();
    let model = MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::tool_call("c1", TOOL, json!({ "path": "notes.txt" })),
            MockStreamEvent::tool_call("c2", "hang", json!({})),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        reply("done"),
    ]);
    // A call that never ends, which `word_count` runs before.
    let hang = |app: &mut App| {
        app.add_open_tool(
            "hang",
            "Never answers.",
            ToolOptions::default(),
            |_: On<ToolCalled<serde_json::Value>>| {},
        );
        app.finish();
        app.cleanup();
        app.update();
    };

    let (mut first, wakes) = app(&store, &root, &model);
    hang(&mut first);
    let agent = first_agent(first.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    first
        .world_mut()
        .trigger(Deliver::new(agent, "Count notes.txt.", DeliveryMode::Steer));
    let hanging = |world: &mut World| {
        let mut started = world.query_filtered::<&ToolRequest, (With<Started>, Without<Settled>)>();
        let hang_runs = started
            .iter(world)
            .any(|it| it.call.function.name.as_str() == "hang");
        hang_runs && total(world) == Some(3)
    };
    run_until(&mut first, &wakes, hanging);
    // Killed: no exit frame runs.
    drop(first);

    let (mut second, wakes) = app(&store, &root, &model);
    hang(&mut second);
    assert_eq!(
        total(second.world_mut()),
        Some(3),
        "restored with the session"
    );
    run_until(&mut second, &wakes, |world| {
        !world.resource::<Seen>().turns.is_empty()
    });
    let seen = second.world().resource::<Seen>();
    // Answered again from its saved result, publishing nothing.
    let restored = seen.outputs.get("c1").cloned();
    assert_eq!(restored, Some(("notes.txt: 3 words".to_owned(), None)));
    assert_eq!(total(second.world_mut()), Some(3));
    std::fs::remove_dir_all(root).ok();
}

/// R5: the terminal view lays the call out, by the view half's renderer,
/// with no terminal: its rows name the file and the count.
#[cfg(feature = "tui")]
#[test]
fn the_terminal_view_draws_the_file_and_its_count() {
    use bevy_ecs::system::RunSystemOnce;
    use rig_tui::{TranscriptLayout, TranscriptView};

    let add_view = |app: &mut App| {
        app.add_plugins((rig_tui::TuiPlugin, WordCountView));
    };
    let (mut app, root) = one_turn(
        "render",
        &[("notes.txt", b"one two three four\n")],
        "notes.txt",
        add_view,
    );
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    let rows = app
        .world_mut()
        .run_system_once(move |layout: TranscriptLayout| {
            let laid_out = layout.lay_out(&TranscriptView::default(), agent, 80);
            let rows = laid_out
                .iter()
                .flat_map(|it| it.rows().iter().map(|row| row.text.clone()));
            rows.collect::<Vec<String>>()
        });
    assert!(rows.is_ok(), "{rows:?}");
    let rows = rows.unwrap_or_default();
    let call = rows
        .iter()
        .position(|row| row.contains("Count") && row.contains("notes.txt"));
    let count = rows.iter().position(|row| row.contains("4 words"));
    assert!(call.is_some() && count > call, "{rows:#?}");
    std::fs::remove_dir_all(root).ok();
}

/// The default `plugins.toml`'s plugins, in its order, as rig's launcher
/// writes it (`src/launcher/plugins.toml` in rig's repository): no
/// published crate lists them, so this copy must follow rig's.
fn add_defaults(app: &mut App) {
    use rig_basics::*;
    use rig_harness::load;
    load::<prompt::DefaultPromptPlugin>(app, "rig-basics", "");
    load::<models::ModelsPlugin>(app, "rig-basics", "");
    load::<login_chatgpt::ChatgptLoginPlugin>(app, "rig-basics", "");
    load::<models::defaults::DefaultsPlugin>(app, "rig-basics", "");
    load::<session_picker::SessionPickerPlugin>(app, "rig-basics", "");
    load::<sessions::SessionsPlugin>(app, "rig-basics", "");
    load::<sessions::agents::AgentsPlugin>(app, "rig-basics", "");
    load::<compaction::CompactionPlugin>(app, "rig-basics", "");
    load::<telemetry::usage::UsagePlugin>(app, "rig-basics", "");
    load::<activity::ActivityPlugin>(app, "rig-basics", "");
    load::<coding_tools::ReadTool>(app, "rig-basics", "");
    load::<coding_tools::EditTool>(app, "rig-basics", "");
    load::<coding_tools::WriteTool>(app, "rig-basics", "");
    load::<coding_tools::ShellTool>(app, "rig-basics", "");
    load::<coding_tools::AttachPlugin>(app, "rig-basics", "");
    load::<reload::ReloadPlugin>(app, "rig-basics", "");
    load::<reload::plugins::PluginsPlugin>(app, "rig-basics", "");
    load::<BasicCommandsPlugin>(app, "rig-basics", "");
    #[cfg(feature = "tui")]
    load::<rig_tui::TuiPlugin>(app, "rig-tui", "");
}

/// R6: the plugin's Gate system is ordered against the default plugins'
/// (`stage_conflicts`, the check `rig plugin check --build` runs), in a
/// run without a terminal of the harness built, not run, in a scratch home.
#[test]
fn stages_are_ordered_with_the_defaults() {
    let root = scratch("stages", &[]);
    let mut app = App::new();
    app.insert_resource(RigHome(Home::at(root.join("home"))))
        .insert_resource(Settings::default())
        .insert_resource(Invoked {
            args: Invocation::default(),
            terminal: false,
        })
        .add_plugins((HeadlessPlugins, RigHarnessPlugins));
    add_defaults(&mut app);
    rig_harness::load::<WordCountPlugin>(&mut app, "rig-wordcount", "");
    // The session plugin sends panics to the session's log; a test shows
    // them, so its failed assertions print.
    drop(std::panic::take_hook());
    app.finish();
    app.cleanup();
    let conflicts = rig_harness::stage_conflicts(app.world_mut());
    assert!(conflicts.is_empty(), "{}", conflicts.join("\n"));
    std::fs::remove_dir_all(root).ok();
}
