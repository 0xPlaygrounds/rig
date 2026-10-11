use std::path::{Path, PathBuf};

use rig_cassette::journal::MemoryStore;
use rig_core::effect::EffectKind;
use rig_core::message::{Message, ToolResultContent};
use rig_core::test_utils::MockCompletionModel;
use rig_harness_test_support::{Ended, app_on, calls, connect, log_ended, reply, run_until};

use super::*;

/// A fresh directory of the test's own, beside the test binary in the
/// target directory.
fn scratch(name: &str) -> PathBuf {
    let exe = std::env::current_exe().unwrap_or_default();
    let base = exe.parent().map(Path::to_path_buf).unwrap_or_default();
    let dir = base.join(format!(
        "corpus-shell-rewrite-{name}-{}",
        std::process::id()
    ));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(dir.join("bin")).ok();
    dir
}

/// A `trash` in `dir/bin` that moves files to `./.trash`.
fn fake_trash(dir: &Path) {
    let trash = dir.join("bin/trash");
    std::fs::write(&trash, "#!/bin/sh\nmkdir -p .trash && mv \"$@\" .trash/\n").ok();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&trash, std::fs::Permissions::from_mode(0o755)).ok();
    }
}

/// Stands in for spec 01's guard: asks the user about every `shell` call
/// about to run whose command has `rm -rf`, as a hold on the call.
fn guard(calls: Query<(Entity, &ToolRequest, &ForAgent), Undecided>, mut commands: Commands) {
    for (call, request, &ForAgent(agent)) in &calls {
        let arguments = &request.call.function.arguments;
        let command = arguments.get("command").and_then(|c| c.as_str());
        let command = command.unwrap_or_default();
        if command.contains("rm -rf") {
            let ask = Ask::yes_no("Run a destructive command?").body(command.to_owned());
            commands.spawn((ask, ForAgent(agent), HoldOf(call)));
        }
    }
}

/// How many questions were asked.
#[derive(Resource, Default)]
struct Asked(usize);

/// The arguments of each tool call the effect log recorded.
#[derive(Resource, Default)]
struct Logged(Vec<String>);

fn log_calls(mut recorded: MessageReader<Recorded>, mut logged: ResMut<Logged>) {
    for Recorded(record) in recorded.read() {
        if let EffectKind::ToolCall { args, .. } = &record.kind {
            logged.0.push(args.clone());
        }
    }
}

/// What the agent's conversation says.
#[derive(Default)]
struct Seen {
    /// The arguments of each call in the conversation.
    called: Vec<String>,
    /// The first text of each tool result in the conversation.
    results: Vec<String>,
    /// The arguments of each call in the effect log.
    logged: Vec<String>,
    /// How many questions were asked.
    asked: usize,
}

/// Runs a turn in `dir` whose model calls `shell` with `rm a.txt`, then
/// with `rm -rf b`, under the rules `rm -rf` -> `trash` and `rm` ->
/// `trash`, with a guard asking about `rm -rf`.
fn run(dir: &Path) -> Option<Seen> {
    let rules = r#"
        [[rule]]
        command = "rm -rf"
        replace = "trash"
        needs_program = "trash"

        [[rule]]
        command = "rm"
        replace = "trash"
        needs_program = "trash"
    "#;
    let rewrites = ShellRewrites::parse(rules).ok()?;
    let path = std::env::var_os("PATH").unwrap_or_default();
    let dirs = std::iter::once(dir.join("bin")).chain(std::env::split_paths(&path));
    let path = std::env::join_paths(dirs).ok()?;
    let processes = Processes::default();
    processes.set(processes.launcher().in_dir(dir).with_env("PATH", path));

    let store = MemoryStore::default();
    let (mut app, wakes) = app_on(&store);
    app.insert_resource(rewrites)
        .insert_resource(processes)
        .insert_resource(ProjectRoot(dir.to_owned()))
        .init_resource::<Ended>()
        .init_resource::<Asked>()
        .init_resource::<Logged>()
        .add_systems(PostUpdate, log_calls)
        .add_observer(log_ended)
        .add_observer(|_: On<Add<Ask>>, mut asked: ResMut<Asked>| asked.0 += 1)
        .add_systems(AgentPass, guard.in_set(Stage::Gate))
        .add_plugins((
            AgentPlugin,
            JournalPlugin,
            rig_coding_tools::ShellTool,
            ShellRewritePlugin,
        ));
    app.finish();
    app.update();
    let model = MockCompletionModel::from_stream_turns([
        calls("c1", SHELL, serde_json::json!({ "command": "rm a.txt" })),
        calls("c2", SHELL, serde_json::json!({ "command": "rm -rf b" })),
        reply("Deleted."),
    ]);
    let world = app.world_mut();
    let agent = world
        .query_filtered::<Entity, With<Agent>>()
        .iter(world)
        .next()?;
    world.entity_mut(agent).insert(connect(&model)?);
    world.trigger(Deliver::new(
        agent,
        "Delete a.txt and b",
        DeliveryMode::Steer,
    ));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });

    let messages = app.world().get::<Conversation>(agent)?.messages();
    let called = messages
        .iter()
        .filter_map(|message| match message {
            Message::Assistant(assistant) => Some(assistant.tool_calls()),
            _ => None,
        })
        .flatten()
        .map(|call| call.function.raw_arguments())
        .collect();
    let results = messages
        .iter()
        .filter_map(|message| match message {
            Message::User { content } => Some(content.iter()),
            _ => None,
        })
        .flatten()
        .filter_map(|item| match item {
            UserContent::ToolResult(result) => result.content.first(),
            _ => None,
        })
        .filter_map(|content| match content {
            ToolResultContent::Text(text) => Some(text.text.clone()),
            _ => None,
        })
        .collect();
    Some(Seen {
        called,
        results,
        logged: app.world().resource::<Logged>().0.clone(),
        asked: app.world().resource::<Asked>().0,
    })
}

#[test]
fn rm_runs_as_trash_the_guard_judges_it_and_the_conversation_keeps_the_models_call() {
    let dir = scratch("rm");
    fake_trash(&dir);
    std::fs::write(dir.join("a.txt"), "a").ok();
    std::fs::create_dir_all(dir.join("b")).ok();

    let seen = run(&dir).unwrap_or_default();

    // What ran is the rewrite: both went to the trash.
    assert!(dir.join(".trash/a.txt").exists());
    assert!(dir.join(".trash/b").exists());
    assert!(!dir.join("a.txt").exists());
    // The guard judged `trash b`, not `rm -rf b`: nobody was asked.
    assert_eq!(seen.asked, 0);
    // The conversation keeps the model's own calls.
    assert_eq!(
        seen.called,
        [r#"{"command":"rm a.txt"}"#, r#"{"command":"rm -rf b"}"#]
    );
    // The effect log records what ran.
    assert_eq!(
        seen.logged,
        [r#"{"command":"trash a.txt"}"#, r#"{"command":"trash b"}"#]
    );
    // Each result says what ran.
    assert_eq!(seen.results.len(), 2, "{:?}", seen.results);
    let wanted = ["(ran as: `trash a.txt`)\n", "(ran as: `trash b`)\n"];
    for (result, wanted) in seen.results.iter().zip(wanted) {
        assert!(result.starts_with(wanted), "{result:?}");
    }
    std::fs::remove_dir_all(&dir).ok();
}
