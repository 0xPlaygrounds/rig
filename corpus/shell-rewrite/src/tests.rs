use std::path::PathBuf;

use rig_cassette::journal::MemoryStore;
use rig_core::message::{Message, ToolResultContent};
use rig_core::test_utils::MockCompletionModel;
use rig_harness_test_support::{app_on, calls, connect, reply, run_until};

use super::*;

/// How many turns ended.
#[derive(Resource, Default)]
struct Ended(usize);

/// A fresh directory of the test's own, under the temp dir.
fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "corpus-shell-rewrite-{name}-{}",
        std::process::id()
    ));
    std::fs::remove_dir_all(&dir).ok();
    dir
}

#[test]
fn the_rules_rewrite_each_simple_command_in_order() {
    let parsed = ShellRewrites::parse(
        r#"
        [[rule]]
        command = "cargo test"
        add = ["--quiet", "-j 4"]

        [[rule]]
        command = "cargo"
        prefix = "timeout 300"

        [[rule]]
        command = "npm install"
        replace = "pnpm install"
        needs_file = "no-such-pnpm-lock.yaml"
        "#,
    );
    assert!(parsed.is_ok(), "{parsed:?}");
    let rewrites = parsed.unwrap_or_default();
    let cases = [
        (
            "cargo test -p x -- --nocapture",
            "timeout 300 cargo test --quiet -j 4 -p x -- --nocapture",
        ),
        ("cargo test --quiet", "timeout 300 cargo test -j 4 --quiet"),
        (
            "cd x && cargo build 2>&1 | tail",
            "cd x && timeout 300 cargo build 2>&1 | tail",
        ),
        ("timeout 9 cargo build", "timeout 9 cargo build"),
        ("echo 'cargo; cargo'", "echo 'cargo; cargo'"),
        ("npm install", "npm install"),
    ];
    let got: Vec<(&str, String)> = cases
        .iter()
        .map(|(command, _)| (*command, rewrites.rewrite(command)))
        .collect();
    let wanted: Vec<(&str, String)> = cases
        .iter()
        .map(|(command, rewritten)| (*command, (*rewritten).to_owned()))
        .collect();
    assert_eq!(got, wanted);
}

/// What the agent's conversation and its model saw.
struct Seen {
    /// The arguments of each call in the conversation.
    called: Vec<String>,
    /// The first text of each tool result in the conversation.
    results: Vec<String>,
    /// The model's second request.
    second: String,
}

/// Runs a turn whose model calls `shell` with `rm a.txt`, with the rule
/// `rm` -> `trash` on, in the working directory.
fn rm_a_txt() -> Option<Seen> {
    let rules = "[[rule]]\ncommand = \"rm\"\nreplace = \"trash\"\nneeds_program = \"trash\"\n";
    let rewrites = ShellRewrites::parse(rules).ok()?;
    let store = MemoryStore::default();
    let (mut app, wakes) = app_on(&store);
    app.insert_resource(rewrites)
        .init_resource::<Ended>()
        .add_observer(|_: On<TurnEnded>, mut ended: ResMut<Ended>| ended.0 += 1)
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
        reply("Deleted."),
    ]);
    let world = app.world_mut();
    let agent = world
        .query_filtered::<Entity, With<Agent>>()
        .iter(world)
        .next()?;
    world.entity_mut(agent).insert(connect(&model)?);
    world.trigger(Deliver::new(agent, "Delete a.txt", DeliveryMode::Steer));
    run_until(&mut app, &wakes, |world| world.resource::<Ended>().0 > 0);

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
    let second = format!("{:?}", model.requests().get(1));
    Some(Seen {
        called,
        results,
        second,
    })
}

#[test]
fn rm_runs_as_trash_while_the_conversation_keeps_the_models_call() {
    // A project with `a.txt`, and a `trash` on PATH that moves files to
    // `./.trash`.
    let dir = scratch("rm");
    let bin = dir.join("bin");
    std::fs::create_dir_all(&bin).ok();
    std::fs::write(dir.join("a.txt"), "a").ok();
    let trash = bin.join("trash");
    std::fs::write(&trash, "#!/bin/sh\nmkdir -p .trash && mv \"$@\" .trash/\n").ok();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&trash, std::fs::Permissions::from_mode(0o755)).ok();
    }
    let path = std::env::var_os("PATH").unwrap_or_default();
    let path = std::env::join_paths(std::iter::once(bin).chain(std::env::split_paths(&path)));
    // SAFETY: no other test of this crate reads or writes the environment,
    // and this one does so before its app starts any thread.
    unsafe { std::env::set_var("PATH", path.unwrap_or_default()) };
    std::env::set_current_dir(&dir).ok();

    let seen = rm_a_txt();

    assert!(dir.join(".trash/a.txt").exists());
    assert!(!dir.join("a.txt").exists());
    let seen = seen.unwrap_or(Seen {
        called: Vec::new(),
        results: Vec::new(),
        second: String::new(),
    });
    // The conversation keeps the model's own call.
    assert_eq!(seen.called, [r#"{"command":"rm a.txt"}"#]);
    // The result, which the next request carried, says what ran.
    assert_eq!(seen.results.len(), 1);
    assert!(
        seen.results
            .iter()
            .all(|result| result.starts_with("(ran as: `trash a.txt`)\n")),
        "{:?}",
        seen.results
    );
    assert!(
        seen.second.contains("(ran as: `trash a.txt`)"),
        "{}",
        seen.second
    );
    std::fs::remove_dir_all(&dir).ok();
}
