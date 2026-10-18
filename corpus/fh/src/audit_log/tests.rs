use std::fs;

use rig_basics::coding_tools::{ShellTool, WriteTool};
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_core::tool::content_hash;
use rig_harness::harness_protocol::{Invocation, SessionId};
use rig_harness_test_support::{
    Ended, ToolOutputs, app_in, connect, first_agent, log_ended, log_tool_outputs, reply,
    run_until, scratch_home, start_call,
};
use serde_json::json;

use super::chain::{self, Entry, Event, Ran, line_hash};
use super::*;

/// The small plugin of the spec's test: refuses `rm -rf` before it runs.
fn no_rm(mut calls: Query<(&ToolRequest, &mut Decision), (ReadyToStart, ToolCalls)>) {
    for (request, mut decision) in &mut calls {
        let command = request.call.function.arguments.get("command");
        let command = command
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default();
        if command.contains("rm -rf") {
            decision.decide(Decider::plugin("test"), Outcome::refused("no rm -rf"));
        }
    }
}

/// Why `rig audit verify` in `home` exits: built as the launcher runs it.
fn verify_command(home: &Home) -> Option<ExitReason> {
    #[derive(Resource, Default)]
    struct Exited(Option<ExitReason>);
    let (mut app, _) = app_in(home, &SessionId::generate());
    let args = Invocation {
        command: Some("audit".to_owned()),
        arguments: vec!["verify".to_owned()],
        ..Invocation::default()
    };
    app.insert_resource(Invoked {
        args,
        terminal: false,
    })
    .add_message::<Notice>()
    .init_resource::<Exited>()
    .add_observer(|exit: On<Exit>, mut exited: ResMut<Exited>| {
        exited.0.get_or_insert(exit.0.clone());
    })
    .add_plugins(AuditLogPlugin);
    app.finish();
    app.update();
    app.world_mut().resource_mut::<Exited>().0.take()
}

/// Today's lines and their entries.
fn read_log(dir: &Path) -> (Vec<String>, Vec<Entry>) {
    let text = fs::read_to_string(chain::file_for(dir, &chain::today())).unwrap_or_default();
    let lines: Vec<String> = text.lines().map(str::to_owned).collect();
    let entries = lines
        .iter()
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect();
    (lines, entries)
}

fn tool_calls(entries: &[Entry]) -> Vec<&Entry> {
    entries
        .iter()
        .filter(|entry| entry.event == Event::ToolCall)
        .collect()
}

#[test]
fn calls_are_logged_chained_and_checked() {
    let home = scratch_home("fh-audit-log");
    let work = home.root().join("work");
    fs::create_dir_all(&work).ok();
    let audit = home.root().join("audit");
    let (mut app, wakes) = app_in(&home, &SessionId::generate());
    app.add_plugins((
        AgentPlugin,
        SessionLogPlugin,
        ShellTool,
        WriteTool,
        AuditLogPlugin,
    ))
    .add_systems(
        AgentPass,
        no_rm.in_set(Stage::Gate).before(AuditSystems::Gate),
    )
    .init_resource::<Ended>()
    .add_observer(log_ended)
    .init_resource::<ToolOutputs>()
    .add_observer(log_tool_outputs);
    app.finish();
    app.update();

    // One reply: a command, a write, and a command the test plugin refuses.
    let model = MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::tool_call("c1", "shell", json!({ "command": "echo hi" })),
            MockStreamEvent::tool_call("c2", "write", json!({ "path": "a.txt", "content": "x" })),
            MockStreamEvent::tool_call("c3", "shell", json!({ "command": "rm -rf x" })),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        reply("Done."),
    ]);
    let world = app.world_mut();
    let agent = first_agent(world).unwrap_or(Entity::PLACEHOLDER);
    if let Some(connection) = connect(&model) {
        world
            .entity_mut(agent)
            .insert((connection, WorkDir(work.clone())));
    }
    world.trigger(Deliver::new(agent, "Go.", DeliveryMode::Steer));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });

    let (lines, entries) = read_log(&audit);
    let calls = tool_calls(&entries);
    let deciders: Vec<_> = calls.iter().map(|entry| entry.decided_by.clone()).collect();
    assert_eq!(deciders, [None, None, Some(Decider::plugin("test"))]);
    let outcomes: Vec<_> = calls.iter().map(|entry| entry.outcome).collect();
    assert_eq!(
        outcomes,
        [Some(Ran::Ran), Some(Ran::Ran), Some(Ran::Refused)],
        "{lines:#?}"
    );
    assert_eq!(calls.first().and_then(|entry| entry.exit_code), Some(0));
    assert_eq!(calls.get(1).map(|entry| entry.files.len()), Some(1));
    let written = calls.get(1).and_then(|entry| entry.files.first());
    assert!(
        written.is_some_and(|file| file.path.ends_with("a.txt")),
        "{written:?}"
    );
    assert_eq!(written.and_then(|file| file.before.clone()), None);
    assert_eq!(
        written.and_then(|file| file.after.clone()),
        Some(content_hash(b"x").0)
    );
    for (before, entry) in lines.iter().zip(entries.iter().skip(1)) {
        assert_eq!(entry.prev, line_hash(before));
    }

    // A subagent's write carries its parent chain.
    let world = app.world_mut();
    let parent = world
        .get::<AgentId>(agent)
        .map(|id| id.0.clone())
        .unwrap_or_default();
    let sub = world
        .spawn((Agent, SpawnedBy(agent), WorkDir(work.clone())))
        .id();
    start_call(
        world,
        sub,
        ("write", "s1"),
        json!({ "path": "b.txt", "content": "y" }),
    );
    run_until(&mut app, &wakes, |world| {
        world.resource::<ToolOutputs>().0.contains_key("s1")
    });
    let (_, entries) = read_log(&audit);
    let last = tool_calls(&entries)
        .last()
        .map(|entry| entry.parents.clone());
    assert_eq!(last, Some(vec![parent]));

    // While the log cannot be written, a write is refused and changes
    // nothing. (The file is put out of reach by a directory in its place,
    // which no permission bit, even root's, writes through.)
    let file = chain::file_for(&audit, &chain::today());
    let aside = audit.join("aside");
    fs::rename(&file, &aside).ok();
    fs::create_dir(&file).ok();
    let world = app.world_mut();
    start_call(
        world,
        agent,
        ("write", "s2"),
        json!({ "path": "c.txt", "content": "z" }),
    );
    run_until(&mut app, &wakes, |world| {
        world.resource::<ToolOutputs>().0.contains_key("s2")
    });
    let output = app
        .world()
        .resource::<ToolOutputs>()
        .0
        .get("s2")
        .cloned()
        .unwrap_or_default();
    assert!(output.contains("audit log cannot be written"), "{output}");
    assert!(!work.join("c.txt").exists());
    fs::remove_dir(&file).ok();
    fs::rename(&aside, &file).ok();

    // The chain holds, until one byte of the middle call's entry changes;
    // `rig audit verify` runs no session of its own.
    let (lines, _) = read_log(&audit);
    assert_eq!(verify_command(&home), Some(ExitReason::Quit));
    assert_eq!(read_log(&audit).0.len(), lines.len());
    let ok = run_command(&audit, &["verify".to_owned()]);
    assert_eq!(ok, Ok(format!("OK, {} entries", lines.len())));
    let text = fs::read_to_string(&file).unwrap_or_default();
    fs::write(
        &file,
        text.replacen("\"content\":\"x\"", "\"content\":\"z\"", 1),
    )
    .ok();
    let at = lines
        .iter()
        .position(|line| line.contains("\"content\":\"x\""))
        .map(|at| at + 1);
    let broken = chain::verify(&audit, None).err().map(|broken| broken.line);
    assert_eq!(broken, at);
    assert!(at.is_some());
    assert_eq!(verify_command(&home), Some(ExitReason::Failed(1)));
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(AuditLogPlugin);
}
