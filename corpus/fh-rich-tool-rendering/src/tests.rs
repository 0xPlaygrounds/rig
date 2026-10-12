use rig_cassette::journal::MemoryStore;
use rig_coding_tools::{EditTool, ShellTool};
use rig_core::completion::{AssistantContent, Message};
use rig_core::message::{ToolFunction, ToolName, ToolResult, UserContent};
use rig_core::streaming::Part;
use rig_core::test_utils::MockCompletionModel;
use rig_harness::rig_ecs::journal::JournalPlugin;
use rig_harness_test_support::{
    Ended, app_on, calls, connect, first_agent, log_ended, reply, run_until, scratch_home,
};
use rig_tui::{CallState, ToolCallView};
use serde_json::json;

use super::*;

/// The transcript lines of `call` with `result`, drawn as the terminal view
/// draws them: by the tool's renderer, else the default look.
fn drawn(
    world: &mut World,
    call: &rig_core::message::ToolCall,
    result: Option<&ToolResult>,
) -> Vec<String> {
    let view = ToolCallView {
        call,
        result,
        state: CallState::Done,
    };
    // A query skips a disabled renderer, as the view's does.
    let mut renderers = world.query::<&ToolRenderer>();
    let renderer = renderers
        .iter(world)
        .find(|renderer| renderer.tool == call.function.name.as_str());
    let lines = match renderer {
        Some(renderer) => (renderer.render)(&view),
        None => view.default_lines(),
    };
    lines.iter().map(ToString::to_string).collect()
}

/// A call of `tool` with `args`.
fn call(tool: &str, args: serde_json::Value) -> Option<rig_core::message::ToolCall> {
    let name = ToolName::new(tool).ok()?;
    Some(rig_core::message::ToolCall::from_wire(
        "c1",
        ToolFunction::new(name, args),
    ))
}

#[test]
fn edits_read_as_diffs_and_commands_as_command_lines() {
    let dir = scratch_home("fh-rich-tool-rendering").root().to_path_buf();
    assert!(std::fs::create_dir_all(&dir).is_ok());
    let source = "fn main() {\n    let a = 1;\n}\n";
    assert!(std::fs::write(dir.join("a.rs"), source).is_ok());
    let edits = json!([{ "old_text": "let a = 1;", "new_text": "let a = 2;" }]);
    let model = MockCompletionModel::from_stream_turns([
        calls("e1", "edit", json!({ "path": "a.rs", "edits": edits })),
        calls("s1", "shell", json!({ "command": "exit 3" })),
        reply("done"),
    ]);
    let connection = connect(&model);
    assert!(connection.is_some(), "the catalog lists the model");
    let Some(connection) = connection else {
        return;
    };
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.insert_resource(Invoked::default())
        .add_plugins((AgentPlugin, JournalPlugin, EditTool, ShellTool, TuiPlugin))
        .add_plugins(RichToolRenderingPlugin)
        .add_tool_renderer("probe", |call| vec![call.header("probe says hi", "")])
        .init_resource::<Ended>()
        .add_observer(log_ended);
    app.finish();
    app.update();
    let world = app.world_mut();
    let agent = first_agent(world);
    assert!(agent.is_some(), "an agent");
    let Some(agent) = agent else {
        return;
    };
    world
        .entity_mut(agent)
        .insert((connection, WorkDir(dir.clone())));
    world.trigger(Deliver::new(
        agent,
        "Change a and fail",
        DeliveryMode::Queue,
    ));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });
    assert_eq!(
        std::fs::read_to_string(dir.join("a.rs")).ok().as_deref(),
        Some("fn main() {\n    let a = 2;\n}\n")
    );

    // Every tool call of the conversation, as the transcript draws it.
    let world = app.world_mut();
    let messages: Vec<Message> = world
        .get::<Conversation>(agent)
        .map(|conversation| conversation.messages().to_vec())
        .unwrap_or_default();
    let results: Vec<&ToolResult> = messages
        .iter()
        .filter_map(|message| match message {
            Message::User { content } => Some(content.iter()),
            _ => None,
        })
        .flatten()
        .filter_map(|item| match item {
            UserContent::ToolResult(result) => Some(result),
            _ => None,
        })
        .collect();
    let mut screen = Vec::new();
    for message in &messages {
        let Message::Assistant(assistant) = message else {
            continue;
        };
        for item in assistant.content.iter() {
            if let AssistantContent::ToolCall(call) = item {
                let result = results
                    .iter()
                    .copied()
                    .find(|result| result.call == call.id);
                screen.extend(drawn(world, call, result));
            }
        }
    }
    let has = |wanted: &dyn Fn(&str) -> bool| screen.iter().any(|line| wanted(line));
    assert!(
        has(&|line| line.starts_with("- ") && line.contains("let a = 1;")),
        "{screen:#?}"
    );
    assert!(
        has(&|line| line.starts_with("+ ") && line.contains("let a = 2;")),
        "{screen:#?}"
    );
    assert!(has(&|line| line.contains("$ exit 3")), "{screen:#?}");
    assert!(has(&|line| line.contains("exit 3 · ")), "{screen:#?}");
    assert!(
        !has(&|line| line.contains("old_text") || line.contains("{\"")),
        "{screen:#?}"
    );

    // Another plugin's renderer is used; a tool without one looks as by default.
    let probe = call("probe", json!({}));
    let other = call("other", json!({ "x": 1 }));
    let (Some(probe), Some(other)) = (probe, other) else {
        return;
    };
    assert!(
        drawn(world, &probe, None)
            .concat()
            .contains("probe says hi")
    );
    assert!(
        drawn(world, &other, None)
            .concat()
            .contains(r#"other {"x":1}"#)
    );

    // A command that printed 100k lines: its last lines and what was cut.
    let Some(big) = call("shell", json!({ "command": "seq 100000" })) else {
        return;
    };
    let numbers: Vec<String> = (1..=100_000).map(|n| n.to_string()).collect();
    let output = format!("{}\n[exit code 0]", numbers.join("\n"));
    let result = big.answer(&rig_core::tool::ToolResult::success(output.into()));
    let lines = drawn(world, &big, Some(&result));
    let shown: Vec<&str> = lines.iter().map(|line| line.trim()).collect();
    assert!(
        shown.contains(&"… 99995 more lines (Ctrl+O to expand)"),
        "{shown:?}"
    );
    assert!(
        shown.ends_with(&[
            "99996",
            "99997",
            "99998",
            "99999",
            "100000",
            "⎿ exit 0 · earlier run"
        ]),
        "{shown:?}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_streaming_write_shows_its_path_before_its_last_chunk() {
    let mut app = App::new();
    app.add_plugins(AgentPlugin)
        .add_plugins(RichToolRenderingPlugin);
    app.finish();
    // A model call still streaming: no outcome yet.
    let reply = app.world_mut().spawn_empty().id();
    let arguments = r#"{"path": "big.txt", "content": "one\ntwo\nthree\nfour\nfive\n"}"#;
    let length = arguments.chars().count();
    let chunks: Vec<String> = (0..10)
        .map(|n| {
            let (from, to) = (n * length / 10, (n + 1) * length / 10);
            arguments.chars().skip(from).take(to - from).collect()
        })
        .collect();
    assert_eq!(chunks.len(), 10);
    let mut seen = None;
    for (index, text) in chunks.into_iter().enumerate() {
        app.world_mut().write_message(StreamDelta {
            effect: reply,
            agent: None,
            kind: DeltaKind::Arguments,
            part: Part::new(1),
            text,
        });
        app.update();
        let preview = app.world().resource::<StreamingCalls>().preview();
        if seen.is_none()
            && preview
                .iter()
                .any(|line| line.to_string().contains("big.txt"))
        {
            seen = Some(index);
        }
    }
    assert!(seen.is_some_and(|index| index < 9), "{seen:?}");
}
