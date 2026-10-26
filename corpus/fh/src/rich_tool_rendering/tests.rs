use std::collections::HashMap;
use std::time::SystemTime;

use rig_basics::coding_tools::{EditTool, ShellTool};
use rig_core::completion::{AssistantContent, Message};
use rig_core::message::{CallId, ToolCall, ToolFunction, ToolName, ToolResult, UserContent};
use rig_core::streaming::parse_partial_arguments;
use rig_core::test_utils::MockCompletionModel;
use rig_core::tool::ToolResultContext;
use rig_harness::rig_ecs::session_log::SessionLogPlugin;
use rig_harness_test_support::MemoryStore;
use rig_harness_test_support::{
    Ended, app_on, calls, connect, first_agent, log_ended, reply, run_until, scratch_home,
};
use rig_tui::{
    CallState, RenderInput, RenderKey, RenderOutput, RenderRequest, ToolCallView, ToolRender,
    TuiPlugin,
};
use serde_json::json;

use super::*;

/// What each tool call published and how long it ran, as the terminal
/// view hands them to a renderer.
#[derive(Resource, Default)]
struct Facts(HashMap<CallId, (ToolResultContext, Duration)>);

fn note_facts(
    settled: On<Add<Settled>>,
    calls: Query<(&ToolRequest, &Published, &Started)>,
    mut facts: ResMut<Facts>,
) {
    if let Ok((request, Published(published), started)) = calls.get(settled.entity) {
        let took = SystemTime::now()
            .duration_since(started.at)
            .unwrap_or_default();
        facts
            .0
            .insert(request.call.id.clone(), (published.clone(), took));
    }
}

/// The lines of `view` as the terminal view draws them: through the
/// render systems, which start from the default look.
fn drawn(world: &mut World, view: &ToolCallView<'_>) -> Vec<String> {
    let input = ToolRender {
        call: view.call.clone(),
        result: view.result.cloned(),
        state: view.state.clone(),
        theme: view.theme,
        facts: view.facts.cloned(),
        took: view.took,
        output: view.output.map(str::to_owned),
        children: view.children.to_vec(),
    };
    let key = RenderKey::Tool(view.call.id.clone());
    let request = RenderRequest {
        input: RenderInput::Tool(input),
        output: RenderOutput {
            lines: view.default_lines(),
            decorations: Vec::new(),
        },
    };
    let mut batch = RenderBatch::default();
    batch.requests.insert(key.clone(), request);
    world.insert_resource(batch);
    world.try_run_schedule(TranscriptRender).ok();
    let mut batch = world.remove_resource::<RenderBatch>().unwrap_or_default();
    let lines = batch.requests.remove(&key).map(|drawn| drawn.output.lines);
    lines.iter().flatten().map(ToString::to_string).collect()
}

/// Another plugin's look, of its `probe` tool.
fn draw_probe(mut batch: ResMut<RenderBatch>) {
    for (call, output) in batch.tools_mut("probe") {
        output.lines = vec![call.view().header("probe says hi", "")];
    }
}

/// A call of `tool` with `args`.
fn call(tool: &str, args: serde_json::Map<String, serde_json::Value>) -> Option<ToolCall> {
    let name = ToolName::new(tool).ok()?;
    Some(ToolCall::from_wire(
        "c1",
        ToolFunction::new(name, serde_json::Value::Object(args)),
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
        calls("s2", "shell", json!({ "command": "seq 100000" })),
        reply("done"),
    ]);
    let connection = connect(&model);
    assert!(connection.is_some(), "the catalog lists the model");
    let Some(connection) = connection else {
        return;
    };
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.add_plugins((
        AgentPlugin,
        SessionLogPlugin,
        EditTool,
        ShellTool,
        TuiPlugin,
    ))
    .add_plugins(RichToolRenderingPlugin)
    .add_systems(TranscriptRender, draw_probe.in_set(RenderSystems::Render))
    .init_resource::<Ended>()
    .init_resource::<Facts>()
    .add_observer(log_ended)
    .add_observer(note_facts);
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
        .map(|conversation| {
            conversation
                .messages()
                .iter()
                .map(|message| (**message).clone())
                .collect()
        })
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
    let facts = world.remove_resource::<Facts>().unwrap_or_default();
    assert_eq!(facts.0.len(), 3, "every call published what they did");
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
                let mut view = ToolCallView::new(call, result, CallState::Done);
                let known = facts.0.get(&call.id);
                view.facts = known.map(|(published, _)| published);
                view.took = known.map(|(_, took)| *took);
                screen.extend(drawn(world, &view));
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
    // The failed command's header says how it ended.
    assert!(
        has(&|line| line.contains("$ exit 3") && line.ends_with("· exit 3")),
        "{screen:#?}"
    );
    assert!(
        !has(&|line| line.contains("old_text") || line.contains("{\"")),
        "{screen:#?}"
    );

    // Another plugin's renderer is used; a tool without one looks as by default.
    let probe = call("probe", serde_json::Map::new());
    let other = call("other", parse_partial_arguments(r#"{"x": 1}"#));
    let (Some(probe), Some(other)) = (probe, other) else {
        return;
    };
    let probe = drawn(world, &ToolCallView::new(&probe, None, CallState::Done));
    assert!(probe.concat().contains("probe says hi"));
    let other = drawn(world, &ToolCallView::new(&other, None, CallState::Done));
    assert!(other.concat().contains(r#"other {"x":1}"#));

    // The command that printed 100k lines: the tool's note of what it
    // cut, then every line it kept, its first and its last, with a mark
    // where lines are missing, then how it ended.
    let at = |wanted: &str| screen.iter().position(|line| line.contains(wanted));
    let (note, end) = (at("of 100000 lines cut"), at("exit 0 · "));
    let kept = note
        .zip(end)
        .and_then(|(note, end)| screen.get(note + 1..end));
    let kept: Vec<&str> = kept
        .unwrap_or_default()
        .iter()
        .map(|line| line.trim())
        .collect();
    let gap = kept
        .iter()
        .any(|line| line.starts_with("[…") && line.contains("lines cut"));
    assert!(
        kept.first() == Some(&"1") && kept.last() == Some(&"100000") && gap,
        "{kept:?}"
    );

    // While a command runs, its output so far.
    let Some(running) = call("shell", parse_partial_arguments(r#"{"command": "make"}"#)) else {
        return;
    };
    let mut view = ToolCallView::new(&running, None, CallState::Running(Some("c".into())));
    view.output = Some("a\nb\nc\n");
    let shown = drawn(world, &view);
    let shown: Vec<&str> = shown.iter().map(|line| line.trim()).collect();
    assert!(shown.ends_with(&["a", "b", "c"]), "{shown:?}");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_streaming_write_shows_its_path_before_its_last_chunk() {
    let mut app = App::new();
    app.add_plugins(RichToolRenderingPlugin);
    app.finish();
    let arguments = r#"{"path": "big.txt", "content": "one\ntwo\nthree\nfour\nfive\n"}"#;
    let length = arguments.chars().count();
    // The call as the view has it after each of 10 chunks.
    let seen = (1..=10).position(|chunks| {
        let so_far: String = arguments.chars().take(chunks * length / 10).collect();
        let Some(call) = call("write", parse_partial_arguments(&so_far)) else {
            return false;
        };
        let view = ToolCallView::new(&call, None, CallState::Streaming);
        drawn(app.world_mut(), &view)
            .concat()
            .contains("write big.txt")
    });
    assert!(seen.is_some_and(|chunk| chunk < 9), "{seen:?}");
}
