use rig_coding_tools::{EditTool, ReadTool, SearchTool, ShellTool, WriteTool};
use rig_core::completion::CompletionRequest;
use rig_core::effect::EffectKind;
use rig_core::test_utils::MockCompletionModel;
use rig_harness_test_support::{
    Collected, MemoryStore, ToolOutputs, app_on, calls, collect, log_tool_outputs, reply,
    run_until, scripted_models,
};
use rig_telemetry::UsagePlugin;
use serde_json::json;

use super::*;

/// A directory of its own for this test, beside the test binary.
fn scratch() -> PathBuf {
    let exe = std::env::current_exe().unwrap_or_default();
    let dir = exe
        .parent()
        .map(|deps| deps.join("corpus-oc-architect-editor-test"))
        .unwrap_or_default();
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).ok();
    dir
}

fn tool_names(request: Option<&CompletionRequest>) -> Vec<String> {
    let tools = request.map(|request| request.tools.iter());
    let names = tools.into_iter().flatten();
    names.map(|tool| tool.name.as_str().to_owned()).collect()
}

fn output(app: &App, id: &str) -> String {
    let outputs = &app.world().resource::<ToolOutputs>().0;
    outputs.get(id).cloned().unwrap_or_default()
}

/// The architect plans and calls `implement`, the editor edits on its own
/// model without the architect's conversation, the rounds per turn are
/// capped, each model's usage is reported apart, and `/architect off`
/// gives the agent its model and tools back.
#[test]
fn the_architect_plans_and_the_editor_edits() {
    let dir = scratch();
    let lib = dir.join("lib.rs");
    std::fs::write(&lib, "pub fn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n").ok();
    let implement = |id: &str, description: &str| {
        calls(
            id,
            IMPLEMENT,
            json!({ "description": description, "files": ["lib.rs"] }),
        )
    };
    let architect = MockCompletionModel::from_stream_turns([
        calls("r1", "read", json!({ "path": "lib.rs" })),
        implement("i1", "rename add to sum"),
        reply("done"),
        implement("i2", "first"),
        implement("i3", "second"),
        implement("i4", "third"),
        implement("i5", "fourth"),
        reply("out of rounds"),
    ]);
    let edit = json!({
        "path": "lib.rs",
        "edits": [{ "old_text": "pub fn add(", "new_text": "pub fn sum(" }],
    });
    let editor = MockCompletionModel::from_stream_turns([
        calls("e1", "edit", edit),
        reply("applied"),
        reply("nothing to do"),
        reply("nothing to do"),
        reply("nothing to do"),
    ]);
    let main = MockCompletionModel::from_stream_turns([reply("back to normal")]);
    let models = [
        ("test/main", &main),
        ("test/architect", &architect),
        ("test/editor", &editor),
    ];
    let settings = "[architect]\narchitect = \"test/architect\"\neditor = \"test/editor\"";

    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.insert_resource(scripted_models(models))
        .insert_resource(Settings::from_toml(settings))
        .insert_resource(ProjectRoot(dir.clone()))
        .init_resource::<ToolOutputs>()
        .add_observer(log_tool_outputs)
        .add_plugins((AgentPlugin, JournalPlugin))
        .add_plugins((ReadTool, SearchTool, EditTool, WriteTool, ShellTool))
        .add_plugins((UsagePlugin, ArchitectEditorPlugin));
    collect::<Notice>(&mut app);
    collect::<EffectRecorded>(&mut app);
    let agent = app.world_mut().spawn(Agent).id();
    app.finish();
    app.update();
    let world = app.world_mut();
    world.trigger(SetModel {
        entity: agent,
        model: "test/main".to_owned(),
    });
    world.trigger(RunCommand::new(agent, "architect on".to_owned()));
    let asked = "Please rename it, ARCHITECT-ONLY-WORDS";
    world.trigger(Deliver::new(agent, asked, DeliveryMode::Queue));
    let idle_after = |model: &MockCompletionModel, count: usize| {
        let model = model.clone();
        move |world: &World| {
            model.request_count() == count && world.get::<ActiveTurn>(agent).is_none()
        }
    };
    run_until(&mut app, &wakes, idle_after(&architect, 3));

    // The architect may only read, and implement.
    let requests = architect.requests();
    let tools = tool_names(requests.first());
    assert!(tools.contains(&"read".to_owned()), "{tools:?}");
    assert!(tools.contains(&IMPLEMENT.to_owned()), "{tools:?}");
    for denied in ["edit", "write", "shell"] {
        assert!(!tools.contains(&denied.to_owned()), "{tools:?}");
    }
    // The editor got the description and the file, not the user's words.
    let first = format!("{:?}", editor.requests().first().map(|r| &r.chat_history));
    assert!(first.contains("rename add to sum"), "{first}");
    assert!(first.contains("pub fn add(a: i32"), "{first}");
    assert!(!first.contains("ARCHITECT-ONLY-WORDS"), "{first}");
    assert!(!tool_names(editor.requests().first()).contains(&IMPLEMENT.to_owned()));
    let written = std::fs::read_to_string(&lib).unwrap_or_default();
    assert!(written.contains("pub fn sum("), "{written}");
    let result = output(&app, "i1");
    assert!(result.contains("- lib.rs: +1 -1"), "{result}");
    assert!(result.contains("applied"), "{result}");
    // The editor's `edit` is recorded under the `implement` call, where
    // the terminal view nests it.
    let records = &app.world().resource::<Collected<EffectRecorded>>().0;
    let named = |name: &str| {
        records
            .iter()
            .find_map(|EffectRecorded(record)| match &record.kind {
                EffectKind::ToolCall { name: tool, .. } if tool == name => Some(record.id),
                _ => None,
            })
    };
    let parent = |id| {
        let record = records
            .iter()
            .find(|EffectRecorded(record)| record.id == id);
        record.and_then(|EffectRecorded(record)| record.parent)
    };
    let under = std::iter::successors(named("edit"), |&id| parent(id));
    assert!(under.into_iter().any(|id| Some(id) == named(IMPLEMENT)));

    // A fourth `implement` in one turn is refused.
    let again = Deliver::new(agent, "Now three tries", DeliveryMode::Queue);
    app.world_mut().trigger(again);
    run_until(&mut app, &wakes, idle_after(&architect, 8));
    assert!(output(&app, "i4").contains("nothing to do"));
    let limited = output(&app, "i5");
    assert!(limited.contains("round limit"), "{limited}");
    assert_eq!(editor.request_count(), 5);

    // `/usage` lists each model with its own count.
    app.world_mut()
        .trigger(RunCommand::new(agent, "usage".to_owned()));
    app.update();
    let notices = &app.world().resource::<Collected<Notice>>().0;
    let usage = notices
        .iter()
        .rev()
        .find(|notice| notice.text.contains("Per model"));
    let usage = usage.map(|notice| notice.text.clone()).unwrap_or_default();
    assert!(
        usage.contains("test/architect:") && usage.contains("8 calls"),
        "{usage}"
    );
    assert!(
        usage.contains("test/editor:") && usage.contains("5 calls"),
        "{usage}"
    );

    // Out of architect mode, the agent's own model and tools are back.
    app.world_mut()
        .trigger(RunCommand::new(agent, "architect off".to_owned()));
    app.world_mut()
        .trigger(Deliver::new(agent, "hello", DeliveryMode::Queue));
    run_until(&mut app, &wakes, idle_after(&main, 1));
    let tools = tool_names(main.requests().first());
    assert!(tools.contains(&"edit".to_owned()), "{tools:?}");
    assert!(!tools.contains(&IMPLEMENT.to_owned()), "{tools:?}");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(ArchitectEditorPlugin);
}
