use rig_coding_tools::{ReadTool, WriteTool};
use rig_core::test_utils::MockCompletionModel;
use rig_harness_test_support::MemoryStore;
use rig_harness_test_support::{
    ToolOutputs, app_on, calls, connect, first_agent, log_tool_outputs, reply, run_until,
    scratch_home,
};
use serde_json::json;

use super::*;

/// The spec's settings: `.txt` files are uppercased, and a line longer
/// than 10 characters is too long.
const SETTINGS: &str = r#"
[format_on_write.formatters]
txt = '''sh -c 'tr a-z A-Z < "$1" > "$1.up" && mv "$1.up" "$1"' sh'''

[format_on_write.linters]
txt = '''awk 'length > 10 { print "W: line too long" }' '''
"#;

/// The agent works in its own `WorkDir`, where its relative path goes.
#[test]
fn a_write_is_formatted_and_linted_before_the_next_read() {
    let dir = scratch_home("format-on-write").root().to_owned();
    std::fs::create_dir_all(&dir).ok();
    let file = "a.txt";
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.insert_resource(Settings::from_toml(SETTINGS))
        .add_plugins((
            AgentPlugin,
            SessionLogPlugin,
            ReadTool,
            WriteTool,
            FormatOnWritePlugin,
        ))
        .init_resource::<ToolOutputs>()
        .add_observer(log_tool_outputs);
    app.finish();
    app.update();
    let model = MockCompletionModel::from_stream_turns([
        calls(
            "w1",
            "write",
            json!({ "path": file, "content": "hello world wide" }),
        ),
        calls("r1", "read", json!({ "path": file })),
        reply("Done."),
    ]);
    let world = app.world_mut();
    let agent = first_agent(world).unwrap_or(Entity::PLACEHOLDER);
    if let Some(connection) = connect(&model) {
        world
            .entity_mut(agent)
            .insert((connection, WorkDir(dir.clone())));
    }
    world.trigger(Deliver::new(agent, "Write a.txt.", DeliveryMode::Steer));
    run_until(&mut app, &wakes, |world| {
        world.resource::<ToolOutputs>().0.contains_key("r1")
    });

    let outputs = &app.world().resource::<ToolOutputs>().0;
    let written = outputs.get("w1").cloned().unwrap_or_default();
    let read = outputs.get("r1").cloned().unwrap_or_default();
    assert!(written.contains("(formatted with sh)"), "{written}");
    assert!(written.contains("1\tHELLO WORLD WIDE"), "{written}");
    assert!(written.contains("W: line too long"), "{written}");
    assert!(read.starts_with("HELLO WORLD WIDE"), "{read}");
    let formatted = std::fs::read_to_string(dir.join(file)).unwrap_or_default();
    assert_eq!(formatted, "HELLO WORLD WIDE");
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(FormatOnWritePlugin);
}
