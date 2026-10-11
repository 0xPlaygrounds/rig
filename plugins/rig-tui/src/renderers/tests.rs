use bevy_ecs::entity_disabling::Disabled;
use rig_core::message::{ToolFunction, ToolName};
use serde_json::{Value, json};

use super::*;

/// The lines of a `tool` call with `args` that answered `output`, as the
/// default look draws it, or for `shell`, the end of its output as its
/// look shows it.
fn drawn(tool: &str, args: Value, output: &str) -> Vec<String> {
    let Ok(name) = ToolName::new(tool) else {
        return Vec::new();
    };
    let call = ToolCall::from_wire("c1", ToolFunction::new(name, args));
    let result = call.answer(&rig_core::tool::ToolResult::success(
        output.to_owned().into(),
    ));
    let view = ToolCallView {
        call: &call,
        result: Some(&result),
    };
    let lines = match tool {
        "shell" => view.tail_lines(RESULT_LINES + 2),
        _ => view.default_lines(),
    };
    lines.iter().map(ToString::to_string).collect()
}

#[test]
fn a_long_result_is_shortened_and_its_cut_marker_stays_in_sight() {
    // `inspect` answers one line of JSON, cut and kept whole in a file.
    let cut = "[Cut at 16384 of 113000 bytes; all of it is in /s/spill/inspect-1.txt: read or \
               search it]";
    let answer = format!("{{\"value\":\"{}\"}}\n{cut}", "x".repeat(16_000));
    let inspect = drawn("inspect", json!({ "method": "world.query" }), &answer);
    assert_eq!(inspect.len(), 3, "{inspect:?}");
    assert!(inspect.iter().all(|line| line.chars().count() < 300));
    assert_eq!(inspect.last(), Some(&format!("    {cut}")));

    // A shell command's cut marker comes first, then the end of its output.
    let marker = "[298000 earlier lines cut; all of the output is in /s/spill/shell-1.txt: read \
                  or search it]";
    let numbers: Vec<String> = (298_001..=300_000).map(|n| n.to_string()).collect();
    let output = format!("{marker}\n{}\n", numbers.join("\n"));
    let shell = drawn("shell", json!({ "command": "seq 1 300000" }), &output);
    let shown = [&format!("  ⎿ {marker}"), "    … 1994 earlier lines"];
    assert!(shell.starts_with(&shown.map(str::to_owned)), "{shell:?}");
    assert_eq!(shell.len(), 8);
    assert_eq!(shell.last().map(String::as_str), Some("    300000"));
}

#[test]
fn a_tool_keeps_its_first_renderer_until_that_one_is_disabled() {
    let Ok(name) = ToolName::new("t") else {
        return;
    };
    let call = ToolCall::from_wire("c1", ToolFunction::new(name, json!({})));
    let view = ToolCallView {
        call: &call,
        result: None,
    };
    let drawn = |app: &mut App| -> Vec<String> {
        let world = app.world_mut();
        let mut renderers = world.query::<&ToolRenderer>();
        let lines = renderers
            .iter(world)
            .flat_map(|renderer| (renderer.render)(&view));
        lines.map(|line| line.to_string()).collect()
    };
    let look = |text: &'static str| move |_: &ToolCallView<'_>| vec![Line::from(text)];
    let mut app = App::new();
    app.add_tool_renderer("t", look("first"))
        .add_tool_renderer("t", look("second"));
    assert_eq!(drawn(&mut app), ["first"]);
    let world = app.world_mut();
    let first: Vec<Entity> = world
        .query_filtered::<Entity, With<ToolRenderer>>()
        .iter(world)
        .collect();
    for renderer in first {
        world.entity_mut(renderer).insert(Disabled);
    }
    app.add_tool_renderer("t", look("second"));
    assert_eq!(drawn(&mut app), ["second"]);
}
