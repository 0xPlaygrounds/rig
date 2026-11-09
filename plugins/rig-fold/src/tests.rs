use rig_core::conversation::pairing::CallRef;
use rig_core::message::{ToolCall, ToolFunction, ToolName};
use rig_tui::{RenderInput, RenderKey, RenderOutput, RenderRequest, ToolCallView, TuiViewPlugin};
use serde_json::json;

use super::*;

/// The app the terminal view runs in, with the plugin and `settings`.
fn app(settings: &str) -> App {
    let mut app = App::new();
    app.insert_resource(Settings::from_toml(settings))
        .add_plugins((TuiViewPlugin, FoldPlugin));
    app
}

/// The call `index` of the reply `m1`.
fn item(index: u32) -> Option<CallItem> {
    let reply = "m1".parse().ok()?;
    Some(CallItem::Reply(CallRef { reply, index }))
}

/// A finished `shell` call, the call `index`, whose result is the numbers
/// from 1 to `count`, an error when `failed`.
fn shell(index: u32, count: usize, failed: bool) -> Option<RenderRequest> {
    let name = ToolName::new("shell").ok()?;
    let call = ToolCall::from_wire(
        format!("c{index}"),
        ToolFunction::new(name, json!({ "command": format!("seq {count}") })),
    );
    let numbers: Vec<String> = (1..=count).map(|n| n.to_string()).collect();
    let mut result = call.answer(&rig_core::tool::ToolResult::success(
        numbers.join("\n").into(),
    ));
    result.is_error = failed;
    let lines = ToolCallView::new(&call, Some(&result), CallState::Done).default_lines();
    let input = ToolRender {
        call: call.clone(),
        result: Some(result),
        state: CallState::Done,
        theme: Default::default(),
        facts: None,
        took: None,
        output: None,
        children: Vec::new(),
        item: item(index),
    };
    let output = RenderOutput {
        lines,
        decorations: Vec::new(),
    };
    Some(RenderRequest {
        input: RenderInput::Tool(input),
        output,
    })
}

/// The rows of each request, by its key, after the render systems ran
/// over them in a pane's batch.
fn drawn(app: &mut App, requests: &[Option<RenderRequest>]) -> Vec<Vec<String>> {
    let pane = app.world_mut().spawn_empty().id();
    let mut batch = RenderBatch {
        pane: Some(pane),
        ..Default::default()
    };
    for (index, request) in (0..).zip(requests.iter().flatten()) {
        batch
            .requests
            .insert(RenderKey::Tool(1, index), request.clone());
    }
    app.insert_resource(batch);
    app.world_mut().run_schedule(TranscriptRender);
    let batch = app.world_mut().remove_resource::<RenderBatch>();
    let mut batch = batch.unwrap_or_default();
    (0..requests.len())
        .map(|index| {
            let request = batch.requests.remove(&RenderKey::Tool(1, index));
            let lines = request.map(|request| request.output.lines);
            lines.iter().flatten().map(ToString::to_string).collect()
        })
        .collect()
}

/// Presses the plugin's key, with `selected` selected in the main pane.
fn press(app: &mut App, selected: Option<CallItem>) {
    let world = app.world_mut();
    let mut panes = world.query_filtered::<Entity, With<MainPane>>();
    let pane = panes.iter(world).next();
    let pane = pane.unwrap_or_else(|| world.spawn((MainPane, TranscriptView::default())).id());
    if let Some(mut view) = world.get_mut::<TranscriptView>(pane) {
        view.selected = selected.map(TranscriptItem::Call);
    }
    let mut bindings = world.query::<(Entity, &KeyBinding)>();
    let binding = bindings
        .iter(world)
        .find(|(_, binding)| binding.label.starts_with("Show the selected tool call"));
    if let Some((entity, _)) = binding {
        world.trigger(KeyPressed {
            entity,
            agent: None,
            released: false,
        });
    }
}

#[test]
fn a_long_result_folds_to_its_head_a_marker_and_its_tail() {
    let mut app = app("");
    let drawn = drawn(&mut app, &[shell(0, 400, false), shell(1, 13, false)]);
    let (long, short) = (drawn.first(), drawn.get(1));
    let (long, short) = (
        long.cloned().unwrap_or_default(),
        short.cloned().unwrap_or_default(),
    );
    let numbers = |range: std::ops::RangeInclusive<usize>| range.map(|n| format!("   {n}"));
    let mut expected = vec![long.first().cloned().unwrap_or_default()];
    expected.extend(numbers(1..=4));
    expected.push("   … 388 lines · ctrl+o shows".to_owned());
    expected.extend(numbers(393..=400));
    assert_eq!(long, expected);
    assert!(
        long.first()
            .is_some_and(|header| header.contains("seq 400"))
    );
    // Folding 13 lines to 12 would hide one line behind a line.
    assert_eq!(short.len(), 14, "{short:?}");
}

#[test]
fn the_key_shows_every_call_or_the_selected_one_whole() {
    let mut app = app("[fold]\nhead = 1\ntail = 2\nkey = \"alt+o\"");
    let calls = [shell(0, 50, false), shell(1, 50, false)];
    let lengths =
        |app: &mut App| -> Vec<usize> { drawn(app, &calls).iter().map(Vec::len).collect() };
    let folded = drawn(&mut app, &calls);
    let marker = folded.first().and_then(|lines| lines.get(2));
    assert_eq!(
        marker.map(String::as_str),
        Some("   … 47 lines · alt+o shows")
    );
    assert_eq!(lengths(&mut app), [5, 5]);
    // Nothing selected: every call shows every line, and again folds them all.
    press(&mut app, None);
    assert_eq!(lengths(&mut app), [51, 51]);
    // The selected call folds alone, and shows whole again.
    press(&mut app, item(1));
    assert_eq!(lengths(&mut app), [51, 5]);
    press(&mut app, item(1));
    assert_eq!(lengths(&mut app), [51, 51]);
    press(&mut app, None);
    press(&mut app, item(0));
    assert_eq!(lengths(&mut app), [51, 5]);
}

#[test]
fn a_failed_call_and_an_export_keep_every_line() {
    let mut app = app("");
    let failed = drawn(&mut app, &[shell(0, 400, true)]);
    let failed = failed.first().map(Vec::as_slice).unwrap_or_default();
    assert_eq!(failed.len(), 401);
    assert_eq!(failed.last().map(String::as_str), Some("   400"));
    // An export's batch has no pane.
    let mut batch = RenderBatch::default();
    if let Some(request) = shell(0, 400, false) {
        batch.requests.insert(RenderKey::Tool(1, 0), request);
    }
    app.insert_resource(batch);
    app.world_mut().run_schedule(TranscriptRender);
    let batch = app.world_mut().remove_resource::<RenderBatch>();
    let request = batch.and_then(|mut batch| batch.requests.remove(&RenderKey::Tool(1, 0)));
    assert_eq!(request.map(|request| request.output.lines.len()), Some(401));
}
