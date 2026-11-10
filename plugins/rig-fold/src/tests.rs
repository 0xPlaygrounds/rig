use rig_core::conversation::pairing::CallRef;
use rig_core::message::{ToolCall, ToolFunction, ToolName};
use rig_core::tool::{ExitStatus, ToolContext};
use rig_harness::main_agent::ShownAgent;
use rig_tui::{
    KeyFocus, RenderInput, RenderKey, RenderOutput, RenderRequest, ToolCallView, TranscriptPane,
    TranscriptSource, TuiViewPlugin,
};
use serde_json::json;

use super::*;

/// The app the terminal view runs in, with `settings`, the plugin added
/// after `before`, and its main pane, which shows the shown agent.
fn app_with(settings: &str, before: impl FnOnce(&mut App)) -> (App, Entity) {
    let mut app = App::new();
    app.insert_resource(Settings::from_toml(settings))
        .add_plugins(TuiViewPlugin);
    before(&mut app);
    app.add_plugins(FoldPlugin);
    let source = TranscriptSource::Shown;
    let pane = TranscriptPane {
        source,
        title: None,
    };
    let pane = app.world_mut().spawn((MainPane, pane)).id();
    (app, pane)
}

fn app(settings: &str) -> (App, Entity) {
    app_with(settings, |_| {})
}

/// The call `index` of the reply `m1`.
fn item(index: u32) -> Option<CallItem> {
    let reply = "m1".parse().ok()?;
    Some(CallItem::Reply(CallRef { reply, index }))
}

/// How a test call ended.
#[derive(Clone, Copy)]
enum End {
    /// It answered.
    Done,
    /// The tool answered with an error.
    Error,
    /// Its command exited with this code, the answer no error.
    Exit(i32),
    /// A stop answered it.
    Stopped,
}

/// A `shell` call, the call `index`, whose result is the numbers from 1 to
/// `count`, ended as `end` says.
fn shell(index: u32, count: usize, end: End) -> Option<RenderRequest> {
    let name = ToolName::new("shell").ok()?;
    let call = ToolCall::from_wire(
        format!("c{index}"),
        ToolFunction::new(name, json!({ "command": format!("seq {count}") })),
    );
    let numbers: Vec<String> = (1..=count).map(|n| n.to_string()).collect();
    let mut result = call.answer(&rig_core::tool::ToolResult::success(
        numbers.join("\n").into(),
    ));
    result.is_error = matches!(end, End::Error | End::Stopped);
    let state = match end {
        End::Stopped => CallState::Stopped,
        _ => CallState::Done,
    };
    let mut context = ToolContext::default();
    if let End::Exit(code) = end {
        context.insert_result(ExitStatus::Exited(code)).ok()?;
    }
    let facts = context.result_context();
    let mut view = ToolCallView::new(&call, Some(&result), state.clone());
    view.facts = Some(&facts);
    let lines = view.default_lines();
    let input = ToolRender {
        call: call.clone(),
        result: Some(result),
        state,
        theme: Default::default(),
        facts: Some(facts),
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

/// The rows of each request after the render systems ran over them in the
/// batch of `pane` showing `agent`; none for an export's, with no pane.
fn drawn(
    app: &mut App,
    (pane, agent): (Option<Entity>, Option<Entity>),
    requests: &[Option<RenderRequest>],
) -> Vec<Vec<String>> {
    let mut batch = RenderBatch {
        pane,
        agent,
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

/// Selects `selected` in `pane`.
fn select(app: &mut App, pane: Entity, selected: Option<CallItem>) {
    if let Some(mut view) = app.world_mut().get_mut::<TranscriptView>(pane) {
        view.selected = selected.map(TranscriptItem::Call);
    }
}

/// Presses the plugin's key, with `selected` selected in the main pane.
fn press(app: &mut App, selected: Option<CallItem>) {
    let world = app.world_mut();
    let mut panes = world.query_filtered::<&mut TranscriptView, With<MainPane>>();
    for mut view in panes.iter_mut(world) {
        view.selected = selected.map(TranscriptItem::Call);
    }
    press_key(app);
}

/// Presses the plugin's key.
fn press_key(app: &mut App) {
    let world = app.world_mut();
    let mut bindings = world.query::<(Entity, &KeyBinding)>();
    let binding = bindings
        .iter(world)
        .find(|(_, binding)| binding.label == LABEL);
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
    let (mut app, pane) = app("");
    let calls = [shell(0, 400, End::Done), shell(1, 13, End::Done)];
    let drawn = drawn(&mut app, (Some(pane), None), &calls);
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
    let (mut app, pane) = app("[fold]\nhead = 1\ntail = 2\nkey = \"alt+o\"");
    let calls = [shell(0, 50, End::Done), shell(1, 50, End::Done)];
    let lengths = |app: &mut App| -> Vec<usize> {
        let drawn = drawn(app, (Some(pane), None), &calls);
        drawn.iter().map(Vec::len).collect()
    };
    let folded = drawn(&mut app, (Some(pane), None), &calls);
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

/// Message ids rise from 0 in each conversation, so two agents' calls can
/// have the same place: showing one whole leaves the other folded.
#[test]
fn a_call_shown_whole_is_the_shown_agents_alone() {
    let (mut app, pane) = app("");
    let world = app.world_mut();
    let (first, second) = (world.spawn_empty().id(), world.spawn_empty().id());
    world.insert_resource(ShownAgent(Some(first)));
    let call = [shell(0, 400, End::Done)];
    let length = |app: &mut App, agent| {
        let drawn = drawn(app, (Some(pane), Some(agent)), &call);
        drawn.first().map_or(0, Vec::len)
    };
    press(&mut app, item(0));
    assert_eq!(length(&mut app, first), 401);
    assert_eq!(length(&mut app, second), 14);
}

#[test]
fn a_failed_or_stopped_call_and_an_export_keep_every_line() {
    let (mut app, pane) = app("");
    for end in [End::Error, End::Exit(3), End::Stopped] {
        let drawn = drawn(&mut app, (Some(pane), None), &[shell(0, 400, end)]);
        let drawn = drawn.first().map(Vec::as_slice).unwrap_or_default();
        assert_eq!(drawn.len(), 401, "{:?}", drawn.first());
        assert_eq!(drawn.last().map(String::as_str), Some("   400"));
    }
    let export = drawn(&mut app, (None, None), &[shell(0, 400, End::Done)]);
    assert_eq!(export.first().map(Vec::len), Some(401));
}

/// The key acts in the transcript pane that has the keys, such as a
/// helper's beside the main one, and falls back to the main pane.
#[test]
fn the_key_acts_in_the_focused_pane() {
    let (mut app, main) = app("");
    let world = app.world_mut();
    let helper = world.spawn_empty().id();
    let side = TranscriptPane {
        source: TranscriptSource::Agent(helper),
        title: None,
    };
    let side = world.spawn(side).id();
    world.insert_resource(KeyFocus(Some(side)));
    select(&mut app, main, item(0));
    select(&mut app, side, item(0));
    press_key(&mut app);
    let call = [shell(0, 400, End::Done)];
    let length = |app: &mut App, shown| drawn(app, shown, &call).first().map_or(0, Vec::len);
    assert_eq!(length(&mut app, (Some(side), Some(helper))), 401);
    assert_eq!(length(&mut app, (Some(main), None)), 14);
    // With no pane focused, the main pane's selection flips.
    app.world_mut().insert_resource(KeyFocus(None));
    press_key(&mut app);
    assert_eq!(length(&mut app, (Some(main), None)), 401);
}

/// With its key taken by another binding the plugin folds nothing, as the
/// lines would have no way back: a binding of any scope bound first, or an
/// input key, which the fold key would take from the input unseen.
#[test]
fn without_its_key_nothing_folds() {
    let taken = |app: &mut App| {
        let binding = KeyBinding::new(KeyPress::ctrl('o'), "Go on");
        app.add_key_binding(binding, |_: On<KeyPressed>| {});
    };
    let call = [shell(0, 400, End::Done)];
    let (mut first, pane) = app_with("", taken);
    let drawn_taken = drawn(&mut first, (Some(pane), None), &call);
    assert_eq!(drawn_taken.first().map(Vec::len), Some(401));
    // Ctrl+E is the input's end of the line.
    let (mut app, pane) = app("[fold]\nkey = \"ctrl+e\"");
    let drawn_input = drawn(&mut app, (Some(pane), None), &call);
    assert_eq!(drawn_input.first().map(Vec::len), Some(401));
}
