use rig_core::message::{ToolFunction, ToolName};
use rig_core::tool::{KeptEnd, OutputCut, ToolContext};
use serde_json::{Value, json};

use super::*;

/// The lines of a `tool` call with `args` that answered `output`, cut as
/// `cut` says, as the default look draws it.
fn drawn(tool: &str, args: Value, output: &str, cut: Option<OutputCut>) -> Vec<String> {
    let Ok(name) = ToolName::new(tool) else {
        return Vec::new();
    };
    let call = ToolCall::from_wire("c1", ToolFunction::new(name, args));
    let result = call.answer(&rig_core::tool::ToolResult::success(
        output.to_owned().into(),
    ));
    let mut context = ToolContext::default();
    if let Some(cut) = cut {
        context.insert_result(cut).ok();
    }
    let facts = context.result_context();
    let mut view = ToolCallView::new(&call, Some(&result), CallState::Done);
    view.facts = Some(&facts);
    let lines = view.default_lines();
    lines.iter().map(ToString::to_string).collect()
}

#[test]
fn a_result_is_drawn_whole_with_its_cut_marker_on_its_side() {
    // `inspect` answers one line of JSON, cut and kept whole in a file.
    let cut = "[Cut at 16384 of 113000 bytes; all of it is in /s/spill/inspect-1.txt: read or \
               search it]";
    let answer = format!("{{\"value\":\"{}\"}}\n{cut}", "x".repeat(16_000));
    let kept = |kept, kept_lines, total_lines| OutputCut {
        kept,
        kept_lines,
        total_lines,
        spill: None,
    };
    let args = json!({ "method": "world.query" });
    let inspect = drawn("inspect", args, &answer, Some(kept(KeptEnd::Start, 1, 1)));
    assert_eq!(inspect.len(), 3, "{inspect:?}");
    assert!(inspect.iter().any(|line| line.chars().count() > 16_000));
    assert_eq!(inspect.last(), Some(&format!("   {cut}")));
    // A command's cut marker comes first, then all of the output it kept.
    let marker = "[298000 of 300000 lines cut, but for lines with errors or failures; all of \
                  the output is in /s/spill/shell-1.txt: read or search it]";
    let numbers: Vec<String> = (298_001..=300_000).map(|n| n.to_string()).collect();
    let output = format!("{marker}\n{}\n", numbers.join("\n"));
    let args = json!({ "command": "seq 1 300000" });
    let shell = drawn(
        "shell",
        args,
        &output,
        Some(kept(KeptEnd::Ends, 2000, 300_000)),
    );
    assert_eq!(shell.get(1), Some(&format!("   {marker}")), "{shell:?}");
    assert_eq!(shell.len(), 2002);
    assert_eq!(shell.last().map(String::as_str), Some("   300000"));
}

#[test]
fn a_calls_row_is_tinted_by_where_it_stands_and_has_no_marker() {
    let Ok(name) = ToolName::new("shell") else {
        return;
    };
    let call = ToolCall::from_wire("c1", ToolFunction::new(name, json!({})));
    let done = call.answer(&rig_core::tool::ToolResult::success("ok".to_owned().into()));
    let mut failed = done.clone();
    failed.is_error = true;
    let theme = Theme::default();
    let header = |result, state| {
        let mut view = ToolCallView::new(&call, result, state);
        view.theme = theme;
        view.header("shell", "ls")
    };
    let rows = [
        (header(None, CallState::Waiting), theme.call_waiting),
        (header(None, CallState::Running(None)), theme.call_running),
        (header(Some(&done), CallState::Done), theme.call_done),
        (header(Some(&failed), CallState::Done), theme.call_failed),
    ];
    for (row, band) in &rows {
        assert_eq!(row.style.bg, Some(*band), "{row}");
        assert_eq!(
            row.to_string().trim_start().split(' ').next(),
            Some("shell")
        );
    }
    let said = |index: usize| rows.get(index).map(|(row, _)| row.to_string());
    assert!(said(1).is_some_and(|row| row.ends_with("· running")));
    assert!(said(3).is_some_and(|row| row.ends_with("· failed")));
}

#[test]
fn a_command_that_exits_non_zero_is_a_failed_call() {
    use rig_core::tool::ExitStatus;

    let Ok(name) = ToolName::new("shell") else {
        return;
    };
    let call = ToolCall::from_wire("c1", ToolFunction::new(name, json!({})));
    let result = call.answer(&rig_core::tool::ToolResult::success(
        "no such file".to_owned().into(),
    ));
    let header = |status: ExitStatus| {
        let mut context = ToolContext::default();
        context.insert_result(status).ok();
        let facts = context.result_context();
        let mut view = ToolCallView::new(&call, Some(&result), CallState::Done);
        view.facts = Some(&facts);
        (view.failed(), view.header("ls", ""))
    };
    let theme = Theme::default();
    let (failed, row) = header(ExitStatus::Exited(2));
    assert!(failed);
    assert_eq!(row.style.bg, Some(theme.call_failed));
    assert!(row.to_string().ends_with("· exit 2"), "{row}");
    let (failed, row) = header(ExitStatus::Exited(0));
    assert!(!failed);
    assert_eq!(row.style.bg, Some(theme.call_done));
}

#[derive(Resource, Default)]
struct RequestsSeen(Vec<(Option<Entity>, u16, Vec<RenderKey>)>);

fn note_requests(batch: Res<RenderBatch>, mut seen: ResMut<RequestsSeen>) {
    seen.0.push((
        batch.pane,
        batch.width,
        batch.requests.keys().cloned().collect(),
    ));
}

#[test]
fn unchanged_messages_do_not_enter_the_render_schedule_again() {
    use rig_core::completion::Message;
    use rig_ecs::agent::Conversation;

    let (mut app, agent) = crate::tests::view_app();
    app.insert_resource(crate::TuiTestScreen::new(80, 24))
        .init_resource::<RequestsSeen>()
        .add_systems(TranscriptRender, note_requests.in_set(RenderSystems::Edit));
    if let Some(mut conversation) = app.world_mut().get_mut::<Conversation>(agent) {
        for n in 0..500 {
            conversation.push(Message::assistant(format!("message {n}")));
        }
    }
    app.update();
    let first: usize = app
        .world()
        .resource::<RequestsSeen>()
        .0
        .iter()
        .map(|(_, _, keys)| keys.len())
        .sum();
    assert!(first >= 500);
    app.world_mut().resource_mut::<RequestsSeen>().0.clear();
    app.world_mut().write_message(crate::RequestRedraw);
    app.update();
    assert!(app.world().resource::<RequestsSeen>().0.is_empty());
    if let Some(mut conversation) = app.world_mut().get_mut::<Conversation>(agent) {
        conversation.push(Message::assistant("new tail"));
    }
    app.update();
    let added: usize = app
        .world()
        .resource::<RequestsSeen>()
        .0
        .iter()
        .map(|(_, _, keys)| keys.len())
        .sum();
    assert!(
        (1..=3).contains(&added),
        "requested {added} items after appending one message"
    );
    use rig_core::streaming::{Part, PartKind, StreamEvent};
    use rig_ecs::agent::{CallOf, Partial, TurnOf};
    use rig_ecs::turn::StreamDelta;
    let turn = app.world_mut().spawn(TurnOf(agent)).id();
    let effect = app
        .world_mut()
        .spawn((CallOf(turn), Partial::default()))
        .id();
    let Ok(name) = ToolName::new("probe") else {
        return;
    };
    let part = Part::new(0);
    for event in [
        StreamEvent::Start {
            part,
            kind: PartKind::ToolCall,
            name: Some(name),
        },
        StreamEvent::Arguments {
            part,
            json: r#"{"path":"a"}"#.to_owned(),
        },
    ] {
        app.world_mut().resource_mut::<RequestsSeen>().0.clear();
        app.world_mut().write_message(StreamDelta {
            effect,
            agent: Some(agent),
            event,
        });
        app.world_mut().write_message(crate::RequestRedraw);
        app.update();
        let seen = &app.world().resource::<RequestsSeen>().0;
        let keys: Vec<&RenderKey> = seen.iter().flat_map(|(_, _, keys)| keys).collect();
        assert_eq!(
            keys.iter()
                .filter(|key| matches!(key, RenderKey::Streaming(_)))
                .count(),
            1
        );
        assert!(
            keys.iter()
                .all(|key| matches!(key, RenderKey::Streaming(_) | RenderKey::Aside(_))),
            "unchanged messages entered the batch: {keys:?}"
        );
    }
    app.world_mut().resource_mut::<RequestsSeen>().0.clear();
    app.world_mut().write_message(crate::RequestRedraw);
    app.update();
    assert!(app.world().resource::<RequestsSeen>().0.is_empty());
}

#[derive(Resource)]
struct Label(&'static str);

fn label_tools(mut batch: ResMut<RenderBatch>, label: Res<Label>) {
    for (_, output) in batch.tools_mut("probe") {
        output.lines = vec![Line::from(label.0)];
    }
}

#[test]
fn saved_panes_and_exports_use_the_same_systems_with_their_own_widths() {
    use crate::{
        DrawnTranscript, Placement, TranscriptLayout, TranscriptPane, TranscriptSource,
        TranscriptView, TuiPanel, TuiTestScreen,
    };
    use ratatui::layout::Constraint;
    use rig_core::completion::{AssistantContent, AssistantMessage, Message};
    use rig_ecs::agent::Conversation;

    let (mut app, agent) = crate::tests::view_app();
    let Ok(name) = ToolName::new("probe") else {
        return;
    };
    let call = ToolCall::from_wire("saved-call", ToolFunction::new(name, json!({})));
    let message = Message::Assistant(AssistantMessage::new(vec![AssistantContent::ToolCall(
        call,
    )]));
    let mut saved = rig_core::conversation::Conversation::default();
    saved.push(message.clone());
    if let Some(mut conversation) = app.world_mut().get_mut::<Conversation>(agent) {
        conversation.push(message);
    }
    app.insert_resource(TuiTestScreen::new(100, 30))
        .insert_resource(Label("first label"))
        .init_resource::<RequestsSeen>()
        .add_systems(TranscriptRender, label_tools.in_set(RenderSystems::Render))
        .add_systems(TranscriptRender, note_requests.in_set(RenderSystems::Edit));
    let pane = app
        .world_mut()
        .spawn((
            TranscriptPane::new(TranscriptSource::Value(saved)),
            TuiPanel::new(Placement::Left(Constraint::Length(31))),
            DrawnTranscript::default(),
        ))
        .id();
    app.update();
    let seen = &app.world().resource::<RequestsSeen>().0;
    assert!(
        seen.iter()
            .any(|(of, width, _)| *of == Some(pane) && *width == 31)
    );
    assert!(
        seen.iter()
            .any(|(of, width, _)| *of != Some(pane) && *width != 31)
    );
    let drawn = app.world().get::<DrawnTranscript>(pane);
    assert!(drawn.is_some_and(|drawn| {
        drawn
            .rows()
            .iter()
            .any(|row| row.text.contains("first label"))
    }));
    let export = TranscriptLayout::lay_out(app.world_mut(), &TranscriptView::default(), agent, 47);
    assert!(export.is_some_and(|drawn| {
        drawn
            .rows()
            .iter()
            .any(|row| row.text.contains("first label"))
    }));
    assert!(
        app.world()
            .resource::<RequestsSeen>()
            .0
            .iter()
            .any(|(pane, width, _)| pane.is_none() && *width == 47)
    );
    app.world_mut().resource_mut::<Label>().0 = "changed label";
    app.world_mut().resource_mut::<RenderRevision>().0 += 1;
    app.update();
    let drawn = app.world().get::<DrawnTranscript>(pane);
    assert!(drawn.is_some_and(|drawn| {
        drawn
            .rows()
            .iter()
            .any(|row| row.text.contains("changed label"))
    }));
}

#[derive(SystemSet, Clone, Debug, PartialEq, Eq, Hash)]
struct OriginalLook;

#[test]
fn a_later_system_can_replace_lines_and_edit_named_decorations() {
    use crate::{TranscriptItem, TranscriptLayout, TranscriptView};
    use rig_core::completion::{AssistantContent, AssistantMessage, Message};
    use rig_ecs::agent::Conversation;

    let (mut app, agent) = crate::tests::view_app();
    let Ok(name) = ToolName::new("probe") else {
        return;
    };
    let call = ToolCall::from_wire("call", ToolFunction::new(name, json!({})));
    let id = call.id.clone();
    if let Some(mut conversation) = app.world_mut().get_mut::<Conversation>(agent) {
        conversation.push(Message::Assistant(AssistantMessage::new(vec![
            AssistantContent::ToolCall(call),
        ])));
    }
    app.insert_resource(Label("original"))
        .add_systems(
            TranscriptRender,
            label_tools
                .in_set(RenderSystems::Render)
                .in_set(OriginalLook),
        )
        .add_systems(
            TranscriptRender,
            (|mut batch: ResMut<RenderBatch>| {
                for (_, output) in batch.tools_mut("probe") {
                    output.lines = vec![Line::from("replacement")];
                }
            })
            .in_set(RenderSystems::Render)
            .after(OriginalLook),
        )
        .add_systems(
            TranscriptRender,
            (|mut batch: ResMut<RenderBatch>| {
                for (_, output) in batch.calls_mut() {
                    output.decorate("note", vec![Line::from("before edit")]);
                }
            })
            .in_set(RenderSystems::Decorate),
        )
        .add_systems(
            TranscriptRender,
            (|mut batch: ResMut<RenderBatch>| {
                for (_, output) in batch.calls_mut() {
                    for (_, lines) in &mut output.decorations {
                        *lines = vec![Line::from("edited note")];
                    }
                }
            })
            .in_set(RenderSystems::Edit),
        );
    let drawn = TranscriptLayout::lay_out(app.world_mut(), &TranscriptView::default(), agent, 80);
    assert!(drawn.is_some());
    let Some(drawn) = drawn else {
        return;
    };
    assert!(drawn.rows().iter().any(|row| row.text == "replacement"));
    let rows = drawn.rows_of(&TranscriptItem::Decoration(id, "note".into()));
    assert!(rows.is_some_and(|rows| drawn.rows().get(rows).is_some_and(
        |rows| rows.len() == 1 && rows.first().is_some_and(|row| row.text == "edited note")
    )));
}

fn label_by_argument(mut batch: ResMut<RenderBatch>) {
    for (call, output) in batch.tools_mut("probe") {
        let n = call.call.function.arguments.get("n").cloned();
        output.lines = vec![Line::from(format!("probe {}", n.unwrap_or_default()))];
    }
}

#[test]
fn calls_sharing_a_provider_id_each_get_their_own_look() {
    use crate::{TranscriptLayout, TranscriptView};
    use rig_core::completion::{AssistantContent, AssistantMessage, Message};
    use rig_ecs::agent::Conversation;

    let (mut app, agent) = crate::tests::view_app();
    let Ok(name) = ToolName::new("probe") else {
        return;
    };
    app.add_systems(
        TranscriptRender,
        label_by_argument.in_set(RenderSystems::Render),
    );
    for n in [1, 2] {
        let call = ToolCall::from_wire("same", ToolFunction::new(name.clone(), json!({ "n": n })));
        let message = Message::Assistant(AssistantMessage::new(vec![AssistantContent::ToolCall(
            call,
        )]));
        if let Some(mut conversation) = app.world_mut().get_mut::<Conversation>(agent) {
            conversation.push(message);
        }
    }
    let export = TranscriptLayout::lay_out(app.world_mut(), &TranscriptView::default(), agent, 60);
    let rows: Vec<String> = export
        .iter()
        .flat_map(|drawn| drawn.rows().iter().map(|row| row.text.trim().to_owned()))
        .collect();
    let probes: Vec<&str> = rows
        .iter()
        .map(String::as_str)
        .filter(|row| row.starts_with("probe"))
        .collect();
    assert_eq!(probes, ["probe 1", "probe 2"], "{rows:#?}");
}
