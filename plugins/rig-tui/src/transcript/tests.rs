use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;
use std::time::SystemTime;

use bevy_ecs::prelude::*;
use bevy_ecs::system::RunSystemOnce;
use ratatui::style::Style;
use ratatui::text::Line;
use rig_core::completion::Message;
use rig_core::message::{
    AssistantContent, AssistantMessage, CallId, ToolCall, ToolFunction, ToolName,
};
use rig_core::streaming::{PartKind, StreamEvent};
use rig_ecs::agent::Conversation;
use rig_ecs::agent::Partial;
use rig_ecs::prelude::{Entry, MessageId, Origin};
use rig_ecs::stage::{HoldOf, Started};
use rig_ecs::tools::ToolRequest;
use rig_ecs::turn::StreamDelta;
use rig_harness::prelude::App;
use rig_harness::prelude::ToolProgress;
use rig_harness::transcript::{EntryOf, TranscriptEntry};
use serde_json::json;

use super::{
    Below, CallFacts, EntryRenderers, Look, Part, Renderers, Scroll, StreamingCalls, Transcript,
};
use crate::render::CallStates;
use crate::tests::{drawn_rows, on_screen, press, view_app};
use crate::{
    AppToolRenderersExt, CallState, DrawnTranscript, Highlight, ItemRow, RenderToolCall, ScrollTo,
    ToolCallView, TranscriptItem, TranscriptView,
};

const WIDTH: u16 = 80;

#[test]
fn a_new_view_follows_the_bottom() {
    let mut scroll = Scroll::default();
    assert_eq!(scroll.place(&[30, 20], 10, WIDTH), (40, Below::Nothing));
    assert_eq!(scroll.place(&[30, 25], 10, WIDTH), (45, Below::Nothing));
}

#[test]
fn a_view_scrolled_up_stays_put_while_rows_arrive() {
    let mut scroll = Scroll::default();
    scroll.up(10);
    assert_eq!(scroll.place(&[30, 20], 10, WIDTH), (30, Below::More));
    // The streaming reply grows, and becomes a message of its own.
    assert_eq!(scroll.place(&[30, 28], 10, WIDTH), (30, Below::New));
    assert_eq!(scroll.place(&[30, 28, 5], 10, WIDTH), (30, Below::New));
}

#[test]
fn scrolling_back_to_the_bottom_follows_again() {
    let mut scroll = Scroll::default();
    scroll.up(10);
    assert_eq!(scroll.place(&[50], 10, WIDTH), (30, Below::More));
    scroll.down(4);
    assert_eq!(scroll.place(&[60], 10, WIDTH), (34, Below::New));
    scroll.down(100);
    assert_eq!(scroll.place(&[60], 10, WIDTH), (50, Below::Nothing));
    assert_eq!(scroll.place(&[70], 10, WIDTH), (60, Below::Nothing));
}

#[test]
fn scrolling_up_stops_at_the_top() {
    let mut scroll = Scroll::default();
    scroll.up(100);
    assert_eq!(scroll.place(&[50], 10, WIDTH), (0, Below::More));
}

#[test]
fn a_resize_keeps_the_same_part_on_top() {
    let mut scroll = Scroll::default();
    scroll.up(25);
    // Row 5 of the second part.
    assert_eq!(scroll.place(&[20, 10, 30], 10, WIDTH), (25, Below::More));
    // At half the width every part takes twice the rows.
    assert_eq!(
        scroll.place(&[40, 20, 60], 10, WIDTH / 2),
        (50, Below::More)
    );
}

#[test]
fn following_again_drops_the_held_row() {
    let mut scroll = Scroll::default();
    scroll.up(10);
    assert_eq!(scroll.place(&[50], 10, WIDTH), (30, Below::More));
    scroll.follow();
    assert_eq!(scroll.place(&[55], 10, WIDTH), (45, Below::Nothing));
}

/// The rows `transcript` draws of `conversation`'s first message with the
/// calls standing as `calls` say, their output so far `outputs`, drawn by
/// `renderers`.
fn drawn(
    transcript: &mut Transcript,
    conversation: &Conversation,
    (calls, outputs): (&BTreeMap<CallId, CallState>, &BTreeMap<CallId, String>),
    renderers: &Renderers<'_>,
) -> String {
    let entries = EntryRenderers::default();
    let (expanded, facts) = (HashSet::new(), CallFacts::default());
    let look = Look {
        renderers,
        entries: &entries,
        calls,
        outputs,
        inside: HashMap::new(),
        expanded: &expanded,
        facts: &facts,
    };
    // The conversation is the same each time.
    transcript.update(Entity::PLACEHOLDER, conversation, false, &look, WIDTH);
    let parts = [Part::Message(0)];
    let visible = transcript.visible(&parts, (WIDTH, 10), &mut Scroll::default(), &[]);
    let rows: Vec<String> = visible.rows.iter().map(ToString::to_string).collect();
    rows.join("\n")
}

/// A conversation of one reply that calls `call`.
fn calling(call: &ToolCall) -> Option<Conversation> {
    let reply = AssistantMessage::new(vec![AssistantContent::ToolCall(call.clone())]);
    let conversation = serde_json::from_value::<Conversation>(json!({
        "messages": [Message::Assistant(reply)],
        "origins": [],
    }));
    conversation.ok()
}

#[test]
fn a_call_is_drawn_again_as_its_holds_go_and_its_progress_moves() {
    let Ok(name) = ToolName::new("edit") else {
        return;
    };
    let call = ToolCall::from_wire("c1", ToolFunction::new(name, json!({ "path": "src/a.rs" })));
    let conversation = calling(&call);
    assert!(conversation.is_some());
    let Some(conversation) = conversation else {
        return;
    };
    let mut world = World::new();
    let effect = world.spawn(ToolRequest { call: call.clone() }).id();
    let lease = "waiting for lease on src/a.rs";
    let hold = world.spawn((Name::new(lease), HoldOf(effect))).id();
    let states = |world: &mut World| {
        let states = world.run_system_once(move |calls: CallStates| calls.of([effect]));
        states.unwrap_or_default()
    };
    let held = states(&mut world);
    let names = vec![lease.to_owned()];
    assert_eq!(held.get(&call.id), Some(&CallState::Held(names)));
    let mut transcript = Transcript::default();
    let none = (BTreeMap::new(), Renderers::default());
    let shown = drawn(&mut transcript, &conversation, (&held, &none.0), &none.1);
    assert!(shown.contains(&format!("held: {lease}")), "{shown}");

    world.despawn(hold);
    let waiting = states(&mut world);
    assert_eq!(waiting.get(&call.id), Some(&CallState::Waiting));
    let shown = drawn(&mut transcript, &conversation, (&waiting, &none.0), &none.1);
    assert!(!shown.contains("held:"), "{shown}");

    let started = Started {
        at: SystemTime::now(),
    };
    let progress = ToolProgress("12 of 40 files".to_owned());
    world.entity_mut(effect).insert((started, progress));
    let running = states(&mut world);
    let line = Some("12 of 40 files".to_owned());
    assert_eq!(running.get(&call.id), Some(&CallState::Running(line)));
    let shown = drawn(&mut transcript, &conversation, (&running, &none.0), &none.1);
    assert!(shown.contains("⎿ 12 of 40 files"), "{shown}");
    world
        .entity_mut(effect)
        .insert(ToolProgress("13 of 40 files".to_owned()));
    let states = states(&mut world);
    let shown = drawn(&mut transcript, &conversation, (&states, &none.0), &none.1);
    assert!(shown.contains("⎿ 13 of 40 files"), "{shown}");
}

#[test]
fn a_running_calls_renderer_gets_all_its_output_so_far() {
    let Ok(name) = ToolName::new("shell") else {
        return;
    };
    let call = ToolCall::from_wire("c1", ToolFunction::new(name, json!({ "command": "make" })));
    let conversation = calling(&call);
    assert!(conversation.is_some());
    let Some(conversation) = conversation else {
        return;
    };
    let look: Arc<RenderToolCall> = Arc::new(|call: &ToolCallView<'_>| {
        let output = call.output.unwrap_or("no output").lines();
        output.map(|line| Line::from(format!("> {line}"))).collect()
    });
    let renderers: Renderers<'_> = HashMap::from([("shell", &look)]);
    let calls = BTreeMap::from([(call.id.clone(), CallState::Running(None))]);
    let mut transcript = Transcript::default();
    let mut outputs = BTreeMap::new();
    for line in ["cc a.c", "cc b.c", "ld app"] {
        let text = outputs.entry(call.id.clone()).or_insert_with(String::new);
        text.push_str(&format!("{line}\n"));
        let shown = drawn(
            &mut transcript,
            &conversation,
            (&calls, &outputs),
            &renderers,
        );
        assert!(shown.ends_with(&format!("> {line}")), "{shown}");
    }
    let shown = drawn(
        &mut transcript,
        &conversation,
        (&calls, &outputs),
        &renderers,
    );
    assert_eq!(shown, "> cc a.c\n> cc b.c\n> ld app");
}

/// Adds `message` to `agent`'s conversation, as a commit does.
fn say(app: &mut App, agent: Entity, message: Message) {
    let commit = move |mut commit: rig_ecs::journal::Commit,
                       mut conversations: Query<&mut Conversation>| {
        if let Ok(mut conversation) = conversations.get_mut(agent) {
            commit.message(agent, &mut conversation, message.clone());
        }
    };
    assert!(app.world_mut().run_system_once(commit).is_ok());
    app.update();
}

/// The id of message `index` of `agent`'s conversation.
fn message_id(app: &App, agent: Entity, index: usize) -> Option<MessageId> {
    let conversation = app.world().get::<Conversation>(agent)?;
    conversation.id(index)
}

/// Where in `lines` the first that contains `text` is.
fn line_of(lines: &[String], text: &str) -> Option<usize> {
    lines.iter().position(|line| line.contains(text))
}

#[test]
fn a_plugin_entry_is_drawn_after_its_message_by_its_origins_look() {
    let (mut app, agent) = view_app();
    app.add_entry_renderer("btw", |entry| {
        let text = if entry.expanded {
            entry.text
        } else {
            entry.title
        };
        vec![Line::from(format!("btw: {text}"))]
    });
    say(&mut app, agent, Message::user("first question"));
    say(&mut app, agent, Message::assistant("first answer"));
    say(&mut app, agent, Message::user("second question"));
    let after = message_id(&app, agent, 1);
    let entry = TranscriptEntry::new("btw", after, "Why tabs?", "The style says so.");
    let entry = app.world_mut().spawn((entry, EntryOf(agent))).id();
    app.update();
    let lines = on_screen(&mut app, agent, 100);
    let order = ["first answer", "btw: Why tabs?", "second question"];
    let order = order.map(|text| line_of(&lines, text));
    assert!(order.iter().all(Option::is_some), "{lines:#?}");
    assert!(order.is_sorted(), "{lines:#?}");
    // Alt+Up selects the newest item, the last message; again, the entry.
    press(&mut app, "alt+up alt+up");
    let selected = app.world().resource::<TranscriptView>().selected.clone();
    assert_eq!(selected, Some(TranscriptItem::Entry(entry)));
    // Ctrl+O expands it: drawn whole.
    press(&mut app, "ctrl+o");
    let lines = on_screen(&mut app, agent, 100);
    assert!(
        line_of(&lines, "btw: The style says so.").is_some(),
        "{lines:#?}"
    );
}

#[test]
fn an_entry_whose_message_was_summarized_is_drawn_right_after_the_summary() {
    let (mut app, agent) = view_app();
    say(&mut app, agent, Message::user("old question"));
    say(&mut app, agent, Message::assistant("old answer"));
    say(&mut app, agent, Message::user("kept prompt"));
    say(&mut app, agent, Message::assistant("kept answer"));
    let after = message_id(&app, agent, 1);
    let entry = TranscriptEntry::new("btw", after, "marked", "");
    app.world_mut().spawn((entry, EntryOf(agent)));
    // A compaction: the summary takes the first two messages' place, and
    // the kept prompt joins it.
    let compact = move |mut commit: rig_ecs::journal::Commit,
                        mut conversations: Query<&mut Conversation>| {
        if let Ok(mut conversation) = conversations.get_mut(agent) {
            let summary = Entry::from(Message::user("the summary"), Origin::plugin("compaction"));
            commit.splice(agent, &mut conversation, ..2, [summary]);
        }
    };
    assert!(app.world_mut().run_system_once(compact).is_ok());
    app.update();
    let lines = on_screen(&mut app, agent, 100);
    let order = ["the summary", "marked", "kept prompt", "kept answer"];
    let order = order.map(|text| line_of(&lines, text));
    assert!(order.iter().all(Option::is_some), "{lines:#?}");
    assert!(order.is_sorted(), "{lines:#?}");
}

#[test]
fn scroll_to_puts_an_item_at_the_top_and_top_reads_it_back() {
    let (mut app, agent) = view_app();
    for index in 0..30 {
        say(&mut app, agent, Message::user(format!("question {index}")));
        say(
            &mut app,
            agent,
            Message::assistant(format!("answer {index}")),
        );
    }
    let Some(hit) = message_id(&app, agent, 21) else {
        return;
    };
    let item = TranscriptItem::Message(hit);
    app.world_mut().trigger(ScrollTo {
        agent,
        item: Some(item.clone()),
        row: ItemRow::Top(1),
    });
    let lines = on_screen(&mut app, agent, 10);
    // The answer's first row is blank; its second, the text.
    assert_eq!(lines.first().map(String::as_str), Some("answer 10"));
    let top = app.world().resource::<TranscriptView>().top().cloned();
    assert_eq!(top, Some((item, 1)));
    // `None` follows the newest output again.
    app.world_mut().trigger(ScrollTo {
        agent,
        item: None,
        row: ItemRow::Top(0),
    });
    let lines = on_screen(&mut app, agent, 10);
    assert_eq!(lines.last().map(String::as_str), Some("answer 29"));
    assert_eq!(app.world().resource::<TranscriptView>().top(), None);
}

#[test]
fn scroll_to_finds_the_row_a_highlight_lights() {
    let (mut app, agent) = view_app();
    for index in 0..30 {
        say(&mut app, agent, Message::user(format!("question {index}")));
        let answer = match index {
            10 => "answer 10\n\nalpha flaky\n\nbeta\n\ngamma flaky".to_owned(),
            _ => format!("answer {index}"),
        };
        say(&mut app, agent, Message::assistant(answer));
    }
    let Some(hit) = message_id(&app, agent, 21) else {
        return;
    };
    app.world_mut().resource_mut::<TranscriptView>().highlights = vec![Highlight {
        text: "FLAKY".to_owned(),
        ..Highlight::default()
    }];
    let mut top_after = |row: ItemRow| {
        let item = Some(TranscriptItem::Message(hit));
        app.world_mut().trigger(ScrollTo { agent, item, row });
        let lines = on_screen(&mut app, agent, 10);
        lines.first().cloned().unwrap_or_default()
    };
    assert_eq!(top_after(ItemRow::Lit(1)), "gamma flaky");
    assert_eq!(top_after(ItemRow::Lit(0)), "alpha flaky");
    // A match on screen stays where it is.
    assert_eq!(top_after(ItemRow::Lit(1)), "alpha flaky");
    // Past the last match, the item's first row: the blank one above it.
    assert_eq!(top_after(ItemRow::Lit(2)), "");
    assert_eq!(top_after(ItemRow::Top(1)), "answer 10");
}

#[test]
fn the_drawn_transcript_has_the_rows_drawn_and_the_matches_lit() {
    let (mut app, agent) = view_app();
    app.init_resource::<DrawnTranscript>();
    let Ok(name) = ToolName::new("shell") else {
        return;
    };
    let call = ToolCall::from_wire(
        "c1",
        ToolFunction::new(name.clone(), json!({ "command": "cargo test" })),
    );
    say(&mut app, agent, Message::user("why is it flaky?"));
    let reply = AssistantMessage::new(vec![AssistantContent::ToolCall(call.clone())]);
    say(&mut app, agent, Message::Assistant(reply));
    let output: Vec<String> = (0..10).map(|test| format!("test {test} flaky")).collect();
    let result = Message::tool_result(call.id.clone(), name, output.join("\n"));
    say(&mut app, agent, result);
    app.world_mut().resource_mut::<TranscriptView>().highlights = vec![Highlight {
        text: "flaky".to_owned(),
        ..Highlight::default()
    }];
    let item = TranscriptItem::Call(call.id.clone());
    let drawn = |app: &mut App| {
        on_screen(app, agent, 5);
        app.world().resource::<DrawnTranscript>().clone()
    };
    let collapsed = drawn(&mut app);
    assert_eq!(collapsed.agent(), Some(agent));
    let rows = collapsed.rows_of(&item).unwrap_or_default();
    let texts = collapsed.rows().get(rows.clone()).unwrap_or_default();
    // Counted over the rows drawn: the output is drawn cut.
    let drawn_matches = texts
        .iter()
        .filter(|row| row.text.contains("flaky"))
        .count();
    assert!((1..10).contains(&drawn_matches), "{texts:#?}");
    assert_eq!(collapsed.lit_in(&item), drawn_matches);
    assert_eq!(collapsed.item_at(rows.start), Some(&item));
    let question = message_id(&app, agent, 0).map(TranscriptItem::Message);
    assert_eq!(question.map(|item| collapsed.lit_in(&item)), Some(1));
    let rows = collapsed.rows().len();
    assert_eq!(collapsed.on_screen(), rows.saturating_sub(5)..rows);
    // Drawn whole, every match is.
    let expanded = &mut app.world_mut().resource_mut::<TranscriptView>().expanded;
    expanded.insert(item.clone());
    assert_eq!(drawn(&mut app).lit_in(&item), 10);
}

#[test]
fn a_highlight_lights_a_renderers_text_and_draws_again() {
    let (mut app, agent) = view_app();
    app.add_tool_renderer("shell", |call| {
        vec![call.header("shell", call.argument("command").unwrap_or_default())]
    });
    let Ok(name) = ToolName::new("shell") else {
        return;
    };
    let call = ToolCall::from_wire(
        "c1",
        ToolFunction::new(name, json!({ "command": "cargo test" })),
    );
    let reply = AssistantMessage::new(vec![AssistantContent::ToolCall(call)]);
    say(&mut app, agent, Message::Assistant(reply));
    let redraw = |app: &mut App| {
        let due = app
            .world_mut()
            .run_system_cached(crate::render::needs_redraw);
        due.unwrap_or(false)
    };
    redraw(&mut app);
    assert!(!redraw(&mut app));
    let lit = Style::new().on_yellow();
    app.world_mut().resource_mut::<TranscriptView>().highlights = vec![Highlight {
        text: "TEST".to_owned(),
        style: lit,
        match_case: false,
    }];
    assert!(redraw(&mut app));
    let rows = drawn_rows(&mut app, agent, 100, Vec::new());
    let spans = rows.iter().flat_map(|row| row.spans.iter());
    let lit: Vec<&str> = spans
        .filter(|span| span.style.bg == lit.bg)
        .map(|span| span.content.as_ref())
        .collect();
    assert_eq!(lit, ["test"]);
}

#[test]
fn a_tool_call_is_drawn_while_its_arguments_stream() {
    let (mut app, agent) = view_app();
    app.add_tool_renderer("write", |call| {
        let path = call.argument("path").unwrap_or("…");
        vec![Line::from(format!("{:?} write {path}", call.state))]
    });
    let Ok(name) = ToolName::new("write") else {
        return;
    };
    let effect = app.world_mut().spawn(Partial::default()).id();
    let part = rig_core::streaming::Part::new(1);
    let stream = |app: &mut App, event: StreamEvent| {
        app.world_mut().write_message(StreamDelta {
            effect,
            agent: Some(agent),
            event,
        });
        app.update();
        let calls = app.world().resource::<StreamingCalls>().of(effect).to_vec();
        let calls = calls.into_iter().map(|(_, call)| call).collect();
        drawn_rows(app, agent, 100, calls)
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
    };
    let opened = StreamEvent::Start {
        part,
        kind: PartKind::ToolCall,
        name: Some(name.clone()),
    };
    let lines = stream(&mut app, opened);
    assert!(line_of(&lines, "Streaming write …").is_some(), "{lines:#?}");
    let json = r#"{"path": "notes.md", "content": "Hel"#.to_owned();
    let lines = stream(&mut app, StreamEvent::Arguments { part, json });
    assert!(
        line_of(&lines, "Streaming write notes.md").is_some(),
        "{lines:#?}"
    );
    let done = ToolCall::from_wire("call-7", ToolFunction::new(name, json!({ "path": "a.md" })));
    let content = AssistantContent::ToolCall(done.clone());
    stream(&mut app, StreamEvent::End { part, content });
    let calls = app.world().resource::<StreamingCalls>().of(effect).to_vec();
    assert_eq!(
        calls.first().map(|(_, call)| call.id.clone()),
        Some(done.id)
    );
    // Once the reply is in the conversation, its streaming calls go.
    app.world_mut().entity_mut(effect).remove::<Partial>();
    app.update();
    assert!(
        app.world()
            .resource::<StreamingCalls>()
            .of(effect)
            .is_empty()
    );
}
