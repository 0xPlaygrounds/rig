use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;
use std::time::SystemTime;

use bevy_ecs::prelude::*;
use bevy_ecs::system::RunSystemOnce;
use ratatui::style::Style;
use ratatui::text::Line;
use rig_core::completion::{CompletionRequest, CompletionResponse, Message, Usage};
use rig_core::effect::{self, EffectId, EffectKind, EffectRecord};
use rig_core::message::{
    AssistantContent, AssistantMessage, CallId, ToolCall, ToolFunction, ToolName,
};
use rig_core::streaming::{PartKind, StreamEvent};
use rig_core::tool::ToolResult;
use rig_ecs::agent::{AgentId, CallOf, Conversation, Partial, TurnOf};
use rig_ecs::prelude::{Aside, AsideId, Author, Entry, Halt, MessageId};
use rig_ecs::stage::{
    Effect, EffectRecorded, EffectRequest, ForAgent, HoldOf, Outcome, Started, SupersededBy,
};
use rig_ecs::tools::ToolRequest;
use rig_ecs::turn::{ModelRequest, StreamDelta};
use rig_harness::note::Note;
use rig_harness::prelude::ToolProgress;
use rig_harness::prelude::{App, Last, Update};
use serde_json::json;

use super::{
    Below, CallFacts, ChildCalls, EntryRenderers, Look, Part, Renderers, Scroll, StreamingCalls,
    Transcript,
};
use crate::render::CallStates;
use crate::tests::{drawn_rows, on_screen, press, view_app};
use crate::{
    AppToolRenderersExt, CallState, DrawnRow, DrawnTranscript, Expanded, Highlight, ItemRow,
    LineSpan, PastEffects, RenderToolCall, ScrollTo, ToolCallView, TranscriptItem,
    TranscriptLayout, TranscriptView,
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
        whole: false,
        note_cut: false,
        facts: &facts,
        children: &ChildCalls::default(),
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
    let conversation =
        rig_core::conversation::Conversation::from_messages([Message::Assistant(reply)]);
    Some(Conversation(conversation))
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

/// Adds `message` to `agent`'s conversation.
fn say(app: &mut App, agent: Entity, message: Message) {
    if let Some(mut conversation) = app.world_mut().get_mut::<Conversation>(agent) {
        conversation.push(message);
    }
    app.update();
}

/// Adds `aside` to `agent`'s conversation; its id.
fn add_aside(app: &mut App, agent: Entity, aside: Aside) -> Option<AsideId> {
    let mut conversation = app.world_mut().get_mut::<Conversation>(agent)?;
    Some(conversation.add_aside(aside))
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
fn a_plugin_note_is_drawn_after_its_message_by_its_plugins_look() {
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
    let note = Note::new("Why tabs?", "The style says so.").aside(Author::plugin("btw"), after);
    let note = add_aside(&mut app, agent, note);
    app.update();
    let lines = on_screen(&mut app, agent, 100);
    let order = ["first answer", "btw: Why tabs?", "second question"];
    let order = order.map(|text| line_of(&lines, text));
    assert!(order.iter().all(Option::is_some), "{lines:#?}");
    assert!(order.is_sorted(), "{lines:#?}");
    // Alt+Up selects the newest item, the last message; again, the note.
    press(&mut app, "alt+up alt+up");
    let selected = app.world().resource::<TranscriptView>().selected.clone();
    assert_eq!(selected, note.map(TranscriptItem::Note));
    // Ctrl+O expands it: drawn whole.
    press(&mut app, "ctrl+o");
    let lines = on_screen(&mut app, agent, 100);
    assert!(
        line_of(&lines, "btw: The style says so.").is_some(),
        "{lines:#?}"
    );
}

#[test]
fn a_note_whose_message_was_summarized_is_drawn_right_after_the_summary() {
    let (mut app, agent) = view_app();
    say(&mut app, agent, Message::user("old question"));
    say(&mut app, agent, Message::assistant("old answer"));
    say(&mut app, agent, Message::user("kept prompt"));
    say(&mut app, agent, Message::assistant("kept answer"));
    let after = message_id(&app, agent, 1);
    let note = Note::new("marked", "").aside(Author::plugin("btw"), after);
    add_aside(&mut app, agent, note);
    // A compaction: the summary takes the first two messages' place.
    if let Some(mut conversation) = app.world_mut().get_mut::<Conversation>(agent) {
        let compaction = Author::plugin("compaction");
        let summary = Entry::new(Message::user("the summary")).by(compaction.clone());
        conversation.splice(..2, [summary], compaction);
    }
    app.update();
    let lines = on_screen(&mut app, agent, 100);
    let order = ["the summary", "marked", "kept prompt", "kept answer"];
    let order = order.map(|text| line_of(&lines, text));
    assert!(order.iter().all(Option::is_some), "{lines:#?}");
    assert!(order.is_sorted(), "{lines:#?}");
}

#[test]
fn the_note_rig_adds_after_a_stop_is_drawn_as_a_stop() {
    let (mut app, agent) = view_app();
    say(&mut app, agent, Message::user("first question"));
    if let Some(mut conversation) = app.world_mut().get_mut::<Conversation>(agent) {
        conversation.halt(Halt::Stopped);
    }
    say(&mut app, agent, Message::user("second question"));
    let lines = on_screen(&mut app, agent, 100);
    let order = [
        "first question",
        "[stopped before it was answered]",
        "second question",
    ];
    let order = order.map(|text| line_of(&lines, text));
    assert!(order.iter().all(Option::is_some), "{lines:#?}");
    assert!(order.is_sorted(), "{lines:#?}");
    assert!(
        !lines
            .iter()
            .any(|line| line.contains("not the user's words")),
        "{lines:#?}"
    );
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
fn a_match_a_wrap_splits_is_lit_and_counted() {
    let (mut app, agent) = view_app();
    app.init_resource::<DrawnTranscript>();
    // Words that fill the 80 columns so "cannot find" wraps after "cannot".
    let filler = "word ".repeat(14);
    say(
        &mut app,
        agent,
        Message::user(format!("{filler}cannot find foo")),
    );
    let lit = Style::new().on_yellow();
    app.world_mut().resource_mut::<TranscriptView>().highlights = vec![Highlight {
        text: "cannot find".to_owned(),
        style: lit,
        match_case: false,
        spans: Vec::new(),
    }];
    let rows = drawn_rows(&mut app, agent, 10, Vec::new());
    let texts: Vec<String> = rows.iter().map(ToString::to_string).collect();
    let split = texts.windows(2).any(|pair| {
        matches!(pair, [above, below] if above.ends_with("cannot") && below.trim_start().starts_with("find"))
    });
    assert!(split, "{texts:#?}");
    let spans = rows.iter().flat_map(|row| row.spans.iter());
    let lit: Vec<&str> = spans
        .filter(|span| span.style.bg == lit.bg)
        .map(|span| span.content.as_ref())
        .collect();
    assert_eq!(lit, ["cannot", "find"]);
    let drawn = app.world().resource::<DrawnTranscript>().clone();
    let question = message_id(&app, agent, 0).map(TranscriptItem::Message);
    assert_eq!(question.map(|item| drawn.lit_in(&item)), Some(1));
}

#[test]
fn a_plugins_own_spans_are_lit_and_counted_and_no_other_text() {
    let (mut app, agent) = view_app();
    app.init_resource::<DrawnTranscript>();
    // Words that fill the 80 columns so "cannot find" wraps after "cannot".
    let filler = "word ".repeat(14);
    let asked = format!("{filler}cannot find foo foobar foo");
    say(&mut app, agent, Message::user(asked));
    for index in 0..20 {
        say(
            &mut app,
            agent,
            Message::assistant(format!("answer {index}")),
        );
    }
    let Some(item) = message_id(&app, agent, 0).map(TranscriptItem::Message) else {
        return;
    };
    // A search of its own, as a regular expression's `cannot find|\bfoo\b`:
    // the foo in foobar is no match.
    let laid = laid_out(&mut app, agent, Expanded::AsShown);
    let lines = laid.and_then(|laid| laid.item(&item)).unwrap_or_default();
    let mut spans = Vec::new();
    for (line, drawn) in lines.lines().enumerate() {
        let text = drawn.text.as_str();
        let words = text
            .match_indices("cannot find")
            .chain(text.match_indices("foo"));
        let whole = words.filter(|(at, found)| {
            let after = text.get(at + found.len()..).unwrap_or_default();
            !after.starts_with(char::is_alphanumeric)
        });
        spans.extend(whole.map(|(at, found)| LineSpan {
            item: Some(item.clone()),
            line,
            bytes: at..at + found.len(),
        }));
    }
    assert_eq!(spans.len(), 3, "{lines:#?}");
    let lit = Style::new().on_yellow();
    app.world_mut().resource_mut::<TranscriptView>().highlights = vec![Highlight {
        style: lit,
        spans,
        ..Highlight::default()
    }];
    let mut lit_after = |row: ItemRow| {
        let item = Some(item.clone());
        app.world_mut().trigger(ScrollTo { agent, item, row });
        let rows = drawn_rows(&mut app, agent, 10, Vec::new());
        let spans = rows.iter().flat_map(|row| row.spans.iter());
        let lit = spans.filter(|span| span.style.bg == lit.bg);
        let lit: Vec<String> = lit.map(|span| span.content.to_string()).collect();
        let top = rows.first().map(ToString::to_string).unwrap_or_default();
        (top, lit)
    };
    // The second and third matches start in the row after the wrap.
    let (top, shown) = lit_after(ItemRow::Lit(1));
    assert!(top.trim_start().starts_with("find foo foobar"), "{top}");
    assert_eq!(shown, ["find", "foo", "foo"]);
    let (top, shown) = lit_after(ItemRow::Lit(0));
    assert!(top.ends_with("cannot"), "{top}");
    assert_eq!(shown, ["cannot", "find", "foo", "foo"]);
    let drawn = app.world().resource::<DrawnTranscript>().clone();
    assert_eq!(drawn.lit_in(&item), 3);
    let rows = drawn.rows_of(&item).unwrap_or_default();
    let rows = drawn.rows().get(rows).unwrap_or_default();
    let counted: Vec<usize> = rows.iter().map(DrawnRow::lit_total).collect();
    assert!(counted.ends_with(&[1, 2]), "{rows:#?}");
}

/// `agent`'s transcript as [`TranscriptLayout`] lays it out at 80
/// columns, with `expanded` items whole.
fn laid_out(app: &mut App, agent: Entity, expanded: Expanded) -> Option<DrawnTranscript> {
    let lay_out = move |layout: TranscriptLayout, view: Res<TranscriptView>| {
        layout.lay_out(&view, agent, WIDTH, expanded)
    };
    app.world_mut().run_system_once(lay_out).ok().flatten()
}

#[test]
fn a_layout_is_the_drawn_transcript_cut_as_drawn_or_whole() {
    let (mut app, agent) = view_app();
    app.init_resource::<DrawnTranscript>();
    // A look that keeps a collapsed output's last lines, as a shell's does.
    app.add_tool_renderer("shell", |call| {
        let mut lines = vec![call.header("shell", "")];
        if call.expanded {
            lines.extend(call.result_lines(usize::MAX));
        } else {
            lines.extend(call.tail_lines(2));
        }
        lines
    });
    let Ok(name) = ToolName::new("shell") else {
        return;
    };
    let call = ToolCall::from_wire("c1", ToolFunction::new(name.clone(), json!({})));
    say(&mut app, agent, Message::user("why does it panic?"));
    let reply = AssistantMessage::new(vec![AssistantContent::ToolCall(call.clone())]);
    say(&mut app, agent, Message::Assistant(reply));
    let output = ["panicked at a.rs", "two", "three", "panicked at b.rs"];
    let result = Message::tool_result(call.id.clone(), name, output.join("\n"));
    say(&mut app, agent, result);
    app.world_mut().resource_mut::<TranscriptView>().highlights = vec![Highlight {
        text: "panicked".to_owned(),
        ..Highlight::default()
    }];
    let item = TranscriptItem::Call(call.id.clone());
    on_screen(&mut app, agent, 5);
    let drawn = app.world().resource::<DrawnTranscript>().clone();
    let shown = laid_out(&mut app, agent, Expanded::AsShown);
    // Laid out as drawn, it is what the frame drew, but on no screen.
    assert_eq!(
        shown.as_ref().map(DrawnTranscript::rows),
        Some(drawn.rows())
    );
    assert!(drawn.is_cut(&item));
    assert_eq!(drawn.lit_in(&item), 1);
    let Some(whole) = laid_out(&mut app, agent, Expanded::All) else {
        return;
    };
    assert!(!whole.is_cut(&item));
    assert_eq!(whole.on_screen(), 0..0);
    // The call alone, whole: its first match is the one the tail cut.
    let call_rows = whole.item(&item).unwrap_or_default();
    let texts: Vec<&str> = call_rows
        .rows()
        .iter()
        .map(|row| row.text.as_str())
        .collect();
    let lit: Vec<usize> = call_rows.rows().iter().map(DrawnRow::lit_total).collect();
    assert_eq!(texts.get(1..).map(<[&str]>::len), Some(4), "{texts:#?}");
    assert_eq!(lit, [0, 1, 0, 0, 1]);
    assert_eq!(call_rows.items().first(), Some(&(item.clone(), 0..5)));
    // Expanded in the view, it is drawn as laid out whole.
    let expanded = &mut app.world_mut().resource_mut::<TranscriptView>().expanded;
    expanded.insert(item.clone());
    on_screen(&mut app, agent, 5);
    let drawn = app.world().resource::<DrawnTranscript>().item(&item);
    assert_eq!(
        drawn.map(|drawn| drawn.rows().to_vec()),
        Some(call_rows.rows().to_vec())
    );
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
        spans: Vec::new(),
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

/// An app with what a [`TranscriptLayout`] reads and no kernel, so the
/// effects a test spawns stay as spawned; and an agent in it.
fn layout_app() -> (App, Entity) {
    let mut app = App::new();
    app.init_resource::<CallFacts>()
        .init_resource::<StreamingCalls>()
        .init_resource::<TranscriptView>()
        .init_resource::<rig_ecs::effects::Effects>()
        .add_message::<EffectRecorded>()
        .add_message::<PastEffects>()
        .add_systems(Update, super::note_effects)
        .add_observer(super::note_agent_name);
    let agent = app
        .world_mut()
        .spawn((
            Name::new("main"),
            AgentId("main".to_owned()),
            Conversation::default(),
        ))
        .id();
    (app, agent)
}

#[test]
fn a_race_draws_the_reply_of_its_member_still_running() {
    let (mut app, agent) = layout_app();
    let world = app.world_mut();
    let turn = world.spawn(TurnOf(agent)).id();
    let streaming = |text: &str| Partial {
        text: text.to_owned(),
        reasoning: String::new(),
    };
    // Losers older and newer than it: a superseded reply is never drawn.
    let first = world.spawn((CallOf(turn), streaming("first loser's"))).id();
    let winner = world.spawn((CallOf(turn), streaming("the winner's"))).id();
    world.spawn((
        CallOf(turn),
        streaming("second loser's"),
        SupersededBy(winner),
    ));
    world.entity_mut(first).insert(SupersededBy(winner));
    world.flush();
    let rows = laid_out(&mut app, agent, Expanded::AsShown);
    let rows: Vec<&str> = rows
        .iter()
        .flat_map(DrawnTranscript::rows)
        .map(|row| row.text.as_str())
        .collect();
    assert!(rows.contains(&"the winner's"), "{rows:?}");
    assert!(!rows.iter().any(|row| row.contains("loser")), "{rows:?}");
}

/// The records of the effects that settled, as the kernel wrote them.
#[derive(Resource, Default)]
struct Records(Vec<EffectRecord>);

/// A model's reply: `text`, then `calls`.
fn reply(text: &str, calls: Vec<ToolCall>) -> Outcome {
    let mut choice = vec![AssistantContent::text(text)];
    choice.extend(calls.into_iter().map(AssistantContent::ToolCall));
    let origin = rig_core::message::Origin::new("test.api", "test", "model-a");
    let reply = CompletionResponse::new(choice, Usage::default(), origin, json!({}));
    Outcome::from(Ok(effect::Outcome::Completion(reply)))
}

/// Spawns `effect`, answered, under `parent`, and runs a frame, so the
/// kernel settles it before the next is made, as a reply settles before
/// its calls start; its id.
fn settle(app: &mut App, parent: Option<EffectId>, effect: impl Bundle) -> Option<EffectId> {
    let spawned = app.world_mut().spawn((effect, Effect::child_of(parent)));
    let id = spawned.get::<Effect>().map(Effect::id);
    app.update();
    id
}

/// A plain effect of a plugin's own, answered.
fn plain(kind: &str) -> (EffectRequest, Outcome) {
    let kind = EffectKind::Custom {
        kind: Arc::from(kind),
        payload: json!(null),
    };
    let request = EffectRequest {
        kind,
        handler: None,
    };
    (request, Outcome::custom(json!(null)))
}

/// The rows a `consult` renderer draws of `agent`'s transcript: one per
/// call under its call, with its depth, name, agent, state and first line.
fn consult_rows(app: &mut App, agent: Entity) -> Vec<String> {
    let rows = laid_out(app, agent, Expanded::AsShown);
    let rows = rows.iter().flat_map(DrawnTranscript::rows);
    rows.map(|row| row.text.clone()).collect()
}

#[test]
fn a_renderer_sees_the_calls_under_its_call_through_any_effect_and_after_a_resume() {
    let (mut app, agent) = view_app();
    let keep = |mut read: MessageReader<EffectRecorded>, mut records: ResMut<Records>| {
        records
            .0
            .extend(read.read().map(|recorded| recorded.0.clone()));
    };
    app.init_resource::<Records>().add_systems(Last, keep);
    let looks = |app: &mut App| {
        app.add_tool_renderer("consult", |view| {
            let children = view.children.iter().map(|child| {
                let agent = child.agent.as_deref().unwrap_or("-");
                let line = child.line.as_deref().unwrap_or("-");
                let (depth, name, state) = (child.depth, &child.name, &child.state);
                Line::from(format!("{depth} {name} {agent} {state:?} {line}"))
            });
            children.collect()
        });
    };
    looks(&mut app);
    let (Ok(consult), Ok(read)) = (ToolName::new("consult"), ToolName::new("read")) else {
        return;
    };
    let consulting = ToolCall::from_wire("c1", ToolFunction::new(consult, json!({})));
    let asked = Message::Assistant(AssistantMessage::new(vec![AssistantContent::ToolCall(
        consulting.clone(),
    )]));
    say(&mut app, agent, asked.clone());
    let id = app.world().get::<AgentId>(agent).cloned();
    let helper = (Name::new("helper"), AgentId("helper".to_owned()));
    let helper = app.world_mut().spawn(helper).id();
    // The real chain: the agent's model call asks for the consult, which
    // spawns a helper (a plain effect) and briefs it (another); the
    // helper's model call, under the brief, asks for a read.
    let model = |text: &str| ModelRequest::new(CompletionRequest::new(Message::user(text)));
    let turn = (model("consult"), ForAgent(agent));
    let turn = settle(&mut app, None, (turn, reply("", vec![consulting.clone()])));
    let answered = Outcome::from(ToolResult::success("both answered".to_owned().into()));
    let consulted = (ToolRequest { call: consulting }, ForAgent(agent), answered);
    let consulted = settle(&mut app, turn, consulted);
    let spawned = settle(&mut app, consulted, plain("subagent"));
    let briefed = settle(&mut app, spawned, (plain("message"), ForAgent(helper)));
    let reading = ToolCall::from_wire("r1", ToolFunction::new(read, json!({ "path": "a.rs" })));
    let asking = (model("look"), ForAgent(helper));
    let asking = (asking, reply("looking", vec![reading.clone()]));
    let asking = settle(&mut app, briefed, asking);
    let source = ToolResult::success("fn main() {}\n".to_owned().into());
    let read = (ToolRequest { call: reading }, ForAgent(helper));
    settle(&mut app, asking, (read, Outcome::from(source)));
    let expected = [
        "0 model helper Done looking",
        "1 read helper Done fn main() {}",
    ];
    assert_eq!(consult_rows(&mut app, agent), expected);

    // Resumed: the same records, read back, and no effect left.
    let records = std::mem::take(&mut app.world_mut().resource_mut::<Records>().0);
    let (mut resumed, agent) = layout_app();
    looks(&mut resumed);
    let world = resumed.world_mut();
    if let Some(id) = id {
        world.entity_mut(agent).insert(id);
    }
    world.spawn((Name::new("helper"), AgentId("helper".to_owned())));
    world.write_message(PastEffects::new(records));
    say(&mut resumed, agent, asked);
    assert_eq!(consult_rows(&mut resumed, agent), expected);
}
