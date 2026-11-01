use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::SystemTime;

use bevy_ecs::prelude::*;
use bevy_ecs::system::RunSystemOnce;
use ratatui::style::Style;
use ratatui::text::Line;
use rig_core::completion::{CompletionRequest, CompletionResponse, Message, Usage};
use rig_core::conversation::{CallRef, TurnEnd};
use rig_core::effect::{self, EffectId, EffectKind, EffectRecord};
use rig_core::message::{
    AssistantContent, AssistantMessage, CallId, Image, Reasoning, ToolCall, ToolFunction, ToolName,
    ToolResult as ToolResultMessage, ToolResultContent,
};
use rig_core::streaming::{PartKind, StreamEvent};
use rig_core::tool::ToolResult;
use rig_ecs::agent::{AgentId, CallOf, Conversation, Partial, TurnOf};
use rig_ecs::prelude::{Aside, AsideId, Author, Entry, Halt, MessageId};
use rig_ecs::stage::{
    Effect, EffectRecorded, EffectRequest, ForAgent, HoldOf, Outcome, Started, SupersededBy,
};
use rig_ecs::tools::ToolRequest;
use rig_ecs::turn::{ModelRequest, TurnCall};
use rig_harness::prelude::ToolProgress;
use rig_harness::prelude::{App, Last, Update};
use rig_harness::rig_core::conversation::Note;
use serde_json::json;

use super::{
    Below, CallFacts, ChildCalls, Look, Part, Pass, Renderers, Scroll, ShownEntries, Stops,
    Transcript,
};
use crate::render::CallStates;
use crate::tests::{drawn_rows, on_screen, press, view_app};
use crate::{
    CallState, ContentKind, DrawnRow, DrawnTranscript, Highlight, ItemRow, LineSpan, LookLine,
    MarkIn, PastEffects, ScrollTo, TextMark, TranscriptItem, TranscriptLayout, TranscriptView,
};

use crate::{RenderBatch, RenderSystems, TranscriptRender};

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
fn a_view_scrolled_up_leaves_its_last_row_to_the_marker() {
    let lines: Vec<LookLine> = (1..=30)
        .map(|n| Line::from(format!("row-{n}")).into())
        .collect();
    let parts = [Part::rows(&lines, &[], usize::from(WIDTH))];
    let transcript = Transcript::default();
    let shown = |scroll: &mut Scroll, height| {
        let visible = transcript.visible(&parts, (WIDTH, height), scroll, &[]);
        let rows: Vec<String> = visible.rows.iter().map(ToString::to_string).collect();
        (rows, visible.below)
    };
    let mut scroll = Scroll::default();
    let (rows, below) = shown(&mut scroll, 10);
    assert_eq!((rows.len(), below), (10, Below::Nothing));
    assert_eq!(rows.last().map(String::as_str), Some("row-30"));
    scroll.up(5);
    let (rows, below) = shown(&mut scroll, 10);
    assert_eq!((rows.len(), below), (9, Below::More));
    assert_eq!(rows.first().map(String::as_str), Some("row-16"));
    // A view of one row keeps it for text.
    let (rows, below) = shown(&mut scroll, 1);
    assert_eq!((rows.len(), below), (1, Below::More));
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
    states: (&BTreeMap<CallRef, CallState>, &BTreeMap<CallRef, String>),
    renderers: &Renderers,
) -> String {
    drawn_at(transcript, conversation, states, renderers, WIDTH)
}

/// [`drawn`] at `width`.
fn drawn_at(
    transcript: &mut Transcript,
    conversation: &Conversation,
    (calls, outputs): (&BTreeMap<CallRef, CallState>, &BTreeMap<CallRef, String>),
    renderers: &Renderers,
    width: u16,
) -> String {
    let (facts, theme) = (CallFacts::default(), crate::Theme::default());
    // The conversation is the same each time.
    let entries = ShownEntries::new(conversation, None);
    let look = Look {
        renderers,
        agent: None,
        calls,
        outputs,
        inside: HashMap::new(),
        facts: &facts,
        theme: &theme,
        children: &ChildCalls::default(),
        marks: &[],
        stops: &Stops::of(conversation, &entries),
    };
    transcript.update(Entity::PLACEHOLDER, &entries, false, &look, width);
    let parts = [Part::Message(0)];
    let visible = transcript.visible(&parts, (width, 10), &mut Scroll::default(), &[]);
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
    // The call is the reply's first, as the kernel would place it.
    let reply = conversation.entries().next().map(|entry| entry.id);
    let at = reply.map(|reply| CallRef { reply, index: 0 });
    let states = |world: &mut World| {
        let state = world.run_system_once(move |calls: CallStates| calls.state(effect));
        let state = state.ok().flatten().map(|(_, state)| state);
        at.zip(state).into_iter().collect::<BTreeMap<_, _>>()
    };
    let held = states(&mut world);
    let names = vec![lease.to_owned()];
    let first = |states: &BTreeMap<CallRef, CallState>| states.values().next().cloned();
    assert_eq!(first(&held), Some(CallState::Held(names)));
    let mut transcript = Transcript::default();
    let none = (BTreeMap::new(), Renderers::default());
    let shown = drawn(&mut transcript, &conversation, (&held, &none.0), &none.1);
    assert!(shown.contains(&format!("held: {lease}")), "{shown}");

    world.despawn(hold);
    let waiting = states(&mut world);
    assert_eq!(first(&waiting), Some(CallState::Waiting));
    let shown = drawn(&mut transcript, &conversation, (&waiting, &none.0), &none.1);
    assert!(!shown.contains("held:"), "{shown}");

    let started = Started {
        at: SystemTime::now(),
    };
    let progress = ToolProgress("12 of 40 files".to_owned());
    world.entity_mut(effect).insert((started, progress));
    let running = states(&mut world);
    let line = Some("12 of 40 files".to_owned());
    assert_eq!(first(&running), Some(CallState::Running(line)));
    let shown = drawn(&mut transcript, &conversation, (&running, &none.0), &none.1);
    assert!(shown.contains("   12 of 40 files"), "{shown}");
    world
        .entity_mut(effect)
        .insert(ToolProgress("13 of 40 files".to_owned()));
    let states = states(&mut world);
    let shown = drawn(&mut transcript, &conversation, (&states, &none.0), &none.1);
    assert!(shown.contains("   13 of 40 files"), "{shown}");
}

/// A table in a reply's reasoning is fit to a new width, as one in its text
/// is, not wrapped as it was laid out for the old width.
#[test]
fn a_table_in_reasoning_is_laid_out_again_when_the_width_changes() {
    let table = "| Tool | What it did |\n| --- | --- |\n| read | Read the first five lines of the \
                 workspace manifest, which name the package and its edition |\n";
    let reasoning = AssistantContent::Reasoning(Reasoning::new(table));
    let reply = Message::Assistant(AssistantMessage::new(vec![reasoning]));
    let conversation = Conversation(rig_core::conversation::Conversation::from_messages([reply]));
    let none = (BTreeMap::new(), BTreeMap::new(), Renderers::default());
    let drawn = |transcript: &mut Transcript, width| {
        drawn_at(
            transcript,
            &conversation,
            (&none.0, &none.1),
            &none.2,
            width,
        )
    };
    let mut resized = Transcript::default();
    drawn(&mut resized, 100);
    assert_eq!(
        drawn(&mut resized, 50),
        drawn(&mut Transcript::default(), 50)
    );
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
    let mut app = App::new();
    crate::renderers::add(&mut app);
    app.add_systems(
        TranscriptRender,
        (|mut batch: ResMut<RenderBatch>| {
            for (input, output) in batch.tools_mut("shell") {
                output.lines = input
                    .output
                    .as_deref()
                    .unwrap_or("no output")
                    .lines()
                    .map(|line| Line::from(format!("> {line}")).into())
                    .collect();
            }
        })
        .in_set(RenderSystems::Render),
    );
    let mut renderers = Renderers::default();
    let Some(reply) = conversation.entries().next().map(|entry| entry.id) else {
        return;
    };
    let at = CallRef { reply, index: 0 };
    let calls = BTreeMap::from([(at, CallState::Running(None))]);
    let mut transcript = Transcript::default();
    let mut outputs = BTreeMap::new();
    for line in ["cc a.c", "cc b.c", "ld app"] {
        let text = outputs.entry(at).or_insert_with(String::new);
        text.push_str(&format!("{line}\n"));
        renderers.pass = Pass::Collect;
        drawn(
            &mut transcript,
            &conversation,
            (&calls, &outputs),
            &renderers,
        );
        renderers.run(app.world_mut());
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
fn an_aside_and_a_message_are_drawn_by_their_looks() {
    let (mut app, agent) = view_app();
    app.add_systems(
        TranscriptRender,
        (|mut batch: ResMut<RenderBatch>| {
            for (input, output) in batch.asides_mut("test.mark") {
                let view = input.view();
                output.lines = {
                    let text = view.aside.value.as_str().unwrap_or_default();
                    vec![Line::from(format!("↳ {text}")).into()]
                };
            }
        })
        .in_set(RenderSystems::Render),
    );
    app.add_systems(
        TranscriptRender,
        (|mut batch: ResMut<RenderBatch>| {
            for (message, output) in batch.messages_mut() {
                let rated =
                    message.id.is_some() && matches!(message.message, Message::Assistant(_));
                output.decorate(
                    "ratings",
                    rated.then(|| Line::from("★ rate it")).into_iter().collect(),
                );
            }
        })
        .in_set(RenderSystems::Decorate),
    );
    say(&mut app, agent, Message::user("first question"));
    say(&mut app, agent, Message::assistant("first answer"));
    if let Some(mut conversation) = app.world_mut().get_mut::<Conversation>(agent) {
        let after = conversation.id(1);
        let mark = Aside::new("test.mark", Author::user(), "marked".into()).after(after);
        conversation.add_aside(mark);
    }
    say(&mut app, agent, Message::user("second question"));
    app.update();
    let lines = on_screen(&mut app, agent, 100);
    let order = [
        "first question",
        "first answer",
        "★ rate it",
        "↳ marked",
        "second question",
    ];
    let order = order.map(|text| line_of(&lines, text));
    assert!(order.iter().all(Option::is_some), "{lines:#?}");
    assert!(order.is_sorted(), "{lines:#?}");
    // The decoration is an item of its own inside the reply's rows.
    crate::tests::keep_drawn(&mut app);
    on_screen(&mut app, agent, 100);
    let drawn = crate::tests::drawn(&app);
    let reply = message_id(&app, agent, 1);
    let decoration = reply.map(|id| TranscriptItem::MessageDecoration(id, "ratings".into()));
    let rows = decoration.and_then(|item| drawn.rows_of(&item));
    let message = reply.and_then(|id| drawn.rows_of(&TranscriptItem::Message(id)));
    assert!(
        rows.zip(message)
            .is_some_and(|(rows, message)| message.start < rows.start && rows.end <= message.end),
        "{drawn:#?}"
    );
}

#[test]
fn a_plugin_note_is_drawn_after_its_message_by_its_plugins_look() {
    let (mut app, agent) = view_app();
    app.add_systems(
        TranscriptRender,
        (|mut batch: ResMut<RenderBatch>| {
            for (input, output) in batch.entries_mut("btw") {
                let entry = input.view();
                output.lines = {
                    vec![
                        Line::from(format!("btw: {}", entry.title)).into(),
                        Line::from(format!("btw: {}", entry.text)).into(),
                    ]
                };
            }
        })
        .in_set(RenderSystems::Render),
    );
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
    let selected = crate::tests::main_view(&app).selected.clone();
    assert_eq!(selected, note.map(TranscriptItem::Aside));
}

#[test]
fn an_image_the_assistant_made_is_drawn() {
    let (mut app, agent) = view_app();
    say(&mut app, agent, Message::user("draw a cat"));
    let reply = AssistantMessage::new(vec![AssistantContent::Image(Image::default())]);
    say(&mut app, agent, Message::Assistant(reply));
    let lines = on_screen(&mut app, agent, 20);
    assert!(line_of(&lines, "[image]").is_some(), "{lines:#?}");
}

#[test]
fn content_renderers_draw_both_sides_blocks_and_decorators_every_call() {
    use rig_core::message::UserContent;
    let (mut app, agent) = view_app();
    crate::tests::keep_drawn(&mut app);
    // An image look that reserves a row under its label, as a plugin
    // placing images does.
    app.add_systems(
        TranscriptRender,
        (|mut batch: ResMut<RenderBatch>| {
            for (input, output) in batch.contents_mut(ContentKind::Image) {
                let view = input.view();
                output.lines = {
                    let by = if view.assistant { "model" } else { "user" };
                    vec![
                        Line::from(format!("img {} by {by}", view.index)).into(),
                        Line::default().into(),
                    ]
                };
            }
        })
        .in_set(RenderSystems::Render),
    )
    .add_systems(
        TranscriptRender,
        (|mut batch: ResMut<RenderBatch>| {
            for (input, output) in batch.tools_mut("shell") {
                let call = input.view();
                output.lines = vec![call.header("shell", "").into()];
            }
        })
        .in_set(RenderSystems::Render),
    )
    .add_systems(
        TranscriptRender,
        (|mut batch: ResMut<RenderBatch>| {
            for (input, output) in batch.calls_mut() {
                let call = input.view();
                output.decorate("names", {
                    vec![Line::from(format!("  ~ {}", call.name()))]
                });
            }
        })
        .in_set(RenderSystems::Decorate),
    );
    let image = UserContent::Image(Image::default());
    let asked = Message::User {
        content: vec![UserContent::text("look"), image],
    };
    say(&mut app, agent, asked);
    let reply = AssistantMessage::new(vec![AssistantContent::Image(Image::default())]);
    say(&mut app, agent, Message::Assistant(reply));
    let (Ok(shell), Ok(read)) = (ToolName::new("shell"), ToolName::new("read")) else {
        return;
    };
    let calls = [("c1", shell), ("c2", read)]
        .map(|(id, name)| ToolCall::from_wire(id, ToolFunction::new(name, json!({}))));
    let reply = calls.clone().map(AssistantContent::ToolCall);
    say(
        &mut app,
        agent,
        Message::Assistant(AssistantMessage::new(reply.to_vec())),
    );
    let lines = on_screen(&mut app, agent, 100);
    for text in ["img 1 by user", "img 0 by model", "~ shell", "~ read"] {
        assert!(line_of(&lines, text).is_some(), "{text}: {lines:#?}");
    }
    // The user's image is an item with its rows, the reserved one too.
    let Some(asked) = message_id(&app, agent, 0) else {
        return;
    };
    let drawn = crate::tests::drawn(&app);
    let rows = drawn.rows_of(&TranscriptItem::Content(asked, 1));
    let texts: Vec<&str> = (rows.unwrap_or_default().map(|row| drawn.rows().get(row)))
        .map(|row| row.map_or("", |row| row.text.as_str()))
        .collect();
    assert_eq!(texts, ["img 1 by user", ""]);
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

/// Ends the turn `agent` runs as a stop does: the request halted, the
/// record ending at the last entry with the calls the stop answered.
fn stop(app: &mut App, agent: Entity, calls: &[&str]) {
    if let Some(mut conversation) = app.world_mut().get_mut::<Conversation>(agent) {
        conversation.halt(Halt::Stopped);
        let at = conversation.entries().next_back().map(|entry| entry.id);
        let calls = calls.iter().map(|id| CallId::from_wire(*id)).collect();
        let why = None;
        conversation.end_turn(TurnEnd::Stopped { why, at, calls });
    }
    app.update();
}

/// Puts `agent`'s conversation back as a session restore reads it, so
/// only what is saved draws it; whether it could.
fn reload(app: &mut App, agent: Entity) -> bool {
    let saved = app
        .world()
        .get::<Conversation>(agent)
        .map(serde_json::to_value);
    let restored = saved
        .and_then(Result::ok)
        .map(serde_json::from_value::<Conversation>);
    let Some(Ok(restored)) = restored else {
        return false;
    };
    app.world_mut().entity_mut(agent).insert(restored);
    app.update();
    true
}

/// How many of `lines` contain `text`.
fn count_of(lines: &[String], text: &str) -> usize {
    lines.iter().filter(|line| line.contains(text)).count()
}

#[test]
fn a_stopped_reply_ends_in_a_stop_at_once_and_after_a_reload() {
    let (mut app, agent) = view_app();
    say(&mut app, agent, Message::user("write an essay"));
    if let Some(mut conversation) = app.world_mut().get_mut::<Conversation>(agent) {
        conversation.open_turn();
    }
    let text = vec![AssistantContent::text("half an essay")];
    let aborted = AssistantMessage::aborted(None, text, "stopped");
    say(&mut app, agent, Message::Assistant(aborted));
    stop(&mut app, agent, &[]);
    let order = ["write an essay", "half an essay", "[stopped]"];
    let lines = on_screen(&mut app, agent, 100);
    let at = order.map(|text| line_of(&lines, text));
    assert!(at.iter().all(Option::is_some), "{lines:#?}");
    assert!(at.is_sorted(), "{lines:#?}");
    // The note the next message brings draws nothing; the stop stays put.
    say(&mut app, agent, Message::user("go on"));
    for _ in 0..2 {
        let lines = on_screen(&mut app, agent, 100);
        let at = ["half an essay", "[stopped]", "go on"].map(|text| line_of(&lines, text));
        assert!(at.iter().all(Option::is_some), "{lines:#?}");
        assert!(at.is_sorted(), "{lines:#?}");
        assert_eq!(count_of(&lines, "[stopped"), 1, "{lines:#?}");
        assert_eq!(count_of(&lines, "Do not carry it out"), 0, "{lines:#?}");
        assert!(reload(&mut app, agent));
    }
}

#[test]
fn a_call_a_stop_cut_while_it_streamed_says_it_never_ran_at_once_and_after_a_reload() {
    let (mut app, agent) = view_app();
    let (Ok(read), Ok(write)) = (ToolName::new("read"), ToolName::new("write")) else {
        return;
    };
    say(&mut app, agent, Message::user("write big.py"));
    if let Some(mut conversation) = app.world_mut().get_mut::<Conversation>(agent) {
        conversation.open_turn();
    }
    // A call whose arguments ended, then one still streaming them.
    let mut cut = Partial::default();
    let ended = rig_core::streaming::Part::new(1);
    cut.follow(&StreamEvent::Start {
        part: ended,
        kind: PartKind::ToolCall,
        name: Some(read.clone()),
    });
    let done = ToolCall::from_wire("r1", ToolFunction::new(read, json!({ "path": "a.md" })));
    let content = AssistantContent::ToolCall(done);
    cut.follow(&StreamEvent::End {
        part: ended,
        content,
    });
    let part = rig_core::streaming::Part::new(0);
    cut.follow(&StreamEvent::Start {
        part,
        kind: PartKind::ToolCall,
        name: Some(write),
    });
    let json = r#"{"path": "big.py", "content": "print("#.to_owned();
    cut.follow(&StreamEvent::Arguments { part, json });
    let calls = cut.calls.iter().map(|call| call.call.id.clone()).collect();
    let aborted = AssistantMessage::aborted(None, cut.stopped(), "stopped");
    say(&mut app, agent, Message::Assistant(aborted));
    if let Some(mut conversation) = app.world_mut().get_mut::<Conversation>(agent) {
        conversation.halt(Halt::Stopped);
        let at = conversation.entries().next_back().map(|entry| entry.id);
        conversation.end_turn(TurnEnd::Stopped {
            why: None,
            at,
            calls,
        });
    }
    for _ in 0..2 {
        let lines = on_screen(&mut app, agent, 100);
        let row = lines.iter().position(|line| line.starts_with(" write"));
        let header = row
            .and_then(|row| lines.get(row..=row + 1))
            .map(<[String]>::concat);
        assert!(
            header
                .is_some_and(|header| header.contains("· stopped while it streamed; nothing ran")),
            "{lines:#?}"
        );
        let read = lines.iter().find(|line| line.starts_with(" read"));
        assert!(
            read.is_some_and(|read| read.contains("· stopped before it ran")),
            "{lines:#?}"
        );
        assert!(reload(&mut app, agent));
    }
}

#[test]
fn a_stop_during_calls_marks_the_calls_it_answered_stopped() {
    let (mut app, agent) = view_app();
    let (Ok(look), Ok(shell), Ok(read)) = (
        ToolName::new("look"),
        ToolName::new("shell"),
        ToolName::new("read"),
    ) else {
        return;
    };
    let call = |id: &str, name: &ToolName| {
        ToolCall::from_wire(id, ToolFunction::new(name.clone(), json!({})))
    };
    let reply = |calls: Vec<ToolCall>| {
        let calls = calls.into_iter().map(AssistantContent::ToolCall).collect();
        Message::Assistant(AssistantMessage::new(calls))
    };
    // An earlier turn's call shares an id with a call the stop answered.
    say(&mut app, agent, Message::user("first"));
    let earlier = call("a", &look);
    say(&mut app, agent, reply(vec![earlier.clone()]));
    let looked = earlier.result(vec![ToolResultContent::text("seen")]);
    say(&mut app, agent, Message::tool_results(vec![looked]));
    say(&mut app, agent, Message::assistant("done"));
    say(&mut app, agent, Message::user("second"));
    if let Some(mut conversation) = app.world_mut().get_mut::<Conversation>(agent) {
        conversation.open_turn();
    }
    let (running, held, failed) = (call("a", &shell), call("b", &shell), call("c", &read));
    say(
        &mut app,
        agent,
        reply(vec![running.clone(), held.clone(), failed.clone()]),
    );
    let results = [
        (running, "partial output", true),
        (held, "Not run: the user stopped the turn.", true),
        (failed, "no such file", true),
    ];
    let results = results.map(|(call, text, error)| ToolResultMessage {
        is_error: error,
        ..call.result(vec![ToolResultContent::text(text)])
    });
    say(&mut app, agent, Message::tool_results(results.to_vec()));
    stop(&mut app, agent, &["a", "b"]);
    let drawn_stopped = |lines: &[String]| {
        let headers: Vec<&str> = lines
            .iter()
            .filter(|line| {
                [" look", " shell", " read"]
                    .iter()
                    .any(|name| line.starts_with(name))
            })
            .map(String::as_str)
            .collect();
        let states: Vec<bool> = headers
            .iter()
            .map(|header| header.contains("· stopped"))
            .collect();
        let failed = headers
            .last()
            .is_some_and(|header| header.contains("· failed"));
        let order = ["no such file", "[stopped before it was answered]"];
        let at = order.map(|text| line_of(lines, text));
        states == [false, true, true, false]
            && failed
            && at.iter().all(Option::is_some)
            && at.is_sorted()
    };
    for _ in 0..2 {
        let lines = on_screen(&mut app, agent, 100);
        assert!(drawn_stopped(&lines), "{lines:#?}");
        assert!(reload(&mut app, agent));
    }
    // A compaction that takes the stopped turn out keeps it drawn so.
    say(&mut app, agent, Message::user("third"));
    let count = app
        .world()
        .get::<Conversation>(agent)
        .map(|it| it.messages().len());
    compact(&mut app, agent, count.unwrap_or(1) - 1, "the summary");
    let lines = on_screen(&mut app, agent, 100);
    assert!(drawn_stopped(&lines), "{lines:#?}");
    assert!(line_of(&lines, "the summary").is_some(), "{lines:#?}");
}

#[test]
fn each_call_of_a_reply_shows_its_own_result_even_when_two_share_an_id() {
    let (mut app, agent) = view_app();
    let (Ok(look), Ok(probe)) = (ToolName::new("look"), ToolName::new("probe")) else {
        return;
    };
    // Two calls of one reply with one id, as a provider may send them.
    let both = [
        ToolCall::from_wire("c9", ToolFunction::new(look, json!({}))),
        ToolCall::from_wire("c9", ToolFunction::new(probe, json!({}))),
    ];
    say(&mut app, agent, Message::user("look and probe"));
    let asked = both.iter().cloned().map(AssistantContent::ToolCall);
    let reply = AssistantMessage::new(asked.collect());
    say(&mut app, agent, Message::Assistant(reply));
    let results = both
        .iter()
        .zip(["looked four", "probed five"])
        .map(|(call, said)| call.result(vec![ToolResultContent::text(said)]));
    say(&mut app, agent, Message::tool_results(results.collect()));
    let lines = on_screen(&mut app, agent, 100);
    let shown = ["looked four", "probed five"].map(|said| count_of(&lines, said));
    assert_eq!(shown, [1, 1], "{lines:#?}");
}

/// A call a stop answered is drawn stopped whatever live state it still
/// has.
#[test]
fn a_stopped_call_wins_over_its_live_state() {
    let Ok(name) = ToolName::new("shell") else {
        return;
    };
    let call = ToolCall::from_wire("c1", ToolFunction::new(name, json!({ "command": "make" })));
    let mut conversation = Conversation::default();
    conversation.open_turn();
    let reply = AssistantMessage::new(vec![AssistantContent::ToolCall(call.clone())]);
    let at = conversation.push(Message::Assistant(reply));
    let calls = vec![call.id.clone()];
    conversation.end_turn(TurnEnd::Stopped {
        why: None,
        at: Some(at),
        calls,
    });
    let call = CallRef {
        reply: at,
        index: 0,
    };
    let calls = BTreeMap::from([(call, CallState::Running(None))]);
    let outputs = BTreeMap::from([(call, "cc a.c".to_owned())]);
    let mut transcript = Transcript::default();
    let renderers = Renderers::default();
    let shown = drawn(
        &mut transcript,
        &conversation,
        (&calls, &outputs),
        &renderers,
    );
    assert!(shown.contains("· stopped"), "{shown}");
    assert!(!shown.contains("running"), "{shown}");
}

#[test]
fn scroll_to_puts_an_item_at_the_top_and_top_reads_it_back() {
    let (mut app, agent) = view_app();
    let pane = crate::tests::main_pane(app.world());
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
        pane,
        item: Some(item.clone()),
        row: ItemRow::Top(1),
    });
    let lines = on_screen(&mut app, agent, 10);
    // The answer's first row is blank; its second, the text.
    assert_eq!(lines.first().map(String::as_str), Some("answer 10"));
    let top = crate::tests::main_view(&app).top().cloned();
    assert_eq!(top, Some((item, 1)));
    // `None` follows the newest output again.
    app.world_mut().trigger(ScrollTo {
        pane,
        item: None,
        row: ItemRow::Top(0),
    });
    let lines = on_screen(&mut app, agent, 10);
    assert_eq!(lines.last().map(String::as_str), Some("answer 29"));
    assert_eq!(crate::tests::main_view(&app).top(), None);
}

#[test]
fn scroll_to_finds_the_row_a_highlight_lights() {
    let (mut app, agent) = view_app();
    let pane = crate::tests::main_pane(app.world());
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
    crate::tests::steer(&mut app, |view| {
        view.highlights = vec![Highlight {
            text: "FLAKY".to_owned(),
            ..Highlight::default()
        }]
    });
    let mut top_after = |row: ItemRow| {
        let item = Some(TranscriptItem::Message(hit));
        app.world_mut().trigger(ScrollTo { pane, item, row });
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
    crate::tests::keep_drawn(&mut app);
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
    crate::tests::steer(&mut app, |view| {
        view.highlights = vec![Highlight {
            text: "flaky".to_owned(),
            ..Highlight::default()
        }]
    });
    let item = TranscriptItem::Call(call.id.clone());
    let drawn = |app: &mut App| {
        on_screen(app, agent, 5);
        crate::tests::drawn(app).clone()
    };
    let drawn = drawn(&mut app);
    assert_eq!(drawn.agent(), Some(agent));
    let rows = drawn.rows_of(&item).unwrap_or_default();
    // The output is drawn whole: every match is lit.
    assert_eq!(drawn.lit_in(&item), 10);
    assert_eq!(drawn.item_at(rows.start), Some(&item));
    let question = message_id(&app, agent, 0).map(TranscriptItem::Message);
    assert_eq!(question.map(|item| drawn.lit_in(&item)), Some(1));
    let rows = drawn.rows().len();
    assert_eq!(drawn.on_screen(), rows.saturating_sub(5)..rows);
}

#[test]
fn a_match_a_wrap_splits_is_lit_and_counted() {
    let (mut app, agent) = view_app();
    crate::tests::keep_drawn(&mut app);
    // Words that fill the 80 columns so "cannot find" wraps after "cannot".
    let filler = "word ".repeat(14);
    say(
        &mut app,
        agent,
        Message::user(format!("{filler}cannot find foo")),
    );
    let lit = Style::new().on_yellow();
    crate::tests::steer(&mut app, |view| {
        view.highlights = vec![Highlight {
            text: "cannot find".to_owned(),
            style: lit,
            match_case: false,
            spans: Vec::new(),
        }]
    });
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
    let drawn = crate::tests::drawn(&app).clone();
    let question = message_id(&app, agent, 0).map(TranscriptItem::Message);
    assert_eq!(question.map(|item| drawn.lit_in(&item)), Some(1));
}

#[test]
fn a_plugins_own_spans_are_lit_and_counted_and_no_other_text() {
    let (mut app, agent) = view_app();
    let pane = crate::tests::main_pane(app.world());
    crate::tests::keep_drawn(&mut app);
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
    let laid = laid_out(&mut app, agent);
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
    crate::tests::steer(&mut app, |view| {
        view.highlights = vec![Highlight {
            style: lit,
            spans,
            ..Highlight::default()
        }]
    });
    let mut lit_after = |row: ItemRow| {
        let item = Some(item.clone());
        app.world_mut().trigger(ScrollTo { pane, item, row });
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
    let drawn = crate::tests::drawn(&app).clone();
    assert_eq!(drawn.lit_in(&item), 3);
    let rows = drawn.rows_of(&item).unwrap_or_default();
    let rows = drawn.rows().get(rows).unwrap_or_default();
    let counted: Vec<usize> = rows.iter().map(DrawnRow::lit_total).collect();
    assert!(counted.ends_with(&[1, 2]), "{rows:#?}");
}

/// `agent`'s transcript as [`TranscriptLayout`] lays it out at 80
/// columns.
fn laid_out(app: &mut App, agent: Entity) -> Option<DrawnTranscript> {
    let world = app.world_mut();
    let view = world
        .query::<&TranscriptView>()
        .iter(world)
        .next()
        .cloned()
        .unwrap_or_default();
    TranscriptLayout::lay_out(world, &view, agent, WIDTH)
}

#[test]
fn a_layout_is_the_drawn_transcript_on_no_screen() {
    let (mut app, agent) = view_app();
    crate::tests::keep_drawn(&mut app);
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
    crate::tests::steer(&mut app, |view| {
        view.highlights = vec![Highlight {
            text: "panicked".to_owned(),
            ..Highlight::default()
        }]
    });
    let item = TranscriptItem::Call(call.id.clone());
    on_screen(&mut app, agent, 5);
    let drawn = crate::tests::drawn(&app).clone();
    let Some(laid) = laid_out(&mut app, agent) else {
        return;
    };
    assert_eq!(laid.rows(), drawn.rows());
    assert_eq!(laid.on_screen(), 0..0);
    // The call alone: its header and every line of its output.
    let call_rows = laid.item(&item).unwrap_or_default();
    let lit: Vec<usize> = call_rows.rows().iter().map(DrawnRow::lit_total).collect();
    assert_eq!(lit, [0, 1, 0, 0, 1]);
    assert_eq!(call_rows.items().first(), Some(&(item, 0..5)));
}

#[test]
fn a_highlight_lights_a_renderers_text_and_draws_again() {
    let (mut app, agent) = view_app();
    app.add_systems(
        TranscriptRender,
        (|mut batch: ResMut<RenderBatch>| {
            for (input, output) in batch.tools_mut("shell") {
                let call = input.view();
                output.lines = vec![
                    call.header("shell", call.argument("command").unwrap_or_default())
                        .into(),
                ];
            }
        })
        .in_set(RenderSystems::Render),
    );
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
    crate::tests::steer(&mut app, |view| {
        view.highlights = vec![Highlight {
            text: "TEST".to_owned(),
            style: lit,
            match_case: false,
            spans: Vec::new(),
        }]
    });
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
    app.add_systems(
        TranscriptRender,
        (|mut batch: ResMut<RenderBatch>| {
            for (input, output) in batch.tools_mut("write") {
                let call = input.view();
                output.lines = {
                    let path = call.argument("path").unwrap_or("…");
                    vec![Line::from(format!("{:?} write {path}", call.state)).into()]
                };
            }
        })
        .in_set(RenderSystems::Render),
    );
    let Ok(name) = ToolName::new("write") else {
        return;
    };
    let turn = app.world_mut().spawn(TurnOf(agent)).id();
    let effect = app
        .world_mut()
        .spawn((CallOf(turn), Partial::default()))
        .id();
    let part = rig_core::streaming::Part::new(1);
    let stream = |app: &mut App, event: StreamEvent| {
        if let Some(mut partial) = app.world_mut().get_mut::<Partial>(effect) {
            partial.follow(&event);
        }
        app.update();
        drawn_rows(app, agent, 100, Vec::new())
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
    let lines = stream(&mut app, StreamEvent::End { part, content });
    assert!(
        line_of(&lines, "Streaming write a.md").is_some(),
        "{lines:#?}"
    );
    // Once the reply is in the conversation, its streaming calls go.
    app.world_mut().entity_mut(effect).remove::<Partial>();
    app.update();
    let lines = drawn_rows(&mut app, agent, 100, Vec::new());
    let lines: Vec<String> = lines.iter().map(ToString::to_string).collect();
    assert!(line_of(&lines, "Streaming write").is_none(), "{lines:#?}");
}

/// An app with what a [`TranscriptLayout`] reads and no kernel, so the
/// effects a test spawns stay as spawned; and an agent in it.
fn layout_app() -> (App, Entity) {
    let mut app = App::new();
    app.init_resource::<CallFacts>()
        .init_resource::<crate::Notices>()
        .init_resource::<crate::Theme>()
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
        calls: Vec::new(),
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
    let rows = laid_out(&mut app, agent);
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
    Outcome::from(Ok(effect::Answered::Completion(reply.into())))
}

/// Spawns `effect`, answered, under `parent`, and runs a frame, so the
/// kernel settles it before the next is made, as a reply settles before
/// its calls start; its id.
fn settle(app: &mut App, parent: Option<EffectId>, effect: impl Bundle) -> Option<EffectId> {
    let spawned = app
        .world_mut()
        .spawn((effect, Effect::with_parent_id(parent)));
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
/// call under its call, with its depth, name, agent, state and first line,
/// each after a `↳ `.
fn consult_rows(app: &mut App, agent: Entity) -> Vec<String> {
    let rows = laid_out(app, agent);
    let rows = rows.iter().flat_map(DrawnTranscript::rows);
    let rows = rows.filter_map(|row| row.text.strip_prefix("↳ "));
    rows.map(str::to_owned).collect()
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
        app.add_systems(
            TranscriptRender,
            (|mut batch: ResMut<RenderBatch>| {
                for (input, output) in batch.tools_mut("consult") {
                    let view = input.view();
                    output.lines = {
                        let children = view.children.iter().map(|child| {
                            let agent = child.agent.as_deref().unwrap_or("-");
                            let line = child.answer.as_deref().unwrap_or("-");
                            let (depth, name, state) = (child.depth, &child.name, &child.state);
                            Line::from(format!("↳ {depth} {name} {agent} {state:?} {line}"))
                        });
                        children.map(LookLine::from).collect()
                    };
                }
            })
            .in_set(RenderSystems::Render),
        );
    };
    looks(&mut app);
    let (Ok(consult), Ok(read)) = (ToolName::new("consult"), ToolName::new("read")) else {
        return;
    };
    let consulting = ToolCall::from_wire("c1", ToolFunction::new(consult, json!({})));
    let id = app.world().get::<AgentId>(agent).cloned();
    let helper = (Name::new("helper"), AgentId("helper".to_owned()));
    let helper = app.world_mut().spawn(helper).id();
    // The real chain: the agent's model call asks for the consult, which
    // spawns a helper (a plain effect) and briefs it (another); the
    // helper's model call, under the brief, asks for a read.
    let model = |text: &str| ModelRequest::new(CompletionRequest::new(Message::user(text)));
    let turn = app.world_mut().spawn(TurnOf(agent)).id();
    let asking = (TurnCall, CallOf(turn), ForAgent(agent));
    settle(&mut app, None, (asking, reply("", vec![consulting])));
    app.update();
    // The kernel spawned the consult as its reply asked for it.
    let records = &app.world().resource::<Records>().0;
    let consulted = records.iter().find_map(|record| match &record.kind {
        EffectKind::ToolCall { name, .. } if name == "consult" => Some(record.id),
        _ => None,
    });
    let asked = app.world().get::<Conversation>(agent);
    let asked = asked.and_then(|conversation| conversation.messages().first().cloned());
    let Some(asked) = asked.map(|asked| (*asked).clone()) else {
        return;
    };
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

/// A mark inside a saved reply's text block is drawn between the words it
/// sits between, the bold on both sides still bold.
#[test]
fn a_mark_is_drawn_inside_a_replys_text_with_its_markdown_kept() {
    use ratatui::style::Modifier;

    let (mut app, agent) = view_app();
    say(&mut app, agent, Message::user("Go on."));
    say(
        &mut app,
        agent,
        Message::assistant("**first part** and **second part**"),
    );
    let Some(id) = message_id(&app, agent, 1) else {
        return;
    };
    // Byte 18: after "**first part** and".
    crate::tests::steer(&mut app, |view| {
        view.marks = vec![TextMark {
            r#in: MarkIn::Message { id, block: 0 },
            at: 18,
            label: "⟳".to_owned(),
            style: Style::new(),
        }]
    });
    let rows = drawn_rows(&mut app, agent, 20, Vec::new());
    let row = rows.iter().find(|row| row.to_string().contains("⟳"));
    assert_eq!(
        row.map(ToString::to_string).as_deref(),
        Some("first part and⟳ second part")
    );
    let bold: Vec<String> = row
        .into_iter()
        .flat_map(|row| row.spans.iter())
        .filter(|span| span.style.add_modifier.contains(Modifier::BOLD))
        .map(|span| span.content.to_string())
        .collect();
    assert_eq!(bold, ["first part", "second part"]);
}

/// A mark on a streaming reply is drawn in it, and gone once cleared.
#[test]
fn a_mark_on_a_streaming_reply_is_drawn_until_cleared() {
    let (mut app, agent) = view_app();
    say(&mut app, agent, Message::user("Go on."));
    let world = app.world_mut();
    let turn = world.spawn(TurnOf(agent)).id();
    let partial = Partial {
        text: "so far more".to_owned(),
        reasoning: String::new(),
        calls: Vec::new(),
    };
    let call = world.spawn((CallOf(turn), partial)).id();
    crate::tests::steer(&mut app, |view| {
        view.marks = vec![TextMark {
            r#in: MarkIn::Streaming(call),
            at: 7,
            label: "[continued] ".to_owned(),
            style: Style::new(),
        }];
    });
    let text = |app: &mut App| -> Vec<String> {
        let rows = drawn_rows(app, agent, 20, Vec::new());
        rows.iter().map(ToString::to_string).collect()
    };
    let rows = text(&mut app);
    assert!(
        rows.iter().any(|row| row == "so far [continued] more"),
        "{rows:?}"
    );
    crate::tests::steer(&mut app, |view| view.marks.clear());
    let rows = text(&mut app);
    assert!(rows.iter().any(|row| row == "so far more"), "{rows:?}");
}

/// Each decorator's rows under a call are an item of their own, inside the
/// call's, holding only its lines whatever order the decorators came in;
/// systems own the names of their contributions.
#[test]
fn each_decorators_rows_are_an_item_inside_its_call() {
    let (mut app, agent) = view_app();
    crate::tests::keep_drawn(&mut app);
    app.add_systems(
        TranscriptRender,
        (|mut batch: ResMut<RenderBatch>| {
            for (input, output) in batch.tools_mut("shell") {
                let call = input.view();
                output.lines = vec![call.header("shell", "").into()];
            }
        })
        .in_set(RenderSystems::Render),
    )
    .add_systems(
        TranscriptRender,
        (|mut batch: ResMut<RenderBatch>| {
            for (input, output) in batch.calls_mut() {
                let _ = input.view();
                output.decorate("late", {
                    vec![Line::from("  late one"), Line::from("  late two")]
                });
            }
        })
        .in_set(RenderSystems::Decorate),
    )
    .add_systems(
        TranscriptRender,
        (|mut batch: ResMut<RenderBatch>| {
            for (input, output) in batch.calls_mut() {
                let _ = input.view();
                output.decorate("early", vec![Line::from("  early")]);
            }
        })
        .in_set(RenderSystems::Decorate),
    );
    let Ok(shell) = ToolName::new("shell") else {
        return;
    };
    let call = ToolCall::from_wire("c1", ToolFunction::new(shell, json!({})));
    let reply = AssistantMessage::new(vec![AssistantContent::ToolCall(call)]);
    say(&mut app, agent, Message::Assistant(reply));
    let lines = on_screen(&mut app, agent, 40);
    assert!(line_of(&lines, "never").is_none(), "{lines:#?}");
    let drawn = crate::tests::drawn(&app);
    let texts = |item: TranscriptItem| -> Vec<String> {
        let rows = drawn.rows_of(&item).unwrap_or_default();
        let rows = rows.filter_map(|row| drawn.rows().get(row));
        rows.map(|row| row.text.trim().to_owned()).collect()
    };
    let id = CallId::from_wire("c1");
    let decoration = |name: &str| TranscriptItem::Decoration(id.clone(), name.to_owned());
    assert_eq!(texts(decoration("late")), ["late one", "late two"]);
    assert_eq!(texts(decoration("early")), ["early"]);
    let call = drawn
        .rows_of(&TranscriptItem::Call(id.clone()))
        .unwrap_or_default();
    for name in ["late", "early"] {
        let rows = drawn.rows_of(&decoration(name)).unwrap_or_default();
        assert!(call.start <= rows.start && rows.end <= call.end, "{name}");
    }
}

/// Puts a summary, `text` by compaction, in place of the first `count`
/// messages of `agent`'s conversation, and what it took out in the agent's
/// [`Replaced`](crate::Replaced); the summary's id.
fn compact(app: &mut App, agent: Entity, count: usize, text: &str) -> Option<MessageId> {
    let world = app.world_mut();
    let mut conversation = world.get_mut::<Conversation>(agent)?;
    let taken: Vec<Entry> = (0..count).filter_map(|at| conversation.entry(at)).collect();
    let compaction = Author::plugin("compaction");
    let summary = Entry::new(Message::user(text)).by(compaction.clone());
    let spliced = conversation.splice(..count, [summary], compaction);
    let summary = spliced.changed.first().copied()?;
    let mut replaced = world
        .get::<crate::Replaced>(agent)
        .cloned()
        .unwrap_or_default();
    replaced.0.insert(summary, taken);
    world.entity_mut(agent).insert(replaced);
    app.update();
    Some(summary)
}

/// The rows of `agent`'s transcript in the main pane at `width` columns.
fn rows_at(app: &mut App, agent: Entity, width: u16) -> Vec<String> {
    use crate::pane::{PaneState, lay_out_panes};
    let pane = crate::tests::main_pane(app.world());
    if let Some(mut canvas) = app.world_mut().get_mut::<crate::PanelCanvas>(pane) {
        canvas.reset(ratatui::layout::Rect::new(0, 0, width, 1000));
    }
    if let Some(mut shown) = app
        .world_mut()
        .get_resource_mut::<rig_harness::main_agent::ShownAgent>()
    {
        shown.0 = Some(agent);
    }
    app.world_mut().run_system_once(lay_out_panes).ok();
    let state = app.world().get::<PaneState>(pane);
    let rows = state.map(|state| state.rows.iter().map(ToString::to_string).collect());
    rows.unwrap_or_default()
}

/// The entries a summary replaced are drawn above it, through the same
/// looks as live ones, after a line that says how many, and laid out again
/// at a new width; none of them is an item to select, rewind to or fork
/// from.
#[test]
fn what_a_summary_replaced_is_drawn_above_it_with_no_items() {
    let (mut app, agent) = view_app();
    crate::tests::keep_drawn(&mut app);
    let Ok(name) = ToolName::new("shell") else {
        return;
    };
    let call = ToolCall::from_wire("c1", ToolFunction::new(name.clone(), json!({})));
    let long = "a long old question ".repeat(6);
    say(
        &mut app,
        agent,
        Message::user(format!("old question: {long}")),
    );
    let reply = AssistantMessage::new(vec![AssistantContent::ToolCall(call.clone())]);
    say(&mut app, agent, Message::Assistant(reply));
    say(
        &mut app,
        agent,
        Message::tool_result(call.id.clone(), name, "old output"),
    );
    say(&mut app, agent, Message::assistant("old answer"));
    say(&mut app, agent, Message::user("kept prompt"));
    let old: Vec<MessageId> = (0..4)
        .filter_map(|at| message_id(&app, agent, at))
        .collect();
    let summary = compact(&mut app, agent, 4, "the summary");
    let lines = on_screen(&mut app, agent, 100);
    let order = [
        "old question",
        "old output",
        "old answer",
        "── 4 messages above were compacted ──",
        "the summary",
        "kept prompt",
    ];
    let order = order.map(|text| line_of(&lines, text));
    assert!(order.iter().all(Option::is_some), "{lines:#?}");
    assert!(order.is_sorted(), "{lines:#?}");

    let drawn = crate::tests::drawn(&app);
    let items: Vec<&TranscriptItem> = drawn.items().iter().map(|(item, _)| item).collect();
    let archived = |item: &&TranscriptItem| match item {
        TranscriptItem::Message(id) => old.contains(id),
        TranscriptItem::Call(_) => true,
        _ => false,
    };
    assert!(!items.iter().any(archived), "{items:#?}");
    let summary = summary.map(TranscriptItem::Message);
    assert!(summary.is_some_and(|it| items.contains(&&it)), "{items:#?}");

    // Narrower, the old question wraps to more rows.
    let wide = rows_at(&mut app, agent, 80);
    let narrow = rows_at(&mut app, agent, 30);
    let question = |rows: &[String]| {
        let first = line_of(rows, "old question");
        let output = line_of(rows, "old output");
        first.zip(output).map(|(first, output)| output - first)
    };
    assert!(
        question(&narrow) > question(&wide),
        "{wide:#?}\n{narrow:#?}"
    );
}

/// A summary of a summary draws what each replaced in its place; what a
/// summary no longer in the conversation replaced is drawn nowhere.
#[test]
fn each_summary_draws_what_it_replaced_while_it_is_in_the_conversation() {
    let (mut app, agent) = view_app();
    for text in ["first", "second", "third", "fourth"] {
        say(&mut app, agent, Message::user(text));
    }
    compact(&mut app, agent, 2, "summary one");
    compact(&mut app, agent, 2, "summary two");
    let lines = on_screen(&mut app, agent, 100);
    let order = [
        "first",
        "second",
        "── 2 messages above were compacted ──",
        "summary one",
        "third",
        "── 3 messages above were compacted ──",
        "summary two",
        "fourth",
    ];
    let order = order.map(|text| line_of(&lines, text));
    assert!(order.iter().all(Option::is_some), "{lines:#?}");
    assert!(order.is_sorted(), "{lines:#?}");

    // A rewind past both summaries.
    if let Some(mut conversation) = app.world_mut().get_mut::<Conversation>(agent) {
        conversation.splice(..1, [], Author::user());
    }
    app.update();
    let lines = on_screen(&mut app, agent, 100);
    assert!(line_of(&lines, "fourth").is_some(), "{lines:#?}");
    let gone = ["first", "summary one", "third", "compacted"];
    assert!(
        gone.iter().all(|text| line_of(&lines, text).is_none()),
        "{lines:#?}"
    );
}

/// What a summary replaced, filled in after the summary went in, is drawn
/// at the next frame.
#[test]
fn filling_what_a_summary_replaced_draws_a_frame() {
    let (mut app, agent) = view_app();
    app.insert_resource(crate::TuiTestScreen::new(80, 30));
    say(&mut app, agent, Message::user("old question"));
    say(&mut app, agent, Message::user("kept prompt"));
    let summary = compact(&mut app, agent, 1, "the summary");
    let screen = |app: &App| app.world().resource::<crate::TuiTestScreen>().rows();
    let taken = app
        .world_mut()
        .get_mut::<crate::Replaced>(agent)
        .and_then(|mut replaced| replaced.0.remove(&summary?));
    app.update();
    assert!(line_of(&screen(&app), "old question").is_none());
    if let (Some(mut replaced), Some(summary), Some(taken)) = (
        app.world_mut().get_mut::<crate::Replaced>(agent),
        summary,
        taken,
    ) {
        replaced.0.insert(summary, taken);
    }
    app.update();
    let rows = screen(&app);
    assert!(line_of(&rows, "old question").is_some(), "{rows:#?}");
}
