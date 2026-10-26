use std::sync::Arc;

use bevy_ecs::system::RunSystemOnce;
use rig_harness::prelude::*;
use rig_harness::rig_memory::MemoryStore;

use super::*;
use crate::tests::{transcript, view_app};

/// Places for plugins' files, all in memory, the session's in `session`.
fn data(session: &MemoryStore) -> PluginData {
    PluginData {
        session: Arc::new(session.clone()),
        project: Arc::new(MemoryStore::default()),
        user: Arc::new(MemoryStore::default()),
    }
}

/// Where in `lines` the first that contains `text` is.
fn line_of(lines: &[String], text: &str) -> Option<usize> {
    lines.iter().position(|line| line.contains(text))
}

#[test]
fn a_notice_stays_before_what_an_edit_adds_to_the_message_it_followed() {
    use rig_core::completion::Message;
    use rig_core::message::UserContent;
    let (mut app, agent) = view_app();
    let note = Deliver::new(agent, "first", DeliveryMode::Inform);
    app.world_mut().trigger(note);
    app.update();
    app.world_mut()
        .write_message(Notice::info(agent, "Interrupted."));
    app.update();
    // A plugin's edit adds a second text to the message, which keeps its id.
    if let Some(mut conversation) = app.world_mut().get_mut::<Conversation>(agent)
        && let Some(mut entry) = conversation.entry(0)
        && let Message::User { content } = std::sync::Arc::make_mut(&mut entry.message)
    {
        content.push(UserContent::text("second"));
        conversation.splice(..1, [entry], Author::plugin("test"));
    }
    app.update();
    let messages = app
        .world()
        .get::<Conversation>(agent)
        .map(|c| c.messages().len());
    assert_eq!(messages, Some(1));
    let lines = transcript(&mut app, agent);
    let order = ["first", "Interrupted.", "second"].map(|text| line_of(&lines, text));
    assert!(order.iter().all(Option::is_some), "{lines:#?}");
    assert!(order.is_sorted(), "{lines:#?}");
}

#[test]
fn notices_are_kept_for_the_build_a_reload_starts_and_for_no_other_run() {
    let session = MemoryStore::default();
    let build = || {
        let (mut app, agent) = view_app();
        app.world_mut()
            .entity_mut(agent)
            .insert(AgentId("a1".to_owned()));
        app.insert_resource(data(&session));
        let loaded = app.world_mut().run_system_once(load_notices);
        assert!(loaded.is_ok());
        (app, agent)
    };
    let shown = |app: &mut App, agent: Entity, text: &str| {
        let lines = transcript(app, agent);
        line_of(&lines, text).is_some()
    };
    // A run that quits keeps nothing for the next run, nor for a fork,
    // which copies the session's files.
    let (mut app, agent) = build();
    app.world_mut()
        .write_message(Notice::warn(agent, "Shown in the run that quit."));
    app.update();
    app.insert_resource(ExitReason::Quit);
    assert!(app.world_mut().run_system_once(keep_notices).is_ok());
    drop(app);
    let (mut app, agent) = build();
    assert!(!shown(&mut app, agent, "Shown in the run that quit."));
    // A reload keeps them for its build, once.
    app.world_mut()
        .write_message(Notice::warn(agent, "Kept across a reload."));
    app.update();
    app.insert_resource(ExitReason::Reload);
    assert!(app.world_mut().run_system_once(keep_notices).is_ok());
    drop(app);
    let (mut app, agent) = build();
    assert!(shown(&mut app, agent, "Kept across a reload."));
    drop(app);
    let (mut app, agent) = build();
    assert!(!shown(&mut app, agent, "Kept across a reload."));
}

#[test]
fn a_notice_sent_as_a_reload_exits_is_kept_for_its_build() {
    let session = MemoryStore::default();
    let (mut app, _) = view_app();
    // As the session log reports a failed write, at the end of the frame.
    let fail = |mut notices: MessageWriter<Notice>| {
        notices.write(Notice::error(None, "The session is no longer saved: full"));
    };
    app.insert_resource(data(&session))
        .insert_resource(ExitReason::Reload)
        .add_systems(
            Last,
            fail.in_set(rig_ecs::WriteLogs)
                .run_if(on_message::<AppExit>),
        );
    app.world_mut().write_message(AppExit::Success);
    app.update();
    drop(app);
    let (mut app, _) = view_app();
    app.insert_resource(data(&session));
    assert!(app.world_mut().run_system_once(load_notices).is_ok());
    let notices = app.world().resource::<Notices>().all();
    let texts: Vec<&str> = notices.iter().map(|notice| notice.text.as_str()).collect();
    assert_eq!(texts, ["The session is no longer saved: full"]);
}

#[test]
fn what_was_typed_and_not_sent_is_kept_across_a_reload() {
    let session = MemoryStore::default();
    let (mut app, _) = view_app();
    app.insert_resource(data(&session))
        .insert_resource(ExitReason::Reload);
    // A key in it, as in a message that came back, never reaches the disk.
    let typed = "/tools OPENAI_API_KEY=sk-proj-abcdefghijklmnop1234";
    crate::tests::edit(&mut app, |input| input.set(typed));
    assert!(app.world_mut().run_system_once(keep_draft).is_ok());
    drop(app);
    let (mut app, _) = view_app();
    app.insert_resource(data(&session));
    assert!(app.world_mut().run_system_once(load_draft).is_ok());
    let kept = "/tools OPENAI_API_KEY=[REDACTED]";
    assert_eq!(crate::tests::input(&app).text(), kept);
    // Once: the build after this one starts empty.
    let (mut app, _) = view_app();
    app.insert_resource(data(&session));
    assert!(app.world_mut().run_system_once(load_draft).is_ok());
    assert!(crate::tests::input(&app).is_empty());
}

/// Adds `message` to `agent`'s conversation.
fn say(app: &mut App, agent: Entity, message: rig_core::completion::Message) {
    if let Some(mut conversation) = app.world_mut().get_mut::<Conversation>(agent) {
        conversation.push(message);
    }
    app.update();
}

#[test]
fn a_notice_says_which_transcript_draws_it_and_where() {
    use rig_core::completion::Message;
    let (mut app, agent) = view_app();
    let id = AgentId("a1".to_owned());
    app.world_mut().entity_mut(agent).insert(id.clone());
    app.world_mut().resource_mut::<ShownAgent>().0 = Some(agent);
    say(&mut app, agent, Message::user("question"));
    say(&mut app, agent, Message::assistant("answer"));
    // About every agent: drawn where the shown agent was.
    app.world_mut()
        .write_message(Notice::info(Option::<Entity>::None, "Every agent."));
    app.update();
    say(&mut app, agent, Message::user("next"));
    let world = app.world();
    let notice = world.resource::<crate::Notices>().all().first();
    let conversation = world.get::<Conversation>(agent);
    assert_eq!(notice.and_then(ShownNotice::drawn_in), Some(&id));
    let slot = |id: &AgentId| notice?.slot(id, conversation?);
    assert_eq!(slot(&id), Some(TranscriptSlot::After(1)));
    assert_eq!(slot(&AgentId("a2".to_owned())), None);
    // Slots order as drawn.
    let order = [
        TranscriptSlot::Top,
        TranscriptSlot::Inside(0, 1),
        TranscriptSlot::After(0),
        TranscriptSlot::Inside(1, 0),
        TranscriptSlot::After(1),
    ];
    assert!(order.is_sorted());
}

#[test]
fn a_notice_about_an_agent_gone_before_it_is_read_is_drawn_in_no_other_transcript() {
    use rig_core::completion::Message;
    let (mut app, agent) = view_app();
    app.world_mut().resource_mut::<ShownAgent>().0 = Some(agent);
    say(&mut app, agent, Message::user("question"));
    // A helper whose model call fails retires in the same pass.
    let helper = app.world_mut().spawn(Agent).id();
    app.update();
    app.world_mut().write_message(Notice::error(
        helper,
        "Your last message was taken out of the conversation; send it again.",
    ));
    app.world_mut().despawn(helper);
    app.update();
    let lines = transcript(&mut app, agent);
    assert!(line_of(&lines, "send it again").is_none(), "{lines:#?}");
    assert!(line_of(&lines, "question").is_some(), "{lines:#?}");
}

#[test]
fn a_notice_is_an_item_a_plugin_scrolls_to() {
    let (mut app, agent) = view_app();
    let pane = crate::tests::main_pane(app.world());
    let question = |index: usize| rig_core::completion::Message::user(format!("question {index}"));
    for index in 0..10 {
        say(&mut app, agent, question(index));
    }
    app.world_mut()
        .write_message(Notice::warn(agent, "Rate limited."));
    app.update();
    for index in 10..30 {
        say(&mut app, agent, question(index));
    }
    let notices = app.world().resource::<crate::Notices>().all().len();
    assert_eq!(notices, 1);
    let item = TranscriptItem::Notice(0);
    app.world_mut().trigger(ScrollTo {
        pane,
        item: Some(item.clone()),
        row: ItemRow::Top(0),
    });
    let lines = crate::tests::on_screen(&mut app, agent, 5);
    assert_eq!(lines.first().map(String::as_str), Some("Rate limited."));
    let top = crate::tests::main_view(&app).top().cloned();
    assert_eq!(top, Some((item, 0)));
}

#[test]
fn a_notice_keeps_its_name_once_older_ones_are_dropped() {
    let (mut app, agent) = view_app();
    let pane = crate::tests::main_pane(app.world());
    say(
        &mut app,
        agent,
        rig_core::completion::Message::user("question"),
    );
    let notice = |app: &mut App, index: usize| {
        let text = format!("notice {index}");
        app.world_mut().write_message(Notice::info(agent, text));
        app.update();
    };
    let scroll_to = |app: &mut App, item: TranscriptItem| {
        let row = ItemRow::Top(0);
        app.world_mut().trigger(ScrollTo {
            pane,
            item: Some(item),
            row,
        });
        crate::tests::on_screen(app, agent, 5).first().cloned()
    };
    (0..KEPT_NOTICES).for_each(|index| notice(&mut app, index));
    let shown = scroll_to(&mut app, TranscriptItem::Notice(KEPT_NOTICES / 2));
    assert_eq!(shown.as_deref(), Some("notice 100"));
    let top = crate::tests::main_view(&app).top().cloned();
    // The oldest goes; the item named at the top still names that notice.
    notice(&mut app, KEPT_NOTICES);
    let item = top.map(|(item, _)| item);
    assert_eq!(item.and_then(|item| scroll_to(&mut app, item)), shown);
}

/// A transcript of notes of `said`, laid out at 12 columns.
fn laid_out_narrow(said: &[&str]) -> Option<DrawnTranscript> {
    let (mut app, agent) = view_app();
    for text in said {
        let note = Deliver::new(agent, *text, DeliveryMode::Inform);
        app.world_mut().trigger(note);
        app.update();
    }
    let lay_out = move |world: &mut World| {
        crate::TranscriptLayout::lay_out(world, &TranscriptView::default(), agent, 12)
    };
    app.world_mut().run_system_once(lay_out).ok().flatten()
}

#[test]
fn rows_a_line_wrapped_into_rejoin_into_it() {
    let said = [
        "alpha beta  gamma delta epsilon zeta",
        "abcdefghijklmnopqrstuvwxyz",
    ];
    let laid_out = laid_out_narrow(&said);
    assert!(laid_out.is_some());
    let Some(drawn) = laid_out else { return };
    let rows: Vec<&str> = drawn.rows().iter().map(|row| row.text.as_str()).collect();
    assert!(
        !rows.iter().any(|row| row.contains("gamma delta")),
        "{rows:#?}"
    );
    let lines: Vec<DrawnLine> = drawn.lines().collect();
    for text in said {
        let line = lines.iter().find(|line| line.text.ends_with(text));
        assert!(line.is_some(), "{text} not rejoined: {lines:#?}");
        let Some(line) = line else { return };
        assert_eq!(line.text, format!("› {text}"));
        assert!(line.rows.len() > 1, "{rows:#?}");
        let continued = drawn.rows().get(line.rows.clone()).unwrap_or_default();
        let continues = continued.iter().map(|row| row.continues.is_some());
        assert!(
            continues.skip(1).all(|continues| continues),
            "{continued:#?}"
        );
    }
    let total: usize = lines.iter().map(|line| line.rows.len()).sum();
    assert_eq!(total, drawn.rows().len());
}

#[test]
fn a_line_says_where_each_of_its_rows_starts_in_it() {
    let laid_out = laid_out_narrow(&["alpha beta  gamma delta epsilon zeta"]);
    assert!(laid_out.is_some());
    let Some(drawn) = laid_out else { return };
    let line = drawn.lines().find(|line| line.text.contains("gamma"));
    assert!(line.is_some(), "{drawn:#?}");
    let Some(line) = line else { return };
    assert_eq!(line.starts.len(), line.rows.len());
    for (row, start) in line.rows.clone().zip(&line.starts) {
        let drawn_row = drawn.rows().get(row).map(|row| row.text.trim_start());
        let from = line.text.get(*start..).unwrap_or_default();
        assert!(
            drawn_row.is_some_and(|text| from.starts_with(text)),
            "{row}: {line:#?}"
        );
        assert_eq!(line.row_at(*start), row);
    }
    let gamma = line.text.find("gamma").unwrap_or_default();
    let row = drawn.rows().get(line.row_at(gamma));
    assert!(
        row.is_some_and(|row| row.text.contains("gamma")),
        "{line:#?}"
    );
    // The blanks a wrap dropped end the row before.
    let blank = line.text.find("  ").unwrap_or_default();
    assert_eq!(line.row_at(blank + 1), line.row_at(blank));
}

#[test]
fn what_comes_back_to_an_agent_not_shown_waits_until_it_is() {
    let (mut app, _) = view_app();
    let other = app.world_mut().spawn(Agent).id();
    for text in ["deploy it", "then tag it"] {
        let text = text.to_owned();
        app.world_mut().write_message(Recalled {
            agent: other,
            text,
            why: None,
        });
        app.update();
    }
    assert_eq!(crate::tests::input(&app).text(), "");
    crate::tests::edit(&mut app, |input| input.set("draft"));
    app.world_mut().insert_resource(ShownAgent(Some(other)));
    app.update();
    let back = "deploy it\n\nthen tag it\n\ndraft";
    assert_eq!(crate::tests::input(&app).text(), back);
}
