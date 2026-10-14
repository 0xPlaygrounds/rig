use bevy_ecs::system::RunSystemOnce;
use rig_core::completion::Message;
use rig_core::conversation::Conversation;
use rig_harness::prelude::*;

use super::ConversationPanel;
use crate::tests::{press, type_text, view_app};
use crate::{Expanded, KeyFocus, TranscriptItem, TranscriptLayout, TranscriptView, TuiTestScreen};

/// Two turns, each answer `lines` lines long.
fn two_turns(lines: usize) -> Conversation {
    let answer = |turn: &str| {
        let lines: Vec<String> = (0..lines).map(|at| format!("{turn} answer {at}")).collect();
        Message::assistant(lines.join("\n\n"))
    };
    Conversation::from_messages([
        Message::user("first question"),
        answer("first"),
        Message::user("second question"),
        answer("second"),
    ])
}

/// A panel opened at the second turn shows it at its top, takes the keys,
/// and Esc closes it and gives the keys back to the input.
#[test]
fn a_conversation_panel_opens_at_its_turn_and_esc_closes_it() {
    let (mut app, _) = view_app();
    app.insert_resource(TuiTestScreen::new(60, 20));
    let conversation = two_turns(20);
    let second = conversation.id(2).map(TranscriptItem::Message);
    let panel = ConversationPanel {
        title: "saved · turn 2".to_owned(),
        conversation,
        at: second,
    };
    let panel = app.world_mut().spawn(panel).id();
    app.world_mut().resource_mut::<KeyFocus>().0 = Some(panel);
    app.update();
    app.update();
    let rows = app.world().resource::<TuiTestScreen>().rows();
    let title = rows.iter().position(|row| row.contains("saved · turn 2"));
    let asked = rows.iter().position(|row| row.contains("second question"));
    assert!(
        matches!((title, asked), (Some(title), Some(asked)) if asked <= title + 2),
        "{rows:#?}"
    );
    assert!(
        !rows.iter().any(|row| row.contains("first answer")),
        "{rows:#?}"
    );

    press(&mut app, "esc");
    assert!(app.world().get_entity(panel).is_err());
    assert_eq!(app.world().resource::<KeyFocus>().0, None);
    type_text(&mut app, "hi");
    assert_eq!(crate::tests::typed(&app), "hi");
}

/// A conversation laid out as a value gives the rows an agent holding it
/// gives.
#[test]
fn a_conversation_value_is_laid_out_as_an_agents() {
    let (mut app, agent) = view_app();
    let conversation = two_turns(2);
    if let Some(mut held) = app
        .world_mut()
        .get_mut::<rig_ecs::agent::Conversation>(agent)
    {
        held.0 = conversation.clone();
    }
    app.update();
    let rows = move |layout: TranscriptLayout, view: Res<TranscriptView>| {
        let texts = |drawn: crate::DrawnTranscript| -> Vec<String> {
            drawn.rows().iter().map(|row| row.text.clone()).collect()
        };
        let live = layout.lay_out(&view, agent, 60, Expanded::All).map(texts);
        let value = layout.lay_out_conversation(&view, &conversation, 60, Expanded::All);
        (live, texts(value))
    };
    let (live, value) = app.world_mut().run_system_once(rows).unwrap_or_default();
    assert!(
        value.iter().any(|row| row.contains("second answer 1")),
        "{value:?}"
    );
    assert_eq!(live, Some(value));
}
