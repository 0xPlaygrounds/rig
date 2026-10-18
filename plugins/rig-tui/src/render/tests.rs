use ratatui::style::Style;
use ratatui::text::Line;

use super::light;
use crate::Highlight;

/// The screen after drawing `shown` as `choosing` says, 120 columns by
/// 20 rows, a line per row.
fn drawn_choice(
    choosing: &mut super::super::pick::PickerInput,
    shown: &super::super::pick::Listing<'_>,
) -> String {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use ratatui::layout::Position;

    let Ok(mut terminal) = Terminal::new(TestBackend::new(120, 20));
    let drawn = terminal.draw(|frame| {
        super::draw_choice(frame, choosing, shown, None);
    });
    assert!(drawn.is_ok());
    let buffer = terminal.backend().buffer();
    let rows = (0..buffer.area.height).map(|y| {
        let cells = (0..buffer.area.width).filter_map(|x| buffer.cell(Position::new(x, y)));
        cells.map(|cell| cell.symbol()).collect::<String>()
    });
    rows.collect::<Vec<_>>().join("\n")
}

#[test]
fn a_question_shows_its_body_choices_and_typed_answer() {
    use super::super::pick::{Item, Listing, PickerInput, Row};

    let mut shown = Listing {
        title: "Allow `rm -rf build`?",
        body: "It matches `rm -rf`.\nType a reason to deny it with.",
        items: ["Deny", "Allow once", "Allow for this session"]
            .map(|label| Item {
                id: label,
                label,
                detail: None,
                checked: false,
                more: None,
            })
            .into(),
        free_text: true,
        ticks: true,
        multiple: false,
        selected: 0,
        edit: None,
        keys: &[],
    };
    let mut choosing = PickerInput {
        filter: "allow".to_owned(),
        selected: 0,
        ticked: Vec::new(),
        scroll: 0,
        draft: None,
        more: false,
        touched: false,
        picked: None,
    };
    // What is typed filters the choices and is an answer of its own.
    assert_eq!(
        shown.rows(&choosing.filter),
        [Row::Item(1), Row::Item(2), Row::Typed]
    );
    let screen = drawn_choice(&mut choosing, &shown);
    for shown in [
        "Allow `rm -rf build`?",
        "It matches `rm -rf`.",
        "› allow",
        "Allow once",
        "Allow for this session",
        "✎ allow",
    ] {
        assert!(screen.contains(shown), "{shown} not in\n{screen}");
    }
    assert!(!screen.contains("Deny"), "{screen}");

    // A body longer than the box scrolls, to its end at most, and leaves
    // the list its rows.
    let body: Vec<String> = (1..=30).map(|n| format!("body line {n}")).collect();
    let body = body.join("\n");
    shown.body = &body;
    let screen = drawn_choice(&mut choosing, &shown);
    assert!(screen.contains("Shift+↑↓ scrolls"), "{screen}");
    assert!(screen.contains("body line 1 "), "{screen}");
    assert!(!screen.contains("body line 30"), "{screen}");
    choosing.scroll = 100;
    let screen = drawn_choice(&mut choosing, &shown);
    assert!(screen.contains("body line 30"), "{screen}");
    assert!(!screen.contains("body line 1 "), "{screen}");
    assert!(screen.contains("Allow for this session"), "{screen}");
    assert_eq!(choosing.scroll, 30 - 8);
}

#[test]
fn a_panel_below_the_input_takes_the_rows_under_the_input_box() {
    use bevy_ecs::prelude::*;
    use ratatui::layout::{Constraint, Rect};

    use crate::tests::view_app;
    use crate::{Editor, Placement, StatusLine, TuiFrame, TuiPanel, TuiTestScreen};

    let (mut app, _) = view_app();
    app.insert_resource(TuiTestScreen::new(80, 24));
    let world = app.world_mut();
    let suggestions = world
        .spawn(TuiPanel::new(Placement::BelowInput(Constraint::Length(1))))
        .id();
    // Too tall to leave the transcript its rows: skipped.
    let tall = TuiPanel::new(Placement::BelowInput(Constraint::Length(30)));
    let tall = world.spawn(tall).id();
    app.update();
    let world = app.world_mut();
    let input = world.query_filtered::<Entity, With<Editor>>().single(world);
    let status = world
        .query_filtered::<Entity, With<StatusLine>>()
        .single(world);
    let frame = world.resource::<TuiFrame>();
    let area = |entity: Result<Entity, _>| entity.ok().and_then(|entity| frame.area(entity));
    let (input, status) = (area(input), area(status));
    assert_eq!(
        area(Ok(suggestions)),
        Some(Rect::new(0, 23, 80, 1)),
        "{frame:?}"
    );
    assert_eq!(area(Ok(tall)).filter(|area| !area.is_empty()), None);
    assert_eq!(input, Some(Rect::new(0, 20, 80, 3)), "{frame:?}");
    assert_eq!(status, Some(Rect::new(0, 19, 80, 1)));
}

#[test]
fn panels_are_laid_out_in_the_order_spawned_whatever_their_entity_index() {
    use bevy_ecs::prelude::*;
    use ratatui::layout::Constraint;

    use crate::tests::view_app;
    use crate::{Placement, TuiFrame, TuiPanel, TuiTestScreen};

    let (mut app, _) = view_app();
    app.insert_resource(TuiTestScreen::new(80, 24));
    let world = app.world_mut();
    let gone: Vec<Entity> = (0..200).map(|_| world.spawn_empty().id()).collect();
    let left = || TuiPanel::new(Placement::Left(Constraint::Length(10)));
    let older = world.spawn(left()).id();
    app.update();
    // Bevy hands out the indices of despawned entities again, in batches.
    let world = app.world_mut();
    gone.into_iter().for_each(|entity| {
        world.despawn(entity);
    });
    let newer = world.spawn(left()).id();
    assert!(newer.index() < older.index());
    app.update();
    let frame = app.world().resource::<TuiFrame>();
    let x = |panel| frame.area(panel).map(|area| area.x);
    assert_eq!((x(older), x(newer)), (Some(0), Some(10)), "{frame:?}");
}

#[test]
fn a_highlight_may_match_case_and_the_one_placed_first_wins() {
    let line = Line::from("Error: error at ERROR");
    let (current, rest) = (Style::new().on_red(), Style::new().on_yellow());
    let highlight = |text: &str, style: Style, match_case: bool| Highlight {
        text: text.to_owned(),
        style,
        match_case,
        spans: Vec::new(),
    };
    // The texts lit in `style`.
    let lit = |highlights: &[Highlight], style: Style| -> Vec<String> {
        let lights = crate::render::Lights {
            highlights,
            ..Default::default()
        };
        let lit = light(std::slice::from_ref(&line), &[None], 0, &lights);
        let spans = lit.iter().flat_map(|row| row.spans.iter());
        let spans = spans.filter(|span| span.style.bg == style.bg);
        spans.map(|span| span.content.to_string()).collect()
    };
    let any_case = [highlight("error", rest, false)];
    assert_eq!(lit(&any_case, rest), ["Error", "error", "ERROR"]);
    let exact = [highlight("error", rest, true)];
    assert_eq!(lit(&exact, rest), ["error"]);
    // The current match, placed first, is set apart from the rest.
    let both = [highlight("Error:", current, true), any_case[0].clone()];
    assert_eq!(lit(&both, current), ["Error:"]);
    assert_eq!(lit(&both, rest), ["error", "ERROR"]);
}

#[test]
fn a_test_screen_shows_the_frame_and_where_each_part_and_item_went() {
    use rig_core::completion::Message;
    use rig_harness::prelude::*;

    use crate::tests::{type_text, view_app};
    use crate::{TerminalWrite, TranscriptItem, TuiFrame, TuiTestScreen};

    let (mut app, agent) = view_app();
    app.insert_resource(TuiTestScreen::new(60, 20));
    crate::tests::keep_drawn(&mut app);
    if let Some(mut conversation) = app.world_mut().get_mut::<Conversation>(agent) {
        conversation.push(Message::user("where is the config read?"));
    }
    app.world_mut()
        .write_message(TerminalWrite(b"\x1b]9;done\x07".to_vec()));
    app.update();
    let world = app.world();
    let screen = world.resource::<TuiTestScreen>();
    let rows = screen.rows();
    assert_eq!(rows.len(), 20);
    assert!(screen.written().ends_with(b"\x1b]9;done\x07"));
    // The message's rows are on screen, where the frame says the
    // transcript is.
    let frame = world.resource::<TuiFrame>().clone();
    assert_eq!((frame.screen.width, frame.screen.height), (60, 20));
    let transcript = frame
        .area(crate::tests::main_pane(world))
        .unwrap_or_default();
    let input = world.try_query_filtered::<Entity, With<crate::MainEditor>>();
    let input = input.and_then(|mut input| input.single(world).ok());
    let input = input
        .and_then(|input| frame.area(input))
        .unwrap_or_default();
    assert!(transcript.height > 0 && input.y > transcript.y);
    let drawn = crate::tests::drawn(&app);
    let id = world.get::<Conversation>(agent).and_then(|c| c.id(0));
    let message = id.and_then(|id| drawn.rows_of(&TranscriptItem::Message(id)));
    let on_screen = message.into_iter().flatten().filter_map(|row| {
        let at = frame.screen_row(drawn, row)?;
        rows.get(usize::from(at))
    });
    let texts: Vec<&String> = on_screen.collect();
    assert!(
        texts
            .iter()
            .any(|row| row.contains("where is the config read?")),
        "{texts:#?} in {rows:#?}"
    );
    assert!(frame.overlays.is_empty());
    // A completion list covers the transcript.
    app.add_command("model", "Choose the model", |_: In<CommandArgs>| {});
    crate::tests::edit(&mut app, |input| input.set(""));
    type_text(&mut app, "/mo");
    let frame = app.world().resource::<TuiFrame>();
    assert_eq!(frame.overlays.len(), 1, "{frame:?}");
    assert!(frame.covered(transcript));
}

/// A repaint asks for a frame and writes its cells again, clipped to the
/// screen, though nothing in them changed.
#[test]
fn a_repaint_writes_its_cells_again_on_the_next_frame() {
    use crate::tests::view_app;
    use crate::{RedrawArea, TuiTestScreen};
    use ratatui::layout::Rect;

    let (mut app, _) = view_app();
    app.insert_resource(TuiTestScreen::new(60, 20));
    app.update();
    app.update();
    assert!(
        app.world()
            .resource::<TuiTestScreen>()
            .repainted()
            .is_empty()
    );
    app.world_mut()
        .write_message(RedrawArea(Rect::new(10, 3, 30, 2)));
    app.world_mut()
        .write_message(RedrawArea(Rect::new(50, 19, 30, 4)));
    app.update();
    assert_eq!(
        app.world().resource::<TuiTestScreen>().repainted(),
        [Rect::new(10, 3, 30, 2), Rect::new(50, 19, 10, 1)]
    );
}
