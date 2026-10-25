use bevy_ecs::system::RunSystemOnce;
use rig_ecs::prelude::*;

use super::super::editor::Editor;
use super::super::history::Sent;
use super::super::view::{self, TuiView};
use super::send;
use rig_harness::commands::{AppCommandsExt, CommandArgs};
use rig_harness::main_agent::ShownAgent;

/// The texts agents accepted.
#[derive(Resource, Default)]
struct Delivered(Vec<String>);

/// Types `text` in the view and presses Enter.
fn enter(app: &mut App, text: &str) {
    let text = text.to_owned();
    let sent = app.world_mut().run_system_once(
        move |mut view: ResMut<TuiView>,
              mut input: crate::MainInput,
              shown: Res<ShownAgent>,
              mut commands: Commands| {
            input.set(text.clone());
            send(&mut view, &mut input, None, shown.0, &mut commands, false);
        },
    );
    assert!(sent.is_ok());
    app.update();
}

#[test]
fn a_refused_command_comes_back_whole_and_enter_sends_it_as_a_message() {
    let mut app = App::new();
    app.add_plugins(AgentPlugin)
        .add_command(
            "quiet",
            "Refuses arguments",
            |In(args): In<CommandArgs>, mut notices: MessageWriter<Notice>| {
                if !args.args.is_empty() {
                    let why = format!("/quiet takes no arguments, not `{}`.", args.args);
                    notices.write(Notice::error(args.agent, why));
                }
            },
        )
        .add_message::<Sent>()
        .init_resource::<TuiView>()
        .init_resource::<view::Notices>()
        .init_resource::<ShownAgent>()
        .init_resource::<Delivered>()
        .add_observer(
            |accepted: On<Add<Started>>,
             messages: Query<&Incoming>,
             mut delivered: ResMut<Delivered>| {
                if let Ok(message) = messages.get(accepted.entity) {
                    delivered.0.push(message.text().to_owned());
                }
            },
        )
        .add_systems(
            Update,
            (view::collect_notices, view::recall_messages).after(AgentLoop),
        );
    app.world_mut()
        .spawn((Editor::default(), crate::MainEditor));
    app.update();
    let world = app.world_mut();
    let agent = world
        .query_filtered::<Entity, With<Agent>>()
        .single(world)
        .ok();
    world.insert_resource(ShownAgent(agent));
    let refused = [
        ("/quiet now", "/quiet takes no arguments, not `now`."),
        ("/gui still doesnt do anything", "Unknown command /gui."),
    ];
    for (typed, why) in refused {
        enter(&mut app, typed);
        let notices = app.world().resource::<view::Notices>();
        let input = crate::tests::input(&app);
        let shown: Vec<&str> = notices
            .all()
            .iter()
            .map(|notice| notice.text.as_str())
            .collect();
        assert_eq!(input.text(), typed);
        assert!(
            matches!(shown.as_slice(), [.., last] if last.starts_with(why)),
            "{shown:?}"
        );
    }
    assert_eq!(
        app.world().resource::<crate::Notices>().all().len(),
        refused.len()
    );
    enter(&mut app, "/gui still doesnt do anything");
    assert_eq!(
        app.world().resource::<Delivered>().0,
        ["/gui still doesnt do anything"]
    );
}

#[test]
fn a_paste_goes_to_the_panel_with_the_keys_not_the_input() {
    use crate::tests::{push, typed, view_app};
    use crate::{KeyFocus, KeyInput, Placement, TuiPanel};
    use crossterm::event::Event;
    use ratatui::layout::Constraint;

    let (mut app, _) = view_app();
    let panel = TuiPanel::new(Placement::Top(Constraint::Length(3)));
    let panel = app.world_mut().spawn(panel).id();
    app.world_mut().resource_mut::<KeyFocus>().0 = Some(panel);
    let paste = |app: &mut App| {
        push(app, KeyInput::Event(Event::Paste("cargo test".to_owned())));
        app.update();
    };
    paste(&mut app);
    assert_eq!(typed(&app), "");
    // Given back the keys, the input takes it.
    app.world_mut().resource_mut::<KeyFocus>().0 = None;
    paste(&mut app);
    assert_eq!(typed(&app), "cargo test");
}

/// Keys a modal filter turned into actions, one at a time, as vim's
/// normal mode would: `d` waits for its motion.
fn normal_mode(mut events: ResMut<crate::PendingInput>, mut deleting: Local<bool>) {
    use crate::{InputAction, KeyInput, KeyPress};
    use crossterm::event::{Event, KeyCode};
    let key = |item: &KeyInput| match item {
        KeyInput::Event(Event::Key(key)) => Some(KeyPress::of(key).code),
        _ => None,
    };
    let mut out = Vec::new();
    for item in std::mem::take(&mut events.0) {
        match (key(&item), std::mem::take(&mut *deleting)) {
            (Some(KeyCode::Esc), _) => {}
            (Some(KeyCode::Char('0')), _) => out.push(KeyInput::Action(InputAction::LineStart)),
            (Some(KeyCode::Char('d')), _) => *deleting = true,
            (Some(KeyCode::Char('w')), true) => {
                out.push(KeyInput::Action(InputAction::DeleteWordAfter));
            }
            _ => out.push(item),
        }
    }
    events.0 = out;
}

#[test]
fn a_filter_turns_the_frames_keys_into_actions_in_their_order() {
    use crate::tests::{press, type_text, typed, view_app};
    use crate::{InputSystems, TuiKeys};

    let (mut app, _) = view_app();
    type_text(&mut app, "hello world");
    app.add_systems(TuiKeys, normal_mode.in_set(InputSystems::Filter));
    // One frame: Esc is dropped, `0` goes to the start, `dw` deletes the
    // word; nothing is typed, and an Esc that reached the view would have
    // interrupted the agent instead.
    press(&mut app, "esc 0 d w");
    assert_eq!(typed(&app), " world");
    // Keys it leaves alone are the view's as before.
    press(&mut app, "x");
    assert_eq!(typed(&app), "x world");
}

/// The input as each key a filter saw found it.
#[derive(Resource, Default)]
struct Seen(Vec<String>);

/// Notes the input as the key being done finds it.
fn note_input(
    events: Res<crate::PendingInput>,
    input: Single<&Editor, With<crate::MainEditor>>,
    mut seen: ResMut<Seen>,
) {
    if !events.0.is_empty() {
        seen.0.push(input.text().to_owned());
    }
}

#[test]
fn a_filter_sees_each_key_with_the_input_the_keys_before_it_left() {
    use crate::tests::{press, view_app};
    use crate::{InputSystems, TuiKeys};

    let (mut app, _) = view_app();
    app.init_resource::<Seen>()
        .add_systems(TuiKeys, note_input.in_set(InputSystems::Filter));
    press(&mut app, "a b");
    assert_eq!(app.world().resource::<Seen>().0, ["", "a"]);
}

/// The searches `/` opened.
#[derive(Resource, Default)]
struct Searches(usize);

/// `/` on an empty input opens a search instead of a slash command.
fn slash_searches(
    mut events: ResMut<crate::PendingInput>,
    input: Single<&Editor, With<crate::MainEditor>>,
    mut searches: ResMut<Searches>,
) {
    use crate::KeyInput;
    use crossterm::event::{Event, KeyCode};
    let slash = |item: &KeyInput| matches!(item, KeyInput::Event(Event::Key(key)) if key.code == KeyCode::Char('/'));
    if input.is_empty() && events.0.first().is_some_and(slash) {
        events.0.remove(0);
        searches.0 += 1;
    }
}

#[test]
fn a_plugin_takes_slash_on_an_empty_input_and_leaves_it_typed_elsewhere() {
    use crate::tests::{press, typed, view_app};
    use crate::{InputSystems, TuiKeys};

    let (mut app, _) = view_app();
    app.init_resource::<Searches>()
        .add_systems(TuiKeys, slash_searches.in_set(InputSystems::Filter));
    press(&mut app, "/");
    assert_eq!(
        (app.world().resource::<Searches>().0, typed(&app)),
        (1, String::new())
    );
    // A key a frame, as a person types: past the start, `/` is typed.
    for key in ["a", "/", "b"] {
        press(&mut app, key);
    }
    assert_eq!(
        (app.world().resource::<Searches>().0, typed(&app)),
        (1, "a/b".to_owned())
    );
}

/// The times the transcript's `/` was pressed.
#[derive(Resource, Default)]
struct Searched(usize);

#[test]
fn the_transcript_has_the_keys_while_an_item_is_selected_until_typing() {
    use crate::tests::{press, typed, view_app};
    use crate::{AppKeyBindingsExt, KeyBinding, KeyPress, KeyPressed, KeyScope, TranscriptItem};
    use crossterm::event::KeyCode;
    use rig_core::message::CallId;

    let (mut app, _) = view_app();
    let slash = KeyBinding::new(KeyPress::key(KeyCode::Char('/')), "Search the transcript");
    let searched = |_: On<KeyPressed>, mut searched: ResMut<Searched>| searched.0 += 1;
    app.init_resource::<Searched>()
        .add_key_binding(slash.in_scope(KeyScope::Transcript), searched);
    let selected = |app: &App| crate::tests::main_view(app).selected.clone();
    let select = |app: &mut App| {
        let item = TranscriptItem::Call(CallId::from_wire("c1"));
        crate::tests::steer(app, |view| view.selected = Some(item));
    };
    select(&mut app);
    press(&mut app, "/");
    assert_eq!(
        (app.world().resource::<Searched>().0, typed(&app)),
        (1, String::new())
    );
    // Typing ends the reading: then `/` is typed, a slash command.
    press(&mut app, "x /");
    assert_eq!(
        (app.world().resource::<Searched>().0, typed(&app)),
        (1, "x/".to_owned())
    );
    assert_eq!(selected(&app), None);
    // Esc on the transcript gives the keys back, and nothing else.
    select(&mut app);
    press(&mut app, "esc");
    assert_eq!((selected(&app), typed(&app)), (None, "x/".to_owned()));
}

#[test]
fn keys_type_into_the_editor_of_the_panel_that_has_them() {
    use ratatui::layout::Constraint;

    use crate::tests::{press, type_text, typed, view_app};
    use crate::{KeyFocus, Placement, TuiPanel};

    let (mut app, _) = view_app();
    let placement = Placement::Over {
        width: Constraint::Length(30),
        height: Constraint::Length(3),
    };
    let search = (Editor::default(), TuiPanel::new(placement));
    let search = app.world_mut().spawn(search).id();
    app.world_mut().resource_mut::<KeyFocus>().0 = Some(search);
    type_text(&mut app, "flaky");
    let text = |app: &App| {
        app.world()
            .get::<Editor>(search)
            .map(|e| e.text().to_owned())
    };
    assert_eq!(text(&app).as_deref(), Some("flaky"));
    assert_eq!(typed(&app), "");
    // Esc gives the keys back to the main editor.
    press(&mut app, "esc");
    type_text(&mut app, "x");
    assert_eq!(
        (text(&app).as_deref(), typed(&app).as_str()),
        (Some("flaky"), "x")
    );
}

/// How many times the view asked to quit.
#[derive(Resource, Default)]
struct Quits(usize);

#[test]
fn ctrl_d_twice_on_an_empty_input_quits_unless_the_moment_passed() {
    use crate::tests::{press, type_text, view_app};
    use rig_harness::prelude::Exit;

    let (mut app, _) = view_app();
    app.init_resource::<Quits>()
        .add_observer(|_: On<Exit>, mut quits: ResMut<Quits>| quits.0 += 1);
    let quits = |app: &App| app.world().resource::<Quits>().0;
    type_text(&mut app, "draft");
    press(&mut app, "ctrl+d ctrl+d");
    assert_eq!(quits(&app), 0, "a typed input is kept");
    press(&mut app, "ctrl+c ctrl+d");
    assert_eq!(quits(&app), 0, "the first press only arms");
    let long_ago = std::time::Instant::now().checked_sub(super::EXIT_WINDOW * 2);
    app.world_mut().resource_mut::<TuiView>().exit_armed = long_ago;
    press(&mut app, "ctrl+d");
    assert_eq!(quits(&app), 0, "once the moment passed, a press arms again");
    press(&mut app, "ctrl+d");
    assert_eq!(quits(&app), 1);
}
