use bevy_ecs::entity_disabling::Disabled;
use crossterm::event::KeyEventKind;

use super::*;
use crate::tests::{Delivered, key_events, press, type_text, typed, view_app};
use crate::view::TuiView;
use crate::{CompletionSource, Placement, TuiPanel};
use ratatui::layout::Constraint;
use rig_harness::commands::{AppCommandsExt, CommandArgs};

/// What the test bindings heard: for each, whether it was a release.
#[derive(Resource, Default)]
struct Heard(Vec<bool>);

fn heard(pressed: On<KeyPressed>, mut heard: ResMut<Heard>) {
    heard.0.push(pressed.released);
}

fn heard_count(app: &App) -> usize {
    app.world().resource::<Heard>().0.len()
}

#[test]
fn keys_read_as_they_are_written() {
    for written in [
        "ctrl+x ctrl+e",
        "alt+shift+left",
        "shift+tab",
        "f5",
        "ctrl+space",
        "A",
    ] {
        let keys: Result<Keys, String> = written.parse();
        assert_eq!(keys.map(|keys| keys.to_string()).as_deref(), Ok(written));
    }
    assert_eq!(
        "Ctrl+E".parse::<Keys>(),
        Ok(Keys::from(KeyPress::ctrl('E')))
    );
    assert_eq!(
        "shift+a".parse::<Keys>(),
        Ok(Keys::from(KeyPress::key(KeyCode::Char('A'))))
    );
    for wrong in ["hyper+x", "ctrl+bogus", "", "f99"] {
        assert!(wrong.parse::<Keys>().is_err(), "{wrong}");
    }
}

#[test]
fn a_global_binding_shadows_the_inputs_and_a_taken_key_stays_until_disabled() {
    let (mut app, _) = view_app();
    app.init_resource::<Heard>();
    let edit = || KeyBinding::new(KeyPress::ctrl('e'), "Edit in $EDITOR");
    // Ctrl+E is the input's end of the line: a global binding shadows it.
    let first = app.add_key_binding(edit(), heard);
    assert!(first.is_some());
    type_text(&mut app, "ab");
    press(&mut app, "ctrl+a ctrl+e x");
    assert_eq!((heard_count(&app), typed(&app)), (1, "xab".to_owned()));
    // A second global one is refused until the first is disabled.
    assert_eq!(app.add_key_binding(edit(), heard), None);
    if let Some(first) = first {
        app.world_mut().entity_mut(first).insert(Disabled);
    }
    assert!(app.add_key_binding(edit(), heard).is_some());
    press(&mut app, "ctrl+e");
    assert_eq!(heard_count(&app), 2);
}

#[test]
fn a_binding_reads_and_edits_the_input_before_it_is_sent() {
    let (mut app, _) = view_app();
    // A template key: the draft is wrapped in a request for a review.
    app.add_key_binding(
        KeyBinding::new(KeyPress::alt('r'), "Ask for a review"),
        |_: On<KeyPressed>, mut input: crate::MainInput| {
            let draft = input.take();
            input.set(format!("Review this: {draft}"));
            input.set_cursor(0);
            input.insert("Please. ");
        },
    );
    type_text(&mut app, "the parser");
    press(&mut app, "alt+r");
    assert_eq!(typed(&app), "Please. Review this: the parser");
    press(&mut app, "enter");
    assert_eq!(
        app.world().resource::<Delivered>().0,
        ["Please. Review this: the parser"]
    );
}

#[test]
fn a_sequence_waits_for_its_last_key() {
    let (mut app, _) = view_app();
    app.init_resource::<Heard>();
    let keys: Result<Keys, String> = "ctrl+x ctrl+e".parse();
    let Ok(keys) = keys else {
        return;
    };
    app.add_key_binding(KeyBinding::new(keys, "Edit in $EDITOR"), heard);
    // Its start may not be bound alone.
    app.add_key_binding(KeyBinding::new(KeyPress::ctrl('x'), "Cut"), heard);
    press(&mut app, "ctrl+x");
    assert_eq!(heard_count(&app), 0);
    press(&mut app, "ctrl+e");
    assert_eq!(heard_count(&app), 1);
    // A sequence that goes wrong is dropped whole.
    press(&mut app, "ctrl+x a");
    assert_eq!((heard_count(&app), typed(&app)), (1, String::new()));
}

#[test]
fn a_focused_panel_takes_its_keys_and_esc_gives_them_back() {
    let (mut app, _) = view_app();
    app.init_resource::<Heard>();
    let panel = TuiPanel::new(Placement::Top(Constraint::Length(3)));
    let panel = app.world_mut().spawn(panel).id();
    // Up in the panel is no conflict with Up in the input.
    let up = KeyBinding::new(KeyPress::key(KeyCode::Up), "Select the file above");
    app.add_key_binding(up.in_scope(KeyScope::Panel(panel)), heard);
    app.world_mut().resource_mut::<KeyFocus>().0 = Some(panel);
    press(&mut app, "up x");
    assert_eq!((heard_count(&app), typed(&app)), (1, String::new()));
    press(&mut app, "esc x up");
    assert_eq!(app.world().resource::<KeyFocus>().0, None);
    assert_eq!((heard_count(&app), typed(&app)), (1, "x".to_owned()));
    // The keys come back to the input when the panel goes, too.
    app.world_mut().resource_mut::<KeyFocus>().0 = Some(panel);
    app.world_mut().entity_mut(panel).remove::<TuiPanel>();
    // At once, so nothing drawn before the next key says a panel has them.
    assert_eq!(app.world().resource::<KeyFocus>().0, None);
    // Up in the input went to the start of its line.
    press(&mut app, "end y");
    assert_eq!(typed(&app), "xy");
    // Only a panel takes the keys.
    let other = app.world_mut().spawn_empty().id();
    app.world_mut().resource_mut::<KeyFocus>().0 = Some(other);
    press(&mut app, "z");
    assert_eq!(typed(&app), "xyz");
}

#[test]
fn a_held_key_is_heard_once_and_again_when_let_go() {
    let (mut app, _) = view_app();
    app.init_resource::<Heard>();
    let talk = KeyBinding::new(KeyPress::ctrl(' '), "Hold to talk").with_releases();
    app.add_key_binding(talk, heard);
    for kind in [
        KeyEventKind::Press,
        KeyEventKind::Repeat,
        KeyEventKind::Repeat,
        KeyEventKind::Release,
    ] {
        key_events(&mut app, "ctrl+space", kind);
    }
    assert_eq!(app.world().resource::<Heard>().0, [false, true]);
}

#[test]
fn a_plugins_binding_and_action_do_built_in_actions() {
    let (mut app, _) = view_app();
    let down = KeyBinding::new(KeyPress::ctrl('o'), "Down");
    app.world_mut().spawn((down, InputAction::Down));
    app.add_command("model", "Choose the model", |_: In<CommandArgs>| {});
    let models = CompletionSource::arg("model").candidate("a/one", "");
    app.world_mut().spawn(models.candidate("a/two", ""));
    let selected = |app: &App| {
        let view = app.world().resource::<TuiView>();
        view.completion
            .as_ref()
            .map(|completion| completion.selected)
    };
    // With a completion open, Down selects its next item.
    type_text(&mut app, "/model a/");
    assert_eq!(selected(&app), Some(0));
    // A filter sees it in the next frame, as the frame's keys start.
    app.update();
    let state = app.world().resource::<crate::InputContext>();
    let listed = state.completion().map(|open| (open.kind, open.items.len()));
    assert_eq!(listed, Some((crate::CompletionKind::Arg(0), 2)));
    press(&mut app, "ctrl+o");
    assert_eq!(selected(&app), Some(1));
    // Without one, it moves the cursor a line down.
    crate::tests::edit(&mut app, |input| input.set("ab\ncd"));
    crate::tests::edit(&mut app, |input| input.set_cursor(1));
    app.update();
    press(&mut app, "ctrl+o");
    assert_eq!(crate::tests::input(&app).cursor(), 4);
    // Put among the keys, it is done too.
    crate::tests::push(&mut app, crate::KeyInput::Action(InputAction::LineStart));
    app.update();
    assert_eq!(crate::tests::input(&app).cursor(), 3);
}
