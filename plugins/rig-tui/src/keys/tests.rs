use bevy_ecs::entity_disabling::Disabled;
use crossterm::event::KeyEventKind;

use super::*;
use crate::TuiInput;
use crate::tests::{Delivered, key_events, press, type_text, typed, view_app};

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
fn a_taken_key_stays_until_its_binding_is_disabled() {
    let (mut app, _) = view_app();
    app.init_resource::<Heard>();
    let edit = || KeyBinding::new(KeyPress::ctrl('e'), "Edit in $EDITOR");
    // Ctrl+E is the end of the line: the binding is refused.
    app.add_key_binding(edit(), heard);
    type_text(&mut app, "ab");
    press(&mut app, "ctrl+a ctrl+e x");
    assert_eq!(typed(&app), "abx");
    assert_eq!(heard_count(&app), 0);
    // Disabled, the view's binding frees the key.
    let world = app.world_mut();
    let built_in: Vec<Entity> = world
        .query::<(Entity, &KeyBinding)>()
        .iter(world)
        .filter(|(_, binding)| binding.keys == Keys::from(KeyPress::ctrl('e')))
        .map(|(entity, _)| entity)
        .collect();
    assert_eq!(built_in.len(), 1);
    for entity in built_in {
        world.entity_mut(entity).insert(Disabled);
    }
    app.add_key_binding(edit(), heard);
    press(&mut app, "ctrl+a ctrl+e y");
    assert_eq!(typed(&app), "yabx");
    assert_eq!(heard_count(&app), 1);
}

#[test]
fn a_binding_reads_and_edits_the_input_before_it_is_sent() {
    let (mut app, _) = view_app();
    // A template key: the draft is wrapped in a request for a review.
    app.add_key_binding(
        KeyBinding::new(KeyPress::alt('r'), "Ask for a review"),
        |_: On<KeyPressed>, mut input: ResMut<TuiInput>| {
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
    let panel = app.world_mut().spawn_empty().id();
    // Up in the panel is no conflict with Up in the input.
    let up = KeyBinding::new(KeyPress::key(KeyCode::Up), "Select the file above");
    app.add_key_binding(up.in_scope(KeyScope::Panel(panel)), heard);
    app.world_mut().resource_mut::<KeyFocus>().0 = Some(panel);
    press(&mut app, "up x");
    assert_eq!((heard_count(&app), typed(&app)), (1, String::new()));
    press(&mut app, "esc x up");
    assert_eq!(app.world().resource::<KeyFocus>().0, None);
    assert_eq!((heard_count(&app), typed(&app)), (1, "x".to_owned()));
    // The keys come back to the input when the panel is gone, too.
    app.world_mut().resource_mut::<KeyFocus>().0 = Some(panel);
    app.world_mut().despawn(panel);
    press(&mut app, "y");
    assert_eq!(typed(&app), "xy");
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
