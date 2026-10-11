use crossterm::event::{Event, KeyEvent, KeyEventKind};
use rig_harness::ask::AskPlugin;
use rig_harness::pick::PickPlugin;
use rig_harness::prelude::*;

use super::*;

/// The frames a panel drew.
#[derive(Resource, Default)]
struct Drawn(usize);

/// The texts agents accepted, in order.
#[derive(Resource, Default)]
pub(crate) struct Delivered(pub(crate) Vec<String>);

/// An agent with the terminal view's input and choices, but no terminal:
/// its keys are the [`TerminalEvent`]s [`press`] writes. Also returns the
/// agent.
pub(crate) fn view_app() -> (App, Entity) {
    let mut app = App::new();
    app.set_error_handler(rig_harness::error::panic)
        .insert_resource(Invoked::default())
        .add_plugins((AgentPlugin, AskPlugin, PickPlugin, TuiPlugin))
        .init_resource::<Delivered>()
        .add_observer(
            |accepted: On<Add<Started>>,
             messages: Query<&Incoming>,
             mut delivered: ResMut<Delivered>| {
                if let Ok(message) = messages.get(accepted.entity) {
                    delivered.0.push(message.text.clone());
                }
            },
        );
    add_view(&mut app);
    app.update();
    let world = app.world_mut();
    let agent = world
        .query_filtered::<Entity, With<Agent>>()
        .single(world)
        .unwrap_or(Entity::PLACEHOLDER);
    (app, agent)
}

/// Writes `kind` events of `keys` (such as `ctrl+x ctrl+e`), then runs a
/// frame.
pub(crate) fn key_events(app: &mut App, keys: &str, kind: KeyEventKind) {
    let parsed: Result<Keys, String> = keys.parse();
    assert!(parsed.is_ok(), "{parsed:?}");
    for press in parsed.iter().flat_map(Keys::presses) {
        let key = KeyEvent::new_with_kind(press.code, press.modifiers, kind);
        app.world_mut()
            .write_message(TerminalEvent(Event::Key(key)));
    }
    app.update();
}

/// Presses `keys`, as [`key_events`] writes them.
pub(crate) fn press(app: &mut App, keys: &str) {
    key_events(app, keys, KeyEventKind::Press);
}

/// Types `text`, a key for each character, then runs a frame.
pub(crate) fn type_text(app: &mut App, text: &str) {
    let keys: Vec<String> = text
        .chars()
        .map(|character| match character {
            ' ' => "space".to_owned(),
            other => other.to_string(),
        })
        .collect();
    press(app, &keys.join(" "));
}

/// The input's text.
pub(crate) fn typed(app: &App) -> String {
    app.world().resource::<TuiInput>().text().to_owned()
}

#[test]
fn a_panel_plugin_is_inert_in_a_print_run() {
    let mut app = App::new();
    // A system that cannot run fails the test.
    app.set_error_handler(rig_harness::error::panic)
        .insert_resource(Invoked::default())
        .add_plugins(TuiPlugin)
        .init_resource::<Drawn>()
        .add_systems(Update, |mut redraw: MessageWriter<RequestRedraw>| {
            redraw.write(RequestRedraw);
        })
        .add_systems(
            PostUpdate,
            (|mut drawn: ResMut<Drawn>| drawn.0 += 1).in_set(TuiSystems::Draw),
        );
    app.update();
    app.update();
    let world = app.world();
    assert_eq!(world.resource::<Drawn>().0, 0);
    assert!(!world.contains_resource::<view::TuiView>());
}

#[test]
fn typed_keys_edit_the_input_and_enter_sends_it() {
    let (mut app, _) = view_app();
    type_text(&mut app, "helo");
    // Back over the `o`, then the missing `l`.
    press(&mut app, "left l");
    assert_eq!(typed(&app), "hello");
    press(&mut app, "ctrl+a");
    type_text(&mut app, "> ");
    press(&mut app, "enter");
    assert_eq!(typed(&app), "");
    assert_eq!(app.world().resource::<Delivered>().0, ["> hello"]);
}

#[test]
fn a_commands_argument_completes_to_a_contributed_candidate() {
    let (mut app, _) = view_app();
    app.add_command("model", "Choose the model", |_: In<CommandArgs>| {});
    let models = ArgCompletion::new("/model")
        .candidate("openai/gpt-5", "")
        .candidate("acme/fast", "no key");
    app.world_mut().spawn(models);
    app.world_mut()
        .spawn(ArgCompletion::new("other").candidate("acme/other", ""));
    type_text(&mut app, "/model acme/");
    press(&mut app, "tab");
    assert_eq!(typed(&app), "/model acme/fast ");
}

#[test]
fn the_terminal_says_when_it_has_the_users_focus() {
    let (mut app, _) = view_app();
    assert_eq!(
        *app.world().resource::<TerminalFocus>(),
        TerminalFocus(true)
    );
    for (event, focused) in [(Event::FocusLost, false), (Event::FocusGained, true)] {
        app.world_mut().write_message(TerminalEvent(event));
        app.update();
        let focus = *app.world().resource::<TerminalFocus>();
        assert_eq!(focus, TerminalFocus(focused));
    }
}
