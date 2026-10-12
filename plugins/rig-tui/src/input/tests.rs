use bevy_ecs::system::RunSystemOnce;
use rig_ecs::prelude::*;

use super::super::editor::TuiInput;
use super::super::history::Sent;
use super::super::view::{self, TranscriptView, TuiView};
use super::send;
use rig_harness::commands::{AppCommandsExt, CommandArgs};
use rig_harness::primary::Shown;

/// The texts agents accepted.
#[derive(Resource, Default)]
struct Delivered(Vec<String>);

/// Types `text` in the view and presses Enter.
fn enter(app: &mut App, text: &str) {
    let text = text.to_owned();
    let sent = app.world_mut().run_system_once(
        move |mut view: ResMut<TuiView>,
              mut input: ResMut<TuiInput>,
              shown: Res<Shown>,
              mut commands: Commands| {
            input.set(text.clone());
            send(&mut view, &mut input, shown.0, &mut commands, false);
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
            "retry",
            "Refuses arguments",
            |In(args): In<CommandArgs>, mut notices: MessageWriter<Notice>| {
                if !args.args.is_empty() {
                    let why = format!("/retry takes no arguments, not `{}`.", args.args);
                    notices.write(Notice::error(args.agent, why));
                }
            },
        )
        .add_message::<Sent>()
        .init_resource::<TuiView>()
        .init_resource::<TranscriptView>()
        .init_resource::<Shown>()
        .init_resource::<TuiInput>()
        .init_resource::<Delivered>()
        .add_observer(
            |accepted: On<Add<Started>>,
             messages: Query<&Incoming>,
             mut delivered: ResMut<Delivered>| {
                if let Ok(message) = messages.get(accepted.entity) {
                    delivered.0.push(message.text.clone());
                }
            },
        )
        .add_systems(
            Update,
            (view::collect_notices, view::recall_messages).after(AgentLoop),
        );
    app.update();
    let world = app.world_mut();
    let agent = world
        .query_filtered::<Entity, With<Agent>>()
        .single(world)
        .ok();
    world.insert_resource(Shown(agent));
    let refused = [
        ("/retry now", "/retry takes no arguments, not `now`."),
        ("/gui still doesnt do anything", "Unknown command /gui."),
    ];
    for (typed, why) in refused {
        enter(&mut app, typed);
        let view = app.world().resource::<TranscriptView>();
        let input = app.world().resource::<TuiInput>();
        let shown: Vec<&str> = view
            .notices()
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
        app.world().resource::<TranscriptView>().notices().len(),
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
    use crate::tests::{typed, view_app};
    use crate::{KeyFocus, Placement, TerminalEvent, TuiPanel};
    use crossterm::event::Event;
    use ratatui::layout::Constraint;

    let (mut app, _) = view_app();
    let panel = TuiPanel::new(Placement::Top(Constraint::Length(3)));
    let panel = app.world_mut().spawn(panel).id();
    app.world_mut().resource_mut::<KeyFocus>().0 = Some(panel);
    let paste = |app: &mut App| {
        let paste = TerminalEvent(Event::Paste("cargo test".to_owned()));
        app.world_mut().write_message(paste);
        app.update();
    };
    paste(&mut app);
    assert_eq!(typed(&app), "");
    // Given back the keys, the input takes it.
    app.world_mut().resource_mut::<KeyFocus>().0 = None;
    paste(&mut app);
    assert_eq!(typed(&app), "cargo test");
}
