use bevy_ecs::system::RunSystemOnce;
use rig_ecs::prelude::*;

use super::super::view::{self, TuiView};
use super::send;

/// The texts sent to agents' models.
#[derive(Resource, Default)]
struct Delivered(Vec<String>);

/// Types `text` in the view and presses Enter.
fn enter(app: &mut App, text: &str) {
    let text = text.to_owned();
    let sent = app.world_mut().run_system_once(
        move |mut view: ResMut<TuiView>, mut commands: Commands| {
            view.editor.set(text.clone());
            send(&mut view, &mut commands, false);
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
        .init_resource::<TuiView>()
        .init_resource::<Delivered>()
        .add_observer(|deliver: On<Deliver>, mut delivered: ResMut<Delivered>| {
            if deliver.command().is_none() {
                delivered.0.push(deliver.text.clone());
            }
        })
        .add_systems(Update, (view::collect_notices, view::recall_messages));
    app.update();
    let world = app.world_mut();
    let agent = world
        .query_filtered::<Entity, With<Agent>>()
        .single(world)
        .ok();
    world.resource_mut::<TuiView>().agent = agent;
    let refused = [
        ("/retry now", "/retry takes no arguments, not `now`."),
        ("/gui still doesnt do anything", "Unknown command /gui."),
    ];
    for (typed, why) in refused {
        enter(&mut app, typed);
        let view = app.world().resource::<TuiView>();
        let shown: Vec<&str> = view
            .notices
            .iter()
            .map(|notice| notice.text.as_str())
            .collect();
        assert_eq!(view.editor.text(), typed);
        assert!(
            matches!(shown.as_slice(), [.., last] if last.starts_with(why)),
            "{shown:?}"
        );
    }
    assert_eq!(
        app.world().resource::<TuiView>().notices.len(),
        refused.len()
    );
    enter(&mut app, "/gui still doesnt do anything");
    assert_eq!(
        app.world().resource::<Delivered>().0,
        ["/gui still doesnt do anything"]
    );
}
