use bevy_app::prelude::*;
use rig::harness_protocol::Invocation;
use rig_ecs::agent::Notice;

use super::{Invoked, LauncherPlugin};

/// An app started with `args` on a terminal or not, through `finish` and
/// `cleanup`.
fn started(args: &[&str], terminal: bool) -> App {
    let mut app = App::new();
    app.add_message::<AppExit>()
        .add_message::<Notice>()
        .insert_resource(Invoked {
            args: Invocation::parse(args).unwrap_or_default(),
            terminal,
        })
        .add_plugins(LauncherPlugin);
    app.finish();
    app.cleanup();
    app
}

#[test]
fn a_run_the_build_cannot_take_exits_with_2_after_its_first_frame() {
    for (args, terminal) in [(&["--nope"][..], false), (&[], true)] {
        let mut app = started(args, terminal);
        app.update();
        assert_eq!(app.should_exit(), Some(AppExit::from_code(2)), "{args:?}");
    }
    let mut fine = started(&["-p", "hi"], false);
    fine.update();
    assert_eq!(fine.should_exit(), None);
}
