use bevy_app::TaskPoolPlugin;
use bevy_app::prelude::*;
use rig::harness_protocol::{Home, Invocation};
use rig_ecs::AgentPlugin;
use rig_ecs::agent::Notice;

use super::{Invoked, LauncherPlugin, LauncherRun};
use crate::host::exit::{Exit, ExitPlugin, ExitReason};

/// An app started with `args` on a terminal or not, through `finish` and
/// `cleanup`.
fn started(args: &[&str], terminal: bool) -> App {
    let mut app = App::new();
    app.add_message::<AppExit>()
        .insert_resource(Invoked {
            args: Invocation::parse(args).unwrap_or_default(),
            terminal,
        })
        .add_plugins((TaskPoolPlugin::default(), AgentPlugin))
        .add_plugins((LauncherPlugin, ExitPlugin));
    app.finish();
    app.cleanup();
    app
}

#[test]
fn a_run_the_build_cannot_take_exits_with_2_after_its_first_frame() {
    for (args, terminal) in [(&["--nope"][..], false), (&[], true)] {
        let mut app = started(args, terminal);
        app.update();
        app.update();
        assert_eq!(app.should_exit(), Some(AppExit::from_code(2)), "{args:?}");
    }
    let mut fine = started(&["-p", "hi"], false);
    fine.update();
    fine.update();
    assert_eq!(fine.should_exit(), None);
}

#[test]
fn an_exit_on_purpose_in_the_first_frame_counts_as_started() {
    let home = Home::at(std::env::temp_dir().join(format!("rig-ready-{}", std::process::id())));
    let ready = |exit: Option<Exit>, plain: Option<AppExit>| {
        let run = home.run(1);
        std::fs::remove_dir_all(run.path()).ok();
        std::fs::create_dir_all(run.path()).ok();
        let mut app = App::new();
        app.add_message::<AppExit>()
            .add_message::<Notice>()
            .insert_resource(Invoked::default())
            .insert_resource(LauncherRun(run.clone()))
            .add_plugins((LauncherPlugin, ExitPlugin));
        app.finish();
        app.cleanup();
        if let Some(exit) = exit {
            app.world_mut().trigger(exit);
        }
        if let Some(plain) = plain {
            app.world_mut().write_message(plain);
        }
        app.update();
        run.ready().exists()
    };
    assert!(ready(None, None));
    assert!(ready(Some(Exit(ExitReason::Failed(2))), None));
    assert!(ready(Some(Exit(ExitReason::Quit)), None));
    assert!(!ready(None, Some(AppExit::from_code(1))), "a crash path");
    std::fs::remove_dir_all(home.root()).ok();
}
