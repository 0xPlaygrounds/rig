use bevy_app::TaskPoolPlugin;
use bevy_app::prelude::*;
use rig_ecs::AgentPlugin;
use rig_ecs::agent::Notice;
use rig_harness_protocol::{Home, Invocation};

use super::{Invoked, LauncherPlugin, LauncherRun, refuse};
use crate::host::exit::{Exit, ExitPlugin, ExitReason, SIGNAL_EXIT_CODE};
use crate::host::session::Scratch;

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
        // Its launcher lives.
        let lock = std::fs::File::create(run.lock());
        let _held = lock.as_ref().map(std::fs::File::try_lock);
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

/// The launcher says a refusal again once it has the terminal back, as a
/// restarted full-screen view kept the screen it was said on. The refused
/// run keeps no files in its session.
#[test]
fn a_refusal_is_told_to_the_launcher() {
    let home = Home::at(std::env::temp_dir().join(format!("rig-refusal-{}", std::process::id())));
    let run = home.run(1);
    std::fs::create_dir_all(run.path()).ok();
    let mut app = App::new();
    app.insert_resource(LauncherRun(run.clone()));
    refuse(&mut app, "this session is plain".to_owned());
    refuse(&mut app, "said once".to_owned());
    let said = std::fs::read_to_string(run.refusal()).ok();
    // A question ends the sentence already.
    let mut asked = App::new();
    asked.insert_resource(LauncherRun(run.clone()));
    refuse(
        &mut asked,
        "unknown tool `shel`; did you mean `shell`?".to_owned(),
    );
    let asked_said = std::fs::read_to_string(run.refusal()).ok();
    std::fs::remove_dir_all(home.root()).ok();
    assert_eq!(said.as_deref(), Some("rig: this session is plain.\n"));
    assert_eq!(
        asked_said.as_deref(),
        Some("rig: unknown tool `shel`; did you mean `shell`?\n")
    );
    assert_eq!(app.world().get_resource::<Scratch>(), Some(&Scratch(None)));
}

/// The launcher is told when the run's front takes the terminal's screen,
/// and only then puts the terminal back after the agent: not for a front
/// that writes lines, such as `--json` typed at a terminal, nor for a
/// refused run.
#[test]
fn the_launcher_hears_when_a_front_takes_the_screen() {
    use crate::host::cli::{AppCliExt, Front};

    let home = Home::at(std::env::temp_dir().join(format!("rig-screen-{}", std::process::id())));
    let screen = |args: &[&str], refused: bool| {
        let run = home.run(1);
        std::fs::remove_dir_all(run.path()).ok();
        std::fs::create_dir_all(run.path()).ok();
        let mut app = App::new();
        app.add_message::<AppExit>()
            .insert_resource(Invoked {
                args: Invocation::parse(args).unwrap_or_default(),
                terminal: true,
            })
            .insert_resource(LauncherRun(run.clone()))
            .add_plugins((TaskPoolPlugin::default(), AgentPlugin))
            .add_plugins((LauncherPlugin, ExitPlugin))
            .add_front(Front::new("view").on_terminal())
            .add_front(Front::new("lines").on_flag("lines", "Write lines"));
        app.finish();
        app.cleanup();
        if refused {
            app.refuse_run("no".to_owned());
        }
        run.screen().exists()
    };
    let taken = [
        screen(&[], false),
        screen(&["--lines"], false),
        screen(&[], true),
    ];
    std::fs::remove_dir_all(home.root()).ok();
    assert_eq!(taken, [true, false, false]);
}

/// An agent whose launcher is gone, such as one killed alone by `kill` or
/// `timeout`, exits as on a signal rather than run on its own.
#[test]
fn an_agent_exits_once_its_launcher_is_gone() {
    use std::time::{Duration, Instant};

    let home = Home::at(std::env::temp_dir().join(format!("rig-gone-{}", std::process::id())));
    let run = home.run(1);
    std::fs::create_dir_all(run.path()).ok();
    let lock = std::fs::File::create(run.lock());
    assert!(lock.as_ref().is_ok_and(|lock| lock.try_lock().is_ok()));
    let mut app = App::new();
    app.add_message::<AppExit>()
        .insert_resource(Invoked::default())
        .insert_resource(LauncherRun(run))
        .add_plugins((TaskPoolPlugin::default(), AgentPlugin))
        .add_plugins((LauncherPlugin, ExitPlugin));
    app.finish();
    app.cleanup();
    let exits_within = |app: &mut App, wait: Duration| {
        let deadline = Instant::now() + wait;
        loop {
            app.update();
            if app.should_exit().is_some() || Instant::now() > deadline {
                return app.should_exit();
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    };
    assert_eq!(exits_within(&mut app, Duration::from_millis(200)), None);
    drop(lock);
    let exit = exits_within(&mut app, Duration::from_secs(5));
    std::fs::remove_dir_all(home.root()).ok();
    assert_eq!(exit, Some(AppExit::from_code(SIGNAL_EXIT_CODE)));
}

/// The launcher finds the mark rig-tools puts on the agent's processes.
#[test]
fn the_processes_carry_the_mark_the_launcher_looks_for() {
    assert_eq!(
        rig_tools::process::LEDGER_VARIABLE,
        rig_harness_protocol::env::PROCESS_LEDGER
    );
}
