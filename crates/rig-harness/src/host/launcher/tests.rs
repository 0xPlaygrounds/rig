#![allow(clippy::disallowed_methods, reason = "test fixtures")]

use bevy_app::TaskPoolPlugin;
use bevy_app::prelude::*;
use bevy_ecs::schedule::IntoScheduleConfigs;
use rig_ecs::AgentPlugin;
use rig_ecs::agent::Notice;
use rig_harness_protocol::{Home, Invocation, SIGNAL_EXIT_CODE};

use super::{Invoked, LauncherPlugin, LauncherRun, refuse};
use crate::host::exit::{Exit, ExitPlugin, ExitReason};
use crate::host::session::Scratch;
use crate::settings::{AppSettingsExt, Settings};

/// An app started with `args` on a terminal or not, through `finish` and
/// `cleanup`.
fn started(args: &[&str], terminal: bool) -> App {
    let mut app = App::new();
    app.add_message::<AppExit>()
        .insert_resource(Settings::default())
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

/// A table of the user's settings that does not read refuses the run,
/// naming the file, the table and the key, rather than running it on that
/// table's defaults.
#[test]
fn a_settings_table_that_does_not_read_refuses_the_run() {
    #[derive(bevy_ecs::prelude::Resource, serde::Deserialize, Default)]
    #[serde(default, deny_unknown_fields)]
    struct Limits {
        #[allow(dead_code, reason = "read only to be checked")]
        lines: usize,
    }
    let mut app = App::new();
    #[derive(bevy_ecs::prelude::Resource, Default)]
    struct Passes(usize);
    app.add_message::<AppExit>()
        .insert_resource(Invoked {
            args: Invocation::parse(&["-p", "hi"]).unwrap_or_default(),
            terminal: false,
        })
        .insert_resource(Settings::from_toml("[limits]\nlinez = 3\n"))
        .add_plugins((TaskPoolPlugin::default(), AgentPlugin))
        .add_setting::<Limits>("limits")
        .add_plugins((LauncherPlugin, ExitPlugin))
        .init_resource::<Passes>()
        .add_systems(
            Update,
            (|mut passes: bevy_ecs::prelude::ResMut<Passes>| passes.0 += 1)
                .in_set(rig_ecs::stage::AgentLoop),
        );
    app.finish();
    app.cleanup();
    for _ in 0..3 {
        if app.should_exit().is_none() {
            app.update();
        }
    }
    assert_eq!(app.should_exit(), Some(AppExit::from_code(2)));
    assert_eq!(app.world().resource::<Passes>().0, 0, "no agent work ran");
    let mut notices = app
        .world_mut()
        .resource_mut::<bevy_ecs::message::Messages<Notice>>();
    let repeated = notices.drain().any(|notice| notice.text.contains("linez"));
    assert!(!repeated, "the refusal said it; no notice says it again");
    let problems = app.world().resource::<Settings>().problems().join("\n");
    assert!(
        problems.contains("settings.toml [limits]") && problems.contains("linez"),
        "{problems}"
    );
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

/// After a reload the agent takes over the jobs its list names while they
/// carry the list's mark, and leaves an id whose processes do not alone:
/// it may name another group by then.
#[cfg(target_os = "linux")]
#[test]
fn jobs_a_reload_kept_are_taken_over_while_marked() {
    use std::os::unix::process::CommandExt;

    let root = std::env::temp_dir().join(format!("rig-harness-adopt-{}", std::process::id()));
    std::fs::create_dir_all(&root).ok();
    let run = rig_harness_protocol::RunDir::at(&root);
    let start = |marked: bool| {
        let mut sleep = std::process::Command::new("sleep");
        sleep.arg("30").process_group(0);
        if marked {
            sleep.env(rig_harness_protocol::env::PROCESS_LEDGER, run.groups());
        }
        sleep.spawn().ok()
    };
    let jobs = [start(true), start(false)];
    assert!(jobs.iter().all(Option::is_some), "could not start the jobs");
    let [Some(mut kept), Some(mut stranger)] = jobs else {
        return;
    };
    let listed = format!("{}\n{}\n", kept.id(), stranger.id());
    std::fs::write(run.groups(), listed).ok();
    let owned_ok = super::own_groups(&run).is_ok();
    let owned = std::fs::read_to_string(run.groups()).unwrap_or_default();
    for job in [&mut kept, &mut stranger] {
        job.kill().ok();
        job.wait().ok();
    }
    std::fs::remove_dir_all(&root).ok();
    assert!(owned_ok, "the list should have been read");
    let owned: Vec<&str> = owned.lines().collect();
    assert_eq!(owned, [kept.id().to_string()]);
}

/// A table a plugin reads in its own `finish`, after the launcher looked,
/// refuses the run all the same.
#[test]
fn a_settings_table_read_in_a_finish_refuses_the_run() {
    #[derive(bevy_ecs::prelude::Resource, serde::Deserialize, Default)]
    #[serde(default, deny_unknown_fields)]
    struct Late {
        #[allow(dead_code, reason = "read only to be checked")]
        lines: usize,
    }
    struct ReadsLate;
    impl Plugin for ReadsLate {
        fn build(&self, _: &mut App) {}
        fn finish(&self, app: &mut App) {
            app.add_setting::<Late>("late");
        }
    }
    let mut app = App::new();
    app.add_message::<AppExit>()
        .insert_resource(Invoked {
            args: Invocation::parse(&["-p", "hi"]).unwrap_or_default(),
            terminal: false,
        })
        .insert_resource(Settings::from_toml("[late]\nlinez = 3\n"))
        .add_plugins((TaskPoolPlugin::default(), AgentPlugin))
        .add_plugins((LauncherPlugin, ExitPlugin, ReadsLate));
    app.finish();
    app.cleanup();
    for _ in 0..3 {
        if app.should_exit().is_none() {
            app.update();
        }
    }
    assert_eq!(app.should_exit(), Some(AppExit::from_code(2)));
}

/// An agent started by a reload that cannot read the list of the jobs
/// the agent before it left running keeps what the list holds, and still
/// lists the jobs it starts itself.
#[cfg(target_os = "linux")]
#[test]
fn a_list_a_reload_cannot_read_is_kept_and_grows() {
    use std::os::unix::process::CommandExt;

    let root = std::env::temp_dir().join(format!("rig-harness-unread-{}", std::process::id()));
    std::fs::create_dir_all(&root).ok();
    let run = rig_harness_protocol::RunDir::at(&root);
    let mut sleep = std::process::Command::new("sleep");
    sleep
        .arg("30")
        .process_group(0)
        .env(rig_harness_protocol::env::PROCESS_LEDGER, run.groups());
    let Ok(mut kept) = sleep.spawn() else {
        std::fs::remove_dir_all(&root).ok();
        return;
    };
    // Bytes that are not text make reading the list fail, while writing it
    // anew would succeed.
    let mut listed = format!("{}\n", kept.id()).into_bytes();
    listed.extend_from_slice(b"\xff\n");
    std::fs::write(run.groups(), &listed).ok();
    let owned = super::own_groups(&run);
    let launcher = rig_tools::process::Launcher::default();
    let started = launcher.spawn("sleep", ["30"], rig_tools::process::Spawn::default());
    let after = std::fs::read(run.groups()).unwrap_or_default();
    let started = started.map(|started| started.id());
    kept.kill().ok();
    kept.wait().ok();
    std::fs::remove_dir_all(&root).ok();
    assert!(owned.is_err(), "reading the list should have failed");
    assert!(after.starts_with(&listed), "the list lost what it held");
    let lines = String::from_utf8_lossy(&after);
    assert!(
        started
            .as_ref()
            .is_ok_and(|started| lines.lines().any(|line| line == started.to_string())),
        "the job {started:?} is not listed: {lines:?}"
    );
}
