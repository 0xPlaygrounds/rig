#![allow(clippy::disallowed_methods, reason = "test fixtures")]

use bevy_app::prelude::*;
use rig_harness_protocol::AgentDescription;

use super::{Describing, report};
use crate::host::cli::{AppCliExt, Cli, Flag, Front};
use crate::host::launcher::Invoked;
use crate::load;
use crate::secrets::Secret;

/// `rig stats`: a headless command that keeps no session.
#[derive(Default)]
struct Stats;

impl Plugin for Stats {
    fn build(&self, app: &mut App) {
        app.add_front(Front::new("stats").sessionless().on_command(
            "stats",
            "[--all]",
            "Count the saved sessions",
        ))
        .add_flag(Flag::new("remote", "Work over SSH").value("HOST:PATH"))
        .add_secret(Secret::new("crypt", "Passphrase"));
    }
}

/// Another plugin's `rig stats`.
#[derive(Default)]
struct OtherStats;

impl Plugin for OtherStats {
    fn build(&self, app: &mut App) {
        app.add_front(Front::new("other").on_command("stats", "", "Other stats"));
    }
}

/// A plugin that takes the launcher's `help`.
#[derive(Default)]
struct Help;

impl Plugin for Help {
    fn build(&self, app: &mut App) {
        app.add_front(Front::new("help").on_command("help", "", "Help"));
    }
}

/// An app with a terminal front and the plugins `add` loads, built.
fn built(add: impl FnOnce(&mut App)) -> App {
    let mut app = App::new();
    app.add_front(Front::new("tui").on_terminal());
    add(&mut app);
    app.finish();
    app.cleanup();
    app
}

#[test]
fn the_report_dispatches_a_command_line_to_its_front() {
    let mut app = built(|app| load::<Stats>(app, "corpus-stats", "path stats"));
    let described = report(app.world_mut());
    let text = described
        .map(|described| described.write())
        .unwrap_or_default();
    let described = AgentDescription::parse(&text);
    let command = described.command("stats");
    assert!(command.is_some_and(|command| !command.terminal && command.sessionless));
    assert_eq!(described.secrets.len(), 1);
    // The launcher reads the user's words against it, and the agent gets
    // the canonical form.
    let args = described.invocation(&["--remote", "h:/p", "stats", "--all"]);
    assert!(args.is_ok(), "{args:?}");
    let mut run = built(|app| {
        app.insert_resource(Invoked {
            args: args.unwrap_or_default(),
            terminal: true,
        });
        load::<Stats>(app, "corpus-stats", "path stats");
    });
    assert!(run.runs_front("stats") && !run.runs_front("tui"));
    assert!(run.world().resource::<Cli>().sessionless());
    let invoked = run.world().resource::<Invoked>();
    assert_eq!(invoked.flag_value("remote"), Some("h:/p"));
    let all = ["--all".to_owned()];
    assert_eq!(invoked.command(), Some(("stats", &all[..])));
}

#[test]
fn two_plugins_with_one_command_or_a_launcher_word_fail_the_build() {
    let mut clash = built(|app| {
        load::<Stats>(app, "corpus-a", "path a");
        load::<OtherStats>(app, "corpus-b", "path b");
    });
    assert_eq!(
        report(clash.world_mut()),
        Err("plugins `corpus-a` and `corpus-b` both add the command `stats`".to_owned())
    );
    let mut reserved = built(|app| {
        app.add_plugins(Help);
    });
    let refused = report(reserved.world_mut()).err().unwrap_or_default();
    assert!(
        refused.contains("a plugin added in main.rs")
            && refused.contains("`help` is the launcher's"),
        "{refused}"
    );
}

#[test]
fn while_describing_no_front_takes_the_run() {
    let mut app = built(|app| {
        app.insert_resource(Describing).insert_resource(Invoked {
            terminal: true,
            ..Invoked::default()
        });
    });
    assert!(!app.runs_front("tui"));
    assert_eq!(crate::host::cli::refusal(&mut app), None);
}

/// Set, with [`rig_harness_protocol::env::DESCRIBE`], on the copy of the
/// test below that describes the app: whose plugins to load.
const CASE: &str = "RIG_TEST_DESCRIBE_CASE";

#[test]
fn a_build_asked_to_describe_itself_writes_its_report_and_runs_no_frame() {
    use rig_harness_protocol::{Home, env};

    let Some(case) = std::env::var_os(CASE) else {
        let name =
            "describe::tests::a_build_asked_to_describe_itself_writes_its_report_and_runs_no_frame";
        let dir = std::env::current_exe()
            .unwrap_or_default()
            .with_file_name(format!("rig-describe-{}", std::process::id()));
        std::fs::create_dir_all(&dir).ok();
        let report = dir.join("report");
        let described = |case: &str| {
            std::fs::remove_file(&report).ok();
            let output = std::env::current_exe().and_then(|test| {
                std::process::Command::new(test)
                    .args(["--exact", name, "--test-threads", "1"])
                    .env(CASE, case)
                    .env(env::DESCRIBE, &report)
                    .env(env::HOME, dir.join("home"))
                    .env_remove(env::SESSION)
                    .output()
            });
            let passed = output.is_ok_and(|output| {
                output.status.success()
                    && String::from_utf8_lossy(&output.stdout).contains("1 passed")
            });
            let text = std::fs::read_to_string(&report).unwrap_or_default();
            (passed, AgentDescription::parse(&text))
        };
        let (passed, report) = described("one");
        let (clash_passed, clash) = described("clash");
        let sessions = Home::at(dir.join("home")).sessions();
        let kept_none = std::fs::read_dir(sessions).map_or(true, |mut dirs| dirs.next().is_none());
        std::fs::remove_dir_all(&dir).ok();
        assert!(passed && report.command("stats").is_some(), "{report:?}");
        assert!(clash_passed && clash == AgentDescription::default());
        assert!(kept_none, "a session directory was made");
        return;
    };
    let mut app = App::new();
    app.add_plugins((crate::HeadlessPlugins, crate::RigHarnessPlugins));
    load::<Stats>(&mut app, "corpus-stats", "path stats");
    if case == "clash" {
        load::<OtherStats>(&mut app, "corpus-other", "path other");
    }
    // No frame runs.
    app.add_systems(Startup, || std::process::abort());
    let exit = crate::run(app);
    let expected = if case == "clash" {
        AppExit::from_code(2)
    } else {
        AppExit::Success
    };
    assert_eq!(exit, expected);
}
