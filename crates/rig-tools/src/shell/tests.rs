use rig_core::tool::{ExitStatus, Tool, ToolContext};

use super::{Shell, ShellArgs};
use crate::process::Launcher;

fn shell(command: &str, spill: Option<crate::Spill>) -> String {
    let shell = Shell {
        spill,
        ..Shell::default()
    };
    futures::executor::block_on(shell.run(ShellArgs::new(command)))
        .map(|ran| ran.output)
        .unwrap_or_default()
}

/// The command runs as the launcher starts it (under its wrapper, with its
/// variables) in the directory the call names, and the call publishes how
/// it ended beside its output.
#[cfg(unix)]
#[test]
fn a_call_runs_as_its_launcher_says_and_publishes_how_it_ended() {
    let dir = std::env::temp_dir().join(format!("rig-tools-launch-{}", std::process::id()));
    std::fs::create_dir_all(dir.join("sub")).ok();
    let shell = Shell {
        launcher: Launcher::default()
            .wrapped(["env", "WRAPPED=yes"])
            .in_dir(&dir)
            .with_env("SET", "set")
            .without_env(["HOME"]),
        ..Shell::default()
    };
    let mut context = ToolContext::new();
    let args = ShellArgs::new("echo \"$WRAPPED $SET ${HOME:-gone} $(basename \"$PWD\")\"; exit 3")
        .in_dir("sub")
        .timeout_secs(5);
    let output = futures::executor::block_on(Tool::call(&shell, &mut context, args));
    assert_eq!(
        output.as_deref().ok(),
        Some("yes set gone sub\n[exit code 3]")
    );
    let status = context.result::<ExitStatus>().ok().flatten();
    assert_eq!(status, Some(ExitStatus::Exited(3)));
    std::fs::remove_dir_all(dir).ok();
}

/// Without a cut, the whole output comes back, for a host that cuts it.
#[cfg(unix)]
#[test]
fn without_a_cut_the_whole_output_comes_back() {
    let shell = Shell {
        cut: None,
        ..Shell::default()
    };
    let ran = futures::executor::block_on(shell.run(ShellArgs::new("seq 1 5000")));
    let ran = ran.ok();
    assert!(ran.as_ref().is_some_and(
        |ran| ran.output.starts_with("1\n2\n") && ran.output.ends_with("5000\n[exit code 0]")
    ));
    assert_eq!(ran.map(|ran| ran.status), Some(ExitStatus::Exited(0)));
}

/// A process that leaves the command's process group keeps the output pipe
/// open after the kill; the call must still return.
#[cfg(target_os = "linux")]
#[test]
fn returns_when_a_process_outside_the_group_holds_the_pipe() {
    use std::time::{Duration, Instant};

    let started = Instant::now();
    let output = shell("setsid sleep 5 & echo started", None);
    assert!(started.elapsed() < Duration::from_secs(3));
    assert!(output.contains("started"), "{output}");
    assert!(output.ends_with("[exit code 0]"), "{output}");
}

/// Output cut by lines or by bytes is kept whole, and `read` reads it back
/// by the path the cut output names.
#[cfg(unix)]
#[test]
fn cut_output_is_read_back_by_its_handle() {
    let dir = std::env::temp_dir().join(format!("rig-tools-spill-{}", std::process::id()));
    let spill = Some(crate::Spill(dir.clone()));
    for (command, lines) in [("seq 1 3000", 3000), ("seq 1 300000", 300000)] {
        let output = shell(command, spill.clone());
        let handle = output
            .lines()
            .next()
            .and_then(|line| line.split_once(" is in "))
            .and_then(|(_, rest)| rest.split_once(':'))
            .map(|(path, _)| path.to_owned())
            .unwrap_or_default();
        let args = crate::ReadArgs::new(handle).lines(lines / 2, 1);
        let read = futures::executor::block_on(Tool::call(
            &crate::Read::default(),
            &mut ToolContext::new(),
            args,
        ));
        let wanted = format!("{}\t{}\n[", lines / 2, lines / 2);
        assert!(
            matches!(&read, Ok(text) if text.contains(&wanted)),
            "{output:.200} {read:?}"
        );
    }
    assert!(shell("seq 1 3", spill).starts_with("1\n2\n3\n"));
    assert_eq!(std::fs::read_dir(&dir).map(Iterator::count).ok(), Some(2));
    std::fs::remove_dir_all(dir).ok();
}
