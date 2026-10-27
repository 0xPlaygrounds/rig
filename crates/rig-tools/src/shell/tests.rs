use rig_core::tool::{ExitStatus, KeptEnd, OutputCut, Tool, ToolContext};

use super::{Shell, ShellArgs, split_status};
use crate::OutputLimit;
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

/// What `read` shows for `args`, empty when it fails.
fn read_text(args: crate::ReadArgs) -> String {
    let (read, mut context) = (crate::Read::default(), ToolContext::new());
    futures::executor::block_on(Tool::call(&read, &mut context, args))
        .ok()
        .and_then(|shown| shown.as_text().map(str::to_owned))
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
        let args = crate::ReadArgs::new(handle.clone()).lines(lines / 2, 1);
        let read = read_text(args);
        let wanted = format!("{}\n[", lines / 2);
        assert!(read.starts_with(&wanted), "{output:.200} {read:?}");
        // The note's own offset and limit read the first lines it cut.
        let number = |after: &str| {
            let first = output.lines().next().unwrap_or_default();
            let rest = first.split_once(after).map_or("", |(_, rest)| rest);
            rest.split(' ').next().and_then(|n| n.parse::<usize>().ok())
        };
        let (offset, limit) = (number("with offset ").unwrap_or(0), number(" and limit "));
        let args = crate::ReadArgs::new(handle).lines(offset, limit.unwrap_or(0));
        let read = read_text(args);
        let gap = format!("lines cut (lines {offset}-");
        assert!(output.contains(&gap), "{output:.300}");
        assert!(read.starts_with(&format!("{offset}\n")), "{read:.100}");
    }
    assert!(shell("seq 1 3", spill).starts_with("1\n2\n3\n"));
    assert_eq!(std::fs::read_dir(&dir).map(Iterator::count).ok(), Some(2));
    std::fs::remove_dir_all(dir).ok();
}

/// Each piece of output reaches the shell's `Live` while the command still
/// runs: here the command waits for a file the sink makes on its first line.
#[cfg(unix)]
#[test]
fn live_output_arrives_before_the_command_ends() {
    use std::sync::{Arc, Mutex, PoisonError};

    use crate::Live;

    let flag = std::env::temp_dir().join(format!("rig-tools-live-{}", std::process::id()));
    std::fs::remove_file(&flag).ok();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let live = Live::new({
        let (seen, flag) = (seen.clone(), flag.clone());
        move |text| {
            std::fs::write(&flag, "").ok();
            seen.lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(text.to_owned());
        }
    });
    let shell = Shell {
        live: Some(live),
        ..Shell::default()
    };
    let command = format!(
        "echo one; while [ ! -e '{}' ]; do sleep 0.05; done; echo two",
        flag.display()
    );
    let ran = futures::executor::block_on(shell.run(ShellArgs::new(command).timeout_secs(5)));
    assert_eq!(ran.map(|ran| ran.status).ok(), Some(ExitStatus::Exited(0)));
    assert_eq!(
        *seen.lock().unwrap_or_else(PoisonError::into_inner),
        ["one\n", "two\n"]
    );
    std::fs::remove_file(flag).ok();
}

/// The command starts through the shell's workspace, in the directory the
/// call names below the launcher's.
#[cfg(unix)]
#[test]
fn the_command_starts_through_the_workspace_in_the_calls_directory() {
    use std::ffi::{OsStr, OsString};
    use std::path::{Path, PathBuf};
    use std::sync::{Arc, Mutex, PoisonError};

    use crate::process::{Process, Spawn};
    use crate::workspace::{Contents, Local, Stat, Workspace};

    /// This machine, noting the directory of every command it starts.
    #[derive(Debug, Default)]
    struct Recording(Mutex<Vec<Option<PathBuf>>>);

    impl Workspace for Recording {
        fn stat(&self, path: &Path) -> std::io::Result<Stat> {
            Local.stat(path)
        }
        fn read(&self, path: &Path, limit: u64) -> std::io::Result<Contents> {
            Local.read(path, limit)
        }
        fn write(&self, path: &Path, bytes: &[u8]) -> std::io::Result<()> {
            Local.write(path, bytes)
        }
        fn remove(&self, path: &Path) -> std::io::Result<()> {
            Local.remove(path)
        }
        fn rename(&self, from: &Path, to: &Path) -> std::io::Result<()> {
            Local.rename(from, to)
        }
        fn canonical(&self, path: &Path) -> PathBuf {
            Local.canonical(path)
        }
        fn files(
            &self,
            root: &Path,
            glob: Option<&str>,
            listing: crate::workspace::Listing,
        ) -> std::io::Result<Vec<PathBuf>> {
            Local.files(root, glob, listing)
        }
        fn spawn(
            &self,
            launcher: &Launcher,
            dir: Option<&Path>,
            program: &OsStr,
            args: &[OsString],
            how: Spawn,
        ) -> std::io::Result<Process> {
            self.0
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(dir.map(Path::to_path_buf));
            Local.spawn(launcher, dir, program, args, how)
        }
    }

    let dir = std::env::temp_dir().join(format!("rig-tools-spawn-{}", std::process::id()));
    std::fs::create_dir_all(dir.join("sub")).ok();
    let recording = Arc::new(Recording::default());
    let shell = Shell {
        workspace: recording.clone(),
        launcher: Launcher::default().in_dir(&dir),
        confine: true,
        ..Shell::default()
    };
    let ran = futures::executor::block_on(shell.run(ShellArgs::new("pwd").in_dir("sub")));
    assert!(ran.is_ok_and(|ran| ran.output.contains("sub\n")));
    let refused = futures::executor::block_on(shell.run(ShellArgs::new("pwd").in_dir("..")));
    assert!(refused.is_err());
    assert_eq!(
        *recording.0.lock().unwrap_or_else(PoisonError::into_inner),
        [Some(dir.join("sub"))]
    );
    std::fs::remove_dir_all(dir).ok();
}

/// The status line splits off a run's output, also after a line the
/// command printed that reads like one.
#[cfg(unix)]
#[test]
fn the_status_line_splits_off_what_the_command_printed() {
    let output = shell("echo one; echo '[exit code 0]'; exit 4", None);
    assert_eq!(
        split_status(&output),
        ("one\n[exit code 0]\n", Some("[exit code 4]"))
    );
    let output = shell("kill -9 $$", None);
    let (printed, status) = split_status(&output);
    assert_eq!(printed, "");
    let status = status.unwrap_or_default();
    assert!(
        status.starts_with("[killed by SIGKILL (signal 9), which this tool did not send"),
        "{status}"
    );
}

/// A call with no `timeout_secs` waits until the command ends, however
/// long it is quiet; one with `timeout_secs` is stopped then, and says so.
#[cfg(unix)]
#[test]
fn a_command_runs_to_its_end_unless_its_timeout_stops_it() {
    let shell = Shell::default();
    let ran = futures::executor::block_on(shell.run(ShellArgs::new("sleep 2; echo built")));
    let output = ran.map(|ran| ran.output).unwrap_or_default();
    assert_eq!(output, "built\n[exit code 0]");
    let args = ShellArgs::new("echo started; sleep 10").timeout_secs(1);
    let ran = futures::executor::block_on(shell.run(args)).ok();
    assert_eq!(
        ran.as_ref().map(|ran| ran.status),
        Some(ExitStatus::TimedOut)
    );
    let output = ran.map(|ran| ran.output).unwrap_or_default();
    assert_eq!(
        split_status(&output),
        ("started\n", Some("[timed out after 1s and stopped]"))
    );
    let args = ShellArgs::new("sleep 10").timeout_secs(1);
    let output = futures::executor::block_on(shell.run(args)).map(|ran| ran.output);
    let output = output.unwrap_or_default();
    assert!(
        output.contains("It printed nothing. If it should have,")
            && output.ends_with("if it needs longer, give a larger `timeout_secs`]"),
        "{output}"
    );
}

/// stdout and stderr interleave only where a line ends: a line one stream
/// writes in pieces is never split by the other's, and a last line without
/// a newline gets one.
#[cfg(unix)]
#[test]
fn the_streams_interleave_only_at_line_ends() {
    let output = shell(
        "printf 'test result:'; sleep 0.2; echo '   Doc-tests' >&2; sleep 0.2; echo ' ok.'; \
         printf out; printf err >&2",
        None,
    );
    assert_eq!(
        output,
        "   Doc-tests\ntest result: ok.\nout\nerr\n[exit code 0]"
    );
}

/// Past its limit, the output keeps its first lines (half of the limit),
/// the lines with errors from the middle with a few after each, and its
/// last lines in the rest, with a line naming the lines missing.
#[cfg(unix)]
#[test]
fn a_cut_keeps_both_ends_and_the_errors_between() {
    let shell = Shell {
        cut: Some(OutputLimit {
            lines: 40,
            bytes: 100 * 1024,
        }),
        ..Shell::default()
    };
    let mut context = ToolContext::new();
    let command = "seq 1 1000; echo 'error[E0308]: mismatched types'; \
                   echo '  --> src/lib.rs:3:5'; seq 1001 3000";
    let output =
        futures::executor::block_on(Tool::call(&shell, &mut context, ShellArgs::new(command)));
    let output = output.unwrap_or_default();
    let mut lines = output.lines();
    let marker = "[2962 of 3002 lines cut, but for lines with errors or failures]";
    assert_eq!(lines.next(), Some(marker));
    let head: Vec<String> = (1..=20).map(|n| n.to_string()).collect();
    let tail: Vec<String> = (2985..=3000).map(|n| n.to_string()).collect();
    let mut wanted = head;
    wanted.extend(
        [
            "[… 980 lines cut (lines 21-1000) …]",
            "error[E0308]: mismatched types",
            "  --> src/lib.rs:3:5",
            "1001",
            "1002",
            "[… 1982 lines cut (lines 1005-2986) …]",
        ]
        .map(str::to_owned),
    );
    wanted.extend(tail);
    wanted.push("[exit code 0]".to_owned());
    assert_eq!(lines.collect::<Vec<_>>(), wanted);
    assert_eq!(
        context.result::<OutputCut>().ok().flatten(),
        Some(OutputCut {
            kept: KeptEnd::Ends,
            kept_lines: 40,
            total_lines: 3002,
            spill: None,
        })
    );
}

/// Once the output is cut, every line kept for the model is cut at 2000
/// characters, on a character boundary, while the spill file keeps it
/// whole: a first line of 300,000 characters keeps its start, and one
/// among short lines too. An error word past the cut still keeps a line
/// from the middle.
#[cfg(unix)]
#[test]
fn a_long_line_keeps_its_start_and_the_spill_keeps_it_whole() {
    let dir = std::env::temp_dir().join(format!("rig-tools-long-{}", std::process::id()));
    let spilling = Shell {
        spill: Some(crate::Spill(dir.clone())),
        ..Shell::default()
    };
    let x = |n: usize| format!("head -c {n} /dev/zero | tr '\\0' x");
    let mut context = ToolContext::new();
    let args = ShellArgs::new(format!("{}; echo", x(300_000)));
    let output = futures::executor::block_on(Tool::call(&spilling, &mut context, args));
    let output = output.unwrap_or_default();
    let mut lines = output.lines();
    let marker = lines.next().unwrap_or_default();
    assert!(
        marker.starts_with("[1 long line cut short; all of the output is in ")
            && marker.ends_with(
                ": `read` it there with offset 1 and limit 1 for the first long line, whose \
                 note says how to read on in it, rather than running the command again]"
            ),
        "{marker}"
    );
    assert_eq!(
        lines.next(),
        Some(format!("{} […]", "x".repeat(2000)).as_str())
    );
    assert_eq!(lines.next(), Some("[exit code 0]"));
    let cut = context.result::<OutputCut>().ok().flatten();
    assert_eq!(
        cut.as_ref().map(|cut| (cut.kept_lines, cut.total_lines)),
        Some((1, 1))
    );
    let spilled = cut.and_then(|cut| cut.spill).unwrap_or_default();
    assert_eq!(
        std::fs::read_to_string(&spilled).ok(),
        Some(format!("{}\n", "x".repeat(300_000)))
    );

    let fits = shell(&format!("{}; echo", x(5000)), None);
    assert_eq!(fits, format!("{}\n[exit code 0]", "x".repeat(5000)));

    let command = format!(
        "seq 1 3000; {}; echo ' test FAILED'; seq 1 3000; yes é | head -n 3000 | tr -d '\\n'; \
         echo",
        x(5000)
    );
    let output = shell(&command, Some(crate::Spill(dir.clone())));
    assert!(
        output.starts_with(
            "[4002 of 6002 lines cut, but for lines with errors or failures, and 2 long lines \
             cut short; all of the output is in "
        ),
        "{output:.300}"
    );
    assert!(
        output.contains(&format!("\n{} […]\n", "x".repeat(300))),
        "{output:.300}"
    );
    assert!(output.contains(&format!("\n{} […]\n", "é".repeat(2000))));
    std::fs::remove_dir_all(dir).ok();
}

/// The cut note's offset and limit name the first lines missing, past an
/// error kept right after the first lines.
#[cfg(unix)]
#[test]
fn the_cut_note_names_the_first_lines_missing() {
    let dir = std::env::temp_dir().join(format!("rig-tools-gap-{}", std::process::id()));
    let shell = Shell {
        spill: Some(crate::Spill(dir.clone())),
        cut: Some(OutputLimit {
            lines: 40,
            bytes: 100 * 1024,
        }),
        ..Shell::default()
    };
    let ran = shell.run(ShellArgs::new("seq 1 20; echo 'test FAILED'; seq 21 3000"));
    let output = futures::executor::block_on(ran)
        .map(|ran| ran.output)
        .unwrap_or_default();
    let first = output.lines().next().unwrap_or_default();
    assert!(
        first.contains("with offset 25 and limit 2000 for the first lines cut"),
        "{first}"
    );
    assert!(
        output.contains("\ntest FAILED\n21\n22\n23\n[… 2961 lines cut (lines 25-2985) …]\n"),
        "{output:.600}"
    );
    std::fs::remove_dir_all(dir).ok();
}

/// Past what the cut keeps of the lines that matter from the middle, it
/// keeps the first and the last of them, each with the lines after it,
/// and says so: the output is cut to the limit however many there are.
#[cfg(unix)]
#[test]
fn a_cut_keeps_the_first_and_last_errors_between() {
    let shell = Shell {
        cut: Some(OutputLimit {
            lines: 80,
            bytes: 100 * 1024,
        }),
        ..Shell::default()
    };
    let mut context = ToolContext::new();
    let command = "seq 1 100; for i in $(seq 1 40); do echo \"error: e$i\"; seq 5; done; \
                   seq 101 400";
    let output =
        futures::executor::block_on(Tool::call(&shell, &mut context, ShellArgs::new(command)));
    let output = output.unwrap_or_default();
    let lines: Vec<&str> = output.lines().collect();
    assert_eq!(
        lines.first().copied(),
        Some("[560 of 640 lines cut, but for the first and last lines with errors or failures]")
    );
    for kept in ["error: e1", "error: e2", "error: e39", "error: e40"] {
        assert!(lines.contains(&kept), "{kept} in {output}");
    }
    assert!(!lines.contains(&"error: e20"), "{output}");
    let cut = context.result::<OutputCut>().ok().flatten();
    assert_eq!(cut.map(|cut| cut.kept_lines), Some(80));
}

/// Terminal escape sequences a program prints, such as colours or a
/// link's markup, are dropped from its output; their text stays.
#[cfg(unix)]
#[test]
fn terminal_escape_sequences_are_dropped_from_the_output() {
    let command = "printf '\\033[1;31merror\\033[0m: \\033[4mbad\\033(B\\033[m\\n'; \
                   printf '\\033]8;;https://example.com\\033\\\\a link\\033]8;;\\a\\n'";
    assert_eq!(shell(command, None), "error: bad\na link\n[exit code 0]");
}

/// A call whose output was cut publishes the cut beside it: the ends kept,
/// how many lines of how many, and the file holding them all.
#[cfg(unix)]
#[test]
fn a_cut_output_publishes_the_cut() {
    let dir = std::env::temp_dir().join(format!("rig-tools-cut-{}", std::process::id()));
    let shell = Shell {
        spill: Some(crate::Spill(dir.clone())),
        cut: Some(OutputLimit {
            lines: 2,
            bytes: 1024,
        }),
        ..Shell::default()
    };
    let mut context = ToolContext::new();
    let output =
        futures::executor::block_on(Tool::call(&shell, &mut context, ShellArgs::new("seq 1 5")));
    let cut = context.result::<OutputCut>().ok().flatten();
    let spill = cut
        .as_ref()
        .and_then(|cut| cut.spill.clone())
        .unwrap_or_default();
    assert!(
        output.is_ok_and(
            |output| output.starts_with("[3 of 5 lines cut,") && output.contains(&spill)
        )
    );
    assert_eq!(
        cut,
        Some(OutputCut {
            kept: KeptEnd::Ends,
            kept_lines: 2,
            total_lines: 5,
            spill: Some(spill.clone())
        })
    );
    assert_eq!(
        std::fs::read_to_string(&spill).ok().as_deref(),
        Some("1\n2\n3\n4\n5\n")
    );
    let mut context = ToolContext::new();
    futures::executor::block_on(Tool::call(&shell, &mut context, ShellArgs::new("seq 1 2"))).ok();
    assert_eq!(context.result::<OutputCut>().ok().flatten(), None);
    std::fs::remove_dir_all(dir).ok();
}

/// A key the command prints is redacted before it is kept, spilled or
/// returned, and a private key block is left out through its end line.
#[cfg(unix)]
#[test]
fn printed_keys_are_neither_returned_nor_spilled() {
    let dir = std::env::temp_dir().join(format!("rig-tools-keys-{}", std::process::id()));
    let printed = "printf 'OPENAI_API_KEY=sk-test-probe-1234567890abcdef\\n\
        -----BEGIN RSA PRIVATE KEY-----\\nMIIEsecretbody\\n-----END RSA PRIVATE KEY-----\\n\
        after\\n'";
    let output = shell(printed, None);
    assert_eq!(
        output,
        "OPENAI_API_KEY=[REDACTED]\n[REDACTED]\nafter\n[exit code 0]"
    );
    let spilled = shell(
        &format!("{printed}; seq 1 3000"),
        Some(crate::Spill(dir.clone())),
    );
    let spill = std::fs::read_dir(&dir)
        .ok()
        .and_then(|mut files| files.next())
        .and_then(Result::ok)
        .and_then(|file| std::fs::read_to_string(file.path()).ok())
        .unwrap_or_default();
    std::fs::remove_dir_all(dir).ok();
    assert!(
        spill.starts_with("OPENAI_API_KEY=[REDACTED]\n[REDACTED]\nafter\n1\n"),
        "{spill:.200}"
    );
    assert!(!spilled.contains("secret"));
}
