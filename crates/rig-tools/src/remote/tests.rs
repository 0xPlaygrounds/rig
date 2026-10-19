use std::path::{Path, PathBuf};
use std::sync::Arc;

use rig_core::tool::{FilesChanged, FilesRead, Scope, Tool, ToolContext, Touches};

use super::{Hosts, Remote};
use crate::process::Launcher;
use crate::workspace::{Local, Workspace};
use crate::{Edit, EditArgs, Read, ReadArgs, Replacement, Search, SearchArgs, Shell, ShellArgs};

/// A fresh directory under the temporary one, with `local` (this
/// machine's project) and `home` (the remote user's home) in it.
fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("rig-tools-remote-{name}-{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    for sub in ["local", "home/strata/src"] {
        std::fs::create_dir_all(dir.join(sub)).ok();
    }
    dir
}

/// A transport that runs its last argument with `sh` in `home`, as the
/// remote user's shell does there.
fn transport(home: &Path) -> Vec<String> {
    vec![
        "sh".to_owned(),
        "-c".to_owned(),
        "cd \"$0\" && HOME=\"$0\" exec sh -c \"$1\"".to_owned(),
        home.display().to_string(),
    ]
}

fn call<T: Tool<Error = rig_core::tool::ToolExecutionError>>(
    tool: &T,
    args: T::Args,
) -> (
    Result<T::Output, rig_core::tool::ToolExecutionError>,
    ToolContext,
) {
    let mut context = ToolContext::new();
    let output = futures::executor::block_on(tool.call(&mut context, args));
    (output, context)
}

/// `read`, `edit`, `search` and `shell` reach a `HOST:PATH` through the
/// host's transport while a plain path stays on this machine, and every
/// path they give back for the remote file keeps the `HOST:` form.
#[test]
fn tools_reach_a_host_path_through_its_transport() {
    let dir = scratch("tools");
    let home = dir.join("home");
    std::fs::write(home.join("strata/src/lib.rs"), "fn old() {}\n").ok();
    std::fs::write(dir.join("local/notes.txt"), "here\n").ok();
    let bench = Remote::new("bench", transport(&home), Launcher::default());
    let workspace: Arc<dyn Workspace> = Arc::new(Hosts::new(Arc::new(Local), bench));
    let base = Some(dir.join("local"));
    let file = "bench:~/strata/src/lib.rs";

    let read = Read {
        workspace: workspace.clone(),
        base: base.clone(),
        confine: false,
        seen: crate::Seen::default(),
    };
    let (shown, context) = call(&read, ReadArgs::new("bench:~/strata/src/../src/lib.rs"));
    assert_eq!(shown.ok().as_deref(), Some("fn old() {}\n"));
    let facts: FilesRead = context.result().ok().flatten().unwrap_or_default();
    let read_paths: Vec<&str> = facts.0.iter().map(|it| it.path.as_str()).collect();
    assert_eq!(read_paths, [file]);
    let (local, _) = call(&read, ReadArgs::new("notes.txt"));
    assert_eq!(local.ok().as_deref(), Some("here\n"));
    let (missing, _) = call(&read, ReadArgs::new("bench:~/strata/gone.rs"));
    let missing = missing
        .err()
        .map(|error| error.to_string())
        .unwrap_or_default();
    assert!(
        missing.contains("bench:~/strata/gone.rs: no such file"),
        "{missing}"
    );

    let edit = Edit {
        workspace: workspace.clone(),
        base: base.clone(),
        confine: false,
        seen: crate::Seen::default(),
    };
    let args = EditArgs::new(file, [Replacement::new("old", "new")]);
    let footprint = edit.footprint(&args);
    let wanted = Touches::Writes(Scope::Paths(vec![PathBuf::from(file)]));
    assert_eq!(footprint, wanted);
    let (edited, context) = call(&edit, args);
    assert!(edited.is_ok(), "{edited:?}");
    let on_disk = std::fs::read_to_string(home.join("strata/src/lib.rs"));
    assert_eq!(on_disk.ok().as_deref(), Some("fn new() {}\n"));
    let changed: FilesChanged = context.result().ok().flatten().unwrap_or_default();
    let changed: Vec<&str> = changed.0.iter().map(|it| it.path.as_str()).collect();
    assert_eq!(changed, [file]);

    let search = Search {
        workspace: workspace.clone(),
        base: base.clone(),
        confine: false,
    };
    let (found, _) = call(&search, SearchArgs::new("fn new").in_path("bench:~/strata"));
    let found = found.unwrap_or_default();
    assert_eq!(found, "bench:~/strata/src/lib.rs:1: fn new() {}");

    let shell = Shell {
        workspace,
        launcher: Launcher::default().in_dir(dir.join("local")),
        ..Shell::default()
    };
    let (ran, _) = call(
        &shell,
        ShellArgs::new("pwd && ls src").in_dir("bench:~/strata"),
    );
    let ran = ran.unwrap_or_default();
    let there = home.join("strata").display().to_string();
    assert!(ran.contains(&there) && ran.contains("lib.rs"), "{ran}");
    assert!(ran.ends_with("[exit code 0]"), "{ran}");
    std::fs::remove_dir_all(&dir).ok();
}

/// A transport that ends with 255 fails a file operation with what it
/// said; the command of a `shell` call there ends with that status.
#[test]
fn a_failed_transport_fails_the_call() {
    let down = Remote::new(
        "down",
        ["sh", "-c", "echo 'no route to host' >&2; exit 255"],
        Launcher::default(),
    );
    let workspace: Arc<dyn Workspace> = Arc::new(Hosts::new(Arc::new(Local), down));
    let read = Read {
        workspace: workspace.clone(),
        base: None,
        confine: false,
        seen: crate::Seen::default(),
    };
    let (failed, _) = call(&read, ReadArgs::new("down:/etc/hostname"));
    let failed = failed
        .err()
        .map(|error| error.to_string())
        .unwrap_or_default();
    assert!(
        failed.contains(
            "the transport to down failed (exit status 255): no route to host (tried twice)"
        ),
        "{failed}"
    );
    let shell = Shell {
        workspace,
        ..Shell::default()
    };
    let (ran, _) = call(&shell, ShellArgs::new("true").in_dir("down:~"));
    let ran = ran.unwrap_or_default();
    assert!(ran.ends_with("[exit code 255]"), "{ran}");
}

/// Paths fold without asking the host, a relative one under its home; a
/// `HOST:` path is kept whole when joined onto a base, and an unknown
/// host's is this machine's. A confined call refuses another host's path.
#[test]
fn host_paths_fold_and_join_as_scp_reads_them() {
    let bench = Remote::new("bench", ["true"], Launcher::default());
    let hosts: &dyn Workspace = &Hosts::new(Arc::new(Local), bench);
    let base = Path::new("/project");
    for (path, resolved) in [
        ("bench:~/a/./b/../c.rs", "bench:~/a/c.rs"),
        ("bench:src/lib.rs", "bench:~/src/lib.rs"),
        ("bench:", "bench:~"),
        ("bench:~/../other", "bench:~/../other"),
        ("bench:/srv/../app", "bench:/app"),
        ("other:~/x", "/project/other:~/x"),
    ] {
        let got = hosts.resolve(Some(base), path);
        assert_eq!(got, Path::new(resolved), "{path}");
    }
    let remote_base = Path::new("bench:~/strata");
    let joined = hosts.resolve(Some(remote_base), "src/lib.rs");
    assert_eq!(joined, Path::new("bench:~/strata/src/lib.rs"));
    let refused = hosts.place(Some(base), true, "bench:~/x").is_err();
    let kept = hosts
        .place(Some(remote_base), true, "src/../lib.rs")
        .is_ok();
    let escaped = hosts.place(Some(remote_base), true, "../lib.rs").is_err();
    assert!(refused && kept && escaped, "{refused} {kept} {escaped}");
    assert!(Remote::new("a:b", ["ssh"], Launcher::default()).is_err());
    assert!(Remote::new("a", Vec::<String>::new(), Launcher::default()).is_err());
}

/// [`transport`], first appending a line to `log` for each call.
fn counted(home: &Path, log: &Path) -> Vec<String> {
    vec![
        "sh".to_owned(),
        "-c".to_owned(),
        "echo call >> \"$1\"; cd \"$0\" && HOME=\"$0\" exec sh -c \"$2\"".to_owned(),
        home.display().to_string(),
        log.display().to_string(),
    ]
}

/// How many lines `log` holds, emptying it.
fn calls(log: &Path) -> usize {
    let count = std::fs::read_to_string(log).map_or(0, |log| log.lines().count());
    std::fs::remove_file(log).ok();
    count
}

/// A `read` is one transport call, as is a `write`, of a new file or over
/// an old one; an `edit` is two (the read the change is made from, then the
/// write), and a `search` that finds files two (the listing, then every
/// file in one script).
#[test]
fn each_file_operation_is_one_transport_call() {
    let dir = scratch("calls");
    let home = dir.join("home");
    let log = dir.join("calls.log");
    std::fs::write(home.join("strata/src/lib.rs"), "fn old() {}\n").ok();
    let bench = Remote::new("bench", counted(&home, &log), Launcher::default());
    let workspace: Arc<dyn Workspace> = Arc::new(Hosts::new(Arc::new(Local), bench));
    let file = "bench:~/strata/src/lib.rs";
    let read = Read {
        workspace: workspace.clone(),
        base: None,
        confine: false,
        seen: crate::Seen::default(),
    };
    let edit = Edit {
        workspace: workspace.clone(),
        base: None,
        confine: false,
        seen: crate::Seen::default(),
    };
    let write = crate::Write {
        workspace: workspace.clone(),
        base: None,
        confine: false,
    };
    let search = Search {
        workspace,
        base: None,
        confine: false,
    };
    let (shown, _) = call(&read, ReadArgs::new(file));
    let read_calls = calls(&log);
    let (edited, _) = call(&edit, EditArgs::new(file, [Replacement::new("old", "new")]));
    let edit_calls = calls(&log);
    let (wrote, _) = call(&write, crate::WriteArgs::new(file, "fn newer() {}\n"));
    let write_calls = calls(&log);
    let new_file = "bench:~/strata/src/new.rs";
    let (created, _) = call(&write, crate::WriteArgs::new(new_file, "fn new() {}\n"));
    let create_calls = calls(&log);
    let (found, _) = call(&search, SearchArgs::new("newer").in_path("bench:~/strata"));
    let search_calls = calls(&log);
    std::fs::remove_dir_all(&dir).ok();
    assert!(shown.is_ok() && edited.is_ok() && found.is_ok());
    assert_eq!(
        wrote.ok().as_deref(),
        Some("Replaced bench:~/strata/src/lib.rs, which had 1 line.")
    );
    assert_eq!(
        created.ok().as_deref(),
        Some("Created bench:~/strata/src/new.rs.")
    );
    assert_eq!(
        (
            read_calls,
            edit_calls,
            write_calls,
            create_calls,
            search_calls
        ),
        (1, 2, 1, 1, 2)
    );
}

/// [`transport`], first appending to `log` how many bytes each call sent
/// back.
fn metered(home: &Path, log: &Path) -> Vec<String> {
    vec![
        "sh".to_owned(),
        "-c".to_owned(),
        "o=$(mktemp) || exit 255; (cd \"$0\" && HOME=\"$0\" exec sh -c \"$2\") > \"$o\"; s=$?; \
         wc -c < \"$o\" >> \"$1\"; cat \"$o\"; rm -f \"$o\"; exit $s"
            .to_owned(),
        home.display().to_string(),
        log.display().to_string(),
    ]
}

/// A `read` of some lines of a long file is sent only those lines, and
/// still says how many lines the file has and publishes the hash of all
/// of it; a last line without a line end counts.
#[test]
fn a_read_of_some_lines_is_sent_only_those() {
    let dir = scratch("excerpt");
    let home = dir.join("home");
    let log = dir.join("sent.log");
    let long: String = (1..=20_000).map(|n| format!("line {n}\n")).collect();
    std::fs::write(home.join("strata/long.log"), &long).ok();
    std::fs::write(home.join("strata/short.txt"), "one\ntwo").ok();
    let bench = Remote::new("bench", metered(&home, &log), Launcher::default());
    let read = Read {
        workspace: Arc::new(Hosts::new(Arc::new(Local), bench)),
        base: None,
        confine: false,
        seen: crate::Seen::default(),
    };
    let args = ReadArgs::new("bench:~/strata/long.log").lines(10_000, 2);
    let (shown, context) = call(&read, args);
    let sent = std::fs::read_to_string(&log).unwrap_or_default();
    let (short, _) = call(&read, ReadArgs::new("bench:~/strata/short.txt").lines(2, 5));
    std::fs::remove_dir_all(&dir).ok();
    assert_eq!(
        shown.ok().as_deref(),
        Some("line 10000\nline 10001\n[9999 more lines; read on with offset 10002]\n")
    );
    let facts: FilesRead = context.result().ok().flatten().unwrap_or_default();
    let hashes: Vec<_> = facts.0.iter().map(|it| it.hash.clone()).collect();
    assert_eq!(hashes, [rig_core::tool::content_hash(long.as_bytes())]);
    let sent: Vec<usize> = sent.lines().filter_map(|n| n.trim().parse().ok()).collect();
    assert!(
        matches!(sent.as_slice(), [bytes] if *bytes < 200),
        "{sent:?}"
    );
    assert_eq!(short.ok().as_deref(), Some("two\n"));
}

/// Reads sent all at once run at most [`super::CALLS_PER_HOST`] at a time
/// on their host, and every one is answered.
#[test]
fn a_host_runs_a_few_file_operations_at_once() {
    let dir = scratch("burst");
    let home = dir.join("home");
    let running = dir.join("running");
    std::fs::create_dir_all(&running).ok();
    for n in 0..8 {
        std::fs::write(home.join(format!("strata/src/{n}.rs")), format!("{n}\n")).ok();
    }
    let seen = dir.join("seen.log");
    let transport = vec![
        "sh".to_owned(),
        "-c".to_owned(),
        "touch \"$1/$$\"; sleep 0.2; ls \"$1\" | wc -l >> \"$2\"; rm \"$1/$$\"; \
         cd \"$0\" && HOME=\"$0\" exec sh -c \"$3\""
            .to_owned(),
        home.display().to_string(),
        running.display().to_string(),
        seen.display().to_string(),
    ];
    let bench = Remote::new("bench", transport, Launcher::default());
    let workspace: Arc<dyn Workspace> = Arc::new(Hosts::new(Arc::new(Local), bench));
    let read = Read {
        workspace,
        base: None,
        confine: false,
        seen: crate::Seen::default(),
    };
    let shown: Vec<Option<String>> = std::thread::scope(|scope| {
        let reads: Vec<_> = (0..8)
            .map(|n| {
                let read = &read;
                scope.spawn(move || {
                    call(read, ReadArgs::new(format!("bench:~/strata/src/{n}.rs")))
                        .0
                        .ok()
                })
            })
            .collect();
        reads
            .into_iter()
            .map(|read| read.join().ok().flatten())
            .collect()
    });
    let most = std::fs::read_to_string(&seen)
        .unwrap_or_default()
        .lines()
        .filter_map(|line| line.trim().parse::<usize>().ok())
        .max();
    std::fs::remove_dir_all(&dir).ok();
    let wanted: Vec<Option<String>> = (0..8).map(|n| Some(format!("{n}\n"))).collect();
    assert_eq!(shown, wanted);
    assert!(
        most.is_some_and(|most| most <= super::CALLS_PER_HOST),
        "{most:?}"
    );
}

/// A read whose transport failed (exit 255) is sent once more; a write
/// never is, and its error says the change may have been made.
#[test]
fn a_failed_transport_is_tried_again_for_a_read_only() {
    let dir = scratch("again");
    let home = dir.join("home");
    let log = dir.join("calls.log");
    let failed = dir.join("failed");
    std::fs::write(home.join("strata/src/lib.rs"), "fn old() {}\n").ok();
    // The first call fails, as a connection the server dropped does.
    let flaky = vec![
        "sh".to_owned(),
        "-c".to_owned(),
        "echo call >> \"$1\"; if [ ! -e \"$2\" ]; then touch \"$2\"; \
         echo 'closed by peer' >&2; exit 255; fi; \
         cd \"$0\" && HOME=\"$0\" exec sh -c \"$3\""
            .to_owned(),
        home.display().to_string(),
        log.display().to_string(),
        failed.display().to_string(),
    ];
    let bench = Remote::new("bench", flaky, Launcher::default());
    let workspace: Arc<dyn Workspace> = Arc::new(Hosts::new(Arc::new(Local), bench));
    let read = Read {
        workspace,
        base: None,
        confine: false,
        seen: crate::Seen::default(),
    };
    let (shown, _) = call(&read, ReadArgs::new("bench:~/strata/src/lib.rs"));
    let read_calls = calls(&log);
    let down = vec![
        "sh".to_owned(),
        "-c".to_owned(),
        "echo call >> \"$0\"; echo 'closed by peer' >&2; exit 255".to_owned(),
        log.display().to_string(),
    ];
    let wrote = Remote::new("bench", down, Launcher::default())
        .and_then(|bench| bench.write(Path::new("~/strata/src/lib.rs"), b"fn new() {}\n"));
    let write_calls = calls(&log);
    let on_disk = std::fs::read_to_string(home.join("strata/src/lib.rs"));
    std::fs::remove_dir_all(&dir).ok();
    assert_eq!(shown.ok().as_deref(), Some("fn old() {}\n"));
    assert_eq!((read_calls, write_calls), (2, 1));
    let refused = wrote
        .err()
        .map(|error| error.to_string())
        .unwrap_or_default();
    assert!(refused.contains("may have been made or not"), "{refused}");
    assert_eq!(on_disk.ok().as_deref(), Some("fn old() {}\n"));
}

/// A command there that runs past its timeout is asked to stop first: its
/// local transport gets `SIGTERM`, the signal `ssh` and `koh exec` hang up
/// the remote command on, and `SIGKILL` only when it is still running
/// after a grace period.
#[cfg(unix)]
#[test]
fn a_timed_out_command_asks_its_transport_to_stop_first() {
    use std::time::{Duration, Instant};

    let dir = scratch("stop");
    let home = dir.join("home");
    let log = dir.join("signals.log");
    // The transport runs the command in the background and waits for it,
    // so a signal reaches its trap at once.
    let transport = |on_term: &str| {
        vec![
            "sh".to_owned(),
            "-c".to_owned(),
            format!("trap '{on_term}' TERM; cd \"$0\" && HOME=\"$0\" sh -c \"$2\" & wait; wait"),
            home.display().to_string(),
            log.display().to_string(),
        ]
    };
    let ended = |transport: Vec<String>| {
        let bench = Remote::new("bench", transport, Launcher::default());
        let shell = Shell {
            workspace: Arc::new(Hosts::new(Arc::new(Local), bench)),
            ..Shell::default()
        };
        let started = Instant::now();
        let args = ShellArgs::new("sleep 30").in_dir("bench:~").timeout_secs(1);
        let ran = futures::executor::block_on(shell.run(args));
        (
            ran.map(|ran| ran.output).unwrap_or_default(),
            started.elapsed(),
        )
    };
    let (hung_up, quick) = ended(transport("echo TERM >> \"$1\"; exit 143"));
    let heard = std::fs::read_to_string(&log).unwrap_or_default();
    // One that will not stop is killed.
    let deaf = vec![
        "sh".to_owned(),
        "-c".to_owned(),
        "trap 'echo ignored >> \"$0\"' TERM; while :; do sleep 0.1; done".to_owned(),
        log.display().to_string(),
    ];
    let (deaf, slow) = ended(deaf);
    let heard_again = std::fs::read_to_string(&log).unwrap_or_default();
    std::fs::remove_dir_all(&dir).ok();
    assert!(
        hung_up.starts_with("[timed out after 1s and killed"),
        "{hung_up}"
    );
    assert_eq!(heard, "TERM\n");
    assert!(quick < crate::process::STOP_GRACE, "{quick:?}");
    assert!(deaf.contains("[timed out after 1s and killed"), "{deaf}");
    assert_eq!(heard_again, "TERM\nignored\n");
    let most = Duration::from_secs(1) + crate::process::STOP_GRACE * 3;
    assert!(
        slow >= crate::process::STOP_GRACE && slow < most,
        "{slow:?}"
    );
}
