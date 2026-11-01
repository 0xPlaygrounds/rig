#![allow(clippy::disallowed_methods, reason = "test fixtures")]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use rig_core::tool::{FilesChanged, FilesRead, Scope, Tool, ToolContext, ToolOutput, Touches};

use super::{Hosts, Remote};
use crate::process::Launcher;
use crate::workspace::{Local, Workspace};
use crate::{
    Edit, EditArgs, List, ListArgs, Read, ReadArgs, Replacement, Search, SearchArgs, Shell,
    ShellArgs, split_status,
};

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
    let bench = Remote::new("bench", transport(&home), Launcher::default);
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
    assert_eq!(
        shown.ok().as_ref().and_then(ToolOutput::as_text),
        Some("fn old() {}\n")
    );
    let facts: FilesRead = context.result().ok().flatten().unwrap_or_default();
    let read_paths: Vec<&str> = facts.0.iter().map(|it| it.path.as_str()).collect();
    assert_eq!(read_paths, [file]);
    let (local, _) = call(&read, ReadArgs::new("notes.txt"));
    assert_eq!(
        local.ok().as_ref().and_then(ToolOutput::as_text),
        Some("here\n")
    );
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
        Launcher::default,
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
/// host's is this machine's. A path starting with `~` is in the home of the
/// host its base is on. A confined call refuses another host's path.
#[test]
fn host_paths_fold_and_join_as_scp_reads_them() {
    let bench = Remote::new("bench", ["true"], Launcher::default);
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
    let home = hosts.resolve(Some(remote_base), "~/notes");
    assert_eq!(home, Path::new("bench:~/notes"));
    // A `HOST:` path is the host's, so it is read off a thread that must
    // not wait; any other is this machine's.
    assert_eq!(hosts.host_for(&home), Some("bench"));
    assert_eq!(hosts.host_for(Path::new("/project/other:~/x")), None);
    let alone = Remote::new("solo", ["true"], Launcher::default).map(|remote| {
        let remote: &dyn Workspace = &remote;
        remote.resolve(Some(Path::new("/srv")), "~/x")
    });
    assert_eq!(alone.ok().as_deref(), Some(Path::new("~/x")));
    assert!(hosts.place(Some(remote_base), true, "~/x").is_err());
    let refused = hosts.place(Some(base), true, "bench:~/x").is_err();
    let kept = hosts
        .place(Some(remote_base), true, "src/../lib.rs")
        .is_ok();
    let escaped = hosts.place(Some(remote_base), true, "../lib.rs").is_err();
    assert!(refused && kept && escaped, "{refused} {kept} {escaped}");
    assert!(Remote::new("a:b", ["ssh"], Launcher::default).is_err());
    assert!(Remote::new("a", Vec::<String>::new(), Launcher::default).is_err());
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
/// an old one, and an `edit` of a file read whole, written or edited
/// before; an
/// edit of a file that changed since is two (the write that finds it
/// changed and sends it back, then the write), as is a `search` that finds
/// files (the listing, then every file in one script).
#[test]
fn each_file_operation_is_one_transport_call() {
    let dir = scratch("calls");
    let home = dir.join("home");
    let log = dir.join("calls.log");
    std::fs::write(home.join("strata/src/lib.rs"), "fn old() {}\n").ok();
    let bench = Remote::new("bench", counted(&home, &log), Launcher::default);
    let workspace: Arc<dyn Workspace> = Arc::new(Hosts::new(Arc::new(Local), bench));
    let file = "bench:~/strata/src/lib.rs";
    let seen = crate::Seen::default();
    let read = Read {
        workspace: workspace.clone(),
        base: None,
        confine: false,
        seen: seen.clone(),
    };
    let edit = Edit {
        workspace: workspace.clone(),
        base: None,
        confine: false,
        seen: seen.clone(),
    };
    let write = crate::Write {
        workspace: workspace.clone(),
        base: None,
        confine: false,
        seen,
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
    let (again, _) = call(
        &edit,
        EditArgs::new(file, [Replacement::new("new", "newest")]),
    );
    let again_calls = calls(&log);
    std::fs::write(
        home.join("strata/src/lib.rs"),
        "fn newest() {}\nfn other() {}\n",
    )
    .ok();
    let (stale, _) = call(
        &edit,
        EditArgs::new(file, [Replacement::new("newest", "new")]),
    );
    let stale_calls = calls(&log);
    let edited_text = std::fs::read_to_string(home.join("strata/src/lib.rs")).ok();
    let (wrote, _) = call(&write, crate::WriteArgs::new(file, "fn newer() {}\n"));
    let write_calls = calls(&log);
    let new_file = "bench:~/strata/src/new.rs";
    let (created, _) = call(&write, crate::WriteArgs::new(new_file, "fn new() {}\n"));
    let create_calls = calls(&log);
    let (fresh, _) = call(
        &edit,
        EditArgs::new(new_file, [Replacement::new("new", "fresh")]),
    );
    let fresh_calls = calls(&log);
    let fresh_text = std::fs::read_to_string(home.join("strata/src/new.rs")).ok();
    let (found, _) = call(&search, SearchArgs::new("newer").in_path("bench:~/strata"));
    let search_calls = calls(&log);
    std::fs::remove_dir_all(&dir).ok();
    assert!(shown.is_ok() && edited.is_ok() && again.is_ok() && stale.is_ok());
    assert!(found.is_ok() && fresh.is_ok());
    assert_eq!(fresh_text.as_deref(), Some("fn fresh() {}\n"));
    assert_eq!(edited_text.as_deref(), Some("fn new() {}\nfn other() {}\n"));
    assert_eq!(
        wrote.ok().as_deref(),
        Some("Replaced bench:~/strata/src/lib.rs, which had 2 lines.")
    );
    assert_eq!(
        created.ok().as_deref(),
        Some("Created bench:~/strata/src/new.rs.")
    );
    assert_eq!(
        (read_calls, edit_calls, again_calls, stale_calls),
        (1, 1, 1, 2)
    );
    assert_eq!(
        (write_calls, create_calls, fresh_calls, search_calls),
        (1, 1, 1, 2)
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
    let bench = Remote::new("bench", metered(&home, &log), Launcher::default);
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
        shown.ok().as_ref().and_then(ToolOutput::as_text),
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
    assert_eq!(
        short.ok().as_ref().and_then(ToolOutput::as_text),
        Some("two\n")
    );
}

/// A `read` of some lines of a file of at most [`crate::MAX_BYTES`] gets
/// the whole file in its one call, so an `edit` after it is one call too,
/// while a larger file is still sent only the lines asked.
#[test]
fn an_edit_after_a_read_of_some_lines_of_a_small_file_is_one_call() {
    let dir = scratch("slice-edit");
    let home = dir.join("home");
    let log = dir.join("calls.log");
    let small: String = (1..=400).map(|n| format!("let x{n} = {n};\n")).collect();
    std::fs::write(home.join("strata/src/lib.rs"), &small).ok();
    let bench = Remote::new("bench", counted(&home, &log), Launcher::default);
    let workspace: Arc<dyn Workspace> = Arc::new(Hosts::new(Arc::new(Local), bench));
    let seen = crate::Seen::default();
    let read = Read {
        workspace: workspace.clone(),
        base: None,
        confine: false,
        seen: seen.clone(),
    };
    let edit = Edit {
        workspace,
        base: None,
        confine: false,
        seen,
    };
    let file = "bench:~/strata/src/lib.rs";
    let (shown, _) = call(&read, ReadArgs::new(file).lines(200, 2));
    let read_calls = calls(&log);
    let (edited, _) = call(
        &edit,
        EditArgs::new(file, [Replacement::new("x200 = 200", "x200 = 0")]),
    );
    let edit_calls = calls(&log);
    let on_disk = std::fs::read_to_string(home.join("strata/src/lib.rs")).unwrap_or_default();
    std::fs::remove_dir_all(&dir).ok();
    assert_eq!(
        shown.ok().as_ref().and_then(ToolOutput::as_text),
        Some("let x200 = 200;\nlet x201 = 201;\n[199 more lines; read on with offset 202]\n")
    );
    assert!(edited.is_ok(), "{edited:?}");
    assert!(on_disk.contains("let x200 = 0;\nlet x201"), "{on_disk}");
    assert_eq!((read_calls, edit_calls), (1, 1));
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
    let bench = Remote::new("bench", transport, Launcher::default);
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
                        .and_then(|shown| shown.as_text().map(str::to_owned))
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

/// A read whose transport failed (exit 255) or was killed, as a dropped
/// connection's is, is sent once more; a write never is: the file is
/// checked again instead, and when that fails too, its error says the
/// write may have happened.
#[cfg(unix)]
#[test]
fn a_failed_transport_is_tried_again_for_a_read_only() {
    let dir = scratch("again");
    let home = dir.join("home");
    let log = dir.join("calls.log");
    let failed = dir.join("failed");
    std::fs::write(home.join("strata/src/lib.rs"), "fn old() {}\n").ok();
    let mut shown = Vec::new();
    // The first call of each fails, as a connection the server dropped
    // does, and as one cut on this machine does.
    for fails in ["echo 'closed by peer' >&2; exit 255", "kill -9 $$"] {
        std::fs::remove_file(&failed).ok();
        let flaky = vec![
            "sh".to_owned(),
            "-c".to_owned(),
            format!(
                "echo call >> \"$1\"; if [ ! -e \"$2\" ]; then touch \"$2\"; {fails}; fi; \
                 cd \"$0\" && HOME=\"$0\" exec sh -c \"$3\""
            ),
            home.display().to_string(),
            log.display().to_string(),
            failed.display().to_string(),
        ];
        let bench = Remote::new("bench", flaky, Launcher::default);
        let workspace: Arc<dyn Workspace> = Arc::new(Hosts::new(Arc::new(Local), bench));
        let read = Read {
            workspace,
            base: None,
            confine: false,
            seen: crate::Seen::default(),
        };
        let (read, _) = call(&read, ReadArgs::new("bench:~/strata/src/lib.rs"));
        shown.push(read.ok().and_then(|read| read.as_text().map(str::to_owned)));
    }
    let read_calls = calls(&log);
    let down = vec![
        "sh".to_owned(),
        "-c".to_owned(),
        "echo call >> \"$0\"; echo 'closed by peer' >&2; exit 255".to_owned(),
        log.display().to_string(),
    ];
    let wrote = Remote::new("bench", down, Launcher::default)
        .and_then(|bench| bench.write(Path::new("~/strata/src/lib.rs"), b"fn new() {}\n"));
    let write_calls = calls(&log);
    let on_disk = std::fs::read_to_string(home.join("strata/src/lib.rs"));
    std::fs::remove_dir_all(&dir).ok();
    let old = Some("fn old() {}\n".to_owned());
    assert_eq!(shown, [old.clone(), old]);
    assert_eq!((read_calls, write_calls), (4, 3));
    let refused = wrote
        .err()
        .map(|error| error.to_string())
        .unwrap_or_default();
    assert!(
        refused.contains(
            "The file was checked again: that failed too (the transport to bench failed (exit \
             status 255): closed by peer (tried twice)), so the write may or may not have \
             happened; read the file before going on"
        ),
        "{refused}"
    );
    assert_eq!(on_disk.ok().as_deref(), Some("fn old() {}\n"));
}

/// A write whose transport dies mid-write, as a dropped link does, fails
/// naming the file and saying whether the write happened, from the file's
/// hash read there again: a write that was lost, and one that landed
/// before the transport was killed. The edit forgets the text it kept of
/// the file, so the next edit starts from the file itself (a read, then the
/// write).
#[cfg(unix)]
#[test]
fn a_write_cut_off_mid_way_says_whether_it_happened() {
    let mut said = Vec::new();
    let mut texts = Vec::new();
    let mut next_calls = Vec::new();
    for (name, dies) in [
        ("lost", "head -c 3 > /dev/null; exit 1"),
        (
            "landed",
            "cd \"$0\" && HOME=\"$0\" sh -c \"$3\"; kill -9 $$",
        ),
    ] {
        let dir = scratch(name);
        let home = dir.join("home");
        let log = dir.join("calls.log");
        let armed = dir.join("armed");
        std::fs::write(home.join("strata/src/lib.rs"), "fn old() {}\n").ok();
        let transport = vec![
            "sh".to_owned(),
            "-c".to_owned(),
            format!(
                "echo call >> \"$1\"; if [ -e \"$2\" ]; then rm -f \"$2\"; {dies}; fi; \
                 cd \"$0\" && HOME=\"$0\" exec sh -c \"$3\""
            ),
            home.display().to_string(),
            log.display().to_string(),
            armed.display().to_string(),
        ];
        let bench = Remote::new("bench", transport, Launcher::default);
        let workspace: Arc<dyn Workspace> = Arc::new(Hosts::new(Arc::new(Local), bench));
        let seen = crate::Seen::default();
        let read = Read {
            workspace: workspace.clone(),
            base: None,
            confine: false,
            seen: seen.clone(),
        };
        let edit = Edit {
            workspace,
            base: None,
            confine: false,
            seen,
        };
        let file = "bench:~/strata/src/lib.rs";
        let (shown, _) = call(&read, ReadArgs::new(file));
        std::fs::write(&armed, "").ok();
        let (cut_off, _) = call(&edit, EditArgs::new(file, [Replacement::new("old", "new")]));
        said.push(
            cut_off
                .err()
                .map(|error| error.to_string())
                .unwrap_or_default(),
        );
        texts.push(std::fs::read_to_string(home.join("strata/src/lib.rs")).unwrap_or_default());
        calls(&log);
        let (again, _) = call(
            &edit,
            EditArgs::new(file, [Replacement::new("fn", "pub fn")]),
        );
        next_calls.push(again.is_ok().then(|| calls(&log)));
        std::fs::remove_dir_all(&dir).ok();
        assert!(shown.is_ok());
    }
    let ended = [
        "the operation on bench ended with exit status 1 and said nothing",
        "the transport to bench was killed by SIGKILL (signal 9), which rig did not send and \
         said nothing",
    ];
    let found = [
        "it does not currently hold what this write sent; the write may or may not have \
         happened; read the file before retrying",
        "it holds what this write sent, so the write happened",
    ];
    let wanted: Vec<String> = ended
        .iter()
        .zip(found)
        .map(|(ended, found)| {
            format!(
                "bench:~/strata/src/lib.rs: the write failed: {ended}. The file was checked \
                 again: {found}"
            )
        })
        .collect();
    assert_eq!(said, wanted);
    assert_eq!(texts, ["fn old() {}\n", "fn new() {}\n"]);
    assert_eq!(next_calls, [Some(2), Some(2)]);
}

/// A command there stops when its call does, however the call ends: when
/// it times out, and when its transport is cut off from outside, as a
/// dropped connection's is. Its whole process group goes, not only the
/// process the transport started, and nothing is left running there.
#[cfg(unix)]
#[test]
fn a_command_there_stops_with_its_call() {
    use std::time::{Duration, Instant};

    let dir = scratch("stop");
    let home = dir.join("home");
    let child = home.join("child");
    // The other machine's side runs in a session of its own, as it does
    // behind `ssh`, so a signal to the transport here never reaches it; it
    // reads the transport's stdin, as `ssh` forwards it.
    let apart = |then: &str| {
        vec![
            "sh".to_owned(),
            "-c".to_owned(),
            format!("exec 3<&0; cd \"$0\" && HOME=\"$0\" setsid sh -c \"$1\" <&3 3<&- & {then}"),
            home.display().to_string(),
        ]
    };
    let mut ended = Vec::new();
    let mut left = Vec::new();
    for (transport, timeout) in [
        (apart("wait"), Some(1)),
        (apart("sleep 1; kill -9 $$"), None),
    ] {
        std::fs::remove_file(&child).ok();
        let bench = Remote::new("bench", transport, Launcher::default);
        let shell = Shell {
            workspace: Arc::new(Hosts::new(Arc::new(Local), bench)),
            ..Shell::default()
        };
        let mut args =
            ShellArgs::new("sleep 300 & echo $! > ~/child; echo started; wait").in_dir("bench:~");
        args.timeout_secs = timeout;
        let ran = futures::executor::block_on(shell.run(args));
        let output = ran.map(|ran| ran.output).unwrap_or_default();
        ended.push(split_status(&output).1.unwrap_or_default().to_owned());
        let pid = std::fs::read_to_string(&child).unwrap_or_default();
        let alive = || {
            std::process::Command::new("kill")
                .args(["-0", pid.trim()])
                .status()
                .is_ok_and(|status| status.success())
        };
        let started = Instant::now();
        while alive() && started.elapsed() < crate::process::STOP_GRACE * 3 {
            std::thread::sleep(Duration::from_millis(50));
        }
        left.push(!pid.trim().is_empty() && alive());
    }
    std::fs::remove_dir_all(&dir).ok();
    assert_eq!(
        ended.first().map(String::as_str),
        Some("[timed out after 1s and stopped, on bench too]")
    );
    assert!(
        ended.get(1).is_some_and(|it| it
            .starts_with("[killed by SIGKILL (signal 9), which this tool did not send")
            && it.ends_with(
                "It was the connection to bench, and the command there was stopped with it]"
            )),
        "{ended:?}"
    );
    assert_eq!(left, [false, false]);
}

/// A file in a remote's home, listed alone, is shown as the remote names
/// it, `~` and all, so a tool given it back reaches that file.
#[test]
fn a_remote_home_file_is_listed_as_named() {
    let dir = scratch("list");
    let home = dir.join("home");
    std::fs::write(home.join("notes.txt"), "remote\n").ok();
    let listed = Remote::new("solo", transport(&home), Launcher::default).map(|remote| {
        let list = List {
            workspace: Arc::new(remote),
            base: None,
            confine: false,
        };
        call(&list, ListArgs::new("~/notes.txt")).0
    });
    std::fs::remove_dir_all(&dir).ok();
    assert_eq!(
        listed.ok().and_then(Result::ok).as_deref(),
        Some("~/notes.txt\n")
    );
}

/// A failed batch read there names each file it was to read, so a search
/// says what it could not read rather than "No matches."; a file under a
/// directory there that may not be searched is not taken for missing.
#[cfg(unix)]
#[test]
fn what_could_not_be_read_there_is_not_taken_for_absent() {
    use std::os::unix::fs::PermissionsExt as _;
    let dir = scratch("unread");
    let home = dir.join("home");
    std::fs::write(home.join("strata/src/lib.rs"), "fn needle() {}\n").ok();
    // Every script that copies files aside, as a batch read does, fails.
    let mut refusing = vec![
        "sh".to_owned(),
        "-c".to_owned(),
        "case \"$1\" in *mktemp*) echo 'reads refused' >&2; exit 1;; esac; \
         cd \"$0\" && HOME=\"$0\" exec sh -c \"$1\""
            .to_owned(),
    ];
    refusing.push(home.display().to_string());
    let bench = Remote::new("bench", refusing, Launcher::default);
    let workspace: Arc<dyn Workspace> = Arc::new(Hosts::new(Arc::new(Local), bench));
    let search = Search {
        workspace: workspace.clone(),
        base: None,
        confine: false,
    };
    let (found, _) = call(&search, SearchArgs::new("needle").in_path("bench:~/strata"));
    let found = found.unwrap_or_default();
    assert!(
        found.starts_with(
            "No matches in the files read.\n[could not read 1 path, so it was not searched: \
             bench:~/strata/src/lib.rs ("
        ) && found.contains("reads refused"),
        "{found}"
    );
    let locked = home.join("locked");
    std::fs::create_dir_all(&locked).ok();
    std::fs::write(locked.join("a.txt"), "a\n").ok();
    let link = home.join("strata/link.txt");
    std::os::unix::fs::symlink("../locked/a.txt", &link).ok();
    std::fs::create_dir_all(locked.join("inner")).ok();
    std::os::unix::fs::symlink("locked/inner", home.join("hidden")).ok();
    // A link to a directory that may be searched hides nothing.
    std::os::unix::fs::symlink("strata", home.join("alias")).ok();
    std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o000)).ok();
    let hidden = std::fs::read(locked.join("a.txt")).is_err();
    let stat = workspace.stat(Path::new("bench:~/locked/a.txt"));
    let linked = workspace.stat(Path::new("bench:~/strata/link.txt"));
    let slashed = workspace.stat(Path::new("bench:~/hidden/"));
    let gone = workspace.stat(Path::new("bench:~/strata/gone/a.txt"));
    let aliased = workspace.stat(Path::new("bench:~/alias/gone.txt"));
    let all = crate::workspace::Listing::All;
    let listing = workspace.files(Path::new("bench:~/locked"), None, all);
    let missing = workspace.files(Path::new("bench:~/strata/gone"), None, all);
    std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o755)).ok();
    std::fs::remove_dir_all(&dir).ok();
    assert!(gone.is_ok_and(|gone| gone == crate::workspace::Stat::Missing));
    assert!(aliased.is_ok_and(|gone| gone == crate::workspace::Stat::Missing));
    let not_found = Some(std::io::ErrorKind::NotFound);
    assert_eq!(missing.err().map(|error| error.kind()), not_found);
    if hidden {
        assert_eq!(
            stat.err().map(|error| error.kind()),
            Some(std::io::ErrorKind::PermissionDenied)
        );
        // Nor a link there whose target such a directory hides.
        assert_eq!(
            linked.err().map(|error| error.kind()),
            Some(std::io::ErrorKind::PermissionDenied)
        );
        // Written with a trailing `/` too.
        assert_eq!(
            slashed.err().map(|error| error.kind()),
            Some(std::io::ErrorKind::PermissionDenied)
        );
        // A directory there that may not be entered is no missing one.
        let kind = listing.err().map(|error| error.kind());
        assert!(kind.is_some() && kind != not_found, "{kind:?}");
    }
}

/// A command that ran on a host, whose output is spilled on this machine,
/// says where a cut failure word is but gives no command for the spill
/// file, which a command there would not find.
#[cfg(unix)]
#[test]
fn a_cut_line_of_a_command_there_gives_no_command_for_the_spill_here() {
    let dir = scratch("spill");
    let bench = Remote::new("bench", transport(&dir.join("home")), Launcher::default);
    let shell = Shell {
        workspace: Arc::new(Hosts::new(Arc::new(Local), bench)),
        spill: Some(crate::Spill(dir.join("spill"))),
        ..Shell::default()
    };
    let command = "seq 1 1500; head -c 3000 /dev/zero | tr '\\0' x; echo ' FAILED'; seq 1 1500";
    let (ran, _) = call(&shell, ShellArgs::new(command).in_dir("bench:~"));
    let ran = ran.unwrap_or_default();
    std::fs::remove_dir_all(&dir).ok();
    assert!(
        ran.contains("[… cut at byte 300 of 3007; `FAILED` is at bytes 3002-3007]"),
        "{ran:.600}"
    );
}

/// Whether the process `pid` runs: there and not a zombie.
#[cfg(target_os = "linux")]
fn alive(pid: u32) -> bool {
    std::fs::read_to_string(format!("/proc/{pid}/stat")).is_ok_and(|stat| {
        !stat
            .rsplit(')')
            .next()
            .unwrap_or_default()
            .trim_start()
            .starts_with('Z')
    })
}

/// Polls `future` once, as an executor that starts it does.
fn start<F: std::future::Future>(future: &mut std::pin::Pin<Box<F>>) -> bool {
    let mut context = std::task::Context::from_waker(std::task::Waker::noop());
    future.as_mut().poll(&mut context).is_ready()
}

/// Waits at most five seconds for `done`; whether it came.
fn soon(done: impl Fn() -> bool) -> bool {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while !done() && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    done()
}

/// A write stopped while its transport is still on its way there, as Esc
/// stops a call, never lands, and its transport is gone.
#[cfg(target_os = "linux")]
#[test]
fn a_stopped_write_never_lands() {
    let dir = scratch("stopped");
    let home = dir.join("home");
    let pid = dir.join("transport.pid");
    let slow = vec![
        "sh".to_owned(),
        "-c".to_owned(),
        "echo $$ > \"$1\"; sleep 3; cd \"$0\" && HOME=\"$0\" exec sh -c \"$2\"".to_owned(),
        home.display().to_string(),
        pid.display().to_string(),
    ];
    let bench = Remote::new("bench", slow, Launcher::default);
    let workspace: Arc<dyn Workspace> = Arc::new(Hosts::new(Arc::new(Local), bench));
    let mut writing = Box::pin(crate::blocking(move || {
        let late = Path::new("bench:~/late.txt");
        workspace
            .write(late, b"AFTER_STOP")
            .map_err(|error| rig_core::tool::ToolExecutionError::other(error.to_string()))
    }));
    assert!(!start(&mut writing));
    assert!(soon(|| pid.exists()));
    drop(writing);
    let transport = std::fs::read_to_string(&pid).ok();
    let transport = transport.and_then(|pid| pid.trim().parse::<u32>().ok());
    std::thread::sleep(std::time::Duration::from_secs(4));
    let landed = home.join("late.txt").exists();
    std::fs::remove_dir_all(&dir).ok();
    assert!(!landed, "the stopped write landed");
    assert!(transport.is_some_and(|pid| !alive(pid)));
}

/// Reads stopped while two run and a third waits for its turn on the
/// host, as Esc stops parallel calls: the third never starts.
#[cfg(unix)]
#[test]
fn a_read_queued_behind_a_stop_never_starts() {
    let dir = scratch("queued");
    let home = dir.join("home");
    let log = dir.join("calls.log");
    std::fs::write(home.join("strata/src/lib.rs"), "fn old() {}\n").ok();
    let slow = vec![
        "sh".to_owned(),
        "-c".to_owned(),
        "echo call >> \"$1\"; sleep 3; cd \"$0\" && HOME=\"$0\" exec sh -c \"$2\"".to_owned(),
        home.display().to_string(),
        log.display().to_string(),
    ];
    let bench = Remote::new("bench", slow, Launcher::default);
    let workspace: Arc<dyn Workspace> = Arc::new(Hosts::new(Arc::new(Local), bench));
    let started = || std::fs::read_to_string(&log).map_or(0, |log| log.lines().count());
    let read = || {
        let workspace = workspace.clone();
        Box::pin(crate::blocking(move || {
            let file = Path::new("bench:~/strata/src/lib.rs");
            workspace
                .read(file, 4096)
                .map_err(|error| rig_core::tool::ToolExecutionError::other(error.to_string()))
        }))
    };
    // Two take the host's turns, then the third waits for one.
    let mut running = [read(), read()];
    assert!(running.iter_mut().all(|read| !start(read)));
    assert!(soon(|| started() == super::CALLS_PER_HOST));
    let mut queued = read();
    assert!(!start(&mut queued));
    // Stopped while it waits, before the turns are given back.
    drop(queued);
    drop(running);
    std::thread::sleep(std::time::Duration::from_secs(4));
    let count = started();
    std::fs::remove_dir_all(&dir).ok();
    assert_eq!(count, super::CALLS_PER_HOST);
}

/// A write whose input reaches the host cut short, as when its call is
/// stopped or the link drops while it is sent, never lands half written.
#[cfg(unix)]
#[test]
fn a_write_cut_short_never_lands() {
    let dir = scratch("short");
    let home = dir.join("home");
    std::fs::write(home.join("strata/src/lib.rs"), "fn old() {}\n").ok();
    let short = vec![
        "sh".to_owned(),
        "-c".to_owned(),
        "head -c 3 | (cd \"$0\" && HOME=\"$0\" exec sh -c \"$1\")".to_owned(),
        home.display().to_string(),
    ];
    let wrote = Remote::new("bench", short, Launcher::default)
        .and_then(|bench| bench.write(Path::new("~/strata/src/lib.rs"), b"fn new() {}\n"));
    let on_disk = std::fs::read_to_string(home.join("strata/src/lib.rs"));
    std::fs::remove_dir_all(&dir).ok();
    assert!(wrote.is_err());
    assert_eq!(on_disk.ok().as_deref(), Some("fn old() {}\n"));
}

/// A host's transport starts under the policy of the moment, so one set
/// after the host was made, such as a sandbox's, holds from the next
/// operation on.
#[cfg(unix)]
#[test]
fn a_policy_set_later_holds_from_the_next_operation() {
    let dir = scratch("policy");
    let home = dir.join("home");
    let mark = dir.join("wrapped");
    std::fs::write(home.join("strata/src/lib.rs"), "fn old() {}\n").ok();
    let policy = Arc::new(std::sync::Mutex::new(Launcher::default()));
    let current = policy.clone();
    let bench = Remote::new("bench", transport(&home), move || {
        current
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    });
    let Ok(bench) = bench else {
        return assert!(bench.is_ok());
    };
    let file = Path::new("~/strata/src/lib.rs");
    let before = bench.read(file, 4096).is_ok() && !mark.exists();
    let wrapper = ["sh", "-c", "touch \"$0\"; exec \"$@\""].map(String::from);
    let wrapper = wrapper.into_iter().chain([mark.display().to_string()]);
    *policy
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = Launcher::default().wrapped(wrapper);
    let after = bench.read(file, 4096).is_ok() && mark.exists();
    std::fs::remove_dir_all(&dir).ok();
    assert!(before && after, "{before} {after}");
}

#[cfg(unix)]
#[test]
fn interrupted_write_payloads_preserve_the_original_for_both_write_paths() -> std::io::Result<()> {
    let dir = scratch("interrupted-payloads");
    let file = dir.join("important.txt");
    let marker = dir.join("attempted");
    let original = "valuable original contents\n".repeat(4096);
    let replacement = "replacement contents\n".repeat(8192);
    for guarded in [false, true] {
        for (forward, status, start) in [
            (0, 255, true),
            (4096, 255, true),
            (4096, 0, true),
            (replacement.len(), 0, true),
            (replacement.len(), 255, true),
            (0, 255, false),
        ] {
            std::fs::write(&file, &original)?;
            let _ = std::fs::remove_file(&marker);
            let transport = r#"
                if [ ! -e "$0/attempted" ]; then
                    touch "$0/attempted"
                    if [ "$3" = yes ]; then head -c "$1" | sh -c "$4"; result=$?;
                    else result=0; fi
                    if [ "$2" -ne 0 ]; then exit "$2"; fi
                    exit "$result"
                fi
                exec sh -c "$4"
            "#;
            let remote = Remote::new(
                "test",
                vec![
                    "sh".into(),
                    "-c".into(),
                    transport.into(),
                    dir.display().to_string(),
                    forward.to_string(),
                    status.to_string(),
                    if start { "yes" } else { "no" }.into(),
                ],
                Launcher::default,
            )?;
            let outcome = if guarded {
                remote
                    .write_if(
                        &file,
                        replacement.as_bytes(),
                        &rig_core::tool::content_hash(original.as_bytes()),
                    )
                    .map(|_| ())
            } else {
                remote.write(&file, replacement.as_bytes())
            };
            let complete = start && forward == replacement.len();
            let expected = if complete { &replacement } else { &original };
            if std::fs::read(&file)? != expected.as_bytes() {
                return Err(std::io::Error::other(format!(
                    "partial replacement committed: guarded={guarded}, forward={forward}"
                )));
            }
            if (status != 0 || !complete) && outcome.is_ok() {
                return Err(std::io::Error::other("failed transfer reported success"));
            }
            if complete && status == 0 {
                outcome?;
            } else if let Err(error) = outcome {
                if error.to_string().contains("write did not happen") {
                    return Err(std::io::Error::other(
                        "verification made an unsupported claim",
                    ));
                }
                if complete && !error.to_string().contains("it holds what this write sent") {
                    return Err(std::io::Error::other("completed write was not recognized"));
                }
            }
        }
    }
    std::fs::remove_dir_all(dir)
}
