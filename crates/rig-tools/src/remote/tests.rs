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
    };
    let (failed, _) = call(&read, ReadArgs::new("down:/etc/hostname"));
    let failed = failed
        .err()
        .map(|error| error.to_string())
        .unwrap_or_default();
    assert!(
        failed.contains("the transport to down failed (exit status 255): no route to host"),
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
