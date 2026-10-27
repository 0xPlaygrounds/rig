#![cfg(unix)]

use std::io;
use std::path::PathBuf;
use std::sync::Arc;

use rig_core::tool::{Tool, ToolContext};
use rig_tools::process::Launcher;
use rig_tools::remote::Remote;
use rig_tools::workspace::{Local, Workspace};
use rig_tools::{Edit, EditArgs, Read, ReadArgs, Replacement, Seen, Write, WriteArgs};

type TestResult = Result<(), Box<dyn std::error::Error>>;

struct Scratch(PathBuf);
impl Scratch {
    fn new() -> io::Result<Self> {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(io::Error::other)?
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("rig-link-audit-{}-{stamp}", std::process::id()));
        std::fs::create_dir(&path)?;
        Ok(Self(path))
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn check(remote: bool, operation: &str, symlink: bool) -> TestResult {
    let scratch = Scratch::new()?;
    let target = scratch.0.join("target.txt");
    let path = scratch.0.join("config.txt");
    std::fs::write(&target, "old value\n")?;
    if symlink {
        std::os::unix::fs::symlink("target.txt", &path)?;
    } else {
        std::fs::write(&path, "old value\n")?;
    }
    // Runs the actual Remote shell scripts on real files, with sh as the
    // transport. No SSH server or mocked filesystem is needed.
    let workspace: Arc<dyn Workspace> = if remote {
        Arc::new(Remote::new(
            "test",
            ["sh", "-c", "exec sh -c \"$1\"", "transport"],
            Launcher::default(),
        )?)
    } else {
        Arc::new(Local)
    };
    let name = path.to_string_lossy().into_owned();
    let seen = Seen::default();
    if operation == "cached_edit" {
        let read = Read {
            workspace: workspace.clone(),
            seen: seen.clone(),
            ..Read::default()
        };
        futures::executor::block_on(read.call(&mut ToolContext::new(), ReadArgs::new(&name)))?;
    }
    let output = if operation == "write" {
        let tool = Write {
            workspace,
            ..Write::default()
        };
        futures::executor::block_on(
            tool.call(&mut ToolContext::new(), WriteArgs::new(name, "new value\n")),
        )?
    } else {
        let tool = Edit {
            workspace,
            seen,
            ..Edit::default()
        };
        futures::executor::block_on(tool.call(
            &mut ToolContext::new(),
            EditArgs::new(name, [Replacement::new("old", "new")]),
        ))?
    };
    let is_link = std::fs::symlink_metadata(&path)?.file_type().is_symlink();
    let actual_target = std::fs::read_to_string(&target)?;
    let actual_path = std::fs::read_to_string(&path)?;
    println!("remote={remote}, operation={operation}, symlink={symlink}: {output:?}");
    println!("link preserved={is_link}; target={actual_target:?}; path={actual_path:?}");
    if actual_path != "new value\n" || (symlink && (!is_link || actual_target != "new value\n")) {
        return Err(io::Error::other(
            "successful operation replaced the symlink without updating its target",
        )
        .into());
    }
    Ok(())
}

#[test]
fn remote_write_preserves_symlink() -> TestResult {
    check(true, "write", true)
}
#[test]
fn remote_uncached_edit_preserves_symlink() -> TestResult {
    check(true, "edit", true)
}
#[test]
fn remote_cached_edit_preserves_symlink() -> TestResult {
    check(true, "cached_edit", true)
}
#[test]
fn local_symlink_control() -> TestResult {
    for operation in ["write", "edit", "cached_edit"] {
        check(false, operation, true)?;
    }
    Ok(())
}
#[test]
fn remote_regular_file_control() -> TestResult {
    for operation in ["write", "edit", "cached_edit"] {
        check(true, operation, false)?;
    }
    Ok(())
}

#[test]
fn remote_resolves_chains_dangling_targets_and_refuses_cycles() -> TestResult {
    let scratch = Scratch::new()?;
    let remote = Remote::new(
        "test",
        ["sh", "-c", "exec sh -c \"$1\"", "transport"],
        Launcher::default(),
    )?;
    let first = scratch.0.join("first");
    let second = scratch.0.join("second");
    let target = scratch.0.join("sub/target with spaces\n");
    std::os::unix::fs::symlink("second", &first)?;
    std::os::unix::fs::symlink(&target, &second)?;
    remote.write(&first, b"created")?;
    if std::fs::read(&target)? != b"created" || !std::fs::symlink_metadata(&first)?.is_symlink() {
        return Err(io::Error::other("chain or newline target was not preserved").into());
    }
    std::fs::remove_file(&second)?;
    std::os::unix::fs::symlink("first", &second)?;
    if remote.write(&first, b"refuse").is_ok() || !std::fs::symlink_metadata(&first)?.is_symlink() {
        return Err(io::Error::other("cyclic link was changed").into());
    }
    Ok(())
}
