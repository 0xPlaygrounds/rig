#![cfg(unix)]

use std::io;
use std::path::PathBuf;

use rig_core::tool::{FilesChanged, Tool, ToolContext, ToolErrorKind};
use rig_tools::{Write, WriteArgs};

type TestResult = Result<(), Box<dyn std::error::Error>>;

struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "rig-confined-write-audit-{}-{stamp}",
            std::process::id()
        ));
        std::fs::create_dir(&dir)?;
        std::fs::create_dir(dir.join("allowed"))?;
        std::fs::create_dir(dir.join("outside"))?;
        Ok(Self(std::fs::canonicalize(dir)?))
    }

    fn tool(&self) -> Write {
        Write {
            base: Some(self.0.join("allowed")),
            confine: true,
            ..Write::default()
        }
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn a_confined_write_must_not_create_a_dangling_symlinks_outside_target() -> TestResult {
    let scratch = Scratch::new()?;
    let target = scratch.0.join("outside/new.txt");
    std::os::unix::fs::symlink("../outside/new.txt", scratch.0.join("allowed/link.txt"))?;
    let tool = scratch.tool();
    let args = WriteArgs::new("link.txt", "escaped write\n");
    println!(
        "confine=true, declared footprint: {:?}",
        tool.footprint(&args)
    );
    let mut context = ToolContext::new();
    let result = futures::executor::block_on(tool.call(&mut context, args));
    println!("write result: {result:?}");
    println!(
        "outside file content: {:?}",
        std::fs::read_to_string(&target)
    );
    println!(
        "published file changes: {:?}",
        context.result::<FilesChanged>()
    );
    if target.exists() {
        return Err(io::Error::other("confined write created a file outside its base").into());
    }
    if !matches!(result, Err(ref error) if error.kind() == ToolErrorKind::PermissionDenied) {
        return Err(io::Error::other("outside target was not refused").into());
    }
    Ok(())
}

#[test]
fn a_confined_write_must_not_overwrite_an_existing_outside_file() -> TestResult {
    let scratch = Scratch::new()?;
    let victim = scratch.0.join("outside/victim.txt");
    std::fs::write(&victim, "original outside content\n")?;
    // The intermediate directory is absent, so this is initially a dangling link.
    // Creating it makes the target's '..' lead back to the existing outside file.
    let link = scratch.0.join("allowed/link.txt");
    std::os::unix::fs::symlink("../outside/missing/../victim.txt", &link)?;
    if std::fs::metadata(&link).is_ok() {
        return Err(io::Error::other("the fixture must start with a dangling symlink").into());
    }
    let tool = scratch.tool();
    let args = WriteArgs::new("link.txt", "overwritten outside the allowed directory\n");
    println!(
        "confine=true, declared footprint: {:?}",
        tool.footprint(&args)
    );
    let mut context = ToolContext::new();
    let result = futures::executor::block_on(tool.call(&mut context, args));
    let after = std::fs::read_to_string(&victim)?;
    println!("write result: {result:?}");
    println!("existing outside file after write: {after:?}");
    println!(
        "published file changes: {:?}",
        context.result::<FilesChanged>()
    );
    if after != "original outside content\n" {
        return Err(
            io::Error::other("confined write overwrote an existing file outside its base").into(),
        );
    }
    if !matches!(result, Err(ref error) if error.kind() == ToolErrorKind::PermissionDenied) {
        return Err(io::Error::other("outside target was not refused").into());
    }
    Ok(())
}

#[test]
fn existing_outside_symlink_and_direct_traversal_are_refused() -> TestResult {
    let scratch = Scratch::new()?;
    let victim = scratch.0.join("outside/victim.txt");
    std::fs::write(&victim, "original outside content\n")?;
    std::os::unix::fs::symlink("../outside/victim.txt", scratch.0.join("allowed/link.txt"))?;
    let tool = scratch.tool();
    for path in ["link.txt", "../outside/victim.txt"] {
        let result = futures::executor::block_on(tool.call(
            &mut ToolContext::new(),
            WriteArgs::new(path, "must not be written"),
        ));
        println!("control {path}: {result:?}");
        if !matches!(result, Err(ref error) if error.kind() == ToolErrorKind::PermissionDenied) {
            return Err(io::Error::other("control did not reject the outside path").into());
        }
    }
    if std::fs::read_to_string(&victim)? != "original outside content\n" {
        return Err(io::Error::other("control changed the outside file").into());
    }
    Ok(())
}

#[test]
fn a_confined_write_can_create_a_file_inside_its_base() -> TestResult {
    let scratch = Scratch::new()?;
    let result = futures::executor::block_on(scratch.tool().call(
        &mut ToolContext::new(),
        WriteArgs::new("new.txt", "allowed content\n"),
    ));
    println!("inside control: {result:?}");
    result?;
    if std::fs::read_to_string(scratch.0.join("allowed/new.txt"))? != "allowed content\n" {
        return Err(io::Error::other("inside control wrote the wrong content").into());
    }
    Ok(())
}

#[test]
fn dangling_link_chains_resolve_before_confinement() -> TestResult {
    let scratch = Scratch::new()?;
    let allowed = scratch.0.join("allowed");
    std::os::unix::fs::symlink("second", allowed.join("first"))?;
    std::os::unix::fs::symlink("nested/new.txt", allowed.join("second"))?;
    futures::executor::block_on(
        scratch
            .tool()
            .call(&mut ToolContext::new(), WriteArgs::new("first", "inside")),
    )?;
    if std::fs::read_to_string(allowed.join("nested/new.txt"))? != "inside"
        || !std::fs::symlink_metadata(allowed.join("first"))?.is_symlink()
    {
        return Err(io::Error::other("in-base dangling chain was not preserved").into());
    }
    std::os::unix::fs::symlink(scratch.0.join("outside/missing-dir"), allowed.join("dir"))?;
    let result = futures::executor::block_on(scratch.tool().call(
        &mut ToolContext::new(),
        WriteArgs::new("dir/new.txt", "outside"),
    ));
    if !matches!(result, Err(ref e) if e.kind() == ToolErrorKind::PermissionDenied)
        || scratch.0.join("outside/missing-dir").exists()
    {
        return Err(io::Error::other("dangling parent directory escaped confinement").into());
    }
    Ok(())
}
