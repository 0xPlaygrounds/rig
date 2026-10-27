#![cfg(feature = "fs")]

use rig_memory::{DirFiles, DirStore, Files};
use std::io;
use std::sync::{Arc, Barrier};

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn concurrent_writes(files: Arc<dyn Files>) -> TestResult {
    const WRITERS: usize = 8;
    let payloads: Vec<Vec<u8>> = (0..WRITERS)
        .map(|i| vec![b'A' + i as u8; (i + 1) * 1024 * 1024])
        .collect();
    let mut errors = 0;
    let mut corrupt = 0;
    for round in 0..10 {
        let barrier = Arc::new(Barrier::new(WRITERS));
        let handles: Vec<_> = payloads
            .iter()
            .map(|payload| {
                let files = files.clone();
                let barrier = barrier.clone();
                let payload = payload.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    files.write_file("state.bin", &payload)
                })
            })
            .collect();
        for handle in handles {
            match handle
                .join()
                .map_err(|_| io::Error::other("writer panicked"))?
            {
                Ok(()) => {}
                Err(error) => {
                    errors += 1;
                    println!("round {round}: write failed: {error}");
                }
            }
        }
        let actual = files.read_file("state.bin")?;
        if !payloads.contains(&actual) {
            corrupt += 1;
            let mut counts = [0usize; WRITERS];
            for byte in &actual {
                if let Some(count) = byte
                    .checked_sub(b'A')
                    .and_then(|i| counts.get_mut(i as usize))
                {
                    *count += 1;
                }
            }
            println!(
                "round {round}: CORRUPT final file: {} bytes, byte counts A-H: {counts:?}",
                actual.len()
            );
        }
    }
    println!("write failures: {errors}; corrupt final files: {corrupt}");
    if errors > 0 || corrupt > 0 {
        return Err(io::Error::other("concurrent writes violated atomic replacement").into());
    }
    Ok(())
}

fn run_test(locked: bool) -> TestResult {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_nanos();
    let dir =
        std::env::temp_dir().join(format!("rig-dirfiles-audit-{}-{stamp}", std::process::id()));
    std::fs::create_dir(&dir)?;
    let files: Arc<dyn Files> = if locked {
        Arc::new(DirStore::open(&dir)?)
    } else {
        Arc::new(DirFiles::new(&dir))
    };
    let result = concurrent_writes(files);
    std::fs::remove_dir_all(dir)?;
    result
}

#[test]
fn dirfiles_concurrent_replacements_are_atomic() -> TestResult {
    run_test(false)
}

#[test]
fn dirstore_concurrent_replacements_control() -> TestResult {
    run_test(true)
}

#[test]
fn process_writer() -> TestResult {
    let Some(dir) = std::env::var_os("RIG_TEST_DIRFILES_DIR") else {
        return Ok(());
    };
    let byte: u8 = std::env::var("RIG_TEST_DIRFILES_BYTE")?.parse()?;
    DirFiles::new(dir).write_file("state.bin", &vec![byte; 1024 * 1024])?;
    Ok(())
}

#[test]
fn independent_processes_replace_complete_files() -> TestResult {
    let dir = std::env::temp_dir().join(format!("rig-dirfiles-processes-{}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    let mut children = Vec::new();
    for byte in b'A'..=b'H' {
        children.push(
            std::process::Command::new(std::env::current_exe()?)
                .args(["--exact", "process_writer"])
                .env("RIG_TEST_DIRFILES_DIR", &dir)
                .env("RIG_TEST_DIRFILES_BYTE", byte.to_string())
                .stdout(std::process::Stdio::null())
                .spawn()?,
        );
    }
    let mut success = true;
    for mut child in children {
        success &= child.wait()?.success();
    }
    let files = DirFiles::new(&dir);
    let actual = files.read_file("state.bin")?;
    let whole = (b'A'..=b'H').any(|b| actual == vec![b; 1024 * 1024]);
    let entries = std::fs::read_dir(&dir)?.count();
    std::fs::remove_dir_all(&dir)?;
    if !success || !whole || entries != 1 {
        return Err(io::Error::other(
            "independent processes failed atomic replacement or leaked a temporary file",
        )
        .into());
    }
    Ok(())
}

#[test]
fn failed_rename_removes_its_temporary_file() -> TestResult {
    let dir = std::env::temp_dir().join(format!("rig-dirfiles-cleanup-{}", std::process::id()));
    std::fs::create_dir_all(dir.join("target"))?;
    let result = DirFiles::new(&dir).write_file("target", b"cannot replace a directory");
    let entries = std::fs::read_dir(&dir)?.count();
    std::fs::remove_dir_all(&dir)?;
    if result.is_ok() || entries != 1 {
        return Err(io::Error::other("failed rename leaked a temporary file").into());
    }
    Ok(())
}
