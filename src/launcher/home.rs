//! The locks launchers sharing one `RIG_HOME` take: the build lock, and
//! each session's launcher lock.

use std::fs::{self, File, TryLockError};
use std::io::ErrorKind;
use std::path::Path;

use rig::harness_protocol::{Home, SessionId};

use super::Result;

/// Takes the lock that marks the launcher of `session` as running, unless
/// another launcher holds it. It is released when the returned file is
/// dropped or the launcher dies.
pub fn hold_session(home: &Home, session: &SessionId) -> Result<Option<File>> {
    let session = home.session(session);
    fs::create_dir_all(session.path())?;
    let file = lock_file(&session.launcher_lock(), true)?;
    match file.try_lock() {
        Ok(()) => Ok(Some(file)),
        Err(TryLockError::WouldBlock) => Ok(None),
        Err(TryLockError::Error(failure)) => Err(failure.into()),
    }
}

/// Removes the staged and trial builds of launchers that are gone, such as
/// one killed with its terminal. Call it holding [`lock`].
pub fn sweep(home: &Home) -> Result<()> {
    let Ok(entries) = fs::read_dir(home.bin()) else {
        return Ok(());
    };
    for entry in entries {
        let entry = entry?;
        let Some(session) = entry.file_name().to_str().and_then(Home::owner_of) else {
            continue;
        };
        let gone = match lock_file(&home.session(&session).launcher_lock(), false) {
            Ok(lock) => lock.try_lock().is_ok(),
            Err(failure) => failure.kind() == ErrorKind::NotFound,
        };
        if gone {
            fs::remove_file(entry.path())?;
        }
    }
    Ok(())
}

/// Waits for, then holds, the lock on generating, building and staging the
/// agent, which every launcher on the root shares. It is released when the
/// returned file is dropped.
pub fn lock(home: &Home) -> Result<File> {
    fs::create_dir_all(home.bin())?;
    let file = lock_file(&home.build_lock(), true)?;
    match file.try_lock() {
        Ok(()) => {}
        Err(TryLockError::WouldBlock) => {
            eprintln!(
                "Waiting for another rig on {} to finish building…",
                home.root().display()
            );
            file.lock()?;
        }
        Err(TryLockError::Error(failure)) => return Err(failure.into()),
    }
    Ok(file)
}

/// Opens a lock file for writing, creating it when `create`.
fn lock_file(path: &Path, create: bool) -> std::io::Result<File> {
    File::options()
        .create(create)
        .write(true)
        .truncate(false)
        .open(path)
}
