//! The locks launchers sharing one `RIG_HOME` take: the build lock, each
//! session's launcher lock, and each launcher's run directory.

use std::fs::{self, File, TryLockError};
use std::time::{Duration, Instant};

use rig_harness_protocol::{Home, RunDir, SessionDir, is_free, lock_file};

use super::Result;
use super::build::BuildOutput;

/// This launcher's [`RunDir`], held locked while it lives. Dropping it
/// kills the process groups an agent left listed there and removes the
/// directory, unless it keeps the log of a run that failed.
pub struct Run {
    dir: RunDir,
    _lock: File,
}

impl Run {
    /// The directory.
    pub fn dir(&self) -> &RunDir {
        &self.dir
    }
}

impl Drop for Run {
    fn drop(&mut self) {
        // Jobs a reload kept for a restart that did not come.
        self.dir.kill_left_groups().ok();
        if !self.dir.log().exists() {
            fs::remove_dir_all(self.dir.path()).ok();
        }
    }
}

/// Creates and holds this launcher's run directory. Call it holding
/// [`lock`], so a [`sweep`] cannot take it before it is locked.
pub fn hold_run(home: &Home) -> Result<Run> {
    let dir = home.run(std::process::id());
    fs::create_dir_all(dir.path())?;
    let lock = lock_file(&dir.lock(), true)?;
    lock.try_lock()?;
    Ok(Run { dir, _lock: lock })
}

/// Removes what launchers that are gone, such as one killed with its
/// terminal, left: their staged and trial builds, their run directories
/// but the logs of failed runs, and the directories of their sessions that
/// saved nothing. Call it holding [`lock`].
pub fn sweep(home: &Home) -> Result<()> {
    if let Ok(entries) = fs::read_dir(home.bin()) {
        for entry in entries {
            let entry = entry?;
            let Some(session) = entry.file_name().to_str().and_then(Home::owner_of) else {
                continue;
            };
            if is_free(&home.session(&session).launcher_lock()) {
                fs::remove_file(entry.path())?;
            }
        }
    }
    if let Ok(entries) = fs::read_dir(home.runs()) {
        for entry in entries {
            let Some(launcher) = entry?.file_name().to_str().and_then(|id| id.parse().ok()) else {
                continue;
            };
            let run = home.run(launcher);
            if is_free(&run.lock()) && !run.log().exists() {
                fs::remove_dir_all(run.path())?;
            }
        }
    }
    if let Ok(entries) = fs::read_dir(home.sessions()) {
        for entry in entries {
            let session = SessionDir::at(entry?.path());
            // Only a launcher's own session: a fork or an import being
            // written has no launcher lock, and a session being deleted
            // is finished by an agent, which can reach its store.
            let lock = session.launcher_lock();
            if lock.exists() && is_free(&lock) && !session.is_saved() && !session.is_deleting() {
                fs::remove_dir_all(session.path())?;
            }
        }
    }
    Ok(())
}

/// How long a headless start waits for the build lock before it says so:
/// another rig's build with nothing to do holds it for a fraction of a
/// second, so runs started together stay quiet, while a real build still
/// says why the start waits.
const QUIET_WAIT: Duration = Duration::from_secs(3);

/// How often a quiet wait tries the lock.
const LOCK_POLL: Duration = Duration::from_millis(50);

/// Waits for, then holds, the lock on generating, building and staging the
/// agent, which every launcher on the root shares. It is released when the
/// returned file is dropped. A wait says so on stderr, at once with
/// [`BuildOutput::Live`], after [`QUIET_WAIT`] with
/// [`BuildOutput::LogOnly`].
pub fn hold_build_lock(home: &Home, output: BuildOutput) -> Result<File> {
    fs::create_dir_all(home.bin())?;
    let file = lock_file(&home.build_lock(), true)?;
    let quiet = match output {
        BuildOutput::Live => Duration::ZERO,
        BuildOutput::LogOnly => QUIET_WAIT,
    };
    take(&file, quiet, || {
        eprintln!(
            "Waiting for another rig on {} to finish building…",
            home.root().display()
        );
    })?;
    Ok(file)
}

/// Takes `lock`, waiting as long as it takes; once the wait outlasted
/// `quiet`, calls `say`.
fn take(lock: &File, quiet: Duration, say: impl FnOnce()) -> Result<()> {
    let deadline = Instant::now() + quiet;
    loop {
        match lock.try_lock() {
            Ok(()) => return Ok(()),
            Err(TryLockError::WouldBlock) if Instant::now() < deadline => {
                std::thread::sleep(LOCK_POLL);
            }
            Err(TryLockError::WouldBlock) => {
                say();
                lock.lock()?;
                return Ok(());
            }
            Err(TryLockError::Error(failure)) => return Err(failure.into()),
        }
    }
}

#[cfg(test)]
mod tests;
