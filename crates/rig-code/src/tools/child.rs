//! Child processes whose output goes to a file, never to the terminal, and
//! that are waited for by polling so no pool thread blocks on them. On Unix
//! each child leads its own session, so it has no controlling terminal to
//! draw on or read from, and stopping it stops everything it started.

use std::fs::File;
use std::path::Path;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};

/// How often a running child is checked.
const POLL: Duration = Duration::from_millis(50);

/// A running child process with stdout and stderr redirected to one file.
/// Dropping it kills the process and, on Unix, every process it started,
/// so cancelling the task that owns it stops them too.
pub(crate) struct LoggedChild(Child);

impl LoggedChild {
    /// Starts `command` with no stdin and its output written to `log`. The
    /// launcher's variables for this agent are not passed on, so an agent
    /// the command starts is not mistaken for this one.
    pub(crate) fn spawn(mut command: Command, log: &Path) -> std::io::Result<Self> {
        if let Some(parent) = log.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = File::create(log)?;
        command
            .env_remove("RIG_DATA_DIR")
            .env_remove("RIG_READY_FILE")
            .env_remove("RIG_NOTICE")
            .env_remove("RIG_LAUNCHER")
            .stdin(Stdio::null())
            .stdout(file.try_clone()?)
            .stderr(file);
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            // SAFETY: `setsid` is async-signal-safe and touches no memory of
            // the parent.
            unsafe {
                command.pre_exec(|| {
                    if libc::setsid() == -1 {
                        return Err(std::io::Error::last_os_error());
                    }
                    Ok(())
                });
            }
        }
        command.spawn().map(Self)
    }

    /// Waits for the process to exit, or `None` once `timeout` has passed.
    pub(crate) async fn wait(&mut self, timeout: Duration) -> std::io::Result<Option<ExitStatus>> {
        let started = Instant::now();
        loop {
            if let Some(status) = self.0.try_wait()? {
                return Ok(Some(status));
            }
            if started.elapsed() >= timeout {
                return Ok(None);
            }
            futures_timer::Delay::new(POLL).await;
        }
    }
}

impl Drop for LoggedChild {
    fn drop(&mut self) {
        #[cfg(unix)]
        if let Ok(group) = libc::pid_t::try_from(self.0.id()) {
            // The child's session is its process group; this also stops
            // what it left running in the background. Nothing to do when
            // the group is already gone.
            // SAFETY: `kill` takes plain integers.
            unsafe {
                libc::kill(-group, libc::SIGKILL);
            }
        }
        if let Ok(None) = self.0.try_wait() {
            // The process may exit between the check and the kill.
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
}
