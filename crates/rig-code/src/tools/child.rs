//! Child processes whose output goes to a file, never to the terminal, and
//! that are waited for by polling so no pool thread blocks on them.

use std::fs::File;
use std::path::Path;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};

/// How often a running child is checked.
const POLL: Duration = Duration::from_millis(50);

/// A running child process with stdout and stderr redirected to one file.
/// Dropping it kills the process, so cancelling the task that owns it
/// stops the process too.
pub(crate) struct LoggedChild(Child);

impl LoggedChild {
    /// Starts `command` with no stdin and its output written to `log`.
    pub(crate) fn spawn(mut command: Command, log: &Path) -> std::io::Result<Self> {
        if let Some(parent) = log.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = File::create(log)?;
        command
            .stdin(Stdio::null())
            .stdout(file.try_clone()?)
            .stderr(file);
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
        if let Ok(None) = self.0.try_wait() {
            // The process may exit between the check and the kill.
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
}
