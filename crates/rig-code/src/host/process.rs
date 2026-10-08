//! Child processes the agent starts and must be able to stop with
//! everything they started: shell commands and the `/reload` build. None of
//! them can read the terminal or write to it.

use std::io;
use std::process::{Child, ChildStderr, ChildStdout, Command, ExitStatus, Stdio};

/// A child process in its own process group, with no input and piped
/// output. Dropping it kills the whole group unless [`Group::stop`] already
/// did.
pub(crate) struct Group {
    /// The leader, until it is reaped.
    child: Option<Child>,
    /// The process group id, until the group is killed.
    #[cfg(unix)]
    group: Option<i32>,
}

impl Group {
    /// Starts `command` with null stdin, piped stdout and stderr, as the
    /// leader of a new process group.
    pub(crate) fn spawn(command: &mut Command) -> io::Result<Self> {
        command
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        #[cfg(unix)]
        std::os::unix::process::CommandExt::process_group(command, 0);
        let child = command.spawn()?;
        Ok(Self {
            #[cfg(unix)]
            group: i32::try_from(child.id()).ok(),
            child: Some(child),
        })
    }

    /// Takes the child's output pipes, to be read elsewhere.
    pub(crate) fn take_output(&mut self) -> (Option<ChildStdout>, Option<ChildStderr>) {
        match &mut self.child {
            Some(child) => (child.stdout.take(), child.stderr.take()),
            None => (None, None),
        }
    }

    /// The leader's exit status once it has exited, reaping it; asking
    /// again is an error. Processes it left in the background keep running
    /// until [`Group::stop`].
    pub(crate) fn try_wait(&mut self) -> io::Result<Option<ExitStatus>> {
        let child = self
            .child
            .as_mut()
            .ok_or_else(|| io::Error::other("the process was already reaped"))?;
        let status = child.try_wait()?;
        if status.is_some() {
            self.child = None;
        }
        Ok(status)
    }

    /// Kills every process left in the group, so pipes they inherited
    /// close, and reaps the leader if it is still running. Call it right
    /// after the leader exits: once the group is empty its id can be reused.
    pub(crate) fn stop(&mut self) {
        #[cfg(unix)]
        if let Some(group) = self.group.take() {
            // SAFETY: `kill` takes no pointers. The group id is the leader's
            // pid; the group still has members or was emptied only just now.
            unsafe {
                libc::kill(-group, libc::SIGKILL);
            }
        }
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

impl Drop for Group {
    fn drop(&mut self) {
        self.stop();
    }
}
