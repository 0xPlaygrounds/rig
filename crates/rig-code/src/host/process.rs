//! Child processes the agent starts and must be able to stop with
//! everything they started: shell commands and the `/reload` build. None of
//! them can read the terminal or write to it.

use std::io;
use std::process::{Child, ChildStderr, ChildStdout, Command, ExitStatus, Stdio};

/// A child process in its own process group, with no input and piped
/// output. Dropping it before it was reaped kills the whole group.
pub(crate) struct Group(Option<Child>);

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
        command.spawn().map(|child| Self(Some(child)))
    }

    /// Takes the child's output pipes, to be read elsewhere.
    pub(crate) fn take_output(&mut self) -> (Option<ChildStdout>, Option<ChildStderr>) {
        match &mut self.0 {
            Some(child) => (child.stdout.take(), child.stderr.take()),
            None => (None, None),
        }
    }

    /// The child's exit status once it has exited. A child that exited is
    /// reaped, and is no longer killed on drop; asking again is an error.
    pub(crate) fn try_wait(&mut self) -> io::Result<Option<ExitStatus>> {
        let child = self
            .0
            .as_mut()
            .ok_or_else(|| io::Error::other("the process was already reaped"))?;
        let status = child.try_wait()?;
        if status.is_some() {
            self.0 = None;
        }
        Ok(status)
    }
}

impl Drop for Group {
    fn drop(&mut self) {
        let Some(child) = &mut self.0 else {
            return;
        };
        #[cfg(unix)]
        if let Ok(group) = i32::try_from(child.id()) {
            // SAFETY: `kill` takes no pointers. The group id is the child's
            // pid (it was started as a group leader) and the child is not
            // reaped yet, so the id cannot have been reused.
            unsafe {
                libc::kill(-group, libc::SIGKILL);
            }
        }
        let _ = child.kill();
        let _ = child.wait();
    }
}
