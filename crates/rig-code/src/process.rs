//! Child processes that never block a task pool thread: a reader thread
//! moves their joined stdout and stderr into a channel, and async code
//! polls that channel and the exit status between timer waits.

use std::{
    io::Read as _,
    process::{Child, Command, ExitStatus, Stdio},
    sync::mpsc::{self, Receiver, TryRecvError},
    time::Duration,
};

/// How long a waiting task sleeps between polls.
pub(crate) const POLL: Duration = Duration::from_millis(50);

/// A running child in its own process group, with stdout and stderr joined
/// into one pipe. Dropping it before the child exits kills the group, so
/// dropping the task that owns it stops the command.
pub(crate) struct Piped {
    child: Child,
    output: Receiver<Vec<u8>>,
    exited: bool,
}

impl Piped {
    /// Spawn `command` with no stdin.
    pub(crate) fn spawn(mut command: Command) -> std::io::Result<Self> {
        let (mut reader, writer) = std::io::pipe()?;
        command
            .stdin(Stdio::null())
            .stdout(writer.try_clone()?)
            .stderr(writer);
        #[cfg(unix)]
        std::os::unix::process::CommandExt::process_group(&mut command, 0);
        let child = command.spawn()?;
        // The command holds the parent's copies of the pipe's write end.
        drop(command);
        let (sender, output) = mpsc::channel();
        std::thread::spawn(move || {
            let mut buffer = [0; 8192];
            while let Ok(read) = reader.read(&mut buffer) {
                let Some(chunk) = buffer.get(..read).filter(|chunk| !chunk.is_empty()) else {
                    break;
                };
                if sender.send(chunk.to_vec()).is_err() {
                    break;
                }
            }
        });
        Ok(Self {
            child,
            output,
            exited: false,
        })
    }

    /// The output that arrived so far, without waiting, and whether the
    /// pipe is still open.
    pub(crate) fn output(&mut self) -> (Vec<u8>, bool) {
        let mut bytes = Vec::new();
        loop {
            match self.output.try_recv() {
                Ok(chunk) => bytes.extend(chunk),
                Err(TryRecvError::Empty) => return (bytes, true),
                Err(TryRecvError::Disconnected) => return (bytes, false),
            }
        }
    }

    /// The exit status, if the child exited.
    pub(crate) fn try_wait(&mut self) -> std::io::Result<Option<ExitStatus>> {
        let status = self.child.try_wait()?;
        self.exited |= status.is_some();
        Ok(status)
    }

    /// Kill the child and everything it started, then reap it.
    pub(crate) fn kill(&mut self) {
        #[cfg(unix)]
        {
            let _ = Command::new("kill")
                .args(["-KILL", "--", &format!("-{}", self.child.id())])
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
        self.exited = true;
    }
}

impl Drop for Piped {
    fn drop(&mut self) {
        if !self.exited {
            self.kill();
        }
    }
}

/// Wait `duration` without holding a pool thread.
pub(crate) async fn sleep(duration: Duration) {
    futures_timer::Delay::new(duration).await;
}
