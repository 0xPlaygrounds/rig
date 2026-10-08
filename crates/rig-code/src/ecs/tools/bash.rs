//! The `bash` tool: a shell command in the working directory, its output in
//! a file of the session directory so nothing reaches the terminal.

use std::{
    fs::File,
    path::PathBuf,
    process::{Child, Command, Stdio},
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant},
};

use futures_timer::Delay;
use serde::Deserialize;
use serde_json::json;

use rig_core::tool::{Tool, ToolContext, ToolExecutionError};

use super::{io_error, tail};
use crate::ecs::{agent::Workdir, paths};

/// Default time a command may run.
const DEFAULT_TIMEOUT_SECS: u64 = 120;
/// Most output bytes returned, from the end.
const OUTPUT_BYTES: usize = 50 * 1024;
/// How often a running command is checked.
const POLL: Duration = Duration::from_millis(50);

/// Runs `sh -c` with stdin closed.
pub struct Bash;

#[derive(Deserialize)]
pub struct BashArgs {
    command: String,
    timeout_secs: Option<u64>,
}

/// Kills the child when dropped, so a cancelled call or a timeout leaves no
/// process behind. On unix the shell leads its own session, so it has no
/// controlling terminal to draw on or read from, and its whole process group
/// is killed, so the commands it started go too.
struct KillOnDrop(Child);

impl Drop for KillOnDrop {
    fn drop(&mut self) {
        // Killing processes that already exited fails harmlessly.
        #[cfg(unix)]
        if let Ok(group) = libc::pid_t::try_from(self.0.id()) {
            // SAFETY: `kill` takes plain numbers; a negative pid names the
            // process group the shell leads.
            unsafe { libc::kill(-group, libc::SIGKILL) };
        }
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// Removes the output file when dropped, also when the call is cancelled.
struct OutputFile(PathBuf);

impl Drop for OutputFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

impl Tool for Bash {
    const NAME: &'static str = "bash";
    type Args = BashArgs;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        format!(
            "Run a shell command with `sh -c` in the working directory. Stdin is closed. \
             Returns the exit code and the last {} KB of stdout and stderr combined. \
             The command is killed after timeout_secs (default {DEFAULT_TIMEOUT_SECS}).",
            OUTPUT_BYTES / 1024
        )
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The shell command"},
                "timeout_secs": {"type": "integer", "description": "Seconds before the command is killed"}
            },
            "required": ["command"]
        })
    }

    async fn call(
        &self,
        context: &mut ToolContext,
        args: BashArgs,
    ) -> Result<String, ToolExecutionError> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let output_file = OutputFile(paths::session_dir().join(format!(
            "bash-{}-{}.out",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        )));
        let output_path = &output_file.0;
        let output = File::create(output_path).map_err(|error| io_error(output_path, error))?;
        let errors = output
            .try_clone()
            .map_err(|error| io_error(output_path, error))?;
        let mut command = Command::new("sh");
        command
            .arg("-c")
            .arg(&args.command)
            .stdin(Stdio::null())
            .stdout(output)
            .stderr(errors);
        #[cfg(unix)]
        // SAFETY: `setsid` is async-signal-safe and touches no memory.
        unsafe {
            std::os::unix::process::CommandExt::pre_exec(&mut command, || {
                if libc::setsid() < 0 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
        if let Some(workdir) = context.scope::<Workdir>() {
            command.current_dir(&workdir.0);
        }
        let mut child =
            KillOnDrop(command.spawn().map_err(|error| {
                ToolExecutionError::other(format!("cannot start `sh`: {error}"))
            })?);
        let timeout = Duration::from_secs(args.timeout_secs.unwrap_or(DEFAULT_TIMEOUT_SECS));
        // A timeout too large to represent never expires.
        let deadline = Instant::now().checked_add(timeout);
        let status = loop {
            match child.0.try_wait() {
                Ok(Some(status)) => break Some(status),
                Ok(None) if deadline.is_some_and(|deadline| Instant::now() >= deadline) => {
                    break None;
                }
                Ok(None) => Delay::new(POLL).await,
                Err(error) => return Err(ToolExecutionError::other(error.to_string())),
            }
        };
        drop(child);
        let text = std::fs::read(output_path)
            .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
            .map_err(|error| io_error(output_path, error))?;
        let text = tail(&text, OUTPUT_BYTES);
        match status {
            Some(status) => Ok(match status.code() {
                Some(code) => format!("exit code {code}\n{text}"),
                None => format!("killed by a signal\n{text}"),
            }),
            None => Err(ToolExecutionError::timeout(format!(
                "the command was killed after {} seconds\n{text}",
                timeout.as_secs()
            ))),
        }
    }
}
