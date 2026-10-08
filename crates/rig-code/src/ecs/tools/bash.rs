//! The `bash` tool: a shell command in the working directory, its output in
//! a file of the session directory so nothing reaches the terminal.

use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
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
/// Most output a command may write before it is killed.
const MAX_OUTPUT_FILE: u64 = 64 * 1024 * 1024;
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
        let end = loop {
            match child.0.try_wait() {
                Ok(Some(status)) => break End::Exited(status),
                Ok(None) if deadline.is_some_and(|deadline| Instant::now() >= deadline) => {
                    break End::TimedOut;
                }
                Ok(None)
                    if std::fs::metadata(output_path)
                        .is_ok_and(|metadata| metadata.len() > MAX_OUTPUT_FILE) =>
                {
                    break End::TooMuchOutput;
                }
                Ok(None) => Delay::new(POLL).await,
                Err(error) => return Err(ToolExecutionError::other(error.to_string())),
            }
        };
        drop(child);
        let text = read_tail(output_path).map_err(|error| io_error(output_path, error))?;
        let text = tail(&text, OUTPUT_BYTES);
        match end {
            End::Exited(status) => Ok(match status.code() {
                Some(code) => format!("exit code {code}\n{text}"),
                None => format!("killed by a signal\n{text}"),
            }),
            End::TimedOut => Err(ToolExecutionError::timeout(format!(
                "the command was killed after {} seconds\n{text}",
                timeout.as_secs()
            ))),
            End::TooMuchOutput => Err(ToolExecutionError::other(format!(
                "the command was killed after writing over {} MB of output\n{text}",
                MAX_OUTPUT_FILE / (1024 * 1024)
            ))),
        }
    }
}

/// Why a command stopped.
enum End {
    Exited(std::process::ExitStatus),
    TimedOut,
    TooMuchOutput,
}

/// The last [`OUTPUT_BYTES`] of the file at `path`, read without loading the
/// rest.
fn read_tail(path: &Path) -> std::io::Result<String> {
    let mut file = File::open(path)?;
    let length = file.metadata()?.len();
    let limit = u64::try_from(OUTPUT_BYTES).unwrap_or(u64::MAX);
    file.seek(SeekFrom::Start(length.saturating_sub(limit)))?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}
