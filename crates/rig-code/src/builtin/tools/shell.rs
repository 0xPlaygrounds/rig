//! The `shell` tool.

use std::process::{Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use rig_core::tool::{PortableTool, ToolExecutionError};
use serde::Deserialize;
use serde_json::json;

use super::{MAX_BYTES, MAX_LINES};
use crate::core::blocking::blocking;
use crate::process::{detach, kill_group};

const DEFAULT_TIMEOUT: u64 = 120;
const MAX_TIMEOUT: u64 = 600;
/// How long output pipes may stay open after the command exits before the
/// rest of its process group is killed.
const PIPE_GRACE: Duration = Duration::from_millis(100);

/// Runs a command with `sh -c` in its own session and process group.
pub struct Shell;

/// Arguments of [`Shell`].
#[derive(Deserialize)]
pub struct ShellArgs {
    command: String,
    timeout_secs: Option<u64>,
}

impl PortableTool for Shell {
    const NAME: &'static str = "shell";
    type Args = ShellArgs;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        format!(
            "Run a shell command with `sh -c` in the current directory. stdin is empty. \
             Returns the exit code and stdout and stderr interleaved, keeping the last {MAX_LINES} lines \
             or {} KB. The command is killed after timeout_secs (default {DEFAULT_TIMEOUT}, \
             at most {MAX_TIMEOUT}).",
            MAX_BYTES / 1024
        )
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The command line."},
                "timeout_secs": {"type": "integer", "description": "Seconds before the command is killed."}
            },
            "required": ["command"]
        })
    }

    async fn call(&self, args: ShellArgs) -> Result<String, ToolExecutionError> {
        let stop = StopOnDrop {
            stopped: Arc::new(AtomicBool::new(false)),
            leader: Arc::new(AtomicU32::new(0)),
        };
        let stopped = Arc::clone(&stop.stopped);
        let leader = Arc::clone(&stop.leader);
        blocking(move || run(args, &stopped, &leader)).await
    }
}

/// Stops the running command when the call is dropped, which is how an
/// interrupted turn, or quitting, cancels it. On unix the process group is
/// killed right away, because the app may exit before the command's thread
/// sees the flag.
struct StopOnDrop {
    stopped: Arc<AtomicBool>,
    /// The running command's pid while it is not reaped, else 0.
    leader: Arc<AtomicU32>,
}

impl Drop for StopOnDrop {
    fn drop(&mut self) {
        self.stopped.store(true, Ordering::Relaxed);
        let leader = self.leader.swap(0, Ordering::SeqCst);
        #[cfg(unix)]
        if leader != 0 {
            crate::process::kill_group_of(leader);
        }
        #[cfg(not(unix))]
        let _ = leader;
    }
}

fn run(
    args: ShellArgs,
    stopped: &AtomicBool,
    leader: &AtomicU32,
) -> Result<String, ToolExecutionError> {
    let timeout = Duration::from_secs(
        args.timeout_secs
            .unwrap_or(DEFAULT_TIMEOUT)
            .clamp(1, MAX_TIMEOUT),
    );
    // stdout and stderr share one pipe, so the output keeps their order.
    let (reader, writer) = std::io::pipe()
        .map_err(|error| ToolExecutionError::other(format!("could not open a pipe: {error}")))?;
    let error_writer = writer
        .try_clone()
        .map_err(|error| ToolExecutionError::other(format!("could not open a pipe: {error}")))?;
    let mut command = Command::new("sh");
    command
        .arg("-c")
        .arg(&args.command)
        .stdin(Stdio::null())
        .stdout(writer)
        .stderr(error_writer);
    detach(&mut command);
    let spawned = command.spawn();
    // The parent's copies of the pipe's write end must close, or reading
    // never ends.
    drop(command);
    let mut child =
        spawned.map_err(|error| ToolExecutionError::other(format!("could not run sh: {error}")))?;
    leader.store(child.id(), Ordering::SeqCst);
    let output = drain(reader);
    let deadline = Instant::now() + timeout;
    let status = loop {
        let exited = child.try_wait();
        if !matches!(exited, Ok(None)) {
            // Reaped, or about to be: its pid may be reused.
            leader.store(0, Ordering::SeqCst);
        }
        match exited {
            Ok(Some(status)) => break Some(status),
            Ok(None) if stopped.load(Ordering::Relaxed) || Instant::now() >= deadline => {
                break None;
            }
            Ok(None) => thread::sleep(Duration::from_millis(20)),
            Err(error) => {
                kill_group(&mut child);
                return Err(ToolExecutionError::other(format!(
                    "waiting for the command failed: {error}"
                )));
            }
        }
    };
    leader.store(0, Ordering::SeqCst);
    // After a normal exit, something the command left running in the
    // background may still hold the output pipes open. Only then is the
    // group killed: a member is alive, so the group id is still its own and
    // cannot have been reused.
    if status.is_none() || !pipe_closed(&output) {
        kill_group(&mut child);
    }
    child.wait().ok();
    let mut output = tail(&joined(output));
    match status {
        Some(status) => match status.code() {
            Some(code) => output.push_str(&format!("[exit code {code}]")),
            None => output.push_str("[killed by a signal]"),
        },
        None if stopped.load(Ordering::Relaxed) => output.push_str("[stopped]"),
        None => output.push_str(&format!(
            "[timed out after {}s and killed]",
            timeout.as_secs()
        )),
    }
    Ok(output)
}

/// Whether the output pipe reached its end within a moment: every process
/// holding it exited or closed it.
fn pipe_closed(output: &JoinHandle<Vec<u8>>) -> bool {
    let deadline = Instant::now() + PIPE_GRACE;
    loop {
        if output.is_finished() || Instant::now() >= deadline {
            return output.is_finished();
        }
        thread::sleep(Duration::from_millis(5));
    }
}

fn drain(mut pipe: impl std::io::Read + Send + 'static) -> JoinHandle<Vec<u8>> {
    thread::spawn(move || {
        let mut bytes = Vec::new();
        pipe.read_to_end(&mut bytes).ok();
        bytes
    })
}

fn joined(reader: JoinHandle<Vec<u8>>) -> String {
    reader
        .join()
        .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
        .unwrap_or_default()
}

/// The end of `output`, at most `MAX_LINES` lines and `MAX_BYTES` bytes.
fn tail(output: &str) -> String {
    let lines: Vec<&str> = output.lines().collect();
    let mut kept = Vec::new();
    let mut bytes = 0;
    for line in lines.iter().rev().take(MAX_LINES) {
        bytes += line.len() + 1;
        if bytes > MAX_BYTES {
            break;
        }
        kept.push(*line);
    }
    let mut text = String::new();
    if kept.len() < lines.len() {
        text.push_str(&format!(
            "[{} earlier lines cut]\n",
            lines.len() - kept.len()
        ));
    }
    for line in kept.iter().rev() {
        text.push_str(line);
        text.push('\n');
    }
    text
}
