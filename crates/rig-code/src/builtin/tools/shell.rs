//! The `shell` tool.

use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use rig_core::tool::{PortableTool, ToolExecutionError};
use serde::Deserialize;
use serde_json::json;

use super::blocking::blocking;
use super::{MAX_BYTES, MAX_LINES};

const DEFAULT_TIMEOUT: u64 = 120;
const MAX_TIMEOUT: u64 = 600;

/// Runs a command with `sh -c` in its own process group.
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
             Returns the exit code and the combined output, keeping the last {MAX_LINES} lines \
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
        let stop = StopOnDrop(Arc::new(AtomicBool::new(false)));
        let stopped = Arc::clone(&stop.0);
        blocking(move || run(args, &stopped)).await
    }
}

/// Tells the running command to stop when the call is dropped, which is how
/// an interrupted turn cancels it.
struct StopOnDrop(Arc<AtomicBool>);

impl Drop for StopOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

fn run(args: ShellArgs, stopped: &AtomicBool) -> Result<String, ToolExecutionError> {
    let timeout = Duration::from_secs(
        args.timeout_secs
            .unwrap_or(DEFAULT_TIMEOUT)
            .clamp(1, MAX_TIMEOUT),
    );
    let mut command = Command::new("sh");
    command
        .arg("-c")
        .arg(&args.command)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(unix)]
    std::os::unix::process::CommandExt::process_group(&mut command, 0);
    let mut child = command
        .spawn()
        .map_err(|error| ToolExecutionError::other(format!("could not run sh: {error}")))?;
    let stdout = drain(child.stdout.take());
    let stderr = drain(child.stderr.take());
    let deadline = Instant::now() + timeout;
    let status = loop {
        match child.try_wait() {
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
    // Also ends anything the command left running in the background, which
    // would otherwise hold the output pipes open.
    kill_group(&mut child);
    child.wait().ok();
    let mut output = joined(stdout);
    output.push_str(&joined(stderr));
    let mut output = tail(&output);
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

fn drain(pipe: Option<impl std::io::Read + Send + 'static>) -> Option<JoinHandle<Vec<u8>>> {
    pipe.map(|mut pipe| {
        thread::spawn(move || {
            let mut bytes = Vec::new();
            pipe.read_to_end(&mut bytes).ok();
            bytes
        })
    })
}

fn joined(reader: Option<JoinHandle<Vec<u8>>>) -> String {
    reader
        .and_then(|reader| reader.join().ok())
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

/// Kills the process group `child` leads, created with `process_group(0)`;
/// elsewhere, the child alone.
#[cfg(unix)]
pub(crate) fn kill_group(child: &mut Child) {
    if let Ok(group) = i32::try_from(child.id()) {
        // SAFETY: `kill` takes plain integers; a negative pid names the
        // process group the child leads, created by `process_group(0)`.
        unsafe {
            libc::kill(-group, libc::SIGKILL);
        }
    }
}

#[cfg(not(unix))]
pub(crate) fn kill_group(child: &mut Child) {
    child.kill().ok();
}
