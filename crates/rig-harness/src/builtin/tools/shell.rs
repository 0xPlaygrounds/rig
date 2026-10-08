//! The `shell` tool.

use std::collections::VecDeque;
use std::io::Read;
use std::process::{Command, ExitStatus, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, sync_channel};
use std::thread;
use std::time::{Duration, Instant};

use rig_core::tool::{PortableTool, ToolExecutionError};
use serde::Deserialize;
use serde_json::json;

use super::{MAX_BYTES, MAX_LINES};
use crate::core::blocking::blocking;
use crate::host::process::{detach, kill_group};

const DEFAULT_TIMEOUT: u64 = 120;
const MAX_TIMEOUT: u64 = 600;
/// How long output pipes may stay open after the command exits before the
/// rest of its process group is killed, and how long reading goes on after
/// that kill.
const PIPE_GRACE: Duration = Duration::from_millis(100);
/// How often the command's state is checked while no output arrives.
const POLL: Duration = Duration::from_millis(20);
/// Output past which the command is killed. Only the last `MAX_BYTES` are
/// kept in memory while it runs.
const OUTPUT_LIMIT: u64 = 10 * 1024 * 1024;

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
             at most {MAX_TIMEOUT}), or once it printed more than {} MB.",
            MAX_BYTES / 1024,
            OUTPUT_LIMIT / 1024 / 1024
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
        let stop = StopOnDrop(Arc::default());
        let running = Arc::clone(&stop.0);
        blocking(move || run(args, &running.stopped, &running.leader)).await
    }
}

/// Stops the running command when the call is dropped, which is how an
/// interrupted turn, or quitting, cancels it. On unix the process group is
/// killed right away, because the app may exit before the command's thread
/// sees the flag.
struct StopOnDrop(Arc<Stop>);

/// What the call and the command's thread share.
#[derive(Default)]
struct Stop {
    /// Set when the call is dropped.
    stopped: AtomicBool,
    /// The running command's pid while it is not reaped, else 0.
    leader: AtomicU32,
}

impl Drop for StopOnDrop {
    fn drop(&mut self) {
        self.0.stopped.store(true, Ordering::Relaxed);
        let leader = self.0.leader.swap(0, Ordering::SeqCst);
        #[cfg(unix)]
        if leader != 0 {
            crate::host::process::kill_group_of(leader);
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
    // A command, such as a nested agent run while working on rig-harness
    // itself, must not act as this agent.
    for name in rig::harness_protocol::env::AGENT_ONLY {
        command.env_remove(name);
    }
    detach(&mut command);
    let spawned = command.spawn();
    // The parent's copies of the pipe's write end must close, or reading
    // never ends.
    drop(command);
    let mut child =
        spawned.map_err(|error| ToolExecutionError::other(format!("could not run sh: {error}")))?;
    leader.store(child.id(), Ordering::SeqCst);
    let chunks = drain(reader);
    let mut tail = Tail::default();
    let deadline = Instant::now() + timeout;
    let end = loop {
        let exited = child.try_wait();
        if !matches!(exited, Ok(None)) {
            // Reaped, or about to be: its pid may be reused.
            leader.store(0, Ordering::SeqCst);
        }
        match exited {
            Ok(Some(status)) => break End::Exited(status),
            Ok(None) if stopped.load(Ordering::Relaxed) => break End::Stopped,
            Ok(None) if Instant::now() >= deadline => break End::TimedOut,
            Ok(None) if tail.written > OUTPUT_LIMIT => break End::TooLong,
            Ok(None) => match chunks.recv_timeout(POLL) {
                Ok(chunk) => tail.push(&chunk),
                Err(RecvTimeoutError::Timeout) => {}
                // The command closed its output but still runs.
                Err(RecvTimeoutError::Disconnected) => thread::sleep(POLL),
            },
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
    // background may still hold the output pipe open; then the rest of its
    // group is killed. The leader was reaped by `try_wait`, so the group id
    // stays its own only while a member lives; the kill follows the reap by
    // at most `PIPE_GRACE`, and an empty group just makes it a no-op. A
    // process that left the group, such as one started with `setsid`,
    // survives the kill and may hold the pipe for good, so reading stops a
    // moment later with what came.
    if !matches!(end, End::Exited(_)) || !tail.read_rest(&chunks) {
        kill_group(&mut child);
        tail.read_rest(&chunks);
    }
    child.wait().ok();
    let mut output = tail.text();
    match end {
        End::Exited(status) => match status.code() {
            Some(code) => output.push_str(&format!("[exit code {code}]")),
            None => output.push_str("[killed by a signal]"),
        },
        End::Stopped => output.push_str("[stopped]"),
        End::TimedOut => output.push_str(&format!(
            "[timed out after {}s and killed]",
            timeout.as_secs()
        )),
        End::TooLong => output.push_str(&format!(
            "[killed after printing more than {} MB]",
            OUTPUT_LIMIT / 1024 / 1024
        )),
    }
    Ok(output)
}

/// How the wait for the command ended.
enum End {
    /// It exited.
    Exited(ExitStatus),
    /// The call was dropped.
    Stopped,
    /// It ran past its timeout.
    TimedOut,
    /// It printed more than [`OUTPUT_LIMIT`].
    TooLong,
}

/// Reads the command's output on a thread of its own, until every writer
/// closed it or the receiver is gone, and sends it on in chunks.
///
/// When the call stops reading while a process that left the group still
/// holds the pipe, this thread stays blocked in `read` until that process
/// writes again or exits: one parked thread per such call. Its next write
/// then finds the receiver gone, the thread drops the read end, and the
/// process gets `EPIPE` or `SIGPIPE`, as after any closed pipe.
fn drain(mut pipe: impl Read + Send + 'static) -> Receiver<Vec<u8>> {
    let (sender, chunks) = sync_channel(16);
    thread::spawn(move || {
        let mut buffer = [0; 8192];
        loop {
            let read = match pipe.read(&mut buffer) {
                Ok(0) => break,
                Ok(read) => read,
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(_) => break,
            };
            let chunk = buffer.iter().take(read).copied().collect();
            if sender.send(chunk).is_err() {
                break;
            }
        }
    });
    chunks
}

/// The end of a command's output.
#[derive(Default)]
struct Tail {
    /// Every byte received.
    written: u64,
    /// The last bytes.
    bytes: VecDeque<u8>,
    /// Whether earlier bytes were dropped.
    cut: bool,
    /// Whole lines dropped before `bytes`.
    cut_lines: usize,
}

impl Tail {
    /// Appends `chunk`, keeping only the last `MAX_BYTES` bytes.
    fn push(&mut self, chunk: &[u8]) {
        self.written += chunk.len() as u64;
        self.bytes.extend(chunk);
        let excess = self.bytes.len().saturating_sub(MAX_BYTES);
        if excess > 0 {
            self.cut_lines += self.bytes.drain(..excess).filter(|&b| b == b'\n').count();
            self.cut = true;
        }
    }

    /// Receives output until the pipe's end, for at most [`PIPE_GRACE`].
    /// Returns whether the end came: every process holding the pipe exited
    /// or closed it.
    fn read_rest(&mut self, chunks: &Receiver<Vec<u8>>) -> bool {
        let deadline = Instant::now() + PIPE_GRACE;
        loop {
            match chunks.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
                Ok(chunk) => self.push(&chunk),
                Err(RecvTimeoutError::Disconnected) => return true,
                Err(RecvTimeoutError::Timeout) => return false,
            }
        }
    }

    /// The kept output, at most `MAX_LINES` lines, after a line saying how
    /// many earlier ones were cut.
    fn text(self) -> String {
        let (front, back) = self.bytes.as_slices();
        let output = String::from_utf8_lossy(&[front, back].concat()).into_owned();
        let mut cut = self.cut_lines;
        // After a cut, the first line kept is only the end of a line.
        let output = match output.split_once('\n') {
            Some((_, rest)) if self.cut => {
                cut += 1;
                rest
            }
            _ => output.as_str(),
        };
        let lines: Vec<&str> = output.lines().collect();
        let kept = lines.len().min(MAX_LINES);
        cut += lines.len() - kept;
        let mut text = String::new();
        if cut > 0 {
            text.push_str(&format!("[{cut} earlier lines cut]\n"));
        }
        for line in lines.iter().skip(lines.len() - kept) {
            text.push_str(line);
            text.push('\n');
        }
        text
    }
}

#[cfg(test)]
mod tests;
