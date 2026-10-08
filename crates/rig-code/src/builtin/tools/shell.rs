//! The `shell` tool.

use std::collections::VecDeque;
use std::io::Read;
use std::process::{Command, ExitStatus, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::thread::{self, JoinHandle};
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
/// rest of its process group is killed.
const PIPE_GRACE: Duration = Duration::from_millis(100);
/// Output past which the command is killed. Only the last `MAX_BYTES` are
/// kept in memory while it runs.
const OUTPUT_LIMIT: u64 = 10 * 1024 * 1024;
/// What the `rig` launcher tells its agent. A command, such as a nested
/// agent run while working on rig-code itself, must not act as this agent.
const LAUNCHER_VARS: [&str; 4] = [
    "RIG_SESSION",
    "RIG_READY_FILE",
    "RIG_NOTICE",
    "RIG_LAUNCHER",
];

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
    for name in LAUNCHER_VARS {
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
    let written = Arc::new(AtomicU64::new(0));
    let output = drain(reader, Arc::clone(&written));
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
            Ok(None) if written.load(Ordering::Relaxed) > OUTPUT_LIMIT => break End::TooLong,
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
    if !matches!(end, End::Exited(_)) || !pipe_closed(&output) {
        kill_group(&mut child);
    }
    child.wait().ok();
    let mut output = output.join().map(Tail::text).unwrap_or_default();
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

/// Whether the output pipe reached its end within a moment: every process
/// holding it exited or closed it.
fn pipe_closed(output: &JoinHandle<Tail>) -> bool {
    let deadline = Instant::now() + PIPE_GRACE;
    loop {
        if output.is_finished() || Instant::now() >= deadline {
            return output.is_finished();
        }
        thread::sleep(Duration::from_millis(5));
    }
}

/// Reads the command's output until every writer closed it, keeping only
/// the last `MAX_BYTES` bytes and counting every byte in `written`.
fn drain(mut pipe: impl Read + Send + 'static, written: Arc<AtomicU64>) -> JoinHandle<Tail> {
    thread::spawn(move || {
        let mut tail = Tail::default();
        let mut buffer = [0; 8192];
        loop {
            let read = match pipe.read(&mut buffer) {
                Ok(0) => break,
                Ok(read) => read,
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(_) => break,
            };
            written.fetch_add(read as u64, Ordering::Relaxed);
            tail.bytes.extend(buffer.iter().take(read));
            let excess = tail.bytes.len().saturating_sub(MAX_BYTES);
            if excess > 0 {
                tail.cut_lines += tail.bytes.drain(..excess).filter(|&b| b == b'\n').count();
                tail.cut = true;
            }
        }
        tail
    })
}

/// The end of a command's output.
#[derive(Default)]
struct Tail {
    /// The last bytes.
    bytes: VecDeque<u8>,
    /// Whether earlier bytes were dropped.
    cut: bool,
    /// Whole lines dropped before `bytes`.
    cut_lines: usize,
}

impl Tail {
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
