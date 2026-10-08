use std::{
    io::Read as _,
    process::{Child, Command, Stdio},
    sync::mpsc::{self, RecvTimeoutError},
    time::{Duration, Instant},
};

use rig_core::tool::{PortableTool, ToolExecutionError};
use serde::Deserialize;

use super::fail;

/// The most output bytes returned, keeping the tail.
const MAX_OUTPUT: usize = 30 * 1024;
/// The timeout when the call names none.
const DEFAULT_TIMEOUT_SECS: u64 = 120;
/// How long output is still collected after the command exits, for
/// background processes that keep the pipe open.
const DRAIN_GRACE: Duration = Duration::from_millis(500);
const POLL: Duration = Duration::from_millis(50);

/// Runs a shell command and returns its combined output and exit status.
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
            "Run a shell command in the working directory, with no stdin. Returns stdout \
             and stderr interleaved (the last {} KB) and the exit status. The command is \
             killed after timeout_secs (default {DEFAULT_TIMEOUT_SECS}).",
            MAX_OUTPUT / 1024
        )
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The command line."},
                "timeout_secs": {"type": "integer", "description": "Seconds before the command is killed."}
            },
            "required": ["command"]
        })
    }

    async fn call(&self, args: ShellArgs) -> Result<String, ToolExecutionError> {
        let timeout = Duration::from_secs(
            args.timeout_secs
                .unwrap_or(DEFAULT_TIMEOUT_SECS)
                .clamp(1, 3600),
        );
        let spawn_failed = |error: std::io::Error| fail(format!("cannot run the command: {error}"));
        let (mut reader, writer) = std::io::pipe().map_err(spawn_failed)?;
        let mut command = shell_command(&args.command);
        command
            .stdin(Stdio::null())
            .stdout(writer.try_clone().map_err(spawn_failed)?)
            .stderr(writer);
        let mut child = command.spawn().map_err(spawn_failed)?;
        // The command holds the parent's copies of the pipe's write end.
        drop(command);

        let (sender, chunks) = mpsc::channel::<Vec<u8>>();
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

        let mut output = Output::default();
        let deadline = Instant::now() + timeout;
        let status = loop {
            output.receive(&chunks, POLL);
            match child.try_wait() {
                Ok(Some(status)) => break Some(status),
                Ok(None) if Instant::now() >= deadline => {
                    kill_group(&mut child);
                    break None;
                }
                Ok(None) => {}
                Err(error) => {
                    kill_group(&mut child);
                    return Err(fail(format!("cannot wait for the command: {error}")));
                }
            }
        };
        let grace = Instant::now() + DRAIN_GRACE;
        while Instant::now() < grace && output.receive(&chunks, POLL) {}

        let mut text = output.into_text();
        match status {
            Some(status) => match status.code() {
                Some(code) => text.push_str(&format!("[exit status {code}]")),
                None => text.push_str("[killed by a signal]"),
            },
            None => text.push_str(&format!(
                "[timed out after {} s; the command was killed]",
                timeout.as_secs()
            )),
        }
        Ok(text)
    }
}

/// The command's output tail.
#[derive(Default)]
struct Output {
    bytes: Vec<u8>,
    cut: bool,
}

impl Output {
    /// Collect what arrives within `wait`. Returns false once the pipe is
    /// closed and empty.
    fn receive(&mut self, chunks: &mpsc::Receiver<Vec<u8>>, wait: Duration) -> bool {
        match chunks.recv_timeout(wait) {
            Ok(chunk) => {
                self.bytes.extend(chunk);
                if self.bytes.len() > 2 * MAX_OUTPUT {
                    let excess = self.bytes.len() - MAX_OUTPUT;
                    self.bytes.drain(..excess);
                    self.cut = true;
                }
                true
            }
            Err(RecvTimeoutError::Timeout) => true,
            Err(RecvTimeoutError::Disconnected) => {
                std::thread::sleep(wait);
                false
            }
        }
    }

    fn into_text(mut self) -> String {
        if self.bytes.len() > MAX_OUTPUT {
            let excess = self.bytes.len() - MAX_OUTPUT;
            self.bytes.drain(..excess);
            self.cut = true;
        }
        let mut text = String::new();
        if self.cut {
            text.push_str("[earlier output cut]\n");
        }
        text.push_str(&String::from_utf8_lossy(&self.bytes));
        if !text.is_empty() && !text.ends_with('\n') {
            text.push('\n');
        }
        text
    }
}

#[cfg(unix)]
fn shell_command(line: &str) -> Command {
    use std::os::unix::process::CommandExt as _;
    let mut command = Command::new("sh");
    command.arg("-c").arg(line).process_group(0);
    command
}

#[cfg(not(unix))]
fn shell_command(line: &str) -> Command {
    let mut command = Command::new("cmd");
    command.arg("/C").arg(line);
    command
}

/// Kill the command and everything it started, then reap it.
fn kill_group(child: &mut Child) {
    #[cfg(unix)]
    {
        let _ = Command::new("kill")
            .args(["-KILL", "--", &format!("-{}", child.id())])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
    let _ = child.kill();
    let _ = child.wait();
}
