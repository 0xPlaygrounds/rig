use std::{
    process::Command,
    time::{Duration, Instant},
};

use rig_core::tool::{PortableTool, ToolExecutionError};
use serde::Deserialize;

use super::fail;
use crate::process::{POLL, Piped, sleep};

/// The most output bytes returned, keeping the tail.
const MAX_OUTPUT: usize = 30 * 1024;
/// The timeout when the call names none.
const DEFAULT_TIMEOUT_SECS: u64 = 120;
/// How long output is still collected after the command exits, for
/// background processes that keep the pipe open.
const DRAIN_GRACE: Duration = Duration::from_millis(500);

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
        // Dropping this future (Esc, or the agent despawned) drops the
        // process, which kills its group.
        let mut process = Piped::spawn(shell_command(&args.command))
            .map_err(|error| fail(format!("cannot run the command: {error}")))?;
        let mut output = Output::default();
        let deadline = Instant::now() + timeout;
        let status = loop {
            output.take(process.output().0);
            match process.try_wait() {
                Ok(Some(status)) => break Some(status),
                Ok(None) if Instant::now() >= deadline => {
                    process.kill();
                    break None;
                }
                Ok(None) => {}
                Err(error) => {
                    process.kill();
                    return Err(fail(format!("cannot wait for the command: {error}")));
                }
            }
            sleep(POLL).await;
        };
        let grace = Instant::now() + DRAIN_GRACE;
        loop {
            let (bytes, open) = process.output();
            output.take(bytes);
            if !open || Instant::now() >= grace {
                break;
            }
            sleep(POLL).await;
        }

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
    /// Append `bytes`, keeping at most twice the returned size.
    fn take(&mut self, bytes: Vec<u8>) {
        self.bytes.extend(bytes);
        if self.bytes.len() > 2 * MAX_OUTPUT {
            let excess = self.bytes.len() - MAX_OUTPUT;
            self.bytes.drain(..excess);
            self.cut = true;
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
    let mut command = Command::new("sh");
    command.arg("-c").arg(line);
    command
}

#[cfg(not(unix))]
fn shell_command(line: &str) -> Command {
    let mut command = Command::new("cmd");
    command.arg("/C").arg(line);
    command
}
