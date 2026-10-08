//! Running shell commands.

use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use rig_core::tool::{PortableTool, ToolExecutionError, ToolOutput};
use serde::Deserialize;
use serde_json::json;

use super::child::LoggedChild;
use super::truncate;

/// Seconds a command may run when the call sets no timeout.
const DEFAULT_TIMEOUT: u64 = 120;

/// Runs a command with the platform shell in the working directory.
pub(super) struct Shell {
    /// Where command output is collected before it is returned.
    pub(super) scratch: PathBuf,
}

#[derive(Deserialize)]
pub(super) struct ShellArgs {
    command: String,
    timeout_secs: Option<u64>,
}

impl PortableTool for Shell {
    const NAME: &'static str = "shell";
    type Args = ShellArgs;
    type Output = ToolOutput;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        format!(
            "Run a shell command in the working directory and return its combined stdout and \
             stderr (the last 50 KB). Stdin is empty and there is no terminal. Processes the \
             command leaves in the background are stopped when it ends. Times out after \
             {DEFAULT_TIMEOUT} seconds unless `timeout_secs` says otherwise."
        )
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "command": { "type": "string", "description": "The command line." },
                "timeout_secs": { "type": "integer", "description": "Seconds before it is killed." }
            },
            "required": ["command"]
        })
    }

    async fn call(&self, args: ShellArgs) -> Result<ToolOutput, ToolExecutionError> {
        static RUNS: AtomicU64 = AtomicU64::new(0);
        let log = self.scratch.join(format!(
            "shell-{}-{}.log",
            std::process::id(),
            RUNS.fetch_add(1, Ordering::Relaxed)
        ));
        let mut command = if cfg!(windows) {
            let mut command = Command::new("cmd");
            command.arg("/C");
            command
        } else {
            let mut command = Command::new("sh");
            command.arg("-c");
            command
        };
        command.arg(&args.command);
        let failed = |error: std::io::Error| ToolExecutionError::other(error.to_string());
        let timeout = args.timeout_secs.unwrap_or(DEFAULT_TIMEOUT);
        let mut child = LoggedChild::spawn(command, &log).map_err(failed)?;
        let status = child
            .wait(Duration::from_secs(timeout))
            .await
            .map_err(failed)?;
        drop(child);
        let output = std::fs::read(&log).unwrap_or_default();
        // The log is only a buffer; a leftover file is harmless.
        let _ = std::fs::remove_file(&log);
        let mut text = truncate(&String::from_utf8_lossy(&output), true);
        if text.is_empty() {
            text.push_str("[no output]");
        }
        match status {
            Some(status) if status.success() => Ok(ToolOutput::text(text)),
            Some(status) => {
                text.push_str(&format!("\n[{status}]"));
                Err(ToolExecutionError::other(text))
            }
            None => {
                text.push_str(&format!("\n[killed after {timeout} seconds]"));
                Err(ToolExecutionError::timeout(text))
            }
        }
    }
}
