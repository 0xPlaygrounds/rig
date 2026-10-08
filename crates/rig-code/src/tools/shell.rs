//! Running shell commands.

use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use rig_core::tool::{PortableTool, ToolExecutionError, ToolOutput};
use serde::Deserialize;
use serde_json::json;

use super::OUTPUT_LIMIT;
use super::child::LoggedChild;

/// Seconds a command may run when the call sets no timeout.
const DEFAULT_TIMEOUT: u64 = 120;
/// A command that writes more output than this is killed, so a runaway
/// command cannot fill the disk.
const OUTPUT_CAP: u64 = 10 * 1024 * 1024;
/// How often the output size is checked.
const CHECK_EVERY: Duration = Duration::from_millis(200);

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
             {DEFAULT_TIMEOUT} seconds unless `timeout_secs` says otherwise, and is killed once \
             it writes over {} MB.",
            OUTPUT_CAP / 1024 / 1024
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
        let started = Instant::now();
        let mut flooded = false;
        let status = loop {
            if let Some(status) = child.wait(CHECK_EVERY).await.map_err(failed)? {
                break Some(status);
            }
            flooded = log.metadata().is_ok_and(|meta| meta.len() > OUTPUT_CAP);
            if flooded || started.elapsed() >= Duration::from_secs(timeout) {
                break None;
            }
        };
        drop(child);
        let mut text = read_tail(&log).unwrap_or_default();
        // The log is only a buffer; a leftover file is harmless.
        let _ = std::fs::remove_file(&log);
        if text.is_empty() {
            text.push_str("[no output]");
        }
        match status {
            Some(status) if status.success() => Ok(ToolOutput::text(text)),
            Some(status) => {
                text.push_str(&format!("\n[{status}]"));
                Err(ToolExecutionError::other(text))
            }
            None if flooded => {
                text.push_str(&format!(
                    "\n[killed after writing over {} MB of output]",
                    OUTPUT_CAP / 1024 / 1024
                ));
                Err(ToolExecutionError::other(text))
            }
            None => {
                text.push_str(&format!("\n[killed after {timeout} seconds]"));
                Err(ToolExecutionError::timeout(text))
            }
        }
    }
}

/// The last [`OUTPUT_LIMIT`] bytes of `log` as text, with a note when the
/// output was longer. Only those bytes are read.
fn read_tail(log: &Path) -> std::io::Result<String> {
    let mut file = std::fs::File::open(log)?;
    let length = file.metadata()?.len();
    let limit = OUTPUT_LIMIT as u64;
    let cut = length > limit;
    if cut {
        file.seek(SeekFrom::Start(length - limit))?;
    }
    let mut bytes = Vec::new();
    file.take(limit).read_to_end(&mut bytes)?;
    let text = String::from_utf8_lossy(&bytes);
    Ok(if cut {
        // The cut may split a character; drop its remains.
        let text = text.trim_start_matches(char::REPLACEMENT_CHARACTER);
        format!("[output cut to its last {OUTPUT_LIMIT} bytes]\n{text}")
    } else {
        text.into_owned()
    })
}
