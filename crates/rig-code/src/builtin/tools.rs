//! The built-in tools: read, edit and write a file, run a shell command, and
//! search files. Each is a Rig tool registered with [`AppExt::add_tool`],
//! as any plugin's tool would be.

use std::fs;
use std::io::Read;
use std::path::Path;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use futures::channel::oneshot;
use regex::RegexBuilder;
use rig_core::tool::{PortableTool, ToolExecutionError};
use serde::Deserialize;
use serde_json::json;

use crate::core::process::{kill, new_group};
use crate::core::registry::AppExt;

/// The most text a tool returns to the model.
const MAX_BYTES: usize = 50 * 1024;
/// The most lines `read` returns at once.
const MAX_LINES: usize = 2000;
/// The most matches `search` returns.
const MAX_MATCHES: usize = 200;

/// Registers [`ReadTool`], [`EditTool`], [`WriteTool`], [`ShellTool`] and
/// [`SearchTool`].
#[derive(Default)]
pub struct BuiltinToolsPlugin;

impl Plugin for BuiltinToolsPlugin {
    fn build(&self, app: &mut App) {
        app.add_tool(ReadTool)
            .add_tool(EditTool)
            .add_tool(WriteTool)
            .add_tool(ShellTool)
            .add_tool(SearchTool);
    }
}

/// An error the model sees in full.
fn failure(message: impl Into<String>) -> ToolExecutionError {
    ToolExecutionError::other(message)
}

fn io_failure(path: &str, error: std::io::Error) -> ToolExecutionError {
    let message = format!("{path}: {error}");
    match error.kind() {
        std::io::ErrorKind::NotFound => ToolExecutionError::not_found(message),
        std::io::ErrorKind::PermissionDenied => ToolExecutionError::permission_denied(message),
        _ => failure(message),
    }
}

/// `text` cut to [`MAX_BYTES`], keeping its start.
fn head(mut text: String) -> String {
    if text.len() > MAX_BYTES {
        let mut end = MAX_BYTES;
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        text.truncate(end);
        text.push_str("\n[output truncated]");
    }
    text
}

/// `text` cut to [`MAX_BYTES`], keeping its end.
fn tail(mut text: String) -> String {
    if text.len() > MAX_BYTES {
        let mut start = text.len() - MAX_BYTES;
        while !text.is_char_boundary(start) {
            start += 1;
        }
        text = format!("[output truncated]\n{}", text.split_off(start));
    }
    text
}

/// Reads a text file.
pub struct ReadTool;

/// [`ReadTool`]'s arguments.
#[derive(Deserialize)]
pub struct ReadArgs {
    path: String,
    offset: Option<usize>,
    limit: Option<usize>,
}

impl PortableTool for ReadTool {
    const NAME: &'static str = "read";
    type Args = ReadArgs;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        format!(
            "Read a text file. Returns at most {MAX_LINES} lines; use offset and limit to \
             read a long file in parts."
        )
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The file to read."},
                "offset": {"type": "integer", "description": "The first line to read, from 1."},
                "limit": {"type": "integer", "description": "How many lines to read."}
            },
            "required": ["path"]
        })
    }

    async fn call(&self, args: ReadArgs) -> Result<String, ToolExecutionError> {
        let text = fs::read_to_string(&args.path).map_err(|error| io_failure(&args.path, error))?;
        let total = text.lines().count();
        let start = args.offset.unwrap_or(1).max(1) - 1;
        if start > 0 && start >= total {
            return Err(ToolExecutionError::invalid_args(format!(
                "offset {} is past the end of {} ({total} lines)",
                start + 1,
                args.path
            )));
        }
        let limit = args.limit.unwrap_or(MAX_LINES).clamp(1, MAX_LINES);
        let lines: Vec<&str> = text.lines().skip(start).take(limit).collect();
        let end = start + lines.len();
        let mut output = head(lines.join("\n"));
        if end < total {
            output.push_str(&format!(
                "\n[lines {}-{end} of {total}; use offset {} to read on]",
                start + 1,
                end + 1
            ));
        }
        Ok(output)
    }
}

/// Replaces one exact piece of text in a file.
pub struct EditTool;

/// [`EditTool`]'s arguments.
#[derive(Deserialize)]
pub struct EditArgs {
    path: String,
    old_text: String,
    new_text: String,
}

impl PortableTool for EditTool {
    const NAME: &'static str = "edit";
    type Args = EditArgs;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "Replace text in a file. old_text must match the file exactly, whitespace included, \
         and occur exactly once; include enough surrounding lines to make it unique."
            .to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The file to edit."},
                "old_text": {"type": "string", "description": "The exact text to replace."},
                "new_text": {"type": "string", "description": "The replacement."}
            },
            "required": ["path", "old_text", "new_text"]
        })
    }

    async fn call(&self, args: EditArgs) -> Result<String, ToolExecutionError> {
        if args.old_text.is_empty() {
            return Err(ToolExecutionError::invalid_args("old_text is empty"));
        }
        let text = fs::read_to_string(&args.path).map_err(|error| io_failure(&args.path, error))?;
        match text.matches(&args.old_text).count() {
            0 => Err(failure(format!(
                "old_text was not found in {}; read the file and copy the text exactly",
                args.path
            ))),
            1 => {
                let edited = text.replacen(&args.old_text, &args.new_text, 1);
                fs::write(&args.path, edited).map_err(|error| io_failure(&args.path, error))?;
                Ok(format!("Edited {}.", args.path))
            }
            count => Err(failure(format!(
                "old_text occurs {count} times in {}; include more lines to make it unique",
                args.path
            ))),
        }
    }
}

/// Writes a whole file, creating it and its directories if needed.
pub struct WriteTool;

/// [`WriteTool`]'s arguments.
#[derive(Deserialize)]
pub struct WriteArgs {
    path: String,
    content: String,
}

impl PortableTool for WriteTool {
    const NAME: &'static str = "write";
    type Args = WriteArgs;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "Write a file, replacing its content. Creates the file and its directories if needed."
            .to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The file to write."},
                "content": {"type": "string", "description": "The whole new content."}
            },
            "required": ["path", "content"]
        })
    }

    async fn call(&self, args: WriteArgs) -> Result<String, ToolExecutionError> {
        if let Some(parent) = Path::new(&args.path).parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent).map_err(|error| io_failure(&args.path, error))?;
        }
        fs::write(&args.path, &args.content).map_err(|error| io_failure(&args.path, error))?;
        Ok(format!(
            "Wrote {} bytes to {}.",
            args.content.len(),
            args.path
        ))
    }
}

/// Runs a shell command with no input, in its own process group, and
/// returns its output. Cancelling the call or a timeout kills the group.
pub struct ShellTool;

/// [`ShellTool`]'s arguments.
#[derive(Deserialize)]
pub struct ShellArgs {
    command: String,
    timeout: Option<u64>,
}

impl PortableTool for ShellTool {
    const NAME: &'static str = "shell";
    type Args = ShellArgs;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "Run a shell command in the working directory and return its output and exit code. \
         It gets no input. The default timeout is 120 seconds."
            .to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The command, run by sh -c."},
                "timeout": {"type": "integer", "description": "Seconds before it is killed."}
            },
            "required": ["command"]
        })
    }

    async fn call(&self, args: ShellArgs) -> Result<String, ToolExecutionError> {
        let mut command = shell(&args.command);
        command
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        new_group(&mut command);
        let child = command
            .spawn()
            .map_err(|error| failure(format!("cannot start the shell: {error}")))?;
        let timeout = Duration::from_secs(args.timeout.unwrap_or(120));
        let cancel = Cancel(Arc::new(AtomicBool::new(false)));
        let cancelled = Arc::clone(&cancel.0);
        let (sender, receiver) = oneshot::channel();
        // The process is waited for on its own thread, so the call can be
        // dropped at any time; dropping `cancel` then kills the process.
        thread::spawn(move || {
            let _ = sender.send(supervise(child, timeout, &cancelled));
        });
        let result = receiver
            .await
            .map_err(|_| failure("the shell command's supervisor stopped"))?;
        drop(cancel);
        result
    }
}

#[cfg(unix)]
fn shell(command: &str) -> Command {
    let mut shell = Command::new("sh");
    shell.arg("-c").arg(command);
    shell
}

#[cfg(not(unix))]
fn shell(command: &str) -> Command {
    let mut shell = Command::new("cmd");
    shell.arg("/C").arg(command);
    shell
}

/// Asks the supervisor to kill the command when the call is dropped.
struct Cancel(Arc<AtomicBool>);

impl Drop for Cancel {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

/// Waits for `child` and collects its output, killing it on timeout or
/// cancellation.
fn supervise(
    mut child: Child,
    timeout: Duration,
    cancelled: &AtomicBool,
) -> Result<String, ToolExecutionError> {
    let stdout = read_on_thread(child.stdout.take());
    let stderr = read_on_thread(child.stderr.take());
    let deadline = Instant::now() + timeout;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Ok(status),
            Ok(None) if cancelled.load(Ordering::Relaxed) => {
                kill(&mut child);
                break Err(ToolExecutionError::cancelled("the command was cancelled"));
            }
            Ok(None) if Instant::now() >= deadline => {
                kill(&mut child);
                break Err(ToolExecutionError::timeout(format!(
                    "the command timed out after {} seconds",
                    timeout.as_secs()
                )));
            }
            Ok(None) => thread::sleep(Duration::from_millis(10)),
            Err(error) => {
                kill(&mut child);
                break Err(failure(format!("cannot wait for the command: {error}")));
            }
        }
    };
    let output = output(
        stdout.join().unwrap_or_default(),
        stderr.join().unwrap_or_default(),
    );
    match status {
        Ok(status) if status.success() => Ok(tail(output)),
        Ok(status) => Err(failure(tail(format!("{}\n{output}", exit(status))))),
        Err(error) => Err(ToolExecutionError::new(
            error.kind(),
            tail(format!("{}\n{output}", error.message())),
        )),
    }
}

fn read_on_thread(pipe: Option<impl Read + Send + 'static>) -> thread::JoinHandle<String> {
    thread::spawn(move || {
        let mut bytes = Vec::new();
        if let Some(mut pipe) = pipe {
            let _ = pipe.read_to_end(&mut bytes);
        }
        String::from_utf8_lossy(&bytes).into_owned()
    })
}

fn output(stdout: String, stderr: String) -> String {
    match (stdout.is_empty(), stderr.is_empty()) {
        (_, true) => stdout,
        (true, false) => stderr,
        (false, false) => format!("{stdout}\n[stderr]\n{stderr}"),
    }
}

fn exit(status: ExitStatus) -> String {
    match status.code() {
        Some(code) => format!("[exit code {code}]"),
        None => format!("[{status}]"),
    }
}

/// Searches file contents with a regular expression, skipping what
/// `.gitignore` and hidden-file rules skip.
pub struct SearchTool;

/// [`SearchTool`]'s arguments.
#[derive(Deserialize)]
pub struct SearchArgs {
    pattern: String,
    path: Option<String>,
    glob: Option<String>,
    ignore_case: Option<bool>,
}

impl PortableTool for SearchTool {
    const NAME: &'static str = "search";
    type Args = SearchArgs;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        format!(
            "Search file contents with a regular expression. Returns `path:line: text` for \
             at most {MAX_MATCHES} matches. Respects .gitignore."
        )
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "A Rust regular expression."},
                "path": {"type": "string", "description": "Directory or file to search; default ."},
                "glob": {"type": "string", "description": "Only files matching this glob, such as *.rs."},
                "ignore_case": {"type": "boolean"}
            },
            "required": ["pattern"]
        })
    }

    async fn call(&self, args: SearchArgs) -> Result<String, ToolExecutionError> {
        let regex = RegexBuilder::new(&args.pattern)
            .case_insensitive(args.ignore_case.unwrap_or(false))
            .build()
            .map_err(|error| ToolExecutionError::invalid_args(error.to_string()))?;
        let root = args.path.as_deref().unwrap_or(".");
        let mut walk = ignore::WalkBuilder::new(root);
        if let Some(glob) = &args.glob {
            let overrides = ignore::overrides::OverrideBuilder::new(root)
                .add(glob)
                .and_then(|builder| builder.build())
                .map_err(|error| ToolExecutionError::invalid_args(error.to_string()))?;
            walk.overrides(overrides);
        }
        let mut matches = Vec::new();
        'files: for entry in walk.build().flatten() {
            if !entry.file_type().is_some_and(|kind| kind.is_file()) {
                continue;
            }
            let Ok(text) = fs::read_to_string(entry.path()) else {
                continue;
            };
            for (number, line) in text.lines().enumerate() {
                if regex.is_match(line) {
                    let line: String = line.chars().take(300).collect();
                    matches.push(format!("{}:{}: {line}", entry.path().display(), number + 1));
                    if matches.len() == MAX_MATCHES {
                        matches.push(format!("[stopped at {MAX_MATCHES} matches]"));
                        break 'files;
                    }
                }
            }
        }
        if matches.is_empty() {
            return Ok("No matches.".to_owned());
        }
        Ok(head(matches.join("\n")))
    }
}
