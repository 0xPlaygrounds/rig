//! The six workspace tools the agent is granted. Relative paths resolve
//! against the task workspace; absolute paths are used as given.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use rig::tool::{DynamicTool, ToolExecutionError, ToolOutput};
use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde_json::json;
use tokio::io::{AsyncRead, AsyncReadExt};

/// Bytes of tool output the model sees; longer output keeps its head and tail.
const OUTPUT_LIMIT: usize = 30_000;
const READ_LINES: usize = 2_000;
const LINE_CHARS: usize = 2_000;
const LIST_LIMIT: usize = 1_000;
const GREP_LIMIT: usize = 300;
const GREP_FILE_BYTES: u64 = 5 * 1024 * 1024;
const BASH_TIMEOUT_SECS: u64 = 180;
const BASH_TIMEOUT_MAX_SECS: u64 = 1_200;
/// How long to keep reading after the shell exits; background children may
/// hold the pipes open indefinitely.
const DRAIN_GRACE: Duration = Duration::from_secs(2);

type ToolResult = Result<ToolOutput, ToolExecutionError>;

/// Every tool, rooted at `workspace`.
pub fn all(workspace: PathBuf) -> Vec<DynamicTool> {
    let root = Arc::new(workspace);
    vec![
        tool(
            &root,
            "bash",
            "Run a bash command in the workspace and return its exit code and combined stdout/stderr. \
             Output over 30000 bytes keeps its head and tail. The command is killed after `timeout_secs` \
             (default 180, max 1200). Run servers in the background with output redirected to a file.",
            json!({
                "type": "object",
                "properties": {
                    "command": { "type": "string", "description": "The bash command." },
                    "timeout_secs": { "type": "integer", "description": "Kill the command after this many seconds." }
                },
                "required": ["command"]
            }),
            bash,
        ),
        tool(
            &root,
            "read_file",
            "Read a text file with line numbers. Returns at most 2000 lines from `offset` (1-based, default 1).",
            json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" },
                    "offset": { "type": "integer", "description": "First line to return, 1-based." },
                    "limit": { "type": "integer", "description": "Maximum number of lines." }
                },
                "required": ["path"]
            }),
            read_file,
        ),
        tool(
            &root,
            "write_file",
            "Create or overwrite a file with `content`, creating parent directories.",
            json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" },
                    "content": { "type": "string" }
                },
                "required": ["path", "content"]
            }),
            write_file,
        ),
        tool(
            &root,
            "edit_file",
            "Replace `old_string` with `new_string` in a file. `old_string` must match exactly once \
             unless `replace_all` is true. Returns the edited region.",
            json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" },
                    "old_string": { "type": "string" },
                    "new_string": { "type": "string" },
                    "replace_all": { "type": "boolean" }
                },
                "required": ["path", "old_string", "new_string"]
            }),
            edit_file,
        ),
        tool(
            &root,
            "list_files",
            "List files under a directory recursively, honouring .gitignore. Directories end in `/`.",
            json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Directory, default the workspace." },
                    "max_depth": { "type": "integer" }
                }
            }),
            list_files,
        ),
        tool(
            &root,
            "grep",
            "Search file contents with a regular expression, honouring .gitignore. Returns `path:line: text`.",
            json!({
                "type": "object",
                "properties": {
                    "pattern": { "type": "string" },
                    "path": { "type": "string", "description": "File or directory, default the workspace." },
                    "glob": { "type": "string", "description": "Only search files matching this glob, e.g. `*.py`." },
                    "case_insensitive": { "type": "boolean" }
                },
                "required": ["pattern"]
            }),
            grep,
        ),
    ]
}

fn tool<A, F, Fut>(
    root: &Arc<PathBuf>,
    name: &'static str,
    description: &'static str,
    parameters: serde_json::Value,
    run: F,
) -> DynamicTool
where
    A: DeserializeOwned + Send + 'static,
    F: Fn(Arc<PathBuf>, A) -> Fut + Copy + Send + Sync + 'static,
    Fut: Future<Output = ToolResult> + Send + 'static,
{
    let root = Arc::clone(root);
    DynamicTool::new(name, description, parameters, move |value| {
        let root = Arc::clone(&root);
        Box::pin(async move {
            let args = serde_json::from_value::<A>(value)
                .map_err(|error| ToolExecutionError::invalid_args(error.to_string()))?;
            run(root, args).await
        })
    })
}

fn resolve(root: &Path, path: &str) -> PathBuf {
    let path = Path::new(path);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    }
}

fn io_error(path: &Path, error: std::io::Error) -> ToolExecutionError {
    let message = format!("{}: {error}", path.display());
    match error.kind() {
        std::io::ErrorKind::NotFound => ToolExecutionError::not_found(message),
        std::io::ErrorKind::PermissionDenied => ToolExecutionError::permission_denied(message),
        _ => ToolExecutionError::other(message),
    }
}

async fn blocking<T: Send + 'static>(
    work: impl FnOnce() -> Result<T, ToolExecutionError> + Send + 'static,
) -> Result<T, ToolExecutionError> {
    tokio::task::spawn_blocking(work)
        .await
        .map_err(|error| ToolExecutionError::other(error.to_string()))?
}

/// Keeps the first and last `OUTPUT_LIMIT / 2` bytes of a stream.
#[derive(Default)]
struct Capture {
    head: Vec<u8>,
    tail: VecDeque<u8>,
    total: usize,
}

impl Capture {
    fn push(&mut self, bytes: &[u8]) {
        self.total += bytes.len();
        for &byte in bytes {
            if self.head.len() < OUTPUT_LIMIT / 2 {
                self.head.push(byte);
            } else {
                if self.tail.len() == OUTPUT_LIMIT / 2 {
                    self.tail.pop_front();
                }
                self.tail.push_back(byte);
            }
        }
    }

    fn render(&self) -> String {
        let kept = self.head.len() + self.tail.len();
        let mut text = String::from_utf8_lossy(&self.head).into_owned();
        if self.total > kept {
            text.push_str(&format!(
                "\n[... {} bytes omitted ...]\n",
                self.total - kept
            ));
        }
        let tail: Vec<u8> = self.tail.iter().copied().collect();
        text.push_str(&String::from_utf8_lossy(&tail));
        text
    }
}

async fn pump(mut reader: impl AsyncRead + Unpin, capture: Arc<Mutex<Capture>>) {
    let mut buffer = [0u8; 8192];
    loop {
        match reader.read(&mut buffer).await {
            Ok(0) | Err(_) => break,
            Ok(read) => {
                if let (Ok(mut capture), Some(bytes)) = (capture.lock(), buffer.get(..read)) {
                    capture.push(bytes);
                }
            }
        }
    }
}

#[derive(Deserialize)]
struct BashArgs {
    command: String,
    timeout_secs: Option<u64>,
}

async fn bash(root: Arc<PathBuf>, args: BashArgs) -> ToolResult {
    let timeout = args
        .timeout_secs
        .unwrap_or(BASH_TIMEOUT_SECS)
        .clamp(1, BASH_TIMEOUT_MAX_SECS);
    let mut command = tokio::process::Command::new("bash");
    command
        .arg("-c")
        .arg(&args.command)
        .current_dir(root.as_path())
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    // Its own process group, so a timeout kills the command's children too.
    #[cfg(unix)]
    command.process_group(0);
    let mut child = command
        .spawn()
        .map_err(|error| ToolExecutionError::other(format!("cannot start bash: {error}")))?;
    let capture = Arc::new(Mutex::new(Capture::default()));
    let mut readers = Vec::new();
    if let Some(stdout) = child.stdout.take() {
        readers.push(tokio::spawn(pump(stdout, Arc::clone(&capture))));
    }
    if let Some(stderr) = child.stderr.take() {
        readers.push(tokio::spawn(pump(stderr, Arc::clone(&capture))));
    }
    let status = match tokio::time::timeout(Duration::from_secs(timeout), child.wait()).await {
        Ok(status) => {
            Some(status.map_err(|error| ToolExecutionError::other(error.to_string()))?)
        }
        Err(_) => {
            kill_group(&mut child).await;
            None
        }
    };
    let drain = futures_join(readers);
    let _drained = tokio::time::timeout(DRAIN_GRACE, drain).await;
    let output = capture
        .lock()
        .map(|capture| capture.render())
        .unwrap_or_default();
    let header = match status {
        Some(status) => match status.code() {
            Some(code) => format!("exit code: {code}"),
            None => "exit code: none (terminated by a signal)".to_string(),
        },
        None => format!("timed out after {timeout}s; the command was killed"),
    };
    Ok(ToolOutput::text(format!("{header}\n{output}")))
}

async fn futures_join(readers: Vec<tokio::task::JoinHandle<()>>) {
    for reader in readers {
        let _finished = reader.await;
    }
}

async fn kill_group(child: &mut tokio::process::Child) {
    #[cfg(unix)]
    if let Some(pid) = child.id() {
        let _killed = tokio::process::Command::new("kill")
            .arg("-KILL")
            .arg(format!("-{pid}"))
            .status()
            .await;
    }
    let _killed = child.kill().await;
}

#[derive(Deserialize)]
struct ReadArgs {
    path: String,
    offset: Option<usize>,
    limit: Option<usize>,
}

async fn read_file(root: Arc<PathBuf>, args: ReadArgs) -> ToolResult {
    let path = resolve(&root, &args.path);
    blocking(move || {
        let bytes = std::fs::read(&path).map_err(|error| io_error(&path, error))?;
        let text = String::from_utf8_lossy(&bytes);
        let lines: Vec<&str> = text.lines().collect();
        let start = args.offset.unwrap_or(1).max(1);
        let limit = args.limit.unwrap_or(READ_LINES).clamp(1, READ_LINES);
        let mut out = String::new();
        for (number, line) in lines.iter().enumerate().skip(start - 1).take(limit) {
            let line: String = line.chars().take(LINE_CHARS).collect();
            out.push_str(&format!("{:>6}\t{line}\n", number + 1));
        }
        let end = (start - 1 + limit).min(lines.len());
        if lines.is_empty() {
            out.push_str("[empty file]\n");
        } else if start > lines.len() {
            out.push_str(&format!(
                "[offset {start} is past the end; the file has {} lines]\n",
                lines.len()
            ));
        } else if end < lines.len() {
            out.push_str(&format!(
                "[showing lines {start}-{end} of {}; pass offset to read more]\n",
                lines.len()
            ));
        }
        Ok(ToolOutput::text(out))
    })
    .await
}

#[derive(Deserialize)]
struct WriteArgs {
    path: String,
    content: String,
}

async fn write_file(root: Arc<PathBuf>, args: WriteArgs) -> ToolResult {
    let path = resolve(&root, &args.path);
    blocking(move || {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|error| io_error(parent, error))?;
        }
        std::fs::write(&path, args.content.as_bytes()).map_err(|error| io_error(&path, error))?;
        Ok(ToolOutput::text(format!(
            "wrote {} bytes to {}",
            args.content.len(),
            path.display()
        )))
    })
    .await
}

#[derive(Deserialize)]
struct EditArgs {
    path: String,
    old_string: String,
    new_string: String,
    #[serde(default)]
    replace_all: bool,
}

async fn edit_file(root: Arc<PathBuf>, args: EditArgs) -> ToolResult {
    let path = resolve(&root, &args.path);
    blocking(move || {
        if args.old_string.is_empty() {
            return Err(ToolExecutionError::invalid_args(
                "old_string is empty; use write_file to create a file",
            ));
        }
        let text = std::fs::read_to_string(&path).map_err(|error| io_error(&path, error))?;
        let matches = text.matches(&args.old_string).count();
        if matches == 0 {
            return Err(ToolExecutionError::not_found(format!(
                "old_string was not found in {}; read the file and copy the text exactly, including whitespace",
                path.display()
            )));
        }
        if matches > 1 && !args.replace_all {
            return Err(ToolExecutionError::invalid_args(format!(
                "old_string matches {matches} times in {}; include more context or set replace_all",
                path.display()
            )));
        }
        let first = text.find(&args.old_string).unwrap_or(0);
        let edited = if args.replace_all {
            text.replace(&args.old_string, &args.new_string)
        } else {
            text.replacen(&args.old_string, &args.new_string, 1)
        };
        std::fs::write(&path, edited.as_bytes()).map_err(|error| io_error(&path, error))?;
        let first_line = text.get(..first).map_or(0, |prefix| prefix.matches('\n').count());
        let shown = args.new_string.matches('\n').count() + 1;
        let from = first_line.saturating_sub(4);
        let mut snippet = String::new();
        for (number, line) in edited.lines().enumerate().skip(from).take(shown + 8) {
            let line: String = line.chars().take(LINE_CHARS).collect();
            snippet.push_str(&format!("{:>6}\t{line}\n", number + 1));
        }
        Ok(ToolOutput::text(format!(
            "replaced {} occurrence(s) in {}\n{snippet}",
            if args.replace_all { matches } else { 1 },
            path.display()
        )))
    })
    .await
}

#[derive(Deserialize)]
struct ListArgs {
    path: Option<String>,
    max_depth: Option<usize>,
}

async fn list_files(root: Arc<PathBuf>, args: ListArgs) -> ToolResult {
    let base = resolve(&root, args.path.as_deref().unwrap_or("."));
    blocking(move || {
        let mut walker = ignore::WalkBuilder::new(&base);
        walker.hidden(false).max_depth(args.max_depth).filter_entry(|entry| {
            entry.file_name() != ".git"
        });
        let mut entries = Vec::new();
        let mut truncated = false;
        for entry in walker.build() {
            let entry = entry.map_err(|error| ToolExecutionError::other(error.to_string()))?;
            if entry.depth() == 0 {
                continue;
            }
            if entries.len() == LIST_LIMIT {
                truncated = true;
                break;
            }
            let relative = entry.path().strip_prefix(&base).unwrap_or(entry.path());
            let is_dir = entry.file_type().is_some_and(|kind| kind.is_dir());
            entries.push(format!(
                "{}{}",
                relative.display(),
                if is_dir { "/" } else { "" }
            ));
        }
        entries.sort();
        let mut out = entries.join("\n");
        if entries.is_empty() {
            out.push_str("[no files]");
        }
        if truncated {
            out.push_str(&format!(
                "\n[stopped at {LIST_LIMIT} entries; list a subdirectory or pass max_depth]"
            ));
        }
        Ok(ToolOutput::text(out))
    })
    .await
}

#[derive(Deserialize)]
struct GrepArgs {
    pattern: String,
    path: Option<String>,
    glob: Option<String>,
    #[serde(default)]
    case_insensitive: bool,
}

async fn grep(root: Arc<PathBuf>, args: GrepArgs) -> ToolResult {
    let base = resolve(&root, args.path.as_deref().unwrap_or("."));
    blocking(move || {
        let regex = regex::RegexBuilder::new(&args.pattern)
            .case_insensitive(args.case_insensitive)
            .build()
            .map_err(|error| ToolExecutionError::invalid_args(error.to_string()))?;
        let mut walker = ignore::WalkBuilder::new(&base);
        walker
            .hidden(false)
            .filter_entry(|entry| entry.file_name() != ".git");
        if let Some(glob) = &args.glob {
            let overrides = ignore::overrides::OverrideBuilder::new(&base)
                .add(glob)
                .and_then(|builder| builder.build())
                .map_err(|error| ToolExecutionError::invalid_args(error.to_string()))?;
            walker.overrides(overrides);
        }
        let mut matches = Vec::new();
        'files: for entry in walker.build().flatten() {
            if !entry.file_type().is_some_and(|kind| kind.is_file()) {
                continue;
            }
            if entry
                .metadata()
                .is_ok_and(|metadata| metadata.len() > GREP_FILE_BYTES)
            {
                continue;
            }
            let Ok(bytes) = std::fs::read(entry.path()) else {
                continue;
            };
            if bytes.iter().take(8192).any(|&byte| byte == 0) {
                continue;
            }
            let text = String::from_utf8_lossy(&bytes);
            let shown = entry.path().strip_prefix(root.as_path()).unwrap_or(entry.path());
            for (number, line) in text.lines().enumerate() {
                if regex.is_match(line) {
                    let line: String = line.chars().take(300).collect();
                    matches.push(format!("{}:{}: {line}", shown.display(), number + 1));
                    if matches.len() == GREP_LIMIT {
                        matches.push(format!(
                            "[stopped at {GREP_LIMIT} matches; narrow the pattern or path]"
                        ));
                        break 'files;
                    }
                }
            }
        }
        if matches.is_empty() {
            return Ok(ToolOutput::text("no matches"));
        }
        Ok(ToolOutput::text(matches.join("\n")))
    })
    .await
}

#[cfg(test)]
mod tests;
