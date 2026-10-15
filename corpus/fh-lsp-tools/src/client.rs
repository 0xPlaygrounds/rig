//! A language server over a child's stdio: JSON-RPC messages framed by
//! `Content-Length` headers, the documents it was given, and what it
//! published since (diagnostics, indexing progress, its last stderr lines).

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::io::{self, BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Stdio};
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread;
use std::time::{Duration, Instant};

use rig_tools::process::{Launcher, kill_group};
use serde_json::{Value, json};

/// How many of a server's last stderr lines are kept for a crash report.
const STDERR_LINES: usize = 12;
/// How often a wait looks at the clock and reports progress.
const TICK: Duration = Duration::from_millis(200);

/// Locks `mutex`, whatever a panicking holder left in it.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Why a request got no answer.
pub(crate) enum Failure {
    /// The server's output ended: it exited or crashed.
    Died,
    /// No answer in time.
    Timeout,
    /// The server answered with an error.
    Error(String),
}

/// What the server told the client, kept by its reader thread.
#[derive(Default)]
pub(crate) struct State {
    /// Whether its output is still open.
    alive: bool,
    /// Answers not taken yet, by request id.
    responses: HashMap<i64, Result<Value, String>>,
    /// The diagnostics last published for each document, by URI.
    pub(crate) diagnostics: HashMap<String, Vec<Value>>,
    /// Documents whose diagnostics were published since they last changed.
    fresh: HashSet<String>,
    /// Work in progress, such as indexing: its title and message, by token.
    progress: BTreeMap<String, (String, String)>,
    /// Its last stderr lines.
    stderr: VecDeque<String>,
}

impl State {
    /// Whether the server reports work in progress, such as indexing.
    pub(crate) fn busy(&self) -> bool {
        !self.progress.is_empty()
    }

    /// Whether the diagnostics of `uri` describe its last content.
    pub(crate) fn fresh(&self, uri: &str) -> bool {
        self.fresh.contains(uri)
    }

    /// The work in progress, as one line, such as `indexing 340/1200`.
    pub(crate) fn progress_line(&self) -> Option<String> {
        let (title, message) = self.progress.values().next()?;
        Some(format!("{title} {message}").trim().to_owned())
    }
}

/// The reader thread's side: the state and a signal when it changes.
#[derive(Default)]
struct Shared {
    state: Mutex<State>,
    changed: Condvar,
}

/// A document the server was given: its version and text.
struct Document {
    version: i64,
    text: String,
}

/// A running language server.
pub(crate) struct Server {
    /// Its program's file name, which names it to the user.
    pub(crate) name: String,
    child: Mutex<Child>,
    stdin: Arc<Mutex<ChildStdin>>,
    shared: Arc<Shared>,
    next_id: AtomicI64,
    /// The documents it was given, by path.
    documents: Mutex<HashMap<PathBuf, Document>>,
}

impl Server {
    /// Starts `command` (a program and its arguments, split at whitespace)
    /// under `launcher` in `root`, and initializes it for that workspace,
    /// waiting at most `timeout` for its answer.
    pub(crate) fn start(
        launcher: &Launcher,
        command: &str,
        root: &Path,
        timeout: Duration,
    ) -> Result<Self, String> {
        let mut words = command.split_whitespace();
        let program = words.next().ok_or("the command is empty")?;
        let name = Path::new(program).file_name().map_or_else(
            || program.to_owned(),
            |name| name.to_string_lossy().into_owned(),
        );
        let mut process = launcher.clone().in_dir(root).command(program);
        process
            .args(words)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let mut child = process
            .spawn()
            .map_err(|failure| format!("could not start `{command}`: {failure}"))?;
        let pipes = (child.stdin.take(), child.stdout.take(), child.stderr.take());
        let (Some(stdin), Some(stdout), Some(stderr)) = pipes else {
            kill_group(&mut child);
            child.wait().ok();
            return Err(format!("`{command}` has no pipes"));
        };
        let shared = Arc::new(Shared::default());
        lock(&shared.state).alive = true;
        let stdin = Arc::new(Mutex::new(stdin));
        let server = Self {
            name,
            child: Mutex::new(child),
            stdin: Arc::clone(&stdin),
            shared: Arc::clone(&shared),
            next_id: AtomicI64::new(1),
            documents: Mutex::new(HashMap::new()),
        };
        let reader = Arc::clone(&shared);
        thread::Builder::new()
            .name("rig-lsp-out".to_owned())
            .spawn(move || read_output(stdout, &reader, &stdin))
            .map_err(|failure| failure.to_string())?;
        let errors = Arc::clone(&shared);
        thread::Builder::new()
            .name("rig-lsp-err".to_owned())
            .spawn(move || read_stderr(stderr, &errors))
            .map_err(|failure| failure.to_string())?;
        let root_uri = uri(root);
        let params = json!({
            "processId": std::process::id(),
            "rootUri": root_uri,
            "rootPath": root,
            "workspaceFolders": [{ "uri": root_uri, "name": "project" }],
            "capabilities": {
                "window": { "workDoneProgress": true },
                "workspace": { "configuration": true, "symbol": {} },
                "textDocument": {
                    "synchronization": {},
                    "publishDiagnostics": {},
                    "definition": { "linkSupport": true },
                    "references": {},
                    "hover": { "contentFormat": ["plaintext", "markdown"] },
                },
            },
        });
        match server.request("initialize", params, timeout) {
            Ok(_) => {}
            Err(Failure::Died) => return Err(server.crash_report()),
            Err(Failure::Timeout) => return Err(format!("`{}` did not initialize", server.name)),
            Err(Failure::Error(why)) => return Err(format!("`{}` refused: {why}", server.name)),
        }
        server
            .notify("initialized", json!({}))
            .map_err(|_| server.crash_report())?;
        Ok(server)
    }

    /// Whether its output is still open and its process has not exited:
    /// a process that died is dead before its output is read to the end.
    pub(crate) fn alive(&self) -> bool {
        let open = lock(&self.shared.state).alive;
        open && matches!(lock(&self.child).try_wait(), Ok(None))
    }

    /// Its process id.
    pub(crate) fn pid(&self) -> u32 {
        lock(&self.child).id()
    }

    /// How it ended and its last stderr lines.
    pub(crate) fn crash_report(&self) -> String {
        let status = lock(&self.child).try_wait().ok().flatten();
        let how = status.map_or_else(
            || "stopped answering".to_owned(),
            |status| format!("exited ({status})"),
        );
        let stderr: Vec<String> = lock(&self.shared.state).stderr.iter().cloned().collect();
        let mut report = format!("language server `{}` {how}", self.name);
        if stderr.is_empty() {
            report.push_str("; it wrote nothing on stderr.");
        } else {
            report.push_str(". Its last stderr lines:\n");
            report.push_str(&stderr.join("\n"));
        }
        report
    }

    /// Sends a notification.
    pub(crate) fn notify(&self, method: &str, params: Value) -> io::Result<()> {
        let message = json!({ "jsonrpc": "2.0", "method": method, "params": params });
        write_message(&self.stdin, &message)
    }

    /// Sends a request and waits at most `timeout` for its answer.
    pub(crate) fn request(
        &self,
        method: &str,
        params: Value,
        timeout: Duration,
    ) -> Result<Value, Failure> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let message = json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params });
        write_message(&self.stdin, &message).map_err(|_| Failure::Died)?;
        let deadline = Instant::now() + timeout;
        let mut state = lock(&self.shared.state);
        loop {
            if let Some(answer) = state.responses.remove(&id) {
                return answer.map_err(Failure::Error);
            }
            if !state.alive {
                return Err(Failure::Died);
            }
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return Err(Failure::Timeout);
            }
            state = self
                .shared
                .changed
                .wait_timeout(state, left)
                .unwrap_or_else(PoisonError::into_inner)
                .0;
        }
    }

    /// Waits until `done` says so, the server dies or `deadline` passes,
    /// calling `tick` with the state at least every [`TICK`]. Whether
    /// `done` said so.
    pub(crate) fn wait(
        &self,
        deadline: Instant,
        mut done: impl FnMut(&State) -> bool,
        mut tick: impl FnMut(&State),
    ) -> bool {
        let mut state = lock(&self.shared.state);
        loop {
            if done(&state) {
                return true;
            }
            tick(&state);
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() || !state.alive {
                return false;
            }
            state = self
                .shared
                .changed
                .wait_timeout(state, left.min(TICK))
                .unwrap_or_else(PoisonError::into_inner)
                .0;
        }
    }

    /// Reads what the server published.
    pub(crate) fn with_state<T>(&self, read: impl FnOnce(&State) -> T) -> T {
        read(&lock(&self.shared.state))
    }

    /// The documents the server was given.
    pub(crate) fn open_documents(&self) -> Vec<PathBuf> {
        lock(&self.documents).keys().cloned().collect()
    }

    /// Gives the server `path` as it is on disk: opens it the first time,
    /// tells it the new content when the file changed since, and closes it
    /// when the file is gone. Whichever tool or command changed it, the
    /// server then describes the file as it is.
    pub(crate) fn sync(&self, path: &Path) -> io::Result<()> {
        let text = std::fs::read_to_string(path).ok();
        let uri = uri(path);
        let mut documents = lock(&self.documents);
        match (documents.get_mut(path), text) {
            (Some(document), Some(text)) if document.text == text => Ok(()),
            (Some(document), Some(text)) => {
                document.version += 1;
                let params = json!({
                    "textDocument": { "uri": uri, "version": document.version },
                    "contentChanges": [{ "text": text }],
                });
                document.text = text;
                lock(&self.shared.state).fresh.remove(&uri);
                self.notify("textDocument/didChange", params)
            }
            (Some(_), None) => {
                documents.remove(path);
                let mut state = lock(&self.shared.state);
                state.fresh.remove(&uri);
                state.diagnostics.remove(&uri);
                drop(state);
                self.notify(
                    "textDocument/didClose",
                    json!({ "textDocument": { "uri": uri } }),
                )
            }
            (None, Some(text)) => {
                let language = language_id(path);
                let params = json!({ "textDocument": {
                    "uri": uri, "languageId": language, "version": 1, "text": text,
                }});
                documents.insert(path.to_path_buf(), Document { version: 1, text });
                lock(&self.shared.state).fresh.remove(&uri);
                self.notify("textDocument/didOpen", params)
            }
            (None, None) => Ok(()),
        }
    }

    /// Asks the server to shut down and exit, and waits at most `grace`
    /// for each step; kills it if it is still running then.
    pub(crate) fn shutdown(&self, grace: Duration) {
        if self.alive() {
            self.request("shutdown", Value::Null, grace).ok();
            self.notify("exit", Value::Null).ok();
        }
        let deadline = Instant::now() + grace;
        let mut child = lock(&self.child);
        while matches!(child.try_wait(), Ok(None)) && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(20));
        }
        if matches!(child.try_wait(), Ok(None)) {
            kill_group(&mut child);
            child.wait().ok();
        }
    }
}

/// A server dropped while it runs is killed with its process group, so no
/// process outlives the session.
impl Drop for Server {
    fn drop(&mut self) {
        let child = self.child.get_mut().unwrap_or_else(PoisonError::into_inner);
        if matches!(child.try_wait(), Ok(None)) {
            kill_group(child);
            child.wait().ok();
        }
    }
}

/// Writes `message` with its `Content-Length` header.
fn write_message(stdin: &Mutex<ChildStdin>, message: &Value) -> io::Result<()> {
    let body = message.to_string();
    let mut stdin = lock(stdin);
    write!(stdin, "Content-Length: {}\r\n\r\n{body}", body.len())?;
    stdin.flush()
}

/// Reads the next message: its headers up to an empty line, then as many
/// bytes as `Content-Length` says. Header lines it does not know are
/// skipped, as is a block without a length; `None` once the output ended.
fn read_message(reader: &mut impl BufRead) -> Option<Value> {
    loop {
        let mut length = None;
        loop {
            let mut line = String::new();
            if reader.read_line(&mut line).ok()? == 0 {
                return None;
            }
            let line = line.trim();
            if line.is_empty() {
                if length.is_some() {
                    break;
                }
                continue;
            }
            if let Some((name, value)) = line.split_once(':')
                && name.trim().eq_ignore_ascii_case("content-length")
            {
                length = value.trim().parse::<usize>().ok();
            }
        }
        let mut body = vec![0; length.unwrap_or_default()];
        reader.read_exact(&mut body).ok()?;
        if let Ok(message) = serde_json::from_slice(&body) {
            return Some(message);
        }
    }
}

/// The reader thread: takes answers, publications and the server's own
/// requests until the output ends, then marks the server dead.
fn read_output(stdout: impl Read, shared: &Shared, stdin: &Mutex<ChildStdin>) {
    let mut reader = BufReader::new(stdout);
    while let Some(message) = read_message(&mut reader) {
        let method = message.get("method").and_then(Value::as_str);
        match (method, message.get("id")) {
            (None, Some(id)) => {
                let Some(id) = id.as_i64() else { continue };
                let answer = match message.get("error") {
                    Some(error) => Err(error
                        .get("message")
                        .and_then(Value::as_str)
                        .unwrap_or("an error")
                        .to_owned()),
                    None => Ok(message.get("result").cloned().unwrap_or(Value::Null)),
                };
                lock(&shared.state).responses.insert(id, answer);
            }
            (Some(method), Some(id)) => {
                // Configuration has an answer per item asked; everything
                // else it asks (progress tokens, registrations) is agreed to.
                let result = match message.pointer("/params/items").and_then(Value::as_array) {
                    Some(items) if method == "workspace/configuration" => {
                        Value::Array(vec![Value::Null; items.len()])
                    }
                    _ => Value::Null,
                };
                let answer = json!({ "jsonrpc": "2.0", "id": id, "result": result });
                write_message(stdin, &answer).ok();
                continue;
            }
            (Some(method), None) => published(method, &message, &mut lock(&shared.state)),
            (None, None) => continue,
        }
        shared.changed.notify_all();
    }
    lock(&shared.state).alive = false;
    shared.changed.notify_all();
}

/// Keeps what a notification published: a document's diagnostics, or the
/// progress of the server's work.
fn published(method: &str, message: &Value, state: &mut State) {
    let params = message.get("params").cloned().unwrap_or_default();
    match method {
        "textDocument/publishDiagnostics" => {
            let uri = params
                .get("uri")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let diagnostics = params
                .get("diagnostics")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            state.fresh.insert(uri.to_owned());
            state.diagnostics.insert(uri.to_owned(), diagnostics);
        }
        "$/progress" => {
            let token = params
                .get("token")
                .map(Value::to_string)
                .unwrap_or_default();
            let value = params.get("value").cloned().unwrap_or_default();
            let text = |key: &str| value.get(key).and_then(Value::as_str).map(str::to_owned);
            let message = text("message")
                .or_else(|| value.get("percentage").map(|percent| format!("{percent}%")))
                .unwrap_or_default();
            match value.get("kind").and_then(Value::as_str) {
                Some("begin") => {
                    let title = text("title").unwrap_or_default().to_lowercase();
                    state.progress.insert(token, (title, message));
                }
                Some("report") => {
                    if let Some(work) = state.progress.get_mut(&token) {
                        work.1 = message;
                    }
                }
                Some("end") => {
                    state.progress.remove(&token);
                }
                _ => {}
            }
        }
        _ => {}
    }
}

/// Keeps the server's last stderr lines.
fn read_stderr(stderr: impl Read, shared: &Shared) {
    for line in BufReader::new(stderr).lines() {
        let Ok(line) = line else { break };
        let mut state = lock(&shared.state);
        if state.stderr.len() == STDERR_LINES {
            state.stderr.pop_front();
        }
        state.stderr.push_back(line);
    }
}

/// The `file://` URI of an absolute path.
pub(crate) fn uri(path: &Path) -> String {
    let mut uri = String::from("file://");
    for byte in path.to_string_lossy().bytes() {
        if byte.is_ascii_alphanumeric() || b"/-_.~".contains(&byte) {
            uri.push(char::from(byte));
        } else {
            uri.push_str(&format!("%{byte:02X}"));
        }
    }
    uri
}

/// The path a `file://` URI names.
pub(crate) fn path_of(uri: &str) -> PathBuf {
    let encoded = uri.strip_prefix("file://").unwrap_or(uri).as_bytes();
    let mut bytes = Vec::with_capacity(encoded.len());
    let mut at = 0;
    while let Some(&byte) = encoded.get(at) {
        let hex = encoded
            .get(at + 1..at + 3)
            .and_then(|hex| std::str::from_utf8(hex).ok())
            .and_then(|hex| u8::from_str_radix(hex, 16).ok());
        match (byte, hex) {
            (b'%', Some(decoded)) => {
                bytes.push(decoded);
                at += 3;
            }
            _ => {
                bytes.push(byte);
                at += 1;
            }
        }
    }
    PathBuf::from(String::from_utf8_lossy(&bytes).into_owned())
}

/// The LSP language id of a file, from its extension.
fn language_id(path: &Path) -> String {
    let extension = path
        .extension()
        .map(|ext| ext.to_string_lossy().into_owned())
        .unwrap_or_default();
    let id = match extension.as_str() {
        "rs" => "rust",
        "ts" => "typescript",
        "tsx" => "typescriptreact",
        "js" => "javascript",
        "jsx" => "javascriptreact",
        "py" => "python",
        "rb" => "ruby",
        "h" => "c",
        "hpp" | "cc" | "cxx" => "cpp",
        "sh" => "shellscript",
        "md" => "markdown",
        other => other,
    };
    id.to_owned()
}
