//! Line-delimited JSON-RPC framing and short-lived probes shared by harnesses.

use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::time::Duration;

use rig_harness::rig_tools::process::Process;
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncWrite, AsyncWriteExt, Lines};

/// Newline-delimited JSON transport over a child process's stdio pipes.
pub(crate) struct Transport<R, W> {
    lines: Lines<tokio::io::BufReader<R>>,
    writer: W,
    next_id: u64,
}

impl<R: AsyncRead + Unpin, W: AsyncWrite + Unpin> Transport<R, W> {
    pub(crate) fn new(reader: R, writer: W) -> Self {
        Self {
            lines: tokio::io::BufReader::new(reader).lines(),
            writer,
            next_id: 0,
        }
    }

    /// Writes `message` as one newline-terminated JSON line.
    pub(crate) async fn write(&mut self, message: &Value) -> Result<(), ()> {
        let mut line = message.to_string();
        line.push('\n');
        self.writer.write_all(line.as_bytes()).await.map_err(|_| ())
    }

    /// Writes a JSON-RPC request and returns its allocated id.
    pub(crate) async fn send_request(
        &mut self,
        method: &str,
        params: Value,
        include_version: bool,
    ) -> Result<u64, ()> {
        self.next_id = self.next_id.saturating_add(1);
        let id = self.next_id;
        let envelope = if include_version {
            json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params })
        } else {
            json!({ "id": id, "method": method, "params": params })
        };
        self.write(&envelope).await?;
        Ok(id)
    }

    /// Writes a JSON-RPC reply for `id`, inserting `id` (and `jsonrpc: "2.0"`
    /// when `include_version` is set) into `answer`.
    pub(crate) async fn reply(
        &mut self,
        id: &Value,
        mut answer: Value,
        include_version: bool,
    ) -> Result<(), ()> {
        if let Some(obj) = answer.as_object_mut() {
            if include_version {
                obj.insert("jsonrpc".to_owned(), "2.0".into());
            }
            obj.insert("id".to_owned(), id.clone());
        }
        self.write(&answer).await
    }

    /// Reads the next JSON object line, skipping non-JSON lines.
    pub(crate) async fn next_object(&mut self) -> Result<Value, ()> {
        loop {
            let line = self.lines.next_line().await.ok().flatten().ok_or(())?;
            if let Ok(message) = serde_json::from_str::<Value>(&line)
                && message.is_object()
            {
                return Ok(message);
            }
        }
    }
}

/// Starts `command`, sends `requests` as NDJSON, and returns the first stdout
/// JSON object matching `predicate` within `timeout`.
pub(crate) fn probe(
    mut command: Command,
    requests: Vec<Value>,
    timeout: Duration,
    predicate: impl Fn(&Value) -> bool + Send + 'static,
) -> Result<Value, &'static str> {
    command
        .env("MISE_OFFLINE", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    let mut process = Process::start_with_stdio(command).map_err(|_| "could not start program")?;
    let (Some(mut stdin), Some(stdout)) = (process.take_stdin(), process.take_stdout()) else {
        return Err("missing stdio pipes");
    };
    let (tx, rx) = std::sync::mpsc::channel();
    let reader = std::thread::spawn(move || {
        for req in requests {
            if writeln!(stdin, "{req}").is_err() {
                return;
            }
        }
        if stdin.flush().is_err() {
            return;
        }
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            if let Ok(msg) = serde_json::from_str::<Value>(&line)
                && predicate(&msg)
            {
                let _ = tx.send(msg);
                return;
            }
        }
    });
    let response = rx.recv_timeout(timeout).ok();
    drop(process);
    let _ = reader.join();
    response.ok_or("timed out")
}
