//! The file-reading tools and the permission hook the permission-control
//! suites share. The hook skips the first tool call and lets later calls run,
//! so a suite proves the model recovers from a refused call by switching tools.

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use rig::agent::{
    AgentHook, DispatchAction, DispatchEvent, HookContext, OutcomeAction, OutcomeEvent,
};
use rig::tool::{Tool, ToolContext};
use serde::Deserialize;
use serde_json::json;

/// A file holding `hello world`, private to one test and deleted on drop.
///
/// Its path never reaches a request: the tool descriptions name `test.txt`.
pub struct ScratchFile {
    path: PathBuf,
}

impl ScratchFile {
    /// Writes a new scratch file under the system temp directory.
    pub fn new() -> anyhow::Result<Self> {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "rig-permission-control-{}-{}.txt",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(&path, "hello world\n")?;
        Ok(Self { path })
    }

    /// The `read_file_head` tool over this file.
    pub fn head(&self) -> ReadFileHead {
        ReadLine::new(self.path.clone())
    }

    /// The `read_file_tail` tool over this file.
    pub fn tail(&self) -> ReadFileTail {
        ReadLine::new(self.path.clone())
    }
}

impl Drop for ScratchFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// The tools take no arguments.
#[derive(Deserialize)]
pub struct ReadFileArgs {}

#[derive(Debug, thiserror::Error)]
#[error("File operation error")]
pub struct FileError;

/// Reads one line of a file with `head -1` or, when `TAIL`, `tail -1`.
pub struct ReadLine<const TAIL: bool> {
    path: PathBuf,
    empty_required: bool,
}

pub type ReadFileHead = ReadLine<false>;
pub type ReadFileTail = ReadLine<true>;

impl<const TAIL: bool> ReadLine<TAIL> {
    const COMMAND: &'static str = if TAIL { "tail" } else { "head" };

    fn new(path: PathBuf) -> Self {
        Self {
            path,
            empty_required: false,
        }
    }

    /// Adds `"required": []` to the parameter schema, as the Groq suite sends it.
    pub fn with_empty_required(self) -> Self {
        Self {
            empty_required: true,
            ..self
        }
    }
}

impl<const TAIL: bool> Tool for ReadLine<TAIL> {
    const NAME: &'static str = if TAIL {
        "read_file_tail"
    } else {
        "read_file_head"
    };
    type Error = FileError;
    type Args = ReadFileArgs;
    type Output = String;

    fn description(&self) -> String {
        let line = if TAIL { "last" } else { "first" };
        format!(
            "Read the {line} line of test.txt using the {} command",
            Self::COMMAND
        )
    }

    fn parameters(&self) -> serde_json::Value {
        if self.empty_required {
            json!({ "type": "object", "properties": {}, "required": [] })
        } else {
            json!({ "type": "object", "properties": {} })
        }
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        _args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        let output = std::process::Command::new(Self::COMMAND)
            .arg("-1")
            .arg(&self.path)
            .output()
            .map_err(|_| FileError)?;
        if !output.status.success() {
            return Err(FileError);
        }
        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    }
}

/// Skips the first tool call with a message pointing at `read_file_tail`,
/// lets every later call run, and records the last tool result.
#[derive(Clone, Default)]
pub struct PermissionHook {
    call_count: Arc<AtomicUsize>,
    last_result: Arc<Mutex<Option<String>>>,
}

impl PermissionHook {
    /// The number of tool calls dispatched so far, skipped ones included.
    pub fn calls(&self) -> usize {
        self.call_count.load(Ordering::SeqCst)
    }

    /// The rendered output of the last tool call that ran.
    pub fn last_result(&self) -> Option<String> {
        self.last_result.lock().ok().and_then(|last| last.clone())
    }
}

impl AgentHook for PermissionHook {
    async fn on_dispatch(&self, _ctx: &HookContext, event: DispatchEvent<'_>) -> DispatchAction {
        let Some(tool_name) = event.tool_name() else {
            return DispatchAction::proceed();
        };
        if self.call_count.fetch_add(1, Ordering::SeqCst) == 0 {
            DispatchAction::skip(format!(
                "Tool '{tool_name}' is currently unavailable. Please use 'read_file_tail' instead to read the file."
            ))
        } else {
            DispatchAction::proceed()
        }
    }

    async fn on_outcome(&self, _ctx: &HookContext, event: OutcomeEvent<'_>) -> OutcomeAction {
        let Some(result) = event.tool_result() else {
            return OutcomeAction::proceed();
        };
        if let Ok(mut last) = self.last_result.lock() {
            *last = Some(result.output().render());
        }
        OutcomeAction::proceed()
    }
}
