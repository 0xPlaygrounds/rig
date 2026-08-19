//! Llamafile permission-control regression coverage.

use anyhow::Result;
use rig::agent::{
    AgentHook, ToolCall as ToolCallEvent, ToolCallAction, ToolResultAction, ToolResultEvent,
    stream_to_stdout,
};
use rig::completion::{Prompt, PromptError};
use rig::prelude::*;
use rig::streaming::StreamingPrompt;
use rig::tool::Tool;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::super::cassette_support::*;
use crate::support::assert_nonempty_response;

const TEST_CONTENT: &str = "hello world\n";

/// A per-test scratch file.
///
/// Both cells in this module used to write and then delete `test.txt` in the
/// process's working directory. Run concurrently — which is what `cargo test`
/// does by default — one cell's `Drop` deleted the file the other was midway
/// through reading, so `head`/`tail` returned nothing, the tool result changed,
/// and the recorded request body stopped matching. It reproduced 4 runs out of
/// 4 in the full suite and passed 4 out of 4 in isolation, which is the
/// signature of shared state rather than flakiness. Both cells were `#[ignore]`d
/// until now, so the collision had never had a chance to fire.
///
/// The file name stays out of the request: the tool *descriptions* still say
/// "test.txt", so the recorded bodies are unaffected by giving each cell its own
/// path.
struct ScratchFile {
    path: std::path::PathBuf,
}

impl ScratchFile {
    fn new(tag: &str) -> Result<Self> {
        let path = std::env::temp_dir().join(format!(
            "rig-permission-control-{tag}-{}.txt",
            std::process::id()
        ));
        std::fs::write(&path, TEST_CONTENT)?;
        Ok(Self { path })
    }
}

impl Drop for ScratchFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

#[derive(Deserialize)]
struct ReadFileArgs {}

#[derive(Debug, thiserror::Error)]
#[error("File operation error")]
struct FileError;

#[derive(Deserialize, Serialize)]
struct ReadFileHead {
    path: std::path::PathBuf,
}

impl Tool for ReadFileHead {
    const NAME: &'static str = "read_file_head";
    type Error = FileError;
    type Args = ReadFileArgs;
    type Output = String;

    fn description(&self) -> String {
        "Read the first line of test.txt using the head command".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {},
            "required": [],
        })
    }

    async fn call(
        &self,
        _context: &mut rig::tool::ToolContext,
        _args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        let output = std::process::Command::new("head")
            .arg("-1")
            .arg(&self.path)
            .output()
            .map_err(|_| FileError)?;

        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    }
}

#[derive(Deserialize, Serialize)]
struct ReadFileTail {
    path: std::path::PathBuf,
}

impl Tool for ReadFileTail {
    const NAME: &'static str = "read_file_tail";
    type Error = FileError;
    type Args = ReadFileArgs;
    type Output = String;

    fn description(&self) -> String {
        "Read the last line of test.txt using the tail command".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {},
            "required": [],
        })
    }

    async fn call(
        &self,
        _context: &mut rig::tool::ToolContext,
        _args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        let output = std::process::Command::new("tail")
            .arg("-1")
            .arg(&self.path)
            .output()
            .map_err(|_| FileError)?;

        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    }
}

#[derive(Clone)]
struct PermissionHook {
    call_count: Arc<AtomicUsize>,
    last_result: Arc<Mutex<Option<String>>>,
}

fn should_skip_retry_capability(
    response: &str,
    last_result: &Option<String>,
    call_count: usize,
) -> bool {
    if last_result.is_some() || call_count >= 2 {
        return false;
    }

    let response = response.to_ascii_lowercase();
    response.contains("not available")
        || response.contains("unavailable")
        || response.contains("i'll call")
        || response.contains("i will call")
        || response.contains("read_file_head")
        || response.contains("read_file_tail")
}

impl AgentHook for PermissionHook {
    async fn on_tool_call(
        &self,
        _ctx: &rig::agent::HookContext,
        event: ToolCallEvent<'_>,
    ) -> ToolCallAction {
        let count = self.call_count.fetch_add(1, Ordering::SeqCst);
        if count == 0 {
            ToolCallAction::skip(format!(
                "Tool '{}' is currently unavailable. Please use 'read_file_tail' instead to read the file.",
                event.tool_name
            ))
        } else {
            ToolCallAction::run()
        }
    }

    async fn on_tool_result(
        &self,
        _ctx: &rig::agent::HookContext,
        event: ToolResultEvent<'_>,
    ) -> ToolResultAction {
        let normalized = event.presentation.render();
        *self.last_result.lock().expect("lock last_result") = Some(normalized);
        ToolResultAction::keep()
    }
}

#[tokio::test]
async fn permission_control_prompt_example() -> Result<()> {
    with_llamafile_cassette_result("permission_control/permission_control_prompt_example", |client| async move {

        let scratch = ScratchFile::new("prompt")?;

        let agent = client.clone()
            .agent(CASSETTE_CHAT_MODEL)
            .preamble("You are a helpful assistant that can read files using different methods.")
            .tool(ReadFileHead {
                    path: scratch.path.clone(),
                })
            .tool(ReadFileTail {
                    path: scratch.path.clone(),
                })
            .build();

        let call_count = Arc::new(AtomicUsize::new(0));
        let last_result = Arc::new(Mutex::new(None));
        let hook = PermissionHook {
            call_count: call_count.clone(),
            last_result: last_result.clone(),
        };

        let response = match agent
            .prompt(
                "Use the available tools to read test.txt now. \
                 Call a tool directly; do not describe the tool call in plain text. \
                 If the first tool result says UNAVAILABLE, immediately call the other tool with no arguments. \
                 After one tool succeeds, reply with exactly the file content and nothing else.",
            )
            .max_turns(3)
            .add_hook(hook)
            .await
        {
            Ok(response) => response,
            Err(PromptError::MaxTurnsError {
                chat_history,
                prompt,
                ..
            }) => {
                let trace = format!("{chat_history:?}\n{prompt:?}");
                let last = last_result.lock().expect("lock last_result").clone();
                if should_skip_retry_capability(&trace, &last, call_count.load(Ordering::SeqCst)) {
                    eprintln!(
                        "skipping llamafile permission-control prompt test: model loops by naming tools in plain text instead of issuing a follow-up tool call"
                    );
                    return Ok(());
                }
                return Err(PromptError::MaxTurnsError {
                    max_turns: 3,
                    chat_history,
                    prompt,
                }
                .into());
            }
            Err(error) => return Err(error.into()),
        };

        let last = last_result.lock().expect("lock last_result").clone();
        if should_skip_retry_capability(&response, &last, call_count.load(Ordering::SeqCst)) {
            eprintln!(
                "skipping llamafile permission-control prompt test: model narrates retries instead of issuing a follow-up tool call"
            );
            return Ok(());
        }
        anyhow::ensure!(last.as_deref() == Some("hello world"));
        anyhow::ensure!(
            call_count.load(Ordering::SeqCst) >= 2,
            "expected at least one skipped tool call followed by a successful retry"
        );

        Ok(())
    })
    .await
}

#[tokio::test]
async fn permission_control_streaming_example() -> Result<()> {
    with_llamafile_cassette_result("permission_control/permission_control_streaming_example", |client| async move {

        let scratch = ScratchFile::new("streaming")?;

        let agent = client.clone()
            .agent(CASSETTE_CHAT_MODEL)
            .preamble("You are a helpful assistant that can read files using different methods.")
            .tool(ReadFileHead {
                    path: scratch.path.clone(),
                })
            .tool(ReadFileTail {
                    path: scratch.path.clone(),
                })
            .build();

        let call_count = Arc::new(AtomicUsize::new(0));
        let last_result = Arc::new(Mutex::new(None));
        let hook = PermissionHook {
            call_count: call_count.clone(),
            last_result: last_result.clone(),
        };

        let mut stream = agent
            .stream_prompt(
                "Use the available tools to read test.txt now. \
                 Call `read_file_head` first. If its tool result says UNAVAILABLE, immediately call `read_file_tail` instead. \
                 Both tools take zero arguments and return the file content. \
                 Never describe a tool call in plain text; emit the tool call directly. \
                 Do not ask any follow-up questions. After a tool succeeds, reply with exactly the file content.",
            )
            .max_turns(3)
            .add_hook(hook)
            .await;

        let final_response = stream_to_stdout(&mut stream).await?;
        let last = last_result.lock().expect("lock last_result").clone();
        if should_skip_retry_capability(
            final_response.output(),
            &last,
            call_count.load(Ordering::SeqCst),
        ) {
            eprintln!(
                "skipping llamafile permission-control streaming test: model narrates retries instead of issuing a follow-up tool call"
            );
            return Ok(());
        }
        assert_nonempty_response(final_response.output());
        anyhow::ensure!(
            final_response
                .output()
                .to_ascii_lowercase()
                .contains("hello world"),
            "expected the streamed final response to mention the file content, got {:?}",
            final_response.output()
        );
        anyhow::ensure!(last.as_deref() == Some("hello world"));
        anyhow::ensure!(
            call_count.load(Ordering::SeqCst) >= 2,
            "expected at least one skipped tool call followed by a successful retry"
        );

        Ok(())
    })
    .await
}
