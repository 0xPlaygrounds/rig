//! Groq permission-control regression coverage.

use anyhow::Result;
use rig::agent::{
    AgentHook, ToolCall as ToolCallEvent, ToolCallAction, ToolResultAction, ToolResultEvent,
    stream_to_stdout,
};
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::Prompt;
use rig::providers::groq;
use rig::streaming::StreamingPrompt;
use rig::tool::Tool;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::support::assert_nonempty_response;

use super::{PERMISSION_CONTROL_PROMPT_MODEL, PERMISSION_CONTROL_STREAMING_MODEL};

const TEST_FILE: &str = "test.txt";
const TEST_CONTENT: &str = "hello world\n";

struct FileCleanup;

impl FileCleanup {
    fn new() -> Result<Self> {
        std::fs::write(TEST_FILE, TEST_CONTENT)?;
        Ok(Self)
    }
}

impl Drop for FileCleanup {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(TEST_FILE);
    }
}

#[derive(Deserialize)]
struct ReadFileArgs {}

#[derive(Debug, thiserror::Error)]
#[error("File operation error")]
struct FileError;

#[derive(Deserialize, Serialize)]
struct ReadFileHead;

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

    async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
        let output = std::process::Command::new("head")
            .arg("-1")
            .arg(TEST_FILE)
            .output()
            .map_err(|_| FileError)?;

        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    }
}

#[derive(Deserialize, Serialize)]
struct ReadFileTail;

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

    async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
        let output = std::process::Command::new("tail")
            .arg("-1")
            .arg(TEST_FILE)
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
#[ignore = "requires GROQ_API_KEY"]
async fn permission_control_prompt_example() -> Result<()> {
    let _cleanup = FileCleanup::new()?;

    let agent = groq::Client::from_env()
        .expect("client should build")
        .agent(PERMISSION_CONTROL_PROMPT_MODEL)
        .preamble("You are a helpful assistant that can read files using different methods.")
        .tool(ReadFileHead)
        .tool(ReadFileTail)
        .build();

    let call_count = Arc::new(AtomicUsize::new(0));
    let last_result = Arc::new(Mutex::new(None));
    let hook = PermissionHook {
        call_count: call_count.clone(),
        last_result: last_result.clone(),
    };

    let _response = agent
        .prompt(
            "Use the available tools to read test.txt now. \
             Do not ask any follow-up questions; just read the file and report its content.",
        )
        .max_turns(5)
        .add_hook(hook)
        .await?;

    let last = last_result.lock().expect("lock last_result").clone();
    anyhow::ensure!(last.as_deref() == Some("hello world"));
    anyhow::ensure!(
        call_count.load(Ordering::SeqCst) >= 2,
        "expected at least one skipped tool call followed by a successful retry"
    );

    Ok(())
}

#[tokio::test]
#[ignore = "requires GROQ_API_KEY"]
async fn permission_control_streaming_example() -> Result<()> {
    let _cleanup = FileCleanup::new()?;

    let agent = groq::Client::from_env()
        .expect("client should build")
        .agent(PERMISSION_CONTROL_STREAMING_MODEL)
        .preamble("You are a helpful assistant that can read files using different methods.")
        .tool(ReadFileHead)
        .tool(ReadFileTail)
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
             Call `read_file_head` first. If it is unavailable, immediately call `read_file_tail` instead. \
             Both tools take zero arguments and return the file content. \
             Do not ask any follow-up questions. After a tool succeeds, reply with the exact file content.",
        )
        .max_turns(5)
        .add_hook(hook)
        .await;

    let final_response = stream_to_stdout(&mut stream).await?;
    let last = last_result.lock().expect("lock last_result").clone();
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
}
