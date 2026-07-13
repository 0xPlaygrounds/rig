//! xAI permission-control regression coverage.

use anyhow::Result;
use rig::agent::{
    AgentHook, ToolCall as ToolCallEvent, ToolCallAction, ToolResultAction, ToolResultEvent,
    stream_to_stdout,
};
use rig::client::CompletionClient;
use rig::completion::{CompletionModel, Prompt};
use rig::providers::xai;
use rig::streaming::StreamingPrompt;
use rig::tool::Tool;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::support::with_xai_cassette_result;
use crate::support::assert_nonempty_response;

const TEST_FILE: &str = "test.txt";
const TEST_CONTENT: &str = "hello world\n";
static PERMISSION_CONTROL_LOCK: LazyLock<tokio::sync::Mutex<()>> =
    LazyLock::new(|| tokio::sync::Mutex::new(()));

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
    type Args = ReadFileArgs;
    type Output = String;

    fn description(&self) -> String {
        "Read the first line of test.txt using the head command".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {},
        })
    }

    async fn call(
        &self,
        _context: &mut rig::tool::ToolContext,
        _args: Self::Args,
    ) -> Result<Self::Output, rig::tool::ToolExecutionError> {
        let output = std::process::Command::new("head")
            .arg("-1")
            .arg(TEST_FILE)
            .output()
            .map_err(|_| rig::tool::ToolExecutionError::from_error(FileError))?;

        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    }
}

#[derive(Deserialize, Serialize)]
struct ReadFileTail;

impl Tool for ReadFileTail {
    const NAME: &'static str = "read_file_tail";
    type Args = ReadFileArgs;
    type Output = String;

    fn description(&self) -> String {
        "Read the last line of test.txt using the tail command".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {},
        })
    }

    async fn call(
        &self,
        _context: &mut rig::tool::ToolContext,
        _args: Self::Args,
    ) -> Result<Self::Output, rig::tool::ToolExecutionError> {
        let output = std::process::Command::new("tail")
            .arg("-1")
            .arg(TEST_FILE)
            .output()
            .map_err(|_| rig::tool::ToolExecutionError::from_error(FileError))?;

        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    }
}

#[derive(Clone)]
struct PermissionHook {
    call_count: Arc<AtomicUsize>,
    last_result: Arc<Mutex<Option<String>>>,
}

impl<M: CompletionModel> AgentHook<M> for PermissionHook {
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
        let normalized = serde_json::from_str::<String>(event.result)
            .unwrap_or_else(|_| event.result.to_string());
        *self.last_result.lock().expect("lock last_result") = Some(normalized);
        ToolResultAction::keep()
    }
}

#[tokio::test]
async fn permission_control_prompt_example() -> Result<()> {
    with_xai_cassette_result(
        "permission_control/permission_control_prompt_example",
        |client| async move {
            let _guard = PERMISSION_CONTROL_LOCK.lock().await;
            let _cleanup = FileCleanup::new()?;

            let agent = client
                .agent(xai::GROK_4)
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
            anyhow::ensure!(call_count.load(Ordering::SeqCst) == 2);

            Ok(())
        },
    )
    .await
}

#[tokio::test]
async fn permission_control_streaming_example() -> Result<()> {
    with_xai_cassette_result(
        "permission_control/permission_control_streaming_example",
        |client| async move {
            let _guard = PERMISSION_CONTROL_LOCK.lock().await;
            let _cleanup = FileCleanup::new()?;

            let agent = client
                .agent(xai::GROK_4)
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
                     Do not ask any follow-up questions; just read the file and report its content.",
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
            anyhow::ensure!(call_count.load(Ordering::SeqCst) == 2);

            Ok(())
        },
    )
    .await
}
