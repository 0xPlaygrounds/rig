//! DeepSeek permission-control regression coverage.

use anyhow::Result;
use rig::agent::{
    AgentHook, HookToolCall, HookToolResult, ToolCallAction, ToolResultAction, stream_to_stdout,
};
use rig::client::CompletionClient;
use rig::completion::{CompletionModel, Prompt};
use rig::providers::deepseek;
use rig::streaming::StreamingPrompt;
use rig::tool::Tool;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::support::with_deepseek_cassette_result;
use crate::support::assert_nonempty_response;

const TEST_CONTENT: &str = "hello world\n";

/// A per-test fixture file. The two tests in this module run in parallel
/// within one target, so a shared path would let one test's cleanup race the
/// other's tool executions. Only the local path is unique; everything on the
/// wire still says `test.txt`, so the cassettes are unaffected.
struct FileCleanup {
    path: std::path::PathBuf,
}

impl FileCleanup {
    fn new(label: &str) -> Result<Self> {
        let path = std::env::temp_dir().join(format!(
            "rig-deepseek-permission-{label}-{}.txt",
            std::process::id()
        ));
        std::fs::write(&path, TEST_CONTENT)?;
        Ok(Self { path })
    }
}

impl Drop for FileCleanup {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

#[derive(Deserialize)]
struct ReadFileArgs {}

#[derive(Deserialize, Serialize)]
struct ReadFileHead {
    #[serde(skip)]
    path: std::path::PathBuf,
}

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
            .arg(&self.path)
            .output()
            .map_err(|error| {
                rig::tool::ToolExecutionError::from_source(rig::tool::ToolErrorKind::Other, error)
            })?;

        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    }
}

#[derive(Deserialize, Serialize)]
struct ReadFileTail {
    #[serde(skip)]
    path: std::path::PathBuf,
}

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
            .arg(&self.path)
            .output()
            .map_err(|error| {
                rig::tool::ToolExecutionError::from_source(rig::tool::ToolErrorKind::Other, error)
            })?;

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
        event: HookToolCall<'_>,
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
        event: HookToolResult<'_>,
    ) -> ToolResultAction {
        let normalized = serde_json::from_str::<String>(event.result)
            .unwrap_or_else(|_| event.result.to_string());
        let mut last = self.last_result.lock().expect("lock last_result");
        *last = Some(normalized);
        ToolResultAction::keep()
    }
}

#[tokio::test]
async fn permission_control_prompt_example() -> Result<()> {
    with_deepseek_cassette_result(
        "permission_control/permission_control_prompt_example",
        |client| async move {
            let _cleanup = FileCleanup::new("prompt")?;

            let agent = client
                .agent(deepseek::DEEPSEEK_V4_FLASH)
                .preamble("You are a helpful assistant that can read files using different methods.")
                .tool(ReadFileHead {
                    path: _cleanup.path.clone(),
                })
                .tool(ReadFileTail {
                    path: _cleanup.path.clone(),
                })
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
            anyhow::ensure!(
                last.as_deref() == Some("hello world"),
                "expected final tool result hello world, got {last:?}"
            );
            anyhow::ensure!(
                call_count.load(Ordering::SeqCst) == 2,
                "expected two tool hook calls"
            );

            Ok(())
        },
    )
    .await
}

#[tokio::test]
async fn permission_control_streaming_example() -> Result<()> {
    with_deepseek_cassette_result(
        "permission_control/permission_control_streaming_example",
        |client| async move {
            let _cleanup = FileCleanup::new("streaming")?;

            let agent = client
                .agent(deepseek::DEEPSEEK_V4_FLASH)
                .preamble("You are a helpful assistant that can read files using different methods.")
                .tool(ReadFileHead {
                    path: _cleanup.path.clone(),
                })
                .tool(ReadFileTail {
                    path: _cleanup.path.clone(),
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
            anyhow::ensure!(
                last.as_deref() == Some("hello world"),
                "expected final tool result hello world, got {last:?}"
            );
            anyhow::ensure!(
                call_count.load(Ordering::SeqCst) == 2,
                "expected two tool hook calls"
            );

            Ok(())
        },
    )
    .await
}
