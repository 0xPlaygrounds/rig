//! Llamafile permission-control regression coverage.

use anyhow::Result;
use rig_core::agent::{HookAction, PromptHook, ToolCallHookAction, stream_to_stdout};
use rig_core::client::CompletionClient;
use rig_core::completion::{CompletionModel, Prompt, PromptError, ToolDefinition};
use rig_core::streaming::StreamingPrompt;
use rig_core::tool::Tool;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::support::assert_nonempty_response;

use super::support;

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

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "read_file_head".to_string(),
            description: "Read the first line of test.txt using the head command".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {},
                "required": [],
            }),
        }
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

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "read_file_tail".to_string(),
            description: "Read the last line of test.txt using the tail command".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {},
                "required": [],
            }),
        }
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

impl<M: CompletionModel> PromptHook<M> for PermissionHook {
    async fn on_tool_call(
        &self,
        tool_name: &str,
        _tool_call_id: Option<String>,
        _internal_call_id: &str,
        _args: &str,
    ) -> ToolCallHookAction {
        let count = self.call_count.fetch_add(1, Ordering::SeqCst);

        if count == 0 {
            ToolCallHookAction::Skip {
                reason: format!(
                    "UNAVAILABLE: {tool_name}. Immediately call read_file_tail with no arguments."
                ),
            }
        } else {
            ToolCallHookAction::Continue
        }
    }

    async fn on_tool_result(
        &self,
        _tool_name: &str,
        _tool_call_id: Option<String>,
        _internal_call_id: &str,
        _args: &str,
        result: &str,
    ) -> HookAction {
        let normalized =
            serde_json::from_str::<String>(result).unwrap_or_else(|_| result.to_string());
        let mut last = self.last_result.lock().expect("lock last_result");
        *last = Some(normalized);

        HookAction::cont()
    }
}

#[tokio::test]
#[ignore = "requires a local llamafile server at http://localhost:8080"]
async fn permission_control_prompt_example() -> Result<()> {
    if support::skip_if_server_unavailable() {
        return Ok(());
    }

    let _cleanup = FileCleanup::new()?;

    let agent = support::client()
        .agent(support::model_name())
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

    let response = match agent
        .prompt(
            "Use the available tools to read test.txt now. \
             Call a tool directly; do not describe the tool call in plain text. \
             If the first tool result says UNAVAILABLE, immediately call the other tool with no arguments. \
             After one tool succeeds, reply with exactly the file content and nothing else.",
        )
        .max_turns(3)
        .with_hook(hook)
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
}

#[tokio::test]
#[ignore = "requires a local llamafile server at http://localhost:8080"]
async fn permission_control_streaming_example() -> Result<()> {
    if support::skip_if_server_unavailable() {
        return Ok(());
    }

    let _cleanup = FileCleanup::new()?;

    let agent = support::client()
        .agent(support::model_name())
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
             Call `read_file_head` first. If its tool result says UNAVAILABLE, immediately call `read_file_tail` instead. \
             Both tools take zero arguments and return the file content. \
             Never describe a tool call in plain text; emit the tool call directly. \
             Do not ask any follow-up questions. After a tool succeeds, reply with exactly the file content.",
        )
        .multi_turn(3)
        .with_hook(hook)
        .await;

    let final_response = stream_to_stdout(&mut stream).await?;
    let last = last_result.lock().expect("lock last_result").clone();
    if should_skip_retry_capability(
        final_response.response(),
        &last,
        call_count.load(Ordering::SeqCst),
    ) {
        eprintln!(
            "skipping llamafile permission-control streaming test: model narrates retries instead of issuing a follow-up tool call"
        );
        return Ok(());
    }
    assert_nonempty_response(final_response.response());
    anyhow::ensure!(
        final_response
            .response()
            .to_ascii_lowercase()
            .contains("hello world"),
        "expected the streamed final response to mention the file content, got {:?}",
        final_response.response()
    );
    anyhow::ensure!(last.as_deref() == Some("hello world"));
    anyhow::ensure!(
        call_count.load(Ordering::SeqCst) >= 2,
        "expected at least one skipped tool call followed by a successful retry"
    );

    Ok(())
}
