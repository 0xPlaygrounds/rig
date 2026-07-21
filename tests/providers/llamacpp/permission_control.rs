use anyhow::Result;
use rig::agent::{
    AgentHook, ToolCall as ToolCallEvent, ToolCallAction, ToolResultAction, ToolResultEvent,
};
use rig::client::CompletionClient;
use rig::completion::Prompt;
use rig::streaming::StreamingPrompt;
use rig::tool::Tool;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::support::{assert_nonempty_response, collect_stream_observation};

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
    ) -> Result<Self::Output, Self::Error> {
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
        })
    }

    async fn call(
        &self,
        _context: &mut rig::tool::ToolContext,
        _args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
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
#[ignore = "requires a local llama.cpp OpenAI-compatible server"]
async fn permission_control_prompt_example() -> Result<()> {
    let _cleanup = FileCleanup::new()?;

    let agent = support::completions_client()
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

    let response = agent
        .prompt(
            "Use the available tools to read test.txt now. \
             Do not ask any follow-up questions; just read the file and report its content.",
        )
        .max_turns(5)
        .add_hook(hook)
        .await?;

    assert_nonempty_response(&response);
    let last = last_result.lock().expect("lock last_result").clone();
    if let Some(last) = last {
        anyhow::ensure!(last == "hello world");
    }
    anyhow::ensure!(call_count.load(Ordering::SeqCst) >= 1);
    Ok(())
}

#[tokio::test]
#[ignore = "requires a local llama.cpp OpenAI-compatible server"]
async fn permission_control_streaming_example() -> Result<()> {
    let _cleanup = FileCleanup::new()?;

    let agent = support::completions_client()
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
             Do not ask any follow-up questions; just read the file and report its content.",
        )
        .max_turns(5)
        .add_hook(hook)
        .await;

    let observation = collect_stream_observation(&mut stream).await;
    anyhow::ensure!(
        observation.errors.is_empty(),
        "streaming permission control produced errors: {:?}",
        observation.errors
    );
    anyhow::ensure!(
        observation.got_final_response,
        "stream should yield a final response; events: {:?}",
        observation.events
    );
    anyhow::ensure!(
        observation.tool_results >= 1,
        "expected at least one streamed tool-result event, got {}; events: {:?}",
        observation.tool_results,
        observation.events
    );
    anyhow::ensure!(
        observation
            .tool_calls
            .iter()
            .any(|tool_name| tool_name == ReadFileHead::NAME),
        "expected stream to include the skipped read_file_head tool call, got {:?}",
        observation.tool_calls
    );

    let last = last_result.lock().expect("lock last_result").clone();
    let final_response = observation
        .final_response_text
        .as_deref()
        .unwrap_or_default();
    assert_nonempty_response(final_response);
    if let Some(last) = last {
        anyhow::ensure!(last == "hello world");
    }
    anyhow::ensure!(call_count.load(Ordering::SeqCst) >= 1);

    Ok(())
}
