//! Groq permission-control regression coverage.

use anyhow::Result;
use rig::agent::{ToolCallAction, ToolResultAction};
use rig::hooks::{HookDecision, HookEntry, HookEvent};
use rig::prelude::*;
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

impl PermissionHook {
    /// The hook record: vetoes the first tool call with a redirect reason, then
    /// records every tool result's normalized presentation.
    fn entry(&self) -> HookEntry {
        let hook = self.clone();
        HookEntry::sync("permission-control", move |event| hook.decide(event))
    }

    fn decide(&self, event: HookEvent) -> HookDecision {
        match event {
            HookEvent::ToolCall { call, .. } => {
                let count = self.call_count.fetch_add(1, Ordering::SeqCst);
                if count == 0 {
                    HookDecision::ToolCall(ToolCallAction::skip(format!(
                        "Tool '{}' is currently unavailable. Please use 'read_file_tail' instead to read the file.",
                        call.function.name
                    )))
                } else {
                    HookDecision::ToolCall(ToolCallAction::run())
                }
            }
            HookEvent::ToolResult { presentation, .. } => {
                let normalized = presentation.render();
                *self.last_result.lock().expect("lock last_result") = Some(normalized);
                HookDecision::ToolResult(ToolResultAction::keep())
            }
            _ => HookDecision::Continue,
        }
    }
}

#[tokio::test]
#[ignore = "requires GROQ_API_KEY"]
async fn permission_control_prompt_example() -> Result<()> {
    let _cleanup = FileCleanup::new()?;

    let agent = rig::providers::groq::Client::from_env()
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
        .runner(
            "Use the available tools to read test.txt now. \
             Do not ask any follow-up questions; just read the file and report its content.",
        )
        .max_turns(5)
        .add_hook(hook.entry())
        .run()
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

    let agent = rig::providers::groq::Client::from_env()
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

    let mut stream =agent
        .runner(
            "Use the available tools to read test.txt now. \
             Call `read_file_head` first. If it is unavailable, immediately call `read_file_tail` instead. \
             Both tools take zero arguments and return the file content. \
             Do not ask any follow-up questions. After a tool succeeds, reply with the exact file content.",
        )
        .max_turns(5)
        .add_hook(hook.entry())
        .stream_run();

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

/// Local stand-in for the deleted `rig::agent::stream_to_stdout` helper:
/// drains a streamed agent run to stdout and returns the final response.
async fn stream_to_stdout(
    stream: &mut rig::stream::AgentRunStream,
) -> Result<rig::agent::PromptResponse, std::io::Error> {
    use futures::StreamExt;
    use rig::message::Text;
    use rig::stream::AgentStreamItem;
    use rig::streaming::StreamedAssistantContent;

    let mut final_res = rig::agent::PromptResponse::empty();
    print!("Response: ");
    while let Some(content) = stream.next().await {
        match content {
            Ok(AgentStreamItem::Assistant(StreamedAssistantContent::Text(Text {
                text, ..
            }))) => {
                print!("{text}");
                std::io::Write::flush(&mut std::io::stdout())?;
            }
            Ok(AgentStreamItem::Assistant(StreamedAssistantContent::Reasoning(reasoning))) => {
                print!("{}", reasoning.display_text());
                std::io::Write::flush(&mut std::io::stdout())?;
            }
            Ok(AgentStreamItem::Final(res)) => final_res = res,
            Ok(AgentStreamItem::ModelTurnRetried { turn }) => {
                print!("\n[model turn {turn} rejected; retry requested]\nResponse: ");
                std::io::Write::flush(&mut std::io::stdout())?;
            }
            Err(err) => eprintln!("Error: {err}"),
            _ => {}
        }
    }

    Ok(final_res)
}
